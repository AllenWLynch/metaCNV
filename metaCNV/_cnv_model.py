import hmmlearn.base
import numpy as np
import nb_regression as nb
import entropy_estimator as entropy_model
from dataclasses import dataclass
from itertools import product


@dataclass
class TransitionMatrix:
    ploidies: np.ndarray
    mat: np.ndarray

    @property
    def n_states(self):
        return len(self.ploidies)


def _get_stationary_state(Q):
    evals, evecs = np.linalg.eig(Q.T)
    evec1 = evecs[:,np.isclose(evals, 1)]
    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    return stationary.real


def _sparse_transition_matrix(alpha = 1e5):
    
    ploidies = [0., 0.25, 0.5, 1, 2, 3, 4, 6, 8]
    stationary_states = [0.05, 0.05, 0.05, 0.7375, 0.05, 0.025, 0.0125, 0.0125,0.0125]
    n_states = len(ploidies)

    pseudocounts = np.ones((n_states, n_states))
    pseudocounts[:,3] += 5
    prior = pseudocounts + np.eye(n_states)*(alpha * np.array(stationary_states))
    
    return TransitionMatrix(
        mat = prior/prior.sum(1, keepdims = True),
        ploidies=np.array(ploidies)
    )


def _make_transition_matrix(base_transition_matrix, n_strains = 1):

    def norm_ploidy(x):
        return round(sum(x)*2)/2

    pi = _get_stationary_state(base_transition_matrix.mat)
    original_ploidy_idx_map = dict(zip(base_transition_matrix.ploidies, range(base_transition_matrix.n_states)))

    hidden_states = list(product(base_transition_matrix.ploidies, repeat = n_strains))

    relative_ploidy = np.array(list(map(norm_ploidy, hidden_states)))
    unique_ploidy_states = np.unique(relative_ploidy)
    num_states = len(unique_ploidy_states)
    new_ploidy_idx_map = dict(zip(unique_ploidy_states, range(num_states)))
    
    transition_matrix = np.zeros((num_states, num_states))

    for hidden_state in hidden_states:

        hidden_state_ploidy = norm_ploidy(hidden_state)
        _from = new_ploidy_idx_map[hidden_state_ploidy]

        logp_hidden_state = np.sum([np.log(pi[original_ploidy_idx_map[hs_ploidy]]) for hs_ploidy in hidden_state])
        
        for to_hidden_state in hidden_states:
            to_hidden_state_ploidy = norm_ploidy(to_hidden_state)
            _to = new_ploidy_idx_map[to_hidden_state_ploidy]
            
            transition_matrix[_from, _to] += np.exp(
                logp_hidden_state + np.sum([
                    np.log(base_transition_matrix.mat[
                        original_ploidy_idx_map[from_hs_ploidy], 
                        original_ploidy_idx_map[to_hs_ploidy]
                        ]) 
                    for from_hs_ploidy, to_hs_ploidy in zip(hidden_state, to_hidden_state)
                ])
            )
    
    return TransitionMatrix(
        ploidies = unique_ploidy_states/n_strains,
        mat = transition_matrix/transition_matrix.sum(1,keepdims = True)
    )


@dataclass
class metaCNVFeatures:
    coverage: np.ndarray
    entropy: np.ndarray
    exposure: np.ndarray
    features: np.ndarray


# PTR effect is *offset*!!!!!!!!!
#dmatrix('coverage + entropy + offset + gc_percent')

def _split_features(X, design_info):

    colkeys = dict(zip(design_info.column_names, range(len(design_info.column_names))))
    
    special_cols = ['coverage','entropy','offset','mutation_rate']
    special = {
        col : X[:,colkeys[col]]
        for col in special_cols
    }
    
    features = X[:, [colkeys[col] for col in design_info.column_names if not col in special_cols]]

    return metaCNVFeatures(
        **special,
        features = features,
    )


def _loglikelihood_coverage(
    ploidies,*,
    coverage,
    offset,
    features,
    beta,
    theta,
    ):

    # N_samples x N_states
    _lambda = np.exp(
        features @ beta[np.newaxis,:].T + offset[:,np.newaxis] + np.log(ploidies)[np.newaxis,:]
    )
    
    return nb.model_log_likelihood(
        y = coverage[:,None],
        _lambda = _lambda,
        theta = theta
    )


def _loglikelihood_entropy(
    ploidies,*,
    entropy,
    coverage,
    mutation_rate,
    n_strains,
    ):

    loglikelihoods = []
    
    for rho in ploidies:
        
        loglikelihoods.append(
            entropy_model.loglikelihood(
                    ploidy = rho*n_strains,
                    entropy = entropy,
                    coverage = coverage,
                    mutation_rate = mutation_rate
                )[:,np.newaxis]
        )

    return np.vstack(loglikelihoods)



class HMMModel(hmmlearn.base.BaseHMM):

    @classmethod
    def get_hmm_model(cls,*,
        n_strains,
        beta, theta, 
        transmat,
        startprob,
        training = True,
        **kw,
    ):

        init_kw = dict(
            n_strains = n_strains,
            ploidies = transmat.ploidies,
            n_components = transmat.n_states, 
            algorithm='map',
            params = 'bd' if training else '',
            init_params = '',
            **kw,
        )

        model = cls(**init_kw)

        model.transmat_ = transmat
        model.startprob_ = startprob
        model.beta_ = beta
        model.theta_ = theta

        return model


    def __init__(self,*,
                 ploidies,
                 n_strains = 1,
                 n_components=1,
                 startprob_prior=1.0, 
                 transmat_prior=1.0,
                 algorithm="viterbi", 
                 random_state=None,
                 n_iter=10, 
                 tol=1e-2, 
                 verbose=False,
                 params='bdst',
                 init_params='bst',
                 implementation="log",
                 design_info = None
                ):

        super().__init__(n_components,
                         startprob_prior=startprob_prior,
                         transmat_prior=transmat_prior,
                         algorithm=algorithm,
                         random_state=random_state,
                         n_iter=n_iter, tol=tol, verbose=verbose,
                         params=params, init_params=init_params,
                         implementation=implementation)

        self.n_strains = n_strains
        self.ploidies = ploidies
        self.design_info = design_info
        

    def fit(self, X, lengths = None):
        self.design_info = X.design_info
        return super().fit(X, lengths = lengths)


    def split_features(self, X):
        return _split_features(X, self.design_info)
    

    def predict_coverage(self, X, lengths = None):

        _, states = self.decode(X, lengths=lengths)
        #cnv_data = self.split_features(X)
        return self.ploidies[states]
        
        '''return ploidies, predict(
                exposure = ploidies,
                features = features,
                beta = self.beta
            )'''


    def _compute_log_likelihood(self, X):
        
        cnv_data = self.split_features(X)

        coverage_ll = _loglikelihood_coverage(
            self.ploidies,
            coverage = cnv_data.coverage,
            offset = cnv_data.offset,
            features = cnv_data.features,
            beta = self.beta_,
            theta = self.theta_,
        )
        
        entropy_ll = _loglikelihood_entropy(
            self.ploidies,
            entropy = cnv_data.entropy,
            coverage = cnv_data.coverage,
            mutation_rate = cnv_data.mutation_rate,
            n_strains = self.n_strains,
        )

        return coverage_ll + entropy_ll
    
    
    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        if 'b' or 'd' in self.params:
            
            cnv_data = self.split_features(stats['X'])
            expected_ploidy = (stats['posteriors'] @ self.ploidies[:,None]).ravel()

            self.beta_, self.theta_ = nb.fit_NB_regression(
                y = cnv_data.coverage,
                features = cnv_data.features,
                exposure = expected_ploidy * np.exp(cnv_data.offset),
                init_beta = self.beta_,
                init_theta = self.theta_,
            )[0]
    

    def _generate_sample_from_state(self, state, random_state=None):
        pass

    def _get_n_fit_scalars_per_param(self):
        return {
            's' : self.n_components - 1,
            't' : self.n_components * (self.n_components - 1),
            'b' : 2,
            'd' : 1,
        }
