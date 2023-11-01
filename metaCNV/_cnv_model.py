import hmmlearn.base
import numpy as np
import metaCNV._nb_regression as nb
import metaCNV._entropy_model as entropy_model
from dataclasses import dataclass
from itertools import product
from patsy import dmatrix
from metaCNV._nb_regression import fit_NB_regression


def _get_design_matrix(regions):

    return dmatrix(
                'coverage + n_reads + entropy + mutation_rate + ' \
                'I(-ori_distance + 0.5) + standardize(np.log(gc_percent)) - 1', 
                data = regions,
                )

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
    
    ploidies = [0., 0.25, 0.5, 2/3, 1, 3/2, 2, 3, 4, 6, 8]
    stationary_states = [0.05, 0.05, 0.05,0.003125,0.75,0.003125, 0.05, 0.025, 0.0125, 0.003125,0.003125]

    assert len(ploidies) == len(stationary_states)
    #assert sum(stationary_states) == 1

    n_states = len(ploidies)

    pseudocounts = np.ones((n_states, n_states))
    pseudocounts[:,4] += 5
    prior = pseudocounts + np.eye(n_states)*(alpha * np.array(stationary_states))
    
    return TransitionMatrix(
        mat = prior/prior.sum(1, keepdims = True),
        ploidies=np.array(ploidies)
    )


def _make_transition_matrix(base_transition_matrix, n_strains = 1):

    def norm_ploidy(x):

        x = sum(x)
    
        return base_transition_matrix.ploidies[
            np.abs( np.log2(x) - np.log2(base_transition_matrix.ploidies * n_strains) ).argmin()
        ]

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
        ploidies = unique_ploidy_states,
        mat = transition_matrix/transition_matrix.sum(1,keepdims = True)
    )


@dataclass
class metaCNVFeatures:
    coverage: np.ndarray
    n_reads : np.ndarray
    entropy: np.ndarray
    mutation_rate : np.ndarray
    features: np.ndarray

    def __len__(self):
        return len(self.n_reads)
    

def _split_features(X, design_info, include_intercept = True):

    def left_pad(x):
        return np.hstack([np.ones((len(x),1)), x])

    colkeys = dict(zip(design_info.column_names, range(len(design_info.column_names))))
    
    special_cols = ['coverage','n_reads','entropy','mutation_rate']
    special = {
        col : X[:,colkeys[col]]
        for col in special_cols
    }
    
    features = X[:, [colkeys[col] for col in design_info.column_names if not col in special_cols]]

    if include_intercept:
        features = left_pad(features)

    return metaCNVFeatures(
        **special,
        features = features,
    )


def _loglikelihood_n_reads(
    ploidies,*,
    n_reads,
    features,
    beta,
    theta,
    eps = 0.01,
    ):

    # N_samples x N_states
    _lambda = np.exp(
        features @ beta[np.newaxis,:].T + np.log(ploidies)[np.newaxis,:]
    )

    # correct for ploidy = 0, or no exposure
    _lambda = np.where(ploidies == 0, eps, _lambda)
    
    ll = nb.model_log_likelihood(
        y = n_reads[:,None],
        _lambda = _lambda,
        theta = theta
    )

    null_mask = n_reads < 0
    ll[null_mask,:] = 0

    return ll


def _loglikelihood_entropy(
    ploidies,*,
    entropy,
    coverage,
    mutation_rate,
    n_strains,
    df = 30,
    ):

    loglikelihoods = []

    n_deletion_states = sum(ploidies*n_strains < 1)
    
    for rho in ploidies:
        
        if rho*n_strains >= 1:
            
            loglikelihoods.append(
                entropy_model.model_log_likelihood(
                        ploidy = np.ones_like(entropy)*np.maximum( rho*n_strains, 1), #cannot have an absolute ploidy less than 1
                        entropy = entropy,
                        coverage = coverage,
                        mutation_rate = mutation_rate,
                        df = df,
                    )[:,np.newaxis]
                )

    ll = np.hstack([np.repeat(loglikelihoods[0], n_deletion_states, axis = 1), *loglikelihoods])

    null_mask = entropy < 0
    ll[null_mask,:] = 0

    return ll



class HMMModel(hmmlearn.base.BaseHMM):

    @classmethod
    def get_hmm_model(cls, 
        X,*,
        n_strains,
        transmat,
        startprob,
        ploidies,
        training = True,
        **kw,
    ):

        init_kw = dict(
            n_strains = n_strains,
            ploidies = ploidies,
            n_components = len(ploidies), 
            algorithm='map',
            params = 'bd' if training else '',
            init_params = '',
            **kw,
        )

        def enrich_single_copy_regions(n_reads):
            median = np.median( n_reads[n_reads >= 0] )

            return np.logical_and(
                n_reads > 0,
                np.abs( np.log(n_reads) - np.log(median) ) < 5
            )
        
        cnvdata = _split_features(X, X.design_info)
        mask = enrich_single_copy_regions(cnvdata.n_reads)

        (beta, theta), _ = fit_NB_regression(
                y = cnvdata.n_reads[mask],
                exposure=np.ones_like(cnvdata.n_reads[mask]),
                features=cnvdata.features[mask],
                weights = np.ones_like(cnvdata.n_reads[mask]),
                fit_beta=True,
                fit_theta=True,
                init_theta = 1.,
            )

        model = cls(**init_kw)

        model.transmat_ = transmat
        model.startprob_ = startprob
        model.beta_ = beta
        model.theta_ = theta

        return model
    

    @staticmethod
    def fit_nb_component(*,
                         X, 
                         posterior_ploidies, 
                         ploidies, 
                         design_info,
                         beta = None, theta = None,
                         filter_function = None,
                         **kw,
                         ):
        
        ploidies = np.maximum(ploidies*np.exp(beta[0]), 0.01)
        
        cnv_data = _split_features(X, design_info, include_intercept=False)
        use_mask = cnv_data.n_reads >= 0 #np.logical_and( , expected_ploidy > 0 )

        if not filter_function is None:
            use_mask = np.logical_and(use_mask, filter_function(cnv_data.n_reads))

        posterior_ploidies = posterior_ploidies[use_mask]
        n_samples = use_mask.sum(); n_ploides = len(ploidies)

        def repeat_expand(x, M):
            return np.repeat( x[use_mask], M, axis = 0)

        features = repeat_expand(cnv_data.features, n_ploides)
        n_reads = repeat_expand(cnv_data.n_reads, n_ploides)

        exposure = np.tile(ploidies, n_samples)
        weights = posterior_ploidies.ravel()
        
        (new_coefs, theta), scores = nb.fit_NB_regression(
            y = n_reads,
            features = features,
            exposure = exposure,
            weights = weights,
            init_beta = beta[1:],
            init_theta = theta,
            **kw,
        )
        
        beta = np.array([beta[0], *new_coefs])
        return beta, theta


    def __init__(self,*,
                 ploidies,
                 coverage_weight = 0.8,
                 entropy_df = 20,
                 n_strains = 1,
                 n_components=1,
                 use_entropy = True,
                 use_coverage = True,
                 startprob_prior=1.0, 
                 transmat_prior=1.0,
                 algorithm="viterbi", 
                 random_state=None,
                 n_iter=25, 
                 tol=1e-2, 
                 verbose=False,
                 params='bdst',
                 init_params='bst',
                 implementation="log",
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
        self.use_entropy = use_entropy
        self.use_coverage = use_coverage
        self.coverage_weight = coverage_weight
        self.entropy_df = entropy_df
        

    def fit(self, X, lengths = None):
        self.design_info = X.design_info
        return super().fit(X, lengths = lengths)


    def split_features(self, X):
        return _split_features(X, self.design_info)
    

    def predict(self, X, lengths = None):

        _, states = self.decode(X, lengths=lengths)
        ploidies = self.ploidies[states]

        cnv_data = self.split_features(X)
        _lambda = np.exp(
                cnv_data.features @ self.beta_.T + np.log(ploidies)
            )

        # correct for ploidy = 0, or no exposure
        _lambda = np.where(ploidies == 0, 0., _lambda)

        return ploidies, _lambda
    

    def _compute_log_likelihood(self, X):
        
        cnv_data = self.split_features(X)

        ll = np.zeros(
            (len(X), self.n_components)
        )
        
        if self.use_coverage:
            coverage_ll = _loglikelihood_n_reads(
                self.ploidies,
                n_reads = cnv_data.n_reads,
                features = cnv_data.features,
                beta = self.beta_,
                theta = self.theta_,
            )
            coverage_ll = np.where(~np.isfinite(coverage_ll), 0, coverage_ll)

            ll = ll + self.coverage_weight*coverage_ll
        
        if self.use_entropy:

            entropy_ll = _loglikelihood_entropy(
                self.ploidies,
                entropy = cnv_data.entropy,
                coverage = np.maximum(cnv_data.coverage,2.),
                mutation_rate = cnv_data.mutation_rate,
                n_strains = self.n_strains,
                df = self.entropy_df,
            )
            entropy_ll = np.where(~np.isfinite(entropy_ll), 0, entropy_ll)
            ll = ll + (1-self.coverage_weight)*entropy_ll

        return ll
    
    
    def _initialize_sufficient_statistics(self):
        stats = super()._initialize_sufficient_statistics()
        stats['X'] = []
        stats['posteriors'] = []

        return stats


    def _accumulate_sufficient_statistics_log(
            self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
        super()._accumulate_sufficient_statistics_log(stats, X, lattice, posteriors, fwdlattice, bwdlattice)

        stats['X'].append(X)
        stats['posteriors'].append(posteriors)

    

    def _do_mstep(self, stats):
        super()._do_mstep(stats)

        posteriors = np.vstack(stats['posteriors'])
        X = np.vstack(stats['X'])

        fit_kw = dict(
            X = X,
            ploidies=self.ploidies,
            posterior_ploidies = posteriors,
            design_info=self.design_info,
            beta = self.beta_,
            theta = self.theta_,
        )

        if 'b' in self.params or 'd' in self.params:

            self.beta_, self.theta_ = self.fit_nb_component(
                    **fit_kw,
                    fit_theta = 'd' in self.params,
                    fit_beta = 'b' in self.params,
                )
            

    def _generate_sample_from_state(self, state, random_state=None):
        pass

    def _get_n_fit_scalars_per_param(self):
        
        return {
            's' : self.n_components - 1,
            't' : (self.transmat_ > 1e-5).sum() - self.n_components,
            'b' : 3,
            'd' : 1,
        }
