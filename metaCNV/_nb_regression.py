
from scipy.special import xlogy, gammaln
from scipy.optimize import minimize, Bounds
import numpy as np
from scipy.special import digamma


def model_log_likelihood(*,y, _lambda, theta):
        return xlogy(y, _lambda) - (y + theta)*np.log(_lambda + theta)\
                + gammaln(y + theta) - gammaln(theta) + theta*np.log(theta) - gammaln(y + 1)


def predict(*, exposure, features, beta):
    return np.exp( np.log(exposure)[:,np.newaxis] + features @ beta[np.newaxis,:].T )\
            .ravel()


def score(*, y, exposure, features, beta, theta, weights):
    
    weighted_ll = model_log_likelihood(
        y = y, 
        _lambda = predict(
            exposure = exposure, 
            features = features, 
            beta = beta
            ), 
        theta = theta
        ).T @ weights
    

    return weighted_ll.item()



def _update_NB_weights(*, y, exposure, features, theta, weights, 
                     init_beta = None
):
    
    y = y[:,np.newaxis]
    weights = weights[:,np.newaxis]
    exposure = exposure[:,np.newaxis]
    n_features = features.shape[1]

    def _objective_jac(params):

        beta = params[np.newaxis,:]
        
        log_lambda = np.log(exposure) + features @ beta.T
        _lambda = np.exp(log_lambda)
        
        # negative binomial likelihood without regularizers
        obj_val = ( y * log_lambda - (y + theta) * np.log(_lambda + theta) ).T @ weights
        
        # jacobian
        error = theta/(_lambda + theta) * (y - _lambda) * weights
        
        dL_dbeta = error.T @ features #- 2*beta
        
        jac = np.squeeze(dL_dbeta)
        
        return -obj_val, -jac
    
        
    def _hess(params):
        
        beta = params[np.newaxis,:]
        
        log_lambda = np.log(exposure) + features @ beta.T
        _lambda = np.exp(log_lambda)
        
        w = -theta * _lambda * (y + theta)/np.square(_lambda + theta) * weights
        
        hess = (w * features).T @ features #- 2
        
        return -hess
    
    
    if init_beta is None:
        init_beta = [0.]*n_features
    
    res = minimize(
            _objective_jac,
            init_beta,
            jac = True,
            method = 'newton-cg',
            hess = _hess,
        )
    
    return res.x


def _update_theta(y, exposure, features, beta, weights, 
                     init_theta = None
                 ):
    
    y = y[:,np.newaxis]
    weights = weights[:,np.newaxis]  
    log_lambda = ( np.log(exposure) + features @ beta.T )[:,np.newaxis]
    _lambda = np.exp(log_lambda)

    def _objective_jac(theta):
        
        # negative binomial likelihood with regularizers
        obj_val = ( y * log_lambda - (y + theta) * np.log(_lambda + theta)\
            + gammaln(y + theta) - gammaln(theta) + theta*np.log(theta) ).T @ weights

        dL_dtheta = ( digamma(y + theta) - digamma(theta) - (y + theta)/(_lambda + theta)\
                    - np.log(_lambda + theta) + (1 + np.log(theta)) ).T @ weights

        return -obj_val, -dL_dtheta

    
    if init_theta is None:
        init_theta = 1.
    
    res = minimize(
            _objective_jac,
            init_theta,
            jac = True,
            method = 'tnc',
            bounds = Bounds(1e-30, 1e5, keep_feasible=True)
         )

    return res.x.item()


def alpha_OLS(*,
    y, exposure, features, beta, weights, init_theta = None,
):
    _lambda = predict(exposure = exposure, features = features, beta = beta)

    squared_residuals = (np.square(y - _lambda) - _lambda)/_lambda

    alpha = ( (weights * _lambda) @ squared_residuals ) /(weights * np.square(_lambda)).sum()

    if alpha<0:
        return 1e5
        
    return 1/alpha



def fit_NB_regression(
    y, exposure, features, weights = None,
    init_beta = None, 
    init_theta = None,
    em_iters = 100, 
    loglike_tol = 1e-8,
    fit_theta = True,
    fit_beta = True,
):
    
    if weights is None:
        weights = np.ones_like(y)

    kwargs = dict(
        y = y,
        exposure = exposure,
        features = features,
        weights = weights,
    )

    if init_beta is None:
        beta = np.array([0.]*features.shape[1])
    else:
        beta = np.array(init_beta)

    if init_theta is None:
        theta = 1.
    else:
        theta = init_theta

    scores = [score(**kwargs, beta = beta, theta = theta)]
    
    if fit_beta and fit_theta:
        for _ in range(em_iters):
            
            beta = _update_NB_weights(**kwargs, init_beta = beta, theta = theta )

            theta = _update_theta(
                **kwargs, beta = beta, init_theta = theta
            )

            curr_score = score(**kwargs, beta = beta, theta = theta)
            if curr_score - scores[-1] < loglike_tol:
                break

            scores.append( curr_score )

    elif fit_beta:
        beta = _update_NB_weights(**kwargs, init_beta = beta, theta = theta )
        scores.append( score(**kwargs, beta = beta, theta = theta) )

    elif fit_theta:
        theta = _update_theta(**kwargs, beta = beta, init_theta = theta )
        scores.append( score(**kwargs, beta = beta, theta = theta) )
    else:
        raise ValueError('Must fit at least one parameter.')

    
    return (beta, theta), scores