
'''
Thank you https://github.com/jasoncoding13/nmf for the code.
'''

import numpy as np
TOL = 1e-4

def random_init(n_components, X):
    """initialize dictionary and representation by random positive number.
    Args
        n_components: int.
        X: array with shape of [feature size, sample size].
    Rets
        D: initialized array with shape of [feature size, n_components].
        R: initialized array with shape of [n_components, sample size].
    """
    n_features, n_samples = X.shape
    avg = np.sqrt(X.mean() / n_components)
    rng = np.random.RandomState(13)
    D = avg * rng.randn(n_features, n_components)
    R = avg * rng.randn(n_components, n_samples)
    np.abs(D, out=D)
    np.abs(R, out=R)
    return D, R


class BaseNMF():
    """Base Class
    """
    def __init__(
            self,
            n_components,
            init,
            tol,
            max_iter,
            skip_iter):
        self.n_components = n_components
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.skip_iter = skip_iter

    def _compute_loss(self, X, D, R):
        return None

    def _update(self, X, D, R):
        return None

    def _init(self, X):
        if self.init == 'random':
            D, R = random_init(self.n_components, X)
        else:
            raise ValueError()
        return D, R

    def fit(self, X, mask = None):
        """
        Args
            X: array with shape of [feature size, sample size].
        Rets
            D: array with shape of [feature size, n_components].
            R: array with shape of [n_components, sample size].
        """
        if mask is None:
            mask = 1.
        
        D, R = self._init(X)
        losses = [self._compute_loss(X, D, R, mask)]
        for iter_ in range(self.max_iter):
            D, R = self._update(X, D, R, mask)
            # check converagence
            if iter_ % self.skip_iter == 0:
                losses.append(self._compute_loss(X, D, R, mask))
                criterion = abs(losses[-1] - losses[-2]) / losses[-2]
                print('iter-{:>4}, criterion-{:0<10.5}, {:>13}'.format(
                        iter_, criterion, losses[-1]))
                if criterion < TOL:
                    break
        return D, R


class WeightedNMF(BaseNMF):
    def __init__(
            self,
            n_components,
            init='random',
            tol=1e-4,
            max_iter=200,
            skip_iter=10):
        super().__init__(n_components, init, tol, max_iter, skip_iter)

    def _compute_loss(self, X, D, R):
        return None

    def _update_weight(self, X, D, R):
        return None

    def _update(self, X, D, R, mask):
        # update W
        W = self._update_weight(X, D, R, mask) * mask
        # update D
        denominator_D = (W * D.dot(R)).dot(R.T)
        denominator_D[denominator_D == 0] = np.finfo(np.float32).eps
        D = D * ((W * X).dot(R.T)) / denominator_D
        
        # update R
        denominator_R = D.T.dot(W * D.dot(R))
        denominator_R[denominator_R == 0] = np.finfo(np.float32).eps
        R = R * (D.T.dot(W * X)) / denominator_R
        R[-1,:] = 1.
        
        return D, R


class L21NMF_intercept(WeightedNMF):
    """L21-NMF
    """
    def _compute_loss(self, X, D, R, mask):
        return np.sum(np.sqrt(np.sum(mask * np.square(X - D.dot(R)), axis=0)))

    def _update_weight(self, X, D, R, mask):
        return 1/np.sqrt(np.sum(mask * np.square(X - D.dot(R)), axis=0))