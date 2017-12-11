import numpy as np
import scipy as sc
from scipy import io
from scipy.linalg import pinv2
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from abc import ABCMeta, abstractmethod
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.externals import six
import warnings
from nonlinearities import *


NONPARAMETRICAL_TRANSROMATIONS = ['linear', 'p2', 'p3', 'p4', 's2', 's3', 's4']


def _center_scale_xy(X, Y, scale=True):
    """ Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


def _nipals_twoblocks_inner_loop(X, Y, x_kind='linear', y_kind='linear', 
                                 x_params=None, y_params=None, max_iter=500, tol=1e-06,
                                 flag_first_iter=True, learning_rate=1.):
    """Inner loop of the iterative NIPALS algorithm.

    Provides an alternative to the svd(X'Y); returns the first left and right
    singular vectors of X'Y.  See PLS for the meaning of the parameters.  It is
    similar to the Power method for determining the eigenvectors and
    eigenvalues of a X'Y.
    """
# STEP 4
    x_score = f(X, kind=x_kind, params=x_params)[:, [0]]
# STEP 5
    y_score = f(Y, kind=y_kind, params=y_params)[:, [0]]
    x_weights_old = 0
    ite = 1
    eps = np.finfo(X.dtype).eps
    # Inner loop of the Wold algo.
# STEP 6
    while True:
# STEP 7        
        y_score_old = y_score.copy()
        x_score_old = x_score.copy()
# STEP 8        
        X_hat = f(X, kind=x_kind, params=x_params)
        Y_hat = f(Y, kind=y_kind, params=y_params)
        # 1.1 Update u: the X weights
        # Mode A regress each X column on y_score
# STEP 9
        x_weights = np.dot(X_hat.T, y_score) / np.dot(y_score.T, y_score)
        # If y_score only has zeros x_weights will only have zeros. In
        # this case add an epsilon to converge to a more acceptable
        # solution
        if np.dot(x_weights.T, x_weights) < eps:
            x_weights += eps
        # 1.2 Normalize u
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) + eps
        # 1.3 Update x_score: the X latent scores
# STEP 10
        x_score = np.dot(X_hat, x_weights)
# STEP 11
        if (x_kind not in NONPARAMETRICAL_TRANSROMATIONS) and flag_first_iter:
            J_t = jacob(X, x_score, kind=x_kind, params=x_params)
            delta_x_params = np.linalg.inv(J_t.T.dot(J_t)).dot(J_t.T).dot(x_score - x_score_old)[:, 0]
# STEP 12
            x_params += learning_rate * delta_x_params
        # 2.1 Update y_weights
        # Mode A regress each Y column on x_score
# STEP 13
        y_weights = np.dot(Y_hat.T, x_score) / np.dot(x_score.T, x_score)
        y_weights /= np.sqrt(np.dot(y_weights.T, y_weights)) + eps
        # 2.2 Update y_score: the Y latent scores
# STEP 14
        y_score = np.dot(Y_hat, y_weights)
# STEP 15
        if (y_kind not in NONPARAMETRICAL_TRANSROMATIONS) and flag_first_iter:
            J_u = jacob(Y, y_score, kind=y_kind, params=y_params)
            delta_y_params = np.linalg.inv(J_u.T.dot(J_u)).dot(J_u.T).dot(y_score - y_score_old)[:, 0]
# STEP 16
            y_params += learning_rate * delta_y_params
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol or Y.shape[1] == 1:
            break
        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached')
            break
        x_weights_old = x_weights
        ite += 1
    return x_weights, y_weights, ite


class _PLS(six.with_metaclass(ABCMeta), BaseEstimator, TransformerMixin,
           RegressorMixin):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm, constructors' parameters
    allow to obtain a specific implementation such as:

    - PLS2 regression, i.e., PLS 2 blocks, mode A, with asymmetric deflation
      and unnormalized y weights such as defined by [Tenenhaus 1998] p. 132.
      With univariate response it implements PLS1.

    We use the terminology defined by [Wegelin et al. 2000].
    This implementation uses the PLS Wold 2 blocks algorithm based on two
    nested loops:
        (i) The outer loop iterate over components.
        (ii) The inner loop estimates the weights vectors. This can be done
        with two algo. the inner loop of the original NIPALS algo.
        
    n_components : int, number of components to keep. (default 2).
    scale : boolean, scale data? (default True)
    max_iter : an integer, the maximum number of iterations (default 500)
        of the NIPALS inner loop (used only if algorithm="nipals")
    tol : non-negative real, default 1e-06
        The tolerance used in the iterative algorithm.
    copy : boolean, default True
        Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effects.

    Attributes
    ----------
    x_weights_ : array, [p, n_components] X block weights vectors.
    y_weights_ : array, [q, n_components] Y block weights vectors.
    x_loadings_ : array, [p, n_components] X block loadings vectors.
    y_loadings_ : array, [q, n_components] Y block loadings vectors.
    x_scores_ : array, [n_samples, n_components] X scores.
    y_scores_ : array, [n_samples, n_components] Y scores.
    x_rotations_ : array, [p, n_components] X block to latents rotations.
    y_rotations_ : array, [q, n_components] Y block to latents rotations.
    coef_ : array, [p, q] The coefficients of the linear model: ``Y = X coef_ + Err``
    n_iter_ : array-like Number of iterations of the NIPALS inner loop for each component.

    """

    @abstractmethod
    def __init__(self, n_components=2, scale=True,
                 x_kind="linear", y_kind='linear', max_iter=500, tol=1e-06,
                learning_rate = 1.):
        self.n_components = n_components
        self.scale = scale
        self.x_kind = x_kind
        self.y_kind = y_kind
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.
        """

        # copy since this will contains the residuals (deflated) matrices
        check_consistent_length(X, Y)
        X = check_array(X, dtype=np.float64, copy=True)
        Y = check_array(Y, dtype=np.float64, copy=True, ensure_2d=False)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        if self.n_components < 1 or self.n_components > p:
            raise ValueError('Invalid number of components: %d' %
                             self.n_components)
        # Scale (in place)
        X, Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale))
        # Residuals (deflated) matrices
        Xk = X.copy()
        Yk = Y.copy()
# STEP 1
        self.x_params = params_initialize(kind=self.x_kind)
        self.y_params = params_initialize(kind=self.y_kind)
        # Results matrices
# STEP 2
        self.x_scores_ = np.zeros((n, self.n_components))
        self.y_scores_ = np.zeros((n, self.n_components))
        self.x_weights_ = np.zeros((p, self.n_components))
        self.y_weights_ = np.zeros((q, self.n_components))
        self.x_loadings_ = np.zeros((p, self.n_components))
        self.y_loadings_ = np.zeros((q, self.n_components))
        self.n_iter_ = []

        # NIPALS algo: outer loop, over components
# STEP 3
        for k in range(self.n_components):
            if np.all(np.dot(Yk.T, Yk) < np.finfo(np.double).eps):
                # Yk constant
                warnings.warn('Y residual constant at iteration %s' % k)
                break
            # 1) weights estimation (inner loop)
            # -----------------------------------
# STEP 17
            x_weights, y_weights, n_iter_ = \
                _nipals_twoblocks_inner_loop(
                    X=Xk, Y=Yk, max_iter=self.max_iter,
                    tol=self.tol, x_kind=self.x_kind, y_kind=self.y_kind,
                    x_params=self.x_params, y_params=self.y_params, flag_first_iter=(k == 0),
                    learning_rate=self.learning_rate)
            self.n_iter_.append(n_iter_)
            # Forces sign stability of x_weights and y_weights
            # Sign undeterminacy issue from svd if algorithm == "svd"
            # and from platform dependent computation if algorithm == 'nipals'
            x_weights, y_weights = svd_flip(x_weights, y_weights.T)
            y_weights = y_weights.T
            # compute scores
            
            Xk_hat = f(Xk, kind=self.x_kind, params=self.x_params)
            Yk_hat = f(Yk, kind=self.y_kind, params=self.y_params)
            
            x_scores = np.dot(Xk_hat, x_weights)
            y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk_hat, y_weights) / y_ss
            # test for null variance
            if np.dot(x_scores.T, x_scores) < np.finfo(np.double).eps:
                warnings.warn('X scores are null at iteration %s' % k)
                break
            # 2) Deflation (in place)
            # ----------------------
            # Possible memory footprint reduction may done here: in order to
            # avoid the allocation of a data chunk for the rank-one
            # approximations matrix which is then subtracted to Xk, we suggest
            # to perform a column-wise deflation.
            #
# STEP 19
            x_loadings = np.dot(Xk_hat.T, x_scores) / np.dot(x_scores.T, x_scores)
            y_loadings = (np.dot(Yk_hat.T, x_scores)
                          / np.dot(x_scores.T, x_scores))
            # - regress Xk's on x_score
            # - subtract rank-one approximations to obtain remainder matrix
# STEP 22
            Xk_hat -= np.dot(x_scores, x_loadings.T)
            # - regress Yk's on x_score, then subtract rank-one approx.
# STEP 23
            Yk_hat -= np.dot(x_scores, y_loadings.T)
# STEP 24
            Xk = finv(Xk_hat, kind=self.x_kind, params=self.x_params)
            Yk = finv(Yk_hat, kind=self.y_kind, params=self.y_params)
            # 3) Store weights, scores and loadings # Notation:
            self.x_scores_[:, k] = x_scores.ravel()  # T
            self.y_scores_[:, k] = y_scores.ravel()  # U
            self.x_weights_[:, k] = x_weights.ravel()  # W
            self.y_weights_[:, k] = y_weights.ravel()  # C
            self.x_loadings_[:, k] = x_loadings.ravel()  # P
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q
        # Such that: X = TP' + Err and Y = UQ' + Err

        # 4) rotations from input space to transformed space (scores)
        # T = X W(P'W)^-1 = XW* (W* : p x k matrix)
        # U = Y C(Q'C)^-1 = YC* (C* : q x k matrix)
        self.x_rotations_ = np.dot(
            self.x_weights_,
            pinv2(np.dot(self.x_loadings_.T, self.x_weights_),
                  check_finite=False))
        if Y.shape[1] > 1:
            self.y_rotations_ = np.dot(
                self.y_weights_,
                pinv2(np.dot(self.y_loadings_.T, self.y_weights_),
                      check_finite=False))
        else:
            self.y_rotations_ = np.ones(1)

        # Estimate regression coefficient
        # Regress Y on T
        # Y = TQ' + Err,
        # Then express in function of X
        # Y = X W(P'W)^-1Q' + Err = XB + Err
        # => B = W*Q' (p x q)
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        # self.coef_ = self.coef_ * self.y_std_
        return self

    def transform(self, X, Y=None):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        Y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        check_is_fitted(self, 'x_mean_')
        X = check_array(X, dtype=FLOAT_DTYPES, copy=True)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        # Apply rotation
        x_scores = np.dot(X, self.x_rotations_)
        if Y is not None:
            Y = check_array(Y, ensure_2d=False, dtype=FLOAT_DTYPES, copy=True)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)
            Y -= self.y_mean_
            Y /= self.y_std_
            y_scores = np.dot(Y, self.y_rotations_)
            return x_scores, y_scores

        return x_scores

    def predict(self, X):
        """Apply the dimension reduction learned on the train data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        copy : boolean, default True
            Whether to copy X and Y, or perform in-place normalization.

        Notes
        -----
        This call requires the estimation of a p x q matrix, which may
        be an issue in high dimensional space.
        """
        check_is_fitted(self, 'x_mean_')
        X = check_array(X, dtype=FLOAT_DTYPES, copy=True)
        # Normalize
        X -= self.x_mean_
        X /= self.x_std_
        X_hat = f(X, kind=self.x_kind, params=self.x_params)
        Ypred_hat = np.dot(X_hat, self.coef_)
        Ypred = finv(Ypred_hat, kind=self.y_kind, params=self.y_params)
        return Ypred * self.y_std_ + self.y_mean_

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of predictors.

        y : array-like, shape = [n_samples, n_targets]
            Target vectors, where n_samples is the number of samples and
            n_targets is the number of response variables.

        Returns
        -------
        x_scores if Y is not given, (x_scores, y_scores) otherwise.
        """
        return self.fit(X, y).transform(X, y)


class PLSNonLinear(_PLS):
    """PLS regression

    PLSRegression implements the PLS 2 blocks regression known as PLS2 or PLS1
    in case of one dimensional response.
    This class inherits from _PLS with mode="A", deflation_mode="regression",
    norm_y_weights=False and algorithm="nipals".

    Parameters
    ----------
    n_components : int, (default 2) Number of components to keep.
    scale : boolean, (default True) whether to scale the data
    max_iter : an integer, (default 500) the maximum number of iterations of the NIPALS inner loop
    tol : non-negative real Tolerance used in the iterative algorithm default 1e-06.
    copy : boolean, default True Whether the deflation should be done on a copy. Let the default
        value to True unless you don't care about side effect

    Attributes
    ----------
    x_weights_ : array, [p, n_components] X block weights vectors.
    y_weights_ : array, [q, n_components] Y block weights vectors.
    x_loadings_ : array, [p, n_components] X block loadings vectors.
    y_loadings_ : array, [q, n_components] Y block loadings vectors.
    x_scores_ : array, [n_samples, n_components] X scores.
    y_scores_ : array, [n_samples, n_components] Y scores.
    x_rotations_ : array, [p, n_components] X block to latents rotations.
    y_rotations_ : array, [q, n_components] Y block to latents rotations.
    coef_ : array, [p, q] The coefficients of the linear model: ``Y = X coef_ + Err``
    n_iter_ : array-like Number of iterations of the NIPALS inner loop for each component.

    Notes
    -----
    Matrices::

        T: x_scores_
        U: y_scores_
        W: x_weights_
        C: y_weights_
        P: x_loadings_
        Q: y_loadings__

    Are computed such that::

        X = T P.T + Err and Y = U Q.T + Err
        T[:, k] = Xk W[:, k] for k in range(n_components)
        U[:, k] = Yk C[:, k] for k in range(n_components)
        x_rotations_ = W (P.T W)^(-1)
        y_rotations_ = C (Q.T C)^(-1)

    where Xk and Yk are residual matrices at iteration k.

    `Slides explaining
    PLS <http://www.eigenvector.com/Docs/Wise_pls_properties.pdf>`_


    For each component k, find weights u, v that optimizes:
    ``max corr(Xk u, Yk v) * std(Xk u) std(Yk u)``, such that ``|u| = 1``

    Note that it maximizes both the correlations between the scores and the
    intra-block variances.

    The residual matrix of X (Xk+1) block is obtained by the deflation on
    the current X score: x_score.

    The residual matrix of Y (Yk+1) block is obtained by deflation on the
    current X score. This performs the PLS regression known as PLS2. This
    mode is prediction oriented.
    """


    def __init__(self, n_components=2, scale=True, x_kind='linear', y_kind='linear', 
                 max_iter=300, tol=1e-06, learning_rate=1.):
        super(PLSNonLinear, self).__init__(
            n_components=n_components, scale=scale,
            max_iter=max_iter, tol=tol, x_kind=x_kind, y_kind=y_kind, 
            learning_rate=learning_rate)