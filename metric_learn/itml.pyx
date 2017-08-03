"""
Information Theoretic Metric Learning, Kulis et al., ICML 2007

ITML minimizes the differential relative entropy between two multivariate
Gaussians under constraints on the distance function,
which can be formulated into a Bregman optimization problem by minimizing the
LogDet divergence subject to linear constraints.
This algorithm can handle a wide variety of constraints and can optionally
incorporate a prior on the distance function.
Unlike some other methods, ITML does not rely on an eigenvalue computation
or semi-definite programming.

Adapted from Matlab code at http://www.cs.utexas.edu/users/pjain/itml/
"""

from __future__ import print_function, absolute_import
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array, check_X_y

from .base_metric import BaseMetricLearner
from .constraints import Constraints
from ._util import vector_norm

cimport cython
cimport numpy as np
from cython cimport floating
from libc.math cimport sqrt as sqrtd
from libc.math cimport fabs as fabsd

cdef extern from "<math.h>" nogil:
  float fabsf(float x)
  float sqrtf(float x)

ctypedef floating (*SQRT)(floating x) nogil
ctypedef floating (*ABS)(floating x) nogil


class ITML(BaseMetricLearner):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               A0=None, verbose=False):
    """Initialize ITML.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables

    max_iter : int, optional

    convergence_threshold : float, optional

    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity

    verbose : bool, optional
        if True, prints information while learning
    """
    self.gamma = gamma
    self.max_iter = max_iter
    self.convergence_threshold = convergence_threshold
    self.A0 = A0
    self.verbose = verbose

  def _process_inputs(self, X, constraints, bounds):
    self.X_ = X = check_array(X)
    # check to make sure that no two constrained vectors are identical
    a,b,c,d = constraints
    no_ident = vector_norm(X[a] - X[b]) > 1e-9
    a, b = a[no_ident], b[no_ident]
    no_ident = vector_norm(X[c] - X[d]) > 1e-9
    c, d = c[no_ident], d[no_ident]
    # init bounds
    if bounds is None:
      self.bounds_ = np.percentile(pairwise_distances(X), (5, 95))
    else:
      assert len(bounds) == 2
      self.bounds_ = bounds
    self.bounds_[self.bounds_==0] = 1e-9
    # init metric
    if self.A0 is None:
      self.L_ = np.identity(X.shape[1], dtype=X.dtype)
    else:
      self.L_ = np.linalg.cholesky(check_array(self.A0)).astype(X.dtype).T
    return a,b,c,d

  @cython.boundscheck(False)
  @cython.wraparound(False)
  @cython.cdivision(True)
  def fit(self, np.ndarray[floating, ndim=2] X, constraints, bounds=None):
    """Learn the ITML model.

    Parameters
    ----------
    X : (n x d) data matrix
        each row corresponds to a single instance
    constraints : 4-tuple of arrays
        (a,b,c,d) indices into X, with (a,b) specifying positive and (c,d)
        negative pairs
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    """
    cdef SQRT sqrt
    if floating is float:
      dtype = np.float32
      sqrt = sqrtf
    else:
      dtype = np.float64
      sqrt = sqrtd
    
    a,b,c,d = self._process_inputs(X, constraints, bounds)
    cdef floating gamma = <floating>self.gamma
    cdef int num_pos = len(a)
    cdef int num_neg = len(c)
    cdef np.ndarray[floating, ndim=1] _lambda = np.zeros(num_pos + num_neg, dtype=dtype)
    cdef np.ndarray[floating, ndim=1] lambdaold = np.zeros_like(_lambda)
    cdef floating gamma_proj = 1. if self.gamma is np.inf else gamma/(gamma+1.)
    cdef np.ndarray[floating, ndim=1] pos_bhat = np.zeros(num_pos, dtype=dtype) + self.bounds_[0]
    cdef np.ndarray[floating, ndim=1] neg_bhat = np.zeros(num_neg, dtype=dtype) + self.bounds_[1]
    cdef np.ndarray[floating, ndim=2] pos_vv = self.X_[a] - self.X_[b]
    cdef np.ndarray[floating, ndim=2] neg_vv = self.X_[c] - self.X_[d]
    cdef np.ndarray[floating, ndim=2] L = self.L_

    cdef np.ndarray[floating, ndim=1] Lv
    cdef np.ndarray[floating, ndim=1] Av
    cdef floating wtw
    cdef floating alpha
    cdef floating beta

    cdef int i
    cdef int it
    cdef int max_iter = self.max_iter
    for it in range(max_iter):
      # update positives
      for i in range(num_pos):
        Lv = L.dot(pos_vv[i])
        wtw = np.sum(Lv ** 2)  # scalar
        alpha = min(_lambda[i], gamma_proj*(1./wtw - 1./pos_bhat[i]))
        _lambda[i] -= alpha
        beta = alpha/(1 - alpha*wtw)
        pos_bhat[i] = 1./((1 / pos_bhat[i]) + (alpha / gamma))
        Av = L.T.dot(Lv)
        if beta >= 0:
          cholupdate(L, Av, sqrt(beta))
        else:
          choldowndate(L, Av, sqrt(-beta))

      # update negatives
      for i in range(num_neg):
        Lv = L.dot(neg_vv[i])
        wtw = np.sum(Lv ** 2)  # scalar
        alpha = min(_lambda[i+num_pos], gamma_proj*(1./neg_bhat[i] - 1./wtw))
        _lambda[i+num_pos] -= alpha
        beta = -alpha/(1 + alpha*wtw)
        neg_bhat[i] = 1./((1 / neg_bhat[i]) - (alpha / gamma))
        Av = L.T.dot(Lv)
        if beta >= 0:
          cholupdate(L, Av, sqrt(beta))
        else:
          choldowndate(L, Av, sqrt(-beta))

      normsum = np.linalg.norm(_lambda) + np.linalg.norm(lambdaold)
      if normsum == 0:
        conv = np.inf
        break
      conv = np.abs(lambdaold - _lambda).sum() / normsum
      if conv < self.convergence_threshold:
        break
      lambdaold = _lambda.copy()
      if self.verbose:
        print('itml iter: %d, conv = %f' % (it, conv))

    if self.verbose:
      print('itml converged at iter: %d, conv = %f' % (it, conv))
    self.n_iter_ = it
    return self

  def transformer(self):
    return self.L_


class ITML_Supervised(ITML):
  """Information Theoretic Metric Learning (ITML)"""
  def __init__(self, gamma=1., max_iter=1000, convergence_threshold=1e-3,
               num_labeled=np.inf, num_constraints=None, bounds=None, A0=None,
               verbose=False):
    """Initialize the learner.

    Parameters
    ----------
    gamma : float, optional
        value for slack variables
    max_iter : int, optional
    convergence_threshold : float, optional
    num_labeled : int, optional
        number of labels to preserve for training
    num_constraints: int, optional
        number of constraints to generate
    bounds : list (pos,neg) pairs, optional
        bounds on similarity, s.t. d(X[a],X[b]) < pos and d(X[c],X[d]) > neg
    A0 : (d x d) matrix, optional
        initial regularization matrix, defaults to identity
    verbose : bool, optional
        if True, prints information while learning
    """
    ITML.__init__(self, gamma=gamma, max_iter=max_iter,
                  convergence_threshold=convergence_threshold,
                  A0=A0, verbose=verbose)
    self.num_labeled = num_labeled
    self.num_constraints = num_constraints
    self.bounds = bounds

  def fit(self, X, y, random_state=np.random):
    """Create constraints from labels and learn the ITML model.

    Parameters
    ----------
    X : (n x d) matrix
        Input data, where each row corresponds to a single instance.

    y : (n) array-like
        Data labels.

    random_state : numpy.random.RandomState, optional
        If provided, controls random number generation.
    """
    X, y = check_X_y(X, y)
    num_constraints = self.num_constraints
    if num_constraints is None:
      num_classes = len(np.unique(y))
      num_constraints = 20 * num_classes**2

    c = Constraints.random_subset(y, self.num_labeled,
                                  random_state=random_state)
    pos_neg = c.positive_negative_pairs(num_constraints,
                                        random_state=random_state)
    return ITML.fit(self, X, pos_neg, bounds=self.bounds)



# Rank-1 Cholesky update implementation from https://github.com/jcrudy/choldate

@cython.cdivision(True)
cdef inline floating hypot(floating x,floating y):
  cdef floating t
  x = fabsf(x) if floating is float else fabsd(x)
  y = fabsf(y) if floating is float else fabsd(y)
  t = x if x < y else y
  x = x if x > y else y
  t = t/x
  return x*sqrtf(1+t*t) if floating is float else x*sqrtd(1+t*t)

@cython.cdivision(True)
cdef inline floating rypot(floating x,floating y):
  cdef floating t
  x = fabsf(x) if floating is float else fabsd(x)
  y = fabsf(y) if floating is float else fabsd(y)
  t = x if x < y else y
  x = x if x > y else y
  t = t/x
  return x*sqrtf(1-t*t) if floating is float else x*sqrtd(1-t*t)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cholupdate(np.ndarray[floating, ndim=2] R, np.ndarray[floating, ndim=1] x, floating scale):
  '''
  Update the upper triangular Cholesky factor R with the rank 1 addition
  implied by x such that:
  R_'R_ = R'R + outer(x,x)
  where R_ is the upper triangular Cholesky factor R after updating.  Note
  that both x and R are modified in place.
  '''
  cdef unsigned int p
  cdef unsigned int k
  cdef unsigned int i
  cdef floating r
  cdef floating c
  cdef floating s

  p = x.shape[0]
  
  if scale != 1:
    for k in range(p):
      x[k] *= scale
  
  for k in range(p):
    r = hypot(R[<unsigned int>k,<unsigned int>k], x[<unsigned int>k])
    c = r / R[<unsigned int>k,<unsigned int>k]
    s = x[<unsigned int>k] / R[<unsigned int>k,<unsigned int>k]
    R[<unsigned int>k,<unsigned int>k] = r
    for i in range(<unsigned int>(k+1),<unsigned int>p):
      R[<unsigned int>k,<unsigned int>i] = (R[<unsigned int>k,<unsigned int>i] + s*x[<unsigned int>i]) / c
      x[<unsigned int>i] = c * x[<unsigned int>i] - s * R[<unsigned int>k,<unsigned int>i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef choldowndate(np.ndarray[floating, ndim=2] R, np.ndarray[floating, ndim=1] x, floating scale):
  '''
  Update the upper triangular Cholesky factor R with the rank 1 subtraction
  implied by x such that:
  R_'R_ = R'R - outer(x,x)
  where R_ is the upper triangular Cholesky factor R after updating.  Note
  that both x and R are modified in place.
  '''
  cdef unsigned int p
  cdef unsigned int k
  cdef unsigned int i
  cdef floating r
  cdef floating c
  cdef floating s

  p = x.shape[0]
  
  if scale != 1:
    for k in range(p):
      x[k] *= scale
  
  for k in range(p):
    r = rypot(R[<unsigned int>k,<unsigned int>k], x[<unsigned int>k])
    c = r / R[<unsigned int>k,<unsigned int>k]
    s = x[<unsigned int>k] / R[<unsigned int>k,<unsigned int>k]
    R[<unsigned int>k,<unsigned int>k] = r
    for i in range(<unsigned int>(k+1),<unsigned int>p):
      R[<unsigned int>k,<unsigned int>i] = (R[<unsigned int>k,<unsigned int>i] - s*x[<unsigned int>i]) / c
      x[<unsigned int>i] = c * x[<unsigned int>i] - s * R[<unsigned int>k,<unsigned int>i]
