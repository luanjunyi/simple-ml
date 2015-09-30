import numpy as np
import pdb
from scipy.optimize import fmin_l_bfgs_b

class LogisticRegression(object):
  """
  Implementation of Logistic Regression classifier.

  Parameters
  ----------

  C : float, optional (default = 1.0)
      Inverse of regularization strength; must be a positive float.
      Like in support vector machines, smaller values specify stronger
      regularization.

  solver : str, 'lbfgs' or 'sgd' (default = 'lbfgs')
      fmin_l_bfgs_b from scipy is used to for lbfgs
      TODO: If solver is sgd, the argument X passed to fix can be a generator.

  sgd_epoch : int, optional (default = 10)
      The number of epoch used for sgd solver
  """
  def __init__(self,
               C = 1.0, #Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
               solver='lbfgs',
               sgd_epoch = 20):
    self.C_ = C
    self.solver_ = solver
    self.w_ = None
    self.sgd_epoch_ = sgd_epoch

  # We'll present as minimization problem here since fmin_l_bfgs_b always minimizes
  def fit(self, X, y):
    self.labels = np.unique(y)
    assert len(self.labels) == 2
    y = np.where(y == self.labels[0], 0, 1)

    if self.solver_ == 'lbfgs':
      self.w_ = self.solve_by_lbfgs(X, y)
    elif self.solver_ == 'sgd':
      self.w_ = self.solve_by_sgd(X, y)
    else:
      raise Exception("unsupported solver: %s" % self.solver_)

  # X is the original input, [1, 1, ..., 1] has not been appended to
  # X yet
  def predict(self, X):
    n, m = X.shape
    X = np.c_[X, [1] * n]
    p = self.predict_with(self.w_, X)
    return np.where( p >= 0.5, self.labels[1], self.labels[0])

  def predict_prob(self, X):
    n, m = X.shape
    X = np.c_[X, [1] * n]
    p = self.predict_with(self.w_, X)
    return p

 # [1, 1, ..., 1] has been appended to X
  def predict_with(self, w, X):
    prod = np.dot(X, w)
    p = 1.0 / (1 + np.exp(-prod))
    return p

  # The minus of log likelyhood of y given X, parametered by w
  def target_func(self, w, X, y):
    regularization = 1.0 / self.C_ * sum(w ** 2)
    prod = np.dot(X, w)
    p = 1.0 / (1 + np.exp(-prod))
    p_eq = (p ** y) * ((1-p) ** (1-y))
    log_prob = sum(np.log(p_eq))
    ret = -log_prob + regularization
    # print "target: %.5f" % ret
    return ret

  # The partial deriviative of log likelyhood on w.
  def derivative(self, w, X, y):
    p = self.predict_with(w, X)
    d = np.sum(X * (p - y)[:, np.newaxis], axis = 0) - 2.0 / self.C_ * w
    return d

  def solve_by_lbfgs(self, X, y):
    n, m = X.shape
    X = np.c_[X, [1] * n]
    n, m = X.shape
    w, _1, _2 = fmin_l_bfgs_b(self.fmin_l_bfgs_b_adapt(self.target_func),
                            fprime = self.fmin_l_bfgs_b_adapt(self.derivative),
                            x0 = np.array([0] * m),
                            args = (X, y))
    return w

  # Technically, we are performing stachastic gradient ascend rather than descend,
  # but SGD is a far more common term
  def solve_by_sgd(self, X, y):
    n, m = X.shape
    X = np.c_[X, [1] * n]
    n, m = X.shape
    w = np.zeros(m)
    a0 = 0.1
    t = 0
    for epoch in xrange(self.sgd_epoch_):
      for i in xrange(n):
        xi = X[i]
        yi = y[i]
        pi = self.predict_with(w, xi)
        alpha = a0 / (1 + 1.0 / self.C_ * a0 * t)
        delta = np.multiply(alpha, (yi - pi)) * xi
        w = w + delta
    return w

  def fmin_l_bfgs_b_adapt(self, func):
    def procedure(params, *args):
      X, y = args
      w = params
      return func(w, X, y)
    return procedure

if __name__ == "__main__":
  X = np.array(
       [[1, 7],
       [-7, -2],
       [19, 33],
       [100, 200],
       [-40, -2],

       [4, 2],
       [5, -1],
       [-8, -100],
       [-5, -30],
       [9,7]])
  y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

  from sklearn.preprocessing import scale
  from sklearn.preprocessing import Normalizer
  
  scaler = Normalizer()
  X = scaler.fit_transform(X.astype(float))
  
  lr = LogisticRegression(C = 1, solver = 'sgd')
  lr.fit(X, y)
  print lr.w_
  pred = lr.predict_prob(X)
  print " ".join(["%.2f" % p for p in pred])

  X_test = np.array(
    [[2, 5],
     [7, 90],
     [10, 13],
     [100, 500],
     [-400, -200],

     [8, 1],
     [9, -10],
     [-9, -10],
     [-50, -59],
     [89,40]])
  X_test = scaler.transform(X_test.astype(float))
  print " ".join(["%.2f" % p for p in lr.predict_prob(X_test)])
       
