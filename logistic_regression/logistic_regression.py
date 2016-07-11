import numpy as np
import pdb
from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse as sps

class LogisticRegression(object):
  """
  Implementation of Logistic Regression classifier.

  Parameters
  ----------

  C : float, optional (default = 1.0)
      Inverse of regularization strength; must be a positive float.
      Like in support vector machines, smaller values specify stronger
      regularization.
  """
  def __init__(self,
               C = 1.0, #Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
               solver='gradient_descend',
               learning_rate=1.0,
               sgd_epoch = 20):
    self.C_ = C
    self.solver_ = solver
    self.w_ = None
    self.sgd_epoch_ = sgd_epoch
    self.learning_rate_ = learning_rate
    self.tol_ = 0.0001

  # We'll present as minimization problem here since fmin_l_bfgs_b always minimizes
  def fit(self, X, y):
    self.labels = np.unique(y)
    assert len(self.labels) == 2
    y = np.where(y == self.labels[0], 0, 1)
    y = y.reshape(-1, 1)

    if self.solver_ == 'sgd':
      self.solve_by_sgd(X, y)
    elif self.solver_ == 'gradient_descend':
      self.solve_by_gradient_descend(X, y)
    else:
      raise Exception("unsupported solver: %s" % self.solver_)

  # X is the original input, [1, 1, ..., 1] has not been appended to
  # X yet
  def predict(self, X):
    n, m = X.shape
    p = self.predict_proba(X)
    return np.where( p >= 0.5, self.labels[1], self.labels[0])

  def predict_proba(self, X):
    return self.predict_with(self.w_, self.b_, X)

  def solve_by_gradient_descend(self, X, y):
    #pdb.set_trace()
    n, m = X.shape
    self.w_ = np.zeros(m).reshape(-1, 1)
    self.b_ = 0
    epoch = 0

    while True:
      p = self.predict_proba(X)
      c = self.cost_func(self.w_, self.b_, p, y)
      print "cost: %f" % c
      dw, db = self.derivative(self.w_, self.b_, X, y)
      if epoch > 0:
        print "epoch: %d, cost: %f, diff: %f" % (epoch, c, prev_cost - c)
        if prev_cost - c < self.tol_:
          break

      self.w_ -= dw * self.learning_rate_
      self.b_ -= db * self.learning_rate_
      prev_cost = c
      epoch += 1

 # [1, 1, ..., 1] has been appended to X
  def predict_with(self, w, b, X):
    dot = np.dot if not sps.issparse(X) else sps.csr_matrix.dot
    prod = dot(X, w) + b
    p = 1.0 / (1 + np.exp(-prod))
    # print 'prod : ', prod
    return p

  # The minus of log likelyhood of y given X, parametered by w
  def cost_func(self, w, b, p, y):
    n = len(y)
    p_eq = (p ** y) * ((1-p) ** (1-y))
    p_eq = np.where(p_eq == 0, 0.000001, p_eq)
    p_eq = np.where(p_eq == 1, 0.999999, p_eq)
    log_prob = sum(np.log(p_eq)) / n

    regularization = (sum(w ** 2) + b ** 2) / (2 * n * self.C_)
    print "loss: %f, regu: %f" % (-log_prob, regularization)
    ret = -log_prob + regularization
    # print "target: %.5f" % ret
    return ret

  # The partial deriviative of log likelyhood on w.
  def derivative(self, w, b, X, y):
    #pdb.set_trace()
    p = self.predict_with(w, b, X)
    n, m = X.shape
    if sps.issparse(X):
      t = sps.diags((p - y).flatten(), 0)
    else:
      t = (p - y)
    dw = ((t * X).sum(axis = 0) / n).reshape(-1, 1) + w / (self.C_ * n)
    db = np.sum(p - y) / n + b / (self.C_ * n)
    return dw, db

  # Technically, we are performing stachastic gradient ascend rather than descend,
  # but SGD is a far more common term
  def solve_by_sgd(self, X, y):
    n, m = X.shape
    self.w_ = np.zeros(m).reshape(-1, 1)
    self.b_ = 0
    epoch = 0
    a0 = self.learning_rate_
    for epoch in xrange(self.sgd_epoch_):
      cost = 0
      for i in xrange(n):
        xi = X[i]
        yi = y[i][0]
        #pdb.set_trace()
        pi = self.predict_with(self.w_, self.b_, xi)[0]
        if pi == 1:
          pi = 0.99999
        if pi == 0:
          pi = 0.00001
        alpha = a0 / (1 + epoch / a0)
        d = np.multiply(alpha, (yi - pi))
        dw = np.multiply(alpha, self.w_ / self.C_) # regularization
        self.w_ += (d * xi).reshape(-1, 1) - dw
        self.b_ += d
        cost += -(yi * np.log(pi) + (1 - yi) * np.log(1 - pi)) / n + \
                (np.sum(self.w_ ** 2) + self.b_ ** 2) / (2.0 * self.C_ * (n ** 2))
      print 'epoch %d, cost: %f' % (epoch, cost)
        

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
  #X = sps.csr_matrix(X)
  y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

  from sklearn.preprocessing import scale
  from sklearn.preprocessing import Normalizer
  
  scaler = Normalizer()
  X = scaler.fit_transform(X.astype(float))
  
  lr = LogisticRegression(C = 10,
                          learning_rate = 0.5,
                          solver = 'sgd',
                          sgd_epoch = 5000)
  lr.fit(X, y)
  print lr.w_
  pred = lr.predict_proba(X)
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
  print " ".join(["%.2f" % p for p in lr.predict_proba(X_test)])
       
