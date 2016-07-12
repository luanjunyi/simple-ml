import numpy as np
import pdb
from scipy.optimize import fmin_l_bfgs_b
import scipy.sparse as sps
from ..common.math import dot
from ..common.math import softmax
from ..common.math import softmax_cross_entropy_loss

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
                 C = 1.0,
                 solver='gradient_descend',
                 learning_rate=1.0,
                 tol = 1e-4,
                 verbose = False,
                 sgd_epoch = 20):
        self.C_ = C
        self.solver_ = solver
        self.w_ = None
        self.sgd_epoch_ = sgd_epoch
        self.learning_rate_ = learning_rate
        self.verbose_ = verbose
        self.tol_ = tol
        self.y_dict = {}
        self.y_labels = []

    # We'll present as minimization problem here since fmin_l_bfgs_b always minimizes
    def fit(self, X, y):
        y = self.translate_y(y)
        if self.solver_ == 'sgd':
            self.solve_by_sgd(X, y)
        elif self.solver_ == 'gradient_descend':
            self.solve_by_gradient_descend(X, y)
        else:
            raise Exception("unsupported solver: %s" % self.solver_)

    def translate_y(self, y):
        self.y_labels = np.unique(y)
        assert len(self.y_labels) >= 2, "unique values of y is less than 2"
        self.y_dict = {v:i for i, v in enumerate(self.y_labels)}
        return np.array([self.y_dict[v] for v in y]).reshape(-1, 1)

    def loss(self, w, b, X, y):
        """
        Return the cross-entropy loss and gradient wrt w and b
        """
        z = dot(X, w) + b
        loss, grad = softmax_cross_entropy_loss(y, z)
        return loss, grad

    def regularization(self, w, n):
        loss = np.sum(w ** 2) / (2 * n)
        grad = w / n
        return loss, grad

    def solve_by_gradient_descend(self, X, y):
        n, d = X.shape
        m = len(self.y_labels)
        self.w_ = np.zeros([d, m])
        self.b_ = np.zeros(m)
        epoch = 0

        while True:
            loss, grad = self.loss(self.w_, self.b_, X, y)
            loss /= n
            grad /= n

            loss_reg, grad_reg_w = self.regularization(self.w_, X.shape[0])
            loss = self.C_ * loss + loss_reg

            dw = self.C_ * dot(X.T, grad) + grad_reg_w
            db = np.sum(grad, axis = 0)

            if epoch > 0:
                if self.verbose_:
                    print "epoch: %d, cost: %f, diff: %f" % (epoch, loss, prev_loss - loss)
                if prev_loss - loss < self.tol_:
                    break
                    prev_loss = loss
            self.w_ -= dw * self.learning_rate_
            self.b_ -= db * self.learning_rate_
            prev_loss = loss
            epoch += 1

    def predict_proba(self, X):
        z = dot(X, self.w_) + self.b_
        return softmax(z)

    def predict(self, X):
        p = self.predict_proba(X)
        return np.argmax(p, axis = 0)

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
                            solver = 'gradient_descend',
                            sgd_epoch = 5000)
    lr.fit(X, y)
    print lr.w_
    pred = lr.predict_proba(X)[:, 1]
    print " ".join(["%.2f" % p for p in pred])

    X_test = np.array(
        [[2, 5],
         [7, 90],
         [1, 130],
         [100, 500],
         [-400, -200],
         [80, 1],
         [9, -200],
         [-9, -100],
         [-50, -590],
         [89,40]])
    X_test = scaler.transform(X_test.astype(float))
    print " ".join(["%.2f" % p for p in lr.predict_proba(X_test)[:, 1]])
       
