from pprint import pprint
import numpy as np
import random, itertools, pdb, scipy, copy, sklearn
from sklearn.metrics import accuracy_score
from ..common.math import softmax
from ..common.math import softmax_cross_entropy_loss

'''
Only support two layer neural networks. Deeper networks should be using
Tensorflow. Only support cross entrophy loss for now. Activation is only ReLU.
'''
class NeurualNetworkClassifier(object):
  def __init__(self,
               hidden_layer_size,
               activation = 'relu',
               cost_function = 'cross_entrophy',
               learning_rate = 0.1,
               weight_std = 1.0,
               alpha = 1.0,
               sgd_batch_size = 64,
               sgd_epoch = 30,
               print_every = 10):
    self.hidden_layer_size = hidden_layer_size
    self.tolerance = 1e-6
    self.learning_rate = learning_rate
    self.weight_std = weight_std
    self.activation = activation
    self.cost_function = cost_function
    self.zero_tol = 1e-5
    self.leaky_const = 0.001
    self.alpha = alpha # L2 regularization term
    self.sgd_epoch = sgd_epoch
    self.sgd_batch_size = sgd_batch_size
    self.debug = False
    self.should_init = True
    self.step_by_step = False
    self.X_test = None
    self.print_every = print_every

  def fit(self, X, y):
    self.initialize(X, y)
    X, y = self.preprocess(X, y)

    n = len(y)

    indexes = range(n)
    for epoch in xrange(self.sgd_epoch):
      random.shuffle(indexes)
      X, y = sklearn.utils.shuffle(X, y)
      cost = 0.0
      for k in xrange(0, n, self.sgd_batch_size):
        cur_idx = indexes[k : k + self.sgd_batch_size]
        X_train = X[cur_idx, :]
        y_train = y[cur_idx]
        loss, grad_W1, grad_b1, grad_W2, grad_b2 = self.loss(X_train, y_train)
        cost += loss
        self.W1 = self.W1 - self.learning_rate * grad_W1
        self.W2 = self.W2 - self.learning_rate * grad_W2
        self.b1 = self.b1 - self.learning_rate * grad_b1
        self.b2 = self.b2 - self.learning_rate * grad_b2

      cost /= n

      #pdb.set_trace()
      yp = self.predict(X)

      if epoch % self.print_every == 0:
        print '[epoch %d], cost: %f, accur: %f' % (epoch, cost, accuracy_score(y, yp))
        if self.X_test is not None:
          yp = self.predict(self.X_test)
          print 'accuracy on testing set: %.4f' % accuracy_score(self.y_test, yp)

  def forward(self, X_train):
    mid1 = np.matmul(X_train, self.W1) + self.b1 # affine 1
    t1 = np.where(mid1 >= 0, mid1, 0) # relu 1
    mid2 = np.matmul(t1, self.W2) + self.b2 # affine 2 
    t2 = np.where(mid2 >= 0, mid2, 0) # relu 2

    return t2, {
      'input': X_train,
      't1': t1,
      'mid1': mid1,
      't2': t2,
      'mid2': mid2,
    }
    
  def backward(self, grad, cache):
    t1 = cache['t1']
    mid1 = cache['mid1']
    t2 = cache['t2']
    mid2 = cache['mid2']
    X = cache['input']

    d_mid2 = np.where(mid2 >= 0, grad, 0) # relu 2
    d_t1 = np.matmul(d_mid2, self.W2.T) # affine 2
    d_b2 = np.sum(d_mid2, axis = 0) # affine 2 bias
    d_W2 = np.matmul(t1.T, d_mid2) # affine 2 weights
    d_mid1 = np.where(mid1 >= 0, d_t1, 0) # relu 1
    d_b1 = np.sum(d_mid1, axis = 0)
    d_W1 = np.matmul(X.T, d_mid1)

    return d_W1, d_b1, d_W2, d_b2
    

  def loss(self, X_train, y_train):
    n = len(y_train)
    # forward pass
    t, cache = self.forward(X_train)
    # get loss
    loss, grad_t = softmax_cross_entropy_loss(y_train, t)
    loss /= n
    grad_t /= n
    dW1, db1, dW2, db2 = self.backward(grad_t, cache)
    # regularization
    loss += 0.5 * self.alpha / n * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
    dW1 += self.alpha * self.W1 / n
    dW2 += self.alpha * self.W2 / n

    return loss, dW1, db1, dW2, db2

  def predict(self, X):
    t, _ = self.forward(X)
    return [self.labels[i] for i in np.argmax(t, axis = 1)]

  def predict_detail(self, X):
    t, _ = self.forward(X)
    prob = softmax(t)
    return prob

  def preprocess(self, X, y):
    X = X.astype(float)
    y = self.load_labels(y)
    return X, y

  def initialize(self, X, y):
    self.input_layer_size = X.shape[1]
    self.output_layer_size = len(np.unique(y))
    self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size) * self.weight_std
    self.b1 = np.zeros([self.hidden_layer_size, ])
    self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size) * self.weight_std
    self.b2 = np.zeros([self.output_layer_size, ])

  def load_labels(self, y):
    t = list(set(y))
    d = dict(zip(t, range(len(t))))
    self.labels = t
    return np.array([d[yi] for yi in y])

if __name__ == "__main__":
  nn = NeurualNetworkClassifier(hidden_layer_size = 20,
                                learning_rate = 1,
                                alpha = 0.00,
                                weight_std = 0.1,
                                print_every = 1,
                                sgd_epoch = 500)
  X = np.array([[0,1],[1,0],[1,1],[0,0]])
  y = np.array([0, 0, 1, 1])
  nn.fit(X, y)
  print nn.predict(X), y
