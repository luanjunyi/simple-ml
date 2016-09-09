from pprint import pprint
import numpy as np
import itertools, pdb, scipy, copy, sklearn
from sklearn.metrics import accuracy_score

class NeurualNetworkClassifier(object):
  def __init__(self,
               hidden_layer_sizes,
               activation = 'relu',
               cost_function = 'svm',
               learning_rate = 0.1,
               alpha = 1.0,
               sgd_batch_size = 10,
               sgd_epoch = 30):
    self.hidden_layer_sizes = hidden_layer_sizes
    self.hidden_layer_num = len(self.hidden_layer_sizes)
    self.tolerance = 1e-6
    self.learning_rate = learning_rate
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

  def sgd_fit(self, X, y):
    X, y = self.preprocess(X, y)
    self.initialize()
    n = len(y)

    for epoch in xrange(self.sgd_epoch):
      X, y = sklearn.utils.shuffle(X, y)
      cost = 0.0
      for k in xrange(0, n, self.sgd_batch_size):
        X_train = X[k:k+self.sgd_batch_size, :]
        y_train = y[k:k+self.sgd_batch_size]
        cost += self.train_on_batch(X_train, y_train, n)

      cost += self.alpha * self.regularization_cost()
      cost /= n

      yp = self.predict(X)
      print '[epoch %d], cost: %f, accur: %f' % (epoch, cost, accuracy_score(y, yp))
      if self.X_test is not None:
        yp = self.predict(self.X_test)
        print 'accuracy on testing set: %.4f' % accuracy_score(self.y_test, yp)


  def train_on_batch(self, X, y, total_train_size):
    delta_w, delta_b = self.zero_deltas(self.weights, self.biases)
    cost = 0.0
    n = len(y)
    for xi, yi in itertools.izip(X, y):
      # forward pass
      output = self.forward(xi)
      cost += self.cost(output, yi)
      if np.isnan(cost):
        pdb.set_trace()

      # backward pass
      dw, db = self.backward(xi, yi)
      self.add_delta(delta_w, delta_b, dw, db, 1.0 / n)

    rdw, rdb = self.regularization_derviative()
    self.add_delta(delta_w, delta_b, rdw, rdb, self.alpha / total_train_size)
    
    self.add_delta(self.weights, self.biases, delta_w, delta_b, self.learning_rate)
    return cost

  # The global(sane) gradient descend doesn't work for NN. So this
  # implementation totally fucks up
  def fit(self, X, y):
    X, y = self.preprocess(X, y)

    if self.should_init:
      self.initialize()

    if self.cost_function == 'mse':
      assert self.hidden_layer_sizes[-1] == len(self.labels)

    n = len(y)

    prev_cost = None
    epoch = 0
    while True:
      #pdb.set_trace()
      delta_w, delta_b = self.zero_deltas(self.weights, self.biases)
      total_cost = 0.0
      for xi, yi in itertools.izip(X, y):
        # forward pass
        output = self.forward(xi)
        cost = self.cost(output, yi)
        if np.isnan(cost):
          pdb.set_trace()
        total_cost += cost
        # backward pass
        dw, db = self.backward(xi, yi)
        self.add_delta(delta_w, delta_b, dw, db, 1.0 / n)

      cur_cost = total_cost / n
      cur_cost += self.alpha * self.regularization_cost() / n
      rdw, rdb = self.regularization_derviative()
      self.add_delta(delta_w, delta_b, rdw, rdb, self.alpha / n)

      if epoch > 0:
        yp = self.predict(X)
        accuracy = accuracy_score(y, yp)
        print "--------------epoch %d: cost %.4f(%f), accuracy: %.4f" % (epoch,
                                                                         cur_cost,
                                                                         (prev_cost if prev_cost else 0) - cur_cost,
                                                                         accuracy)
        if self.X_test is not None:
          yp = self.predict(self.X_test)
          print 'accuracy on testing set: %.4f' % accuracy_score(self.y_test, yp)

        if self.step_by_step:
          raw_input('press to continue...')

        if np.isnan(cur_cost) or prev_cost - cur_cost < self.tolerance:
          print 'cost is %f, improve is (%f)' % (cur_cost, prev_cost - cur_cost)
          cmd = raw_input('go on?')
          if cmd == 'pdb':
            pdb.set_trace()
            continue # usually something went wrong, like the cost increased, so we skip updating w and b.
          elif cmd == 'q':
            break
          else:
            try:
              r = float(cmd)
              self.learning_rate = r
              print 'set learning rate to %f' % r
            except Exception, err:
              pass
    
      # print "=========dw=========="
      # pprint(delta_w)
      # print "=========db=========="
      # pprint(delta_b)
      self.last_weights = copy.deepcopy(self.weights)
      self.last_biases = copy.deepcopy(self.biases)
      self.add_delta(self.weights, self.biases, delta_w, delta_b, self.learning_rate)
      # print "=========weights========"
      # pprint(self.weights)
      # print "=========biases========="
      # pprint(self.biases)
      prev_cost = cur_cost
      epoch += 1

    return self

  def predict(self, X):
    y = []
    for x in X:
      output = self.forward(x)
      idx = np.argmax(output)
      y.append(self.labels[idx])
    return y

  def predict_detail(self, X):
    output = []
    for x in X:
      output.append(self.forward(x))
    return output

  # Implementation
  def preprocess(self, X, y):
    X = X.astype(float)
    self.input_layer_size = X.shape[1]
    self.output_layer_size = len(np.unique(y))
    assert (self.output_layer_size == 1 or self.output_layer_size == len(np.unique(y)))
    y = self.load_labels(y)
    return X, y

  def initialize(self):
    self.weights = []
    self.biases = []
    prev_sz = self.input_layer_size
    for sz in self.hidden_layer_sizes:
      w = np.random.randn(sz, prev_sz) / np.sqrt(prev_sz)
      self.weights.append(w)
      self.biases.append(np.zeros(sz).reshape(-1, 1))
      prev_sz = sz

  def regularization_cost(self):
    return np.sum([np.sum(w * w) for w in self.weights]) / 2.0

  def regularization_derviative(self):
    dw = [np.copy(-w) for w in self.weights]
    db = [np.zeros(b.shape) for b in self.biases]
    return dw, db

  def assign_params(self, w, b):
    self.weights = copy.deepcopy(w)
    self.biases = copy.deepcopy(b)

  def copy_state(self):
    return {
      'weights': copy.deepcopy(self.weights),
      'biases': copy.deepcopy(self.biases),
      'neurons': copy.deepcopy(self.neurons),
    }

  def assign_state(self, state):
    self.weights = copy.deepcopy(state['weights'])
    self.biases = copy.deepcopy(state['biases'])
    self.neurons = copy.deepcopy(state['neurons'])

  def activate(self, x):
    if self.activation == "relu":
      return np.where(x > self.zero_tol, x, 0.0)
    elif self.activation == "leaky-relu":
      return np.where(x > self.zero_tol, x, self.leaky_const * x)
    elif self.activation == "sigmoid":
      x = np.where(x > 100, 100, x)
      x = np.where(x < -100, -100, x)
      return scipy.special.expit(x)
    else:
      raise Exception("unknown activation function(%s)" % self.activation)

  # derivative of activation function
  def sigma_prime(self, z, a):
    if self.activation == "relu":
      return np.where(z > self.zero_tol, 1.0, 0.0)
    elif self.activation == "leaky-relu":
      return np.where(z > self.zero_tol, 1, self.leaky_const)
    elif self.activation == "sigmoid":
      return a * (1 - a)
    else:
      raise Exception("unknown activation function(%s)" % self.activation)

  def cost(self, out, y):
    out = np.array(out)

    if self.cost_function == 'svm':
      c = out + 1.0 - out[y]
      c[y] = 0.0
      c = np.where(c > self.zero_tol, c, 0)
      return np.sum(c)
    elif self.cost_function == 'cross-entropy':
      t = np.zeros(len(out)).reshape(-1, 1)
      t[y] = 1
      out = np.where(out > self.zero_tol, out, self.zero_tol)
      out = np.where(out < 1, out, 1 - self.zero_tol)

      return -np.sum(t * np.log(out) + (1 - t) * np.log(1 - out))
    elif self.cost_function == 'softmax':
      return -np.log(np.exp(out[y]) / sum(np.exp(out)))
    elif self.cost_function == 'mse':
      t = np.zeros(len(out)).reshape(-1, 1)
      t[y] = 1
      return np.sum((t - out) ** 2)
    elif self.cost_function == 'mse-single':
      assert out.shape == (1, 1)
      out = sum(out[0])
      return (out - y) ** 2
    else:
      raise Exception("unknown cost function(%s)" % str(self.cost_function))

  def dc_da(self, a, y):
    dc_da = None
    if self.cost_function == 'svm':
      dc_da = np.where(1 + a - a[y] > self.zero_tol, 1, 0)
      t = np.where(1 + a - a[y] > self.zero_tol, -1, 0)
      t[y] = 0
      dc_da[y] = np.sum(t)
    elif self.cost_function == 'cross-entropy':
      t = np.zeros(len(a)).reshape(-1, 1)
      t[y] = 1
      a = np.where(a > self.zero_tol, a, self.zero_tol)
      a = np.where(a < 1, a, 1 - self.zero_tol)
      dc_da = t / a - (1 - t) / (1 - a)
    elif self.cost_function == 'softmax':
      l = sum(np.exp(a))
      p = np.exp(a) / l
      dc_da = p
      dc_da[y] = p[y] - 1
    elif self.cost_function == 'mse':
      t = np.zeros(len(a)).reshape(-1, 1)
      t[y] = 1
      dc_da = 2 * (a - t)
    elif self.cost_function == 'mse-single':
      assert a.shape == (1, 1)
      return 2 * (sum(a) - y)
    else:
      raise Exception("unknown cost function (%s)" % str(self.cost_function))

    return dc_da

  def forward(self, x):
    #import pdb; pdb.set_trace()
    self.neurons = []
    prev = x.reshape(-1, 1)
    for layer in xrange(self.hidden_layer_num):
      z = np.dot(self.weights[layer], prev) + self.biases[layer]
      a = self.activate(z)
      self.neurons.append([z, a])
      prev = a

    return prev

  def backward(self, x, y):
    if self.debug:
      pdb.set_trace()

    dw = []
    db = []
    delta_z = [None] * (self.hidden_layer_num)
    # equation 1: at output layer dc/dz = dc/da * da/dz
    # da/dz = sigma_prime(z) = 1(z > 0)
    # dc/da is complicated for LeRU
    z = self.neurons[-1][0]
    a = self.neurons[-1][1]

    if self.cost_function == 'cross-entropy' and self.activation == 'sigmoid':
      t = np.zeros(len(a)).reshape(-1, 1)
      t[y] = 1
      delta_z[-1] = t - a
    else:
      da_dz = self.sigma_prime(z, a)
      dc_da = self.dc_da(a, y)
      delta_z[-1] = (-dc_da) * da_dz
    # equation 2: dc/dz1 = [Wt (inner) dc/dz2] * da/dz1
    for i in xrange(self.hidden_layer_num - 2, -1, -1):
      z = self.neurons[i][0]
      a = self.neurons[i][1]
      da_dz = self.sigma_prime(z, a)
      delta_z[i] = np.dot(self.weights[i+1].T, delta_z[i+1]) * da_dz
    # equation 3: dc/db = dc/dz
    for i in xrange(0, self.hidden_layer_num):
      db.append(delta_z[i])
    # equation 4: dc/dw2 = a1 (outer) dc/dz2
    for i in xrange(0, self.hidden_layer_num):
      layer_in = x.reshape(-1, 1) if i == 0 else self.neurons[i-1][1]
      t = np.outer(delta_z[i], layer_in)
      dw.append(t)
    return dw, db

  def avg_cost(self, X, y):
    cost = 0.0
    for xi, yi in itertools.izip(X, y):
      output = self.forward(xi)
      ci = self.cost(output, yi)
      cost += ci
    return cost / len(y)

  def zero_deltas(self, weights, biases):
    return [np.zeros(w.shape) for w in weights], [np.zeros(b.shape) for b in biases]

  def add_delta(self, delta_w, delta_b, dw, db, rate = 1.0):
    assert len(delta_w) == len(dw), "delta_w and dw length didn't match"
    assert len(delta_b) == len(db), "delta_b and db length didn't match"
    for i in xrange(len(dw)):
      delta_w[i] = delta_w[i] + dw[i] * rate # can't use += here, will call __iadd__ 
                                             # which is unpredictable
    for i in xrange(len(db)):
      delta_b[i] = delta_b[i] + db[i] * rate

  def load_labels(self, y):
    t = list(set(y))
    d = dict(zip(t, range(len(t))))
    self.labels = t
    return np.array([d[yi] for yi in y])

  def numerical_derivative(self, X, y):
    dw = []
    db = []
    h = 1e-5

    for w in self.weights:
      m, n = w.shape
      dw_cur = np.zeros(w.shape)
      for i in xrange(m):
        for j in xrange(n):
          ori = w[i, j]
          w[i, j] = ori - h
          c0 = self.avg_cost(X, y)
          w[i, j] = ori + h
          c1 = self.avg_cost(X, y)
          w[i, j] = ori
          cur = (c1 - c0) / (2*h)
          dw_cur[i, j] = cur
      dw.append(dw_cur)

    for b in self.biases:
      db_cur = np.zeros(len(b))
      for i, b_cur in enumerate(b):
        ori = b_cur
        b[i] = ori - h
        c0 = self.avg_cost(X, y)
        b[i] = ori + h
        c1 = self.avg_cost(X, y)
        b[i] = ori
        cur = (c1 - c0) / (2*h)
        db_cur[i] = cur
      db.append(db_cur)

    return dw, db

  def gradient_check(self, X, y):
    self.input_layer_size = X.shape[1]
    self.output_layer_size = self.hidden_layer_sizes[-1]
    self.initialize()
    delta_w, delta_b = self.zero_deltas(self.weights, self.biases)
    for xi, yi in itertools.izip(X, y):
      # forward pass
      self.forward(xi)
      # backward pass
      dw, db = self.backward(xi, yi)
      self.add_delta(delta_w, delta_b, dw, db)

    num_dw, num_db = self.numerical_derivative(X, y)

    print "dw: \n", delta_w, "\n=========\n", num_dw
    print "db: \n", delta_b, "\n=========\n", num_db

    
    dw_error = [np.abs(dw - ndw) / np.maximum(dw, ndw) for dw, ndw in \
                itertools.izip(delta_w, num_dw) ]
    db_error = [np.abs(db - ndb) / np.maximum(db, ndb) for db, ndb in \
                itertools.izip(delta_b, num_db) ]
    print '==========================='
    pprint(dw_error)
    pprint(db_error)

  def go_back(self):
    self.weights = copy.deepcopy(self.last_weights)
    self.biases = copy.deepcopy(self.last_biases)

if __name__ == "__main__":
  nn = NeurualNetworkClassifier()
  X = np.array([[0,1],[1,0],[1,1],[0,0]])
  y = np.array([1, 1, 0, 0])
  nn.fit(X, y)
  print nn.predict(X), y
