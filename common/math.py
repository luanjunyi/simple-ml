import numpy as np
from scipy.sparse import issparse

def dot(a, b):
    """
    Dot multiply, support numpy ndarray and sparse array from scipy
    """
    if issparse(a) or issparse(b):
        return a * b
    else:
        return np.dot(a, b)

def softmax(z):
    """
    z could be a nxm matrix or nx1 vector or 1xn vector.
    In case z is a matrix, the softmax is calculated for each row
    """
    origin_shape = z.shape
    reshaped = False
    if len(z.shape) == 1 or z.shape[1] == 1:
        z = z.reshape([1, -1])
        reshaped = True
    t = np.amax(z, axis=1, keepdims=True)
    z = z - t
    e = np.exp(z)
    sm = e / np.sum(e, axis=1, keepdims=True)
    return sm.reshape(origin_shape) if reshaped else sm

def softmax_cross_entropy_loss(y, p):
    """
    y is the true labels of length n
    p is n x m with p[i, j] the score before softmax that row[i] is label[j]

    First apply softmax on p, then return the cross entropy loss and the derivatives
    of the loss wrt p
    """
    n, m = p.shape
    y = y.flatten()
    assert n == len(y), "length of y must equals p.shape[0], see the doc string of the function"
    prob = softmax(p)
    l = -np.log(np.maximum(prob, 1e-7)) # avoid cases when prob is 0, log on that will be -inf
    t = np.zeros_like(l)
    t[np.arange(n), y] = 1
    loss = np.sum(t * l)
    grad = -(t - prob)

    return loss, grad
