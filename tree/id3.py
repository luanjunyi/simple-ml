from collections import defaultdict
import numpy as np
import pandas as pd

class ID3(object):
  def __init__(self):
    pass

  def fit(self, X, y):
    self.root_ = self.build_node(X, y, set(range(X.shape[1])), range(X.shape[0]))

  def predict_prob(self, X):
    return [self.predict_one_prob(self.root_, x) for x in X]

  def predict(self, X):
    return [self.predict_one(self.root_, x) for x in X]

  def predict_one_prob(self, node, x):
    if node.is_leave:
      return node.prob
    val = x[node.split_column]
    if val in node.children:
      return self.predict_one_prob(node.children[val], x)
    else:
      return node.prob

  def predict_one(self, node, x):
    prob = self.predict_one_prob(node, x)
    return sorted(prob.items(), key = lambda t: -t[1])[0][0]

  def build_node(self, X, y, column_candidates, record_indexes):
    #import pdb; pdb.set_trace()
    if len(record_indexes) == 0:
      raise Exception('record_indexes is empty')

    H = ID3.entropy(y[record_indexes])

    if len(column_candidates) == 0 or len(np.unique(y[record_indexes])) == 1:
      return Node.make_node(y[record_indexes], entropy=H, is_leave=True)

    ret = Node.make_node(y[record_indexes], entropy=H, is_leave=False)
    info_gain = {}
    column_entropy = {}
    gain_ratio = {}
    for col in column_candidates:
      split_indexes = self.split(X, record_indexes, col)
      cur_H = 0
      for index in split_indexes:
        cur_H += len(index) * 1.0 / len(record_indexes) * ID3.entropy(y[index])
      info_gain[col] = H - cur_H
      column_entropy[col] = ID3.entropy(X[record_indexes, col])
      gain_ratio[col] = 0 if column_entropy[col] == 0 else \
                        info_gain[col] / column_entropy[col]

    col, gain = sorted(gain_ratio.items(), key=lambda t: -t[1])[0]
    split_indexes = self.split(X, record_indexes, col)
    ret.children = {}
    ret.split_column = col
    ret.gain = gain

    #pdb.set_trace()
    for index in split_indexes:
      candidates_for_child = column_candidates.copy()
      candidates_for_child.remove(col)
      child = self.build_node(X, y, candidates_for_child, index)
      val = X[index[0]][col]
      ret.children[val] = child

    return ret

  def split(self, X, cur_index, col):
    value_idx = defaultdict(list)
    for i in cur_index:
      value_idx[X[i, col]].append(i)
    return value_idx.values()

  @staticmethod
  def entropy(y):
    y = pd.Series(y)
    n = len(y) * 1.0
    value_counts = y.value_counts()
    probs = value_counts.to_dict()
    for v in probs:
      probs[v] = probs[v] / n
    entropy = sum([-p * np.log2(p) for p in probs.values()])
    return entropy

class Node(object):
  def __init__(self, is_leave, entropy):
    self.entropy = entropy
    self.col = None
    self.children = None
    self.is_leave = is_leave
    self.prob = None
    self.split_column = None
    self.gain = None

  @staticmethod
  def make_node(y, entropy, is_leave=False):
    ret = Node(is_leave=True, entropy=entropy)
    ret.is_leave = is_leave
    n = float(len(y))
    value_counts = pd.Series(y).value_counts()
    prob = value_counts.to_dict()
    for v in prob:
      prob[v] = prob[v] / n
    ret.prob = prob
    return ret
