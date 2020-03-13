# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:19:52 2020

@author: Andrei
"""

from collections import Counter
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.utils.data
from torch_model_base import TorchModelBase
from torch_rnn_classifier import TorchRNNClassifier, TorchRNNClassifierModel
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier

from nltk.corpus import wordnet as wn
from nltk.tree import Tree

import nli
import os
import utils


###############################################################################
###############################################################################
###############################################################################
####                                                                       ####
####                       Utility code section                            ####
####                                                                       ####
###############################################################################
###############################################################################
###############################################################################

from datetime import datetime as dt
lst_log = []
log_fn = dt.now().strftime("logs/%Y%m%d_%H%M_log.txt")

def P(s=''):
  lst_log.append(s)
  print(s, flush=True)
  try:
    with open(log_fn, 'w') as f:
      for item in lst_log:
        f.write("{}\n".format(item))
  except:
    pass
  return

def Pr(s=''):
  print('\r' + str(s), end='', flush=True)
  
  
def prepare_grid_search(params_grid, nr_trials):
  import itertools

  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)

  params = []
  values = []
  for k in params_grid:
    params.append(k)
    assert type(params_grid[k]) is list, 'All grid-search params must be lists. Error: {}'.format(k)
    values.append(params_grid[k])
  combs = list(itertools.product(*values))
  n_options = len(combs)
  grid_iterations = []
  for i in range(n_options):
    comb = combs[i]
    func_kwargs = {}
    for j,k in enumerate(params):
      func_kwargs[k] = comb[j]
    grid_iterations.append(func_kwargs)
  idxs = np.arange(n_options)
  np.random.shuffle(idxs)
  idxs = idxs[:nr_trials]
  P("Generated {} random grid-search iters out of a total of iters".format(
      len(idxs), n_options))
  return [grid_iterations[i] for i in idxs]

        
  
###############################################################################
###############################################################################
###############################################################################
####                                                                       ####
####                      END utility code section                         ####
####                                                                       ####
###############################################################################
###############################################################################
###############################################################################


GLOVE_HOME = os.path.join('data', 'glove.6B')

DATA_HOME = os.path.join("data", "nlidata")

SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

MULTINLI_HOME = os.path.join(DATA_HOME, "multinli_1.0")

ANNOTATIONS_HOME = os.path.join(DATA_HOME, "multinli_1.0_annotations")


def wordnet_features(t1, t2, methodname):
  pairs = []
  words1 = t1.leaves()
  words2 = t2.leaves()
  for w1, w2 in product(words1, words2):
    hyps = [h for ss in wn.synsets(w1) for h in getattr(ss, methodname)()]
    syns = wn.synsets(w2)
    if set(hyps) & set(syns):
      pairs.append((w1,w2))
  return Counter(pairs)

def hypernym_features(t1, t2):
  return wordnet_features(t1,t2, 'hypernyms')

def hyponym_features(t1, t2):
  return wordnet_features(t1, t2,'hyponyms')


def glove_leaves_phi(t1, t2, np_func=np.sum):
  """Represent `tree` as a combination of the vector of its words.
  
  Parameters
  ----------
  t1 : nltk.Tree   
  t2 : nltk.Tree   
  np_func : function (default: np.sum)
      A numpy matrix operation that can be applied columnwise, 
      like `np.mean`, `np.sum`, or `np.prod`. The requirement is that 
      the function take `axis=0` as one of its arguments (to ensure
      columnwise combination) and that it return a vector of a 
      fixed length, no matter what the size of the tree is.
  
  Returns
  -------
  np.array
          
  """    
  prem_vecs = _get_tree_vecs(t1, glove_lookup, np_func)  
  hyp_vecs = _get_tree_vecs(t2, glove_lookup, np_func)  
  return np.concatenate((prem_vecs, hyp_vecs))
    
    
def _get_tree_vecs(tree, lookup, np_func):
  allvecs = np.array([lookup[w] for w in tree.leaves() if w in lookup])    
  if len(allvecs) == 0:
      dim = len(next(iter(lookup.values())))
      feats = np.zeros(dim)    
  else:       
      feats = np_func(allvecs, axis=0)      
  return feats

def fit_softmax_with_crossvalidation(X, y):
  """A MaxEnt model of dataset with hyperparameter cross-validation.
  
  Parameters
  ----------
  X : 2d np.array
      The matrix of features, one example per row.
      
  y : list
      The list of labels for rows in `X`.   
  
  Returns
  -------
  sklearn.linear_model.LogisticRegression
      A trained model instance, the best model found.
  
  """    
  basemod = LogisticRegression(
      fit_intercept=True, 
      solver='liblinear', 
      multi_class='auto')
  cv = 3
  param_grid = {'C': [0.4, 0.6, 0.8, 1.0],
                'penalty': ['l1','l2']}    
  best_mod = utils.fit_classifier_with_crossvalidation(
      X, y, basemod, cv, param_grid)
  return best_mod


def fit_shallow_neural_classifier_with_crossvalidation(X, y):    
  basemod = TorchShallowNeuralClassifier(max_iter=50)
  cv = 3
  param_grid = {'hidden_dim': [25, 50, 100]}
  best_mod = utils.fit_classifier_with_crossvalidation(
      X, y, basemod, cv, param_grid)
  return best_mod




class TorchRNNSentenceEncoderDataset(torch.utils.data.Dataset):
  def __init__(self, sequences, seq_lengths, y, prepare=False):
    if not prepare:
      self.prem_seqs, self.hyp_seqs = sequences
      self.prem_lengths, self.hyp_lengths = seq_lengths
    else:
      prems, hyps = zip(*sequences)
      self.prem_seqs = [x for x in prems]
      self.hyp_seqs = [x for x in hyps]
      self.prem_lengths = [len(x) for x in prems]
      self.hyp_lengths = [len(x) for x in hyps]
      if type(y[0]) == str:
        classes_ = sorted(set(y))
        n_classes_ = len(classes_)
        class2index = dict(zip(self.classes_, range(n_classes_)))
        y = [class2index[label] for label in y]
            
    self.y = y
    assert len(self.prem_seqs) == len(self.y)

  @staticmethod
  def collate_fn(batch):
    X_prem, X_hyp, prem_lengths, hyp_lengths, y = zip(*batch)
    prem_lengths = torch.LongTensor(prem_lengths)
    hyp_lengths = torch.LongTensor(hyp_lengths)
    y = torch.LongTensor(y)
    return (X_prem, X_hyp), (prem_lengths, hyp_lengths), y

  def __len__(self):
    return len(self.prem_seqs)

  def __getitem__(self, idx):
    return (self.prem_seqs[idx], self.hyp_seqs[idx],
            self.prem_lengths[idx], self.hyp_lengths[idx],
            self.y[idx])
      
        
        
class TorchRNNSentenceEncoderClassifierModel(TorchRNNClassifierModel):
  def __init__(self, vocab_size, embed_dim, embedding, use_embedding,
        hidden_dim, output_dim, bidirectional, device):
    super(TorchRNNSentenceEncoderClassifierModel, self).__init__(
        vocab_size, embed_dim, embedding, use_embedding,
        hidden_dim, output_dim, bidirectional, device)
    self.hypothesis_rnn = nn.LSTM(
        input_size=self.embed_dim,
        hidden_size=hidden_dim,
        batch_first=True,
        bidirectional=self.bidirectional)
    if bidirectional:
        classifier_dim = hidden_dim * 2 * 2
    else:
        classifier_dim = hidden_dim * 2
    self.classifier_layer = nn.Linear(
        classifier_dim, output_dim)

  def forward(self, X, seq_lengths):
    X_prem, X_hyp = X
    prem_lengths, hyp_lengths = seq_lengths
    prem_state = self.rnn_forward(X_prem, prem_lengths, self.rnn)
    hyp_state = self.rnn_forward(X_hyp, hyp_lengths, self.hypothesis_rnn)
    state = torch.cat((prem_state, hyp_state), dim=1)
    logits = self.classifier_layer(state)
    return logits        
  
  
class TorchRNNSentenceEncoderClassifier(TorchRNNClassifier):
  
  def build_dataset(self, X, y):
    X_prem, X_hyp = zip(*X)
    X_prem, prem_lengths = self._prepare_dataset(X_prem)
    X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
    return TorchRNNSentenceEncoderDataset(
        (X_prem, X_hyp), (prem_lengths, hyp_lengths), y)

  def build_graph(self):
    return TorchRNNSentenceEncoderClassifierModel(
        len(self.vocab),
        embedding=self.embedding,
        embed_dim=self.embed_dim,
        use_embedding=self.use_embedding,
        hidden_dim=self.hidden_dim,
        output_dim=self.n_classes_,
        bidirectional=self.bidirectional,
        device=self.device)

  def predict_proba(self, X):
    with torch.no_grad():
      X_prem, X_hyp = zip(*X)
      X_prem, prem_lengths = self._prepare_dataset(X_prem)
      X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
      preds = self.model((X_prem, X_hyp), (prem_lengths, hyp_lengths))
      preds = torch.softmax(preds, dim=1).cpu().numpy()
      return preds
        
if __name__ == '__main__':
  utils.fix_random_seeds()
  
  glove_lookup = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.50d.txt'))  
  
#  reader10 = nli.SNLITrainReader(SNLI_HOME, samp_percentage=0.10, random_state=42)
#  snli_iterator = iter(reader10.read())
#  snli_ex = next(snli_iterator)
#  
#  annotator_labels: list of str
#  captionID: str
#  gold_label: str
#  pairID: str
#  sentence1: str
#  sentence1_binary_parse: nltk.tree.Tree
#  sentence1_parse: nltk.tree.Tree
#  sentence2: str
#  sentence2_binary_parse: nltk.tree.Tree
#  sentence2_parse: nltk.tree.Tree  
#  print(snli_ex)
#  
#  t1 = Tree.fromstring("""(S (NP (D the) (N puppy)) (VP moved))""")
#  t2 = Tree.fromstring("""(S (NP (D the) (N dog)) (VP danced))""")
#  
#  h1 = hypernym_features(t1, t2)
  
  
  nli.experiment(
      train_reader=nli.SNLITrainReader(
          SNLI_HOME, samp_percentage=0.10, random_state=42), 
      phi=glove_leaves_phi,
      train_func=fit_shallow_neural_classifier_with_crossvalidation,
      assess_reader=nli.SNLIDevReader(SNLI_HOME),
      random_state=42,
      vectorize=False)  # Ask `experiment` not to featurize; we did it already.  
  
  
  