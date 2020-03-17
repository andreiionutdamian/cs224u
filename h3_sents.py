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

import torch as th
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
from functools import partial
import pickle
from sklearn.feature_extraction import DictVectorizer

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


def get_object_params(obj, n=None):
  """
  Parameters
  ----------
  obj : any type
    the inspected object.
  n : int, optional
    the number of params that are returned. The default is None
    (all params returned).
  Returns
  -------
  out_str : str
    the description of the object 'obj' in terms of parameters values.
  """
  
  out_str = obj.__class__.__name__+"("
  n_added_to_log = 0
  for _iter, (prop, value) in enumerate(vars(obj).items()):
    if type(value) in [int, float, bool]:
      out_str += prop+'='+str(value) + ','
      n_added_to_log += 1
    elif type(value) in [str]:
      out_str += prop+"='" + value + "',"
      n_added_to_log += 1
    
    if n is not None and n_added_to_log >= n:
      break
  #endfor
  
  out_str = out_str[:-1] if out_str[-1]==',' else out_str
  out_str += ')'
  return out_str  

  
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


def raw_data_load(reader_name):
  fn = "data/{}.npy".format(reader_name)
  P("Loading raw data '{}'...".format(fn))
  data = np.load(fn, allow_pickle=True)
  P("Loaded data {}".format(data.shape))
  return data

def raw_data_builder(reader):
  data_name = reader.__class__.__name__+"_{}".format(int(reader.data_size))
  data = []
  cnt = 0
  fn = "data/{}.npy".format(data_name)
  if os.path.isfile(fn):
    return raw_data_load(data_name)
  P("Running raw data builder '{}' using {}...".format(
      data_name, reader))
  for ex in reader.read():
    t1 = ex.sentence1_parse
    t2 = ex.sentence2_parse
    label = ex.gold_label
    data.append((t1, t2, label))
    cnt += 1
    if cnt % 10:
      Pr("Processed {:.1f}% examples so far...\t\t".format(
          cnt / reader.data_size * 100))
  np_arr = np.array(data)
  P("\nSaving {}...".format(fn))
  np.save(fn, np_arr)
  P("Raw data saved.")
  return data


def featurize_and_save(raw_data, phi, data_name, vectorize=True, vectorizer=None):
  feats, y = [] , []
  n_obs = len(raw_data)
  fn = "data/{}.pkl".format(data_name)
  P("Running featurizer '{}' on {} obs...".format(
      phi.__name__ if str(type(phi)) == 'function' else phi , n_obs))
  for i, ex in enumerate(raw_data):
    t1,t2, label = ex
    sents_feats = phi(t1, t2)
    feats.append(sents_feats)
    y.append(label)
    Pr("Processed/featurized {:.1f}% examples so far...\t\t".format((i+1)/n_obs*100))

  if vectorize:
    if vectorizer == None:
      P("\nVectorizing with new DictVectorizer...")        
      vectorizer = DictVectorizer(sparse=True)
      feat_matrix = vectorizer.fit_transform(feats)
      
    else:
      P("\nVectorizing using pre-trained vectorizer {} with {} feats...".format(
          vectorizer,
          len(vectorizer.feature_names_)))        
      feat_matrix = vectorizer.transform(feats)
    P("Vectorized using {} feats".format(len(vectorizer.feature_names_)))
  else:
    feat_matrix = feats
    P("")

  data = (feat_matrix, y), vectorizer
  P("Writing data to {}...".format(fn))
  with open(fn, 'wb') as fh:
    pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
  P("Done writing data")
  return data



def load_featurized_data(data_name):
  import pickle
  fn = "data/{}.pkl".format(data_name)
  if not os.path.isfile(fn):
    return None
  P("Loading data '{}'...".format(data_name))
  with open(fn, 'rb') as fh:
    (X,y), vectorizer = pickle.load(fh)
  P("Loaded X: {}, y: {}, vectorizer: {} ({} feats)".format(
      X.shape if hasattr(X,'shape') else len(X), 
      y.shape if hasattr(y,'shape') else len(y),
      vectorizer,
      len(vectorizer.feature_names_) if vectorizer else 0,
      ))
  
  return (X,y), vectorizer


def maybe_load_data(data_name):
  if data_name in globals():
    data = globals()[data_name]
    P("{} found in globals".format(data_name))
  else:
    data = load_featurized_data(data_name)
    P("{} {}already loaded in globals".format(data_name, "" if data else "NOT "))
  return data


def prepare_data(data_name, reader, phi, vectorize, vectorizer):
  results = maybe_load_data(data_name)
  if results is None:
    raw_data = raw_data_builder(
        reader=reader, 
        )        
    results = featurize_and_save(
        raw_data=raw_data, 
        phi=phi,
        data_name=data_name,
        vectorize=vectorize,
        vectorizer=vectorizer,
        )
  return results


def train_test_experiment(model, x_train, y_train, x_test, y_test,
                          score_func=utils.safe_macro_f1):
  P("Training with {}...".format(model.__class__.__name__))
  mod = model.fit(x_train, y_train)
  # Predictions:
  predictions = mod.predict(x_test)
  # Report:
  P(nli.classification_report(y_test, predictions, digits=3))
  # Return the overall score and experimental info:
  return score_func(y_test, predictions)
  


def word_cross_product_phi(t1, t2):
  """Basis for cross-product features. This tends to produce pretty 
  dense representations.
  
  Parameters
  ----------
  t1, t2 : `nltk.tree.Tree`
      As given by `str2tree`.
      
  Returns
  -------
  defaultdict
      Maps each (w1, w2) in the cross-product of `t1.leaves()` and 
      `t2.leaves()` to its count. This is a multi-set cross-product
      (repetitions matter).
  
  """
  return Counter([(w1, w2) for w1, w2 in product(t1.leaves(), t2.leaves())])

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


def baseline_softmax_cross_words(X,y):
  model = LogisticRegression(
      C = 0.4,
      penalty='l1',
      fit_intercept=True, 
      solver='liblinear', 
      multi_class='auto')
  model.fit(X,y)
  return model


def fit_shallow_neural_classifier_with_crossvalidation(X, y):    
  basemod = TorchShallowNeuralClassifier(max_iter=50)
  cv = 3
  param_grid = {'hidden_dim': [25, 50, 100]}
  best_mod = utils.fit_classifier_with_crossvalidation(
      X, y, basemod, cv, param_grid)
  return best_mod


def sentence_encoding_rnn_phi(t1, t2):
  """Map `t1` and `t2` to a pair of lits of leaf nodes."""
  return (t1.leaves(), t2.leaves())

def get_sentence_encoding_vocab(X, n_words=None):    
  wc = Counter([w for pair in X for ex in pair for w in ex])
  wc = wc.most_common(n_words) if n_words else wc.items()
  vocab = {w for w, c in wc}
  vocab.add("$UNK")
  return sorted(vocab)


def fit_sentence_encoding_rnn(X, y):   
  vocab = get_sentence_encoding_vocab(X, n_words=10000)
  mod = TorchRNNSentenceEncoderClassifier(
      vocab, 
      batch_size=64,
      hidden_dim=64, 
      bidirectional=False,
      max_iter=30,
      use_embedding=True)
  mod.fit(X, y)
  return mod    


class TorchRNNSentenceEncoderDataset(th.utils.data.Dataset):
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
    prem_lengths = th.LongTensor(prem_lengths)
    hyp_lengths = th.LongTensor(hyp_lengths)
    y = th.LongTensor(y)
    return (X_prem, X_hyp), (prem_lengths, hyp_lengths), y

  def __len__(self):
    return len(self.prem_seqs)

  def __getitem__(self, idx):
    return (self.prem_seqs[idx], self.hyp_seqs[idx],
            self.prem_lengths[idx], self.hyp_lengths[idx],
            self.y[idx])
      
        
        
class TorchRNNSentenceEncoderClassifierModel(TorchRNNClassifierModel):
  def __init__(self, vocab_size, embed_dim, embedding, use_embedding,
        hidden_dim, output_dim, bidirectional, device, dropout):
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
    if dropout > 0:
      self.dropout = th.nn.Dropout(dropout)
    else:
      self.dropout = None
    self.classifier_layer = nn.Linear(
        classifier_dim, output_dim)
    return
    

  def forward(self, X, seq_lengths):
    X_prem, X_hyp = X
    prem_lengths, hyp_lengths = seq_lengths
    prem_state = self.rnn_forward(X_prem, prem_lengths, self.rnn)
    hyp_state = self.rnn_forward(X_hyp, hyp_lengths, self.hypothesis_rnn)
    state = th.cat((prem_state, hyp_state), dim=1)
    if self.dropout is not None:
      state = self.dropout(state)
    logits = self.classifier_layer(state)
    return logits        
  
  
class TorchRNNSentenceEncoderClassifier(TorchRNNClassifier):
  
  def __init__(self, 
               vocab,
               embedding=None,
               use_embedding=True,
               embed_dim=50,
               bidirectional=False,
               dropout=0,
               **kwargs):
    self.dropout = dropout
    super().__init__(
        vocab=vocab, 
        embedding=embedding,
        use_embedding=use_embedding,
        embed_dim=embed_dim,
        bidirectional=bidirectional,
        **kwargs)
    return
  
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
        device=self.device,
        dropout=self.dropout)

  def predict_proba(self, X, verbose=False):
    self.model.eval()
    with th.no_grad():
      X_prem, X_hyp = zip(*X)
      X_prem, prem_lengths = self._prepare_dataset(X_prem)
      X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
      preds = self.model((X_prem, X_hyp), (prem_lengths, hyp_lengths))
      preds = th.softmax(preds, dim=1).cpu().numpy()

      X_prem = th.nn.utils.rnn.pad_sequence(X_prem, batch_first=True)
      X_hyp = th.nn.utils.rnn.pad_sequence(X_hyp, batch_first=True)
      all_preds = []
      b_size = 64
      n_batches = X_prem.shape[0] // b_size
      dl = th.utils.data.DataLoader(
          th.utils.data.TensorDataset(
              X_prem, X_hyp,
              prem_lengths, hyp_lengths),
          batch_size=b_size)
      for i_batch, (X_b_p, X_b_h, p_l_b, h_l_b) in enumerate(dl):
        X_batch = X_b_p, X_b_h
        seq_lengths_batch = p_l_b, h_l_b
        preds = self.model(X_batch, seq_lengths_batch)
        preds = th.softmax(preds, dim=1).cpu().numpy()
        if verbose:
          print("\r    Prediction {:.1f}%".format((i_batch) / n_batches * 100), 
                flush=True, end='')
        all_preds.append(preds)
      np_preds = np.concatenate(all_preds)
      if verbose:
        print("\rDone full prediction on {} observations".format(np_preds.shape[0]))
      return np_preds
    
    



def get_glove_sents(t1, t2):
  load_global_glove()
  assert "glove_lookup" in globals()
  s1 = t1.leaves()
  s2 = t2.leaves()
  sr1 = [glove_lookup[w] for w in s1 if w in glove_lookup]
  sr2 = [glove_lookup[w] for w in s2 if w in glove_lookup]
  
  if len(sr1) == 0:
    sr1 = [np.zeros(GLOVE_DIM)]
  if len(sr2) == 0:
    sr2 = [np.zeros(GLOVE_DIM)]
  return np.array(sr1).astype(np.float32), np.array(sr2).astype(np.float32)

def fit_rnn_glove_sents(X, y):
  model = TorchRNNSentenceEncoderClassifier(
      vocab={}, 
      batch_size=64,
      hidden_dim=64, 
      bidirectional=True,
      max_iter=30,
      use_embedding=False,
      dropout=0.5,
      )
  model.fit(X, y)
  return model



def load_global_glove():
  if "glove_lookup" not in globals():
    P("Loading GloVe-{}...".format(GLOVE_DIM))
    glove_lookup = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(GLOVE_DIM)))  
    P("GloVe-{} loaded.".format(GLOVE_DIM))
    globals()['glove_lookup'] = glove_lookup
  return
  




if __name__ == '__main__':
  utils.fix_random_seeds()
  GLOVE_DIM = 300
  SAMPLING = None # 0.1
  TRAIN_CROSS_BASELINE = False
  
  if TRAIN_CROSS_BASELINE:
    train_cross, cross_vectorizer = prepare_data(
        data_name='train_cross',
        reader=nli.SNLITrainReader(SNLI_HOME, samp_percentage=SAMPLING, random_state=42),
        phi=word_cross_product_phi,
        vectorize=True,
        vectorizer=None,
        )
      
    dev_cross, _ = prepare_data(
        data_name='dev_cross',
        reader=nli.SNLIDevReader(SNLI_HOME),
        phi=word_cross_product_phi,
        vectorize=True,
        vectorizer=cross_vectorizer,
        )
  
    P("=" * 70)
    P("Running baseline log-reg with word cross...")
    """
    Training with LogisticRegression...
                   precision    recall  f1-score   support
    
    contradiction      0.784     0.765     0.774      3278
       entailment      0.742     0.812     0.775      3329
          neutral      0.725     0.672     0.697      3235
    
         accuracy                          0.750      9842
        macro avg      0.750     0.750     0.749      9842
     weighted avg      0.750     0.750     0.749      9842  
    """
    base1 = train_test_experiment(
        model=LogisticRegression(
            C = 0.4,
            penalty='l1',
            fit_intercept=True, 
            solver='liblinear', 
            multi_class='auto'
        ), 
        x_train=train_cross[0], 
        y_train=train_cross[1], 
        x_test=dev_cross[0], 
        y_test=dev_cross[1],    
        score_func=utils.safe_macro_f1,
        )
    

  train_glove_sents, _ = prepare_data(
      data_name='train_glove{}_sents'.format(GLOVE_DIM),
      reader=nli.SNLITrainReader(SNLI_HOME, samp_percentage=SAMPLING, random_state=42),
      phi=get_glove_sents,
      vectorize=False,
      vectorizer=None,
      )


  dev_glove_sents, _ = prepare_data(
      data_name='dev_glove{}_sents'.format(GLOVE_DIM),
      reader=nli.SNLIDevReader(SNLI_HOME),
      phi=get_glove_sents,
      vectorize=False,
      vectorizer=None,
      )
  
  if "glove_lookup" in globals():
    del glove_lookup
  

  P("=" * 70)
  P("Running sent encoders with pre-trained embeddings...")
  base1 = train_test_experiment(
      model=TorchRNNSentenceEncoderClassifier(
          vocab={}, 
          batch_size=64,
          hidden_dim=64, 
          bidirectional=True,
          max_iter=30,
          use_embedding=False,
          dropout=0.5,
      ), 
      x_train=train_glove_sents[0], 
      y_train=train_glove_sents[1], 
      x_test=dev_glove_sents[0], 
      y_test=dev_glove_sents[1],
      score_func=utils.safe_macro_f1,
      )
