# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:28:37 2020

@author: Andrei
"""
import json
import numpy as np
import os
import pandas as pd
import nli
import utils
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_model_base import TorchModelBase




DATA_HOME = 'data'

NLIDATA_HOME = os.path.join(DATA_HOME, 'nlidata')

wordentail_filename = os.path.join(
    NLIDATA_HOME, 'nli_wordentail_bakeoff_data.json')

GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')






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



def find_logreg_params(train_data, dev_data):
  baseline_vector_combo_funcs = [
      concat, 
      summar,
      ]
  
  for vcf in baseline_vector_combo_funcs:
    X_train, y_train = nli.word_entail_featurize(
        data=train_data,  
        vector_func=l_glv, 
        vector_combo_func=vcf
        )
    X_dev, y_dev = nli.word_entail_featurize(
        data=dev_data,  
        vector_func=l_glv, 
        vector_combo_func=vcf
        )
    model = fit_softmax_with_crossvalidation(X_train, y_train)
    predictions = model.predict(X_dev)
    P(nli.classification_report(y_dev, predictions, digits=3))
  return


###############################################################################
###############################################################################
###############################################################################
####                                                                       ####
####                       Utility code section                            ####
####                                                                       ####
###############################################################################
###############################################################################
###############################################################################
import torch as th
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
from collections import OrderedDict
from time import time
import textwrap
from sklearn.metrics import classification_report
import vsm

lst_log = []
_date = dt.now().strftime("%Y%m%d_%H%M")
log_fn = dt.now().strftime("logs/"+_date+"_log.txt")

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

  
def prepare_grid_search(params_grid, valid_fn, nr_trials):
  import itertools


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
  P("Filtering {} grid-search options...".format(len(grid_iterations)))
  cleaned_iters = [x for x in grid_iterations if valid_fn(x)]
  n_options = len(cleaned_iters)
  idxs = np.arange(n_options)
  np.random.shuffle(idxs)
  idxs = idxs[:nr_trials]
  P("Generated {} random grid-search iters out of a total of {} iters".format(
      len(idxs), n_options))
  return [cleaned_iters[i] for i in idxs]


def add_res(dct, model_name, score, **kwargs):
  n_existing = len(dct['MODEL'])
  dct['MODEL'].append(model_name)
  dct['SCORE'].append(score)
  for key in kwargs:
    if key not in dct:
      dct[key] = ['-' ] * n_existing
    dct[key].append(kwargs[key])
  for k in dct:
    if len(dct[k]) < (n_existing + 1):
      dct[k] = dct[k] + [' '] * ((n_existing + 1) - len(dct[k]))
  return dct


def maybe_add_top_model(top_models, model, score, k=5):
  if len(top_models) < k:
    top_models.append([model, score])
  else:
    for i in range(k):
      if top_models[i][1] < score:
        for jj in range(k-1, i, -1):
          top_models[jj] = top_models[jj-1]
        top_models[i][0] = model
        top_models[i][1] = score
        break          
  return sorted(top_models, key=lambda x: x[1], reverse=True)
       

tm = [['a', 100], ['e', 99], ['dddd', 98], ['dd', 52], ['gg', 51]] 

tm = maybe_add_top_model(tm, 'kk', 52)  
###############################################################################
###############################################################################
###############################################################################
####                                                                       ####
####                      END utility code section                         ####
####                                                                       ####
###############################################################################
###############################################################################
###############################################################################

def maybe_find_glove_replacement(w):
  w = w.lower()
  lw = len(w)
  if lw < 4:
    return None
  else:   
    nw1 = ''
    found = False
    for i in range(lw//2 + 2):
      nw = w[:-(i+1)]
      if nw in GLOVE:
        nw1 = nw
        found = True
        break
    nw2 = ''
    for i in range(lw//2 + 2):
      nw = w[(i+1):]
      if nw in GLOVE:
        nw2 = nw
        found = True
        break
    if found:
      return nw1 if len(nw1) > len(nw2) else nw2
    return None

def l_glv_rep(w):    
  """Return lower `w`'s GloVe representation if available, else return 
  a replacement (zeros vector is nothing is found)."""
  if w in GLOVE:
    return GLOVE[w]
  else:
    nw = maybe_find_glove_replacement(w)
    v = np.random.uniform(low=-1e-4, high=1e-4, size=GLOVE_DIM)
    if nw is not None:
      v = v + GLOVE[nw]
    return v
    
def l_glv(w):    
  """Return lower `w`'s GloVe representation if available, else return 
  a zeros vector."""
  return GLOVE.get(w.lower(), np.zeros(GLOVE_DIM))

  
def concat(u, v):
  return np.concatenate((u,v))

def summar(u, v):
  return u + v

def arr(u,v):
  return np.array((u,v))


def test_glove_vs_data(trn, dev):
  train_words = set()
  for x in trn:
    train_words.add(x[0][0])
    train_words.add(x[0][1])

  dev_words = set()
  for x in dev:
    dev_words.add(x[0][0])
    dev_words.add(x[0][1])
  
  glove_train = [x for x in trn if (x[0][0] in GLOVE) and (x[0][1] in GLOVE)]
  glove_dev = [x for x in dev if (x[0][0] in GLOVE) and (x[0][1] in GLOVE)]
  out_train = [x for x in trn if (x[0][0] not in GLOVE) or (x[0][1] not in GLOVE)]
  out_dev = [x for x in dev if (x[0][0] not in GLOVE) or (x[0][1] not in GLOVE)]
  
  P("\nGlove train: {} ({:.1f}%)".format(len(glove_train), len(glove_train)/len(trn)*100))
  P("\nGlove dev: {} ({:.1f}%)".format(len(glove_dev), len(glove_dev)/len(dev)*100))
  
  miss_train = [x.lower() for x in train_words if x.lower() not in GLOVE] 
  miss_dev = [x.lower() for x in dev_words if x.lower() not in GLOVE]
  P("\nTrain has {} words that are not in GLOVE: {}...".format(len(miss_train), miss_train[:5]))
  positives = 0
  negatives = 0
  for i, w in enumerate(miss_train):
    for x in trn:
      if x[0][0] == w or x[0][1] == w:
        if x[1]:
          positives += 1
        else:
          negatives += 1
        if x[1] == 1 and positives < 5:
          P("  {}".format(x))
        if x[1] == 0 and negatives < 5:
          P("  {}".format(x))
  P("\nPos vs neg: {} vs {}".format(positives, negatives))
  P("\n\nDev has {} words that are not in GLOVE: {}...".format(len(miss_dev), miss_dev[:5]))
  positives = 0
  negatives = 0
  for i, w in enumerate(miss_dev):
    for x in dev:
      if x[0][0] == w or x[0][1] == w:
        if x[1]:
          positives += 1
        else:
          negatives += 1
        if x[1] == 1 and positives < 5:
          P("  {}".format(x))
        if x[1] == 0 and negatives < 5:
          P("  {}".format(x))
  P("\nPos vs neg: {} vs {}".format(positives, negatives))
  res = {
      'glove_train' : glove_train,
      'glove_dev' : glove_dev,
      'out_train' : out_train,
      'out_dev' : out_dev
      }
  return res, (miss_train, miss_dev)


class ConstrativeLoss(th.nn.Module):
  def __init__(self, margin=0.2):
    super(ConstrativeLoss, self).__init__()
    self.margin = margin
    
    
  def forward(self, dist, gold):
    th_d_sq = th.pow(dist, 2)
    th_d_sqm = th.pow(th.clamp(self.margin - dist, 0), 2)
    loss = (1 - gold) * th_d_sq + gold * th_d_sqm
    return loss.mean()
  
  def __repr__(self):
    s = self.__class__.__name__ + "(margin={})".format(
        self.margin,
        )
    return s  


class FocalLoss(th.nn.Module):
  def __init__(self, alpha=4, gamma=2):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma

  def forward(self, inputs, targets):
    BCE_loss = th.nn.functional.binary_cross_entropy_with_logits(
        inputs, 
        targets, 
        reduction='none',
        )

    pt = th.exp(-BCE_loss)
    F_loss = self.alpha * th.pow(1 - pt, self.gamma) * BCE_loss
    return th.mean(F_loss)
      
  def __repr__(self):
    s = self.__class__.__name__ + "(alpha={}, gamma={})".format(
        self.alpha,
        self.gamma,
        )
    return s


class InputPlaceholder(th.nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.input_dim = input_dim
    
  def forward(self, inputs):
    return inputs
  
  def __repr__(self):
    s = self.__class__.__name__ + "(input_dim={})".format(
        self.input_dim,
        )
    return s
  
class L2_Normalizer(th.nn.Module):
  def __init__(self,):
    super().__init__()
    
  def forward(self, inputs):
    return th.nn.functional.normalize(inputs, p=2, dim=1)

class PathsCombiner(th.nn.Module):
  def __init__(self, input_dim, method, activ=None):
    super().__init__()
    self.method = method
    self.input_dim = input_dim
    self.activ = self.get_activation(activ)
    if method in ['sub','abs','sqr', 'add']:
      self.output_dim = input_dim
    elif method == 'cat':
      self.output_dim = input_dim * 2
    elif method == 'eucl':
      self.output_dim = 1
    else:
      raise ValueError("Unknown combine method '{}'".format(method))
    return
    
  def forward(self, paths):
    path1 = paths[0]
    path2 = paths[1]
#    if self.norm_each:
#      path1 = th.nn.functional.normalize(path1, p=2, dim=1)
#      path2 = th.nn.functional.normalize(path2, p=2, dim=1)
      
    if self.method == 'sub':
      th_x = path1 - path2
    elif self.method == 'add':
      th_x = path1 + path2
    elif self.method == 'cat':
      th_x = th.cat((path1, path2), dim=1)
    elif self.method == 'abs':
      th_x = (path1 - path2).abs()
    elif self.method == 'sqr':
      th_x = th.pow(path1 - path2, 2)
    elif self.method == 'eucl':
      th_x = th.pairwise_distance(path1, path2, keepdim=True)    
    
    if self.activ is not None:
      th_x = self.activ(th_x)
      
    return th_x
  
  def get_activation(self, act):
    if act == 'relu':
      return th.nn.ReLU()
    elif act == 'tanh':
      return th.nn.Tanh()
    elif act == 'selu':
      return th.nn.SELU()
    elif act == 'sigmoid':
      return th.nn.Sigmoid()
    else:
      return None
  
  
  def __repr__(self):
    s = self.__class__.__name__ + "(input_dim={}x2, output_dim={}, method='{}', act={})".format(
        self.input_dim,
        self.output_dim,
        self.method,
        self.activ,
        )
    return s

class ThWordEntailModel(th.nn.Module):
  def __init__(self,
               input_dim,
               siam_lyrs,
               siam_norm,
               siam_bn,
               comb_activ,
               separate_paths,
               layers,
               input_drop,
               other_drop,
               bn_inputs,
               bn,
               smethod,
               loss_type,
               device,
               activ='relu',
               ):
    super().__init__()
    self.device = device
    self.has_input_drop = input_drop
    self.has_input_bn = bn_inputs
    self.smethod = smethod
    self.separate = separate_paths
    self.loss_type = loss_type
    self.siam_norm = siam_norm
    self.siam_bn = siam_bn
    self.comb_activ = comb_activ
    
    if self.loss_type == 'cl' and (layers != [] or siam_lyrs == [] or separate_paths):
      raise ValueError("Cannot have siamese nets with CL with this config: layers={}  siam_lyrs={} sep={}".format(
          layers, siam_lyrs, separate_paths))
      
    if other_drop != 0 and layers == [] and siam_lyrs == []:
      raise ValueError("Cannot have dropout on no layers...")

    if self.separate:
      paths = [[],[]]
    else:
      paths = [[]]
    self.path_input = input_dim
    
    for path_no in range(len(paths)):
      last_output = self.path_input
      paths[path_no].append(InputPlaceholder(self.path_input))
      if input_drop > 0:
        paths[path_no].append(th.nn.Dropout(input_drop))
      if bn_inputs:
        paths[path_no].append(th.nn.BatchNorm1d(last_output))
      if len(siam_lyrs) > 0:
        for i, layer in enumerate(siam_lyrs):
          paths[path_no].append(th.nn.Linear(last_output, layer, bias=not self.siam_bn))
          if self.siam_bn:
            paths[path_no].append(th.nn.BatchNorm1d(layer))
          if i < (len(siam_lyrs) - 1):
            paths[path_no].append(self.get_activation(activ))
            if other_drop > 0:
              paths[path_no].append(th.nn.Dropout(other_drop))
          last_output = layer
      if self.siam_norm:
        paths[path_no].append(L2_Normalizer())
    if self.separate :
      self.path1_layers = th.nn.ModuleList(paths[0])
      self.path2_layers = th.nn.ModuleList(paths[1])
    else:
      self.siam_layers = th.nn.ModuleList(paths[0])
      
    
    siam_combine = PathsCombiner(
        last_output, 
        method=self.smethod, 
        activ=self.comb_activ,
        )
    last_output = siam_combine.output_dim
    post_lyrs = [siam_combine]    
    if self.loss_type != 'cl':
      for i, layer in enumerate(layers):
        post_lyrs.append(th.nn.Linear(last_output, layer, bias=not bn))
        if bn:
          post_lyrs.append(th.nn.BatchNorm1d(layer))
        post_lyrs.append(self.get_activation(activ))
        if other_drop > 0:
          post_lyrs.append(th.nn.Dropout(other_drop))
        last_output = layer
      post_lyrs.append(th.nn.Linear(last_output, 1))
    self.post_layers = th.nn.ModuleList(post_lyrs)
    return
  
  def forward(self, inputs):
    th_path1 = inputs[:,0]
    th_path2 = inputs[:,1]

    if self.separate:
      if len(self.path1_layers) > 0:
        for th_layer in self.path1_layers:
          th_path1 = th_layer(th_path1)
        for th_layer in self.path2_layers:
          th_path2 = th_layer(th_path2)
    else:
      if len(self.siam_layers) > 0:
        for th_layer in self.siam_layers:
          th_path1 = th_layer(th_path1)
        for th_layer in self.siam_layers:
          th_path2 = th_layer(th_path2)

    
    th_x = (th_path1, th_path2)    
    # first layer in post-siam must be the combination layer
    for layer in self.post_layers:
      th_x = layer(th_x)    
    return th_x
    
  
  def get_activation(self, act):
    if act == 'relu':
      return th.nn.ReLU()
    elif act == 'tanh':
      return th.nn.Tanh()
    elif act == 'selu':
      return th.nn.SELU()
    elif act == 'sigmoid':
      return th.nn.Sigmoid()
    else:
      raise ValueError("Unknown activation function '{}'".format(act))
  
      
      


class WordEntailClassifier():
  def __init__(self, 
               model_name,
               siam_lyrs,
               s_l2,
               s_bn,
               c_act,
               separ,
               layers,                 
               inp_drp,
               o_drp,
               bn,
               bn_inp,
               activ,
               x_dev,
               y_dev,
               s_comb,
               rev,
               lr,
               loss,
               bal,
               cl_m=1,
               lr_decay=0.5,
               batch=256,
               l2_strength=0,
               max_epochs=10000,
               max_patience=10,
               max_fails=40,
               optim=th.optim.Adam,
               device=th.device("cuda" if th.cuda.is_available() else "cpu"),
               ):
    self.model_name = model_name
    self.layers = layers
    self.siam_lyrs = siam_lyrs
    self.input_drop = inp_drp
    self.other_drop = o_drp
    self.bn = bn
    self.bn_inputs = bn_inp
    self.activ=activ    
    self.max_epochs = max_epochs
    self.x_dev = x_dev
    self.y_dev = np.array(y_dev)
    self.batch_size = batch
    self.max_patience = max_patience
    self.optimizer = optim
    self.siamese_method = s_comb
    self.reverse_target = rev
    self.device = device
    self.lr = lr
    self.lr_decay = lr_decay
    self.l2_strength = l2_strength
    self.max_fails = max_fails
    self.separate_paths = separ
    self.loss_type = loss
    self.siam_norm = s_l2
    self.siam_bn = s_bn
    self.comb_activ = c_act
    self.margin = cl_m
    self.use_balancing = bal
    if loss == 'cl' and not rev:
      raise ValueError("CL must receive reversed targets")
    return
  
  
  def define_graph(self):
    model = ThWordEntailModel(
        input_dim=self.input_dim,
        siam_lyrs=self.siam_lyrs,
        separate_paths=self.separate_paths,
        layers=self.layers,
        input_drop=self.input_drop,
        other_drop=self.other_drop,
        bn=self.bn,
        bn_inputs=self.bn_inputs,
        activ=self.activ,
        device=self.device,
        smethod=self.siamese_method,
        siam_norm=self.siam_norm,
        loss_type=self.loss_type,
        siam_bn=self.siam_bn,
        comb_activ=self.comb_activ,
        )
    return model    
      
  
  def fit(self, X, y):
    utils.fix_random_seeds()
    # Data prep:
    X = np.array(X).astype(np.float32)
    n_obs = X.shape[0]
    # here is a trick: we consider the words that entail those that have
    # minimal distance if using siamese
    np_y = np.array(y).reshape(-1,1)
    if self.reverse_target:
      np_y = 1 - np_y
    self.input_dim = X.shape[-1]
    X = th.tensor(X, dtype=th.float32)
    y = th.tensor(np_y, dtype=th.float32)
    dataset = th.utils.data.TensorDataset(X, y)
    
    sampler = None
    if self.use_balancing:
      cls_0 = (np_y == 0).sum()
      cls_1 = np_y.shape[0] - cls_0
      cls_weights = 1 / np.array([cls_0, cls_1])
      weights = [cls_weights[i] for i in np_y.ravel()]
      sampler = th.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(weights))
      
    dataloader = th.utils.data.DataLoader(
        dataset, 
        batch_size=self.batch_size, 
        shuffle=sampler is None,
        pin_memory=True,
        sampler=sampler,
        )
    # Optimization:
    if self.loss_type == 'bce':
      loss = th.nn.BCEWithLogitsLoss()
    elif self.loss_type == 'cl':
      loss = ConstrativeLoss(margin=self.margin)
    elif self.loss_type == 'fl':
      loss = FocalLoss()
    else:
      raise ValueError('unknown loss {}'.format(self.loss_type))
    if not hasattr(self, "model"):
      self.model = self.define_graph()
      P("Initialized model {}:\n{}\n{}".format(
          self.model_name,
          textwrap.indent(str(self.model), " " * 2),
          textwrap.indent("Loss: " + (str(loss) if loss.__class__.__name__ != 'method' else loss.__name__) + "\n", " " * 2),
          ))

    else:
      P("\rFitting already loaded model...\t\t\t", end='', flush=True)
    self.model.to(self.device)
    self.model.train()
    optimizer = self.optimizer(
        self.model.parameters(),
        lr=self.lr,
        weight_decay=self.l2_strength)
    # Train:
    patience = 0
    fails = 0
    best_f1 = 0
    best_fn = ''
    best_epoch = -1
    self.errors = []
    for epoch in range(1, self.max_epochs+1):
      epoch_error = 0.0
      for batch_iter, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(self.device, non_blocking=True)
        y_batch = y_batch.to(self.device, non_blocking=True)
        batch_preds = self.model(X_batch)
        err = loss(batch_preds, y_batch)
        epoch_error += err.item()
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        Pr("Training epoch {} - {:.1f}% - Patience {}/{},  Fails {}/{}\t\t\t\t\t".format(
            epoch, 
            (batch_iter + 1) / (n_obs // self.batch_size + 1) * 100,
            patience, self.max_patience,
            fails, self.max_fails))
      # end epoch
      predictions = self.predict(self.x_dev)
      macrof1 = utils.safe_macro_f1(self.y_dev, predictions)
      # resume training
      self.model.train()
      if macrof1 > best_f1:
        patience = 0
        fails = 0
        last_best_fn = best_fn
        best_fn = "models/{}_e{:03}_F{:.4f}.th".format(
            self.model_name, epoch, macrof1)
        best_epoch = epoch
        P("\rFound new best macro-f1 {:.4f} > {:.4f} at epoch {}. \t\t\t".format(macrof1, best_f1, epoch))
        best_f1 = macrof1
        th.save(self.model.state_dict(), best_fn)
        th.save(optimizer.state_dict(), best_fn + '.optim')
        if last_best_fn != '':
          try:
            os.remove(last_best_fn)
            os.remove(last_best_fn + '.optim')
          except:
            pass
      else:
        patience += 1
        fails += 1
        Pr("Finished epoch {}. Current score {:.3f} < {:.3f}. Patience {}/{},  Fails {}/{}".format(
            epoch, macrof1, best_f1, patience, self.max_patience, fails, self.max_fails))
        if patience > self.max_patience:
          lr_old = optimizer.param_groups[0]['lr'] 
          lr_new = lr_old * self.lr_decay
          self.model.load_state_dict(th.load(best_fn))
          optimizer.load_state_dict(th.load(best_fn + '.optim'))
          for param_group in optimizer.param_groups:
            param_group['lr'] = lr_new
          P("\nPatience reached {}/{}  -  reloaded from ep {} reduced lr from {:.1e} to {:.1e}".format(
              patience, self.max_patience, best_epoch, lr_old, lr_new))
          patience = 0
          
        if fails > self.max_fails:
          P("\nMax fails {}/{} reached!".format(fails, self.max_fails))
          break
          
      self.errors.append(epoch_error)
    # end all epochs
    if best_fn != '':
      P("Loading model from epoch {} with macro-f1 {:.4f}".format(best_epoch, best_f1))
      self.model.load_state_dict(th.load(best_fn))            
    return self
  

  def predict_proba(self, X):
    self.model.eval()
    with th.no_grad():
      self.model.to(self.device)
      X = th.tensor(X, dtype=th.float).to(self.device)
      preds = self.model(X)
      if self.loss_type in ['bce', 'fl']:
        result = th.sigmoid(preds).cpu().numpy().ravel()
        if self.reverse_target:
          result = 1 - result
      else:
        result = preds.cpu().numpy().ravel()
      return result


  def predict(self, X):
    probs = self.predict_proba(X)
    if self.loss_type in ['bce', 'fl']:
      classes = (probs >= 0.5).astype(np.int8)
    elif self.loss_type == 'cl':
      thr = self.margin / 2
      classes = (probs >= thr).astype(np.int8)
      if self.reverse_target:
        # we have to reverse
        classes = 1 - classes
    else:
      raise ValueError('UNK LOSS')
    return classes
  
  
  

  
  
  
  
class EnsembleWrapper():
  def __init__(self, models, use_proba=False):
    self.models = models
    self.use_proba = use_proba
    names = [x.model_name.split('_')[1] for x in self.models if hasattr(x, "model_name")]
    self.model_name = "E_" + "_".join(names)
    P("Initialized ensemble model with {} models: {}".format(
        len(self.models),
        names))
  
  def predict(self, x):
    preds = []
    for model in self.models:
      if self.use_proba:
        if model.loss_type != 'cl':
          model_preds = model.predict_proba(x)
        else:
          model_preds = []
      else:
        model_preds = model.predict(x)
      if len(model_preds) > 0:
        preds.append(model_preds)        
    final_preds_cat = np.vstack(preds).T
    final_preds = final_preds_cat.mean(axis=1)
    return (final_preds >= 0.5).astype(np.uint8)
  
    
  def fit(self, x, y):
    P("****** Ensemble model passing fit call ******")
    return self
    




def get_baselines(dct_res, trn, dev):
  baseline_model_facts = {
      "BaseLR_C6L2": lambda: LogisticRegression(fit_intercept=True, 
                                                solver='liblinear', 
                                                multi_class='auto',
                                                C=0.6,
                                                penalty='l2'),
      "BaseLR_C4L1": lambda: LogisticRegression(fit_intercept=True, 
                                                solver='liblinear', 
                                                multi_class='auto',
                                                C=0.4,
                                                penalty='l1'),
      "BaseNN_50" : lambda: TorchShallowNeuralClassifier(hidden_dim=50, eta=0.005),
      "BaseNN_150" : lambda: TorchShallowNeuralClassifier(hidden_dim=150, eta=0.005),
      "BaseNN_300" : lambda: TorchShallowNeuralClassifier(hidden_dim=300, eta=0.005),
  }
  
  baseline_vector_combo_funcs = [
      concat, 
  #    summar,
      ]
  for vf in [l_glv, l_glv_rep]:
    for vcf in baseline_vector_combo_funcs:
      for model_name in baseline_model_facts:
        P("=" * 70)
        P("Running baseline model '{}' with '{}'".format(
            model_name, vcf.__name__))
        model = baseline_model_facts[model_name]()
        x_d, y_d = nli.word_entail_featurize(
            data=dev, 
            vector_func=vf, 
            vector_combo_func=vcf
            )
        res = nli.wordentail_experiment(
            train_data=trn,
            assess_data=dev,
            vector_func=vf,
            vector_combo_func=vcf,
            model=model,
            )
        score = res['macro-F1']
        y_pred = model.predict(x_d)
        report = classification_report(y_d, y_pred, digits=3, output_dict=True)
        assert score == report['macro avg']['f1-score']
        P_REC = report['1']['recall']
        add_res(dct_res, model_name, score, P_REC=P_REC, VF=vf.__name__)
        df = pd.DataFrame(dct_results).sort_values('SCORE')
        P("\nResults so far:\n{}\n".format(df))
  return dct_res


def run_grid_search(dct_res, trn, dev):
  grid_non_CL = {
      "siam_lyrs" : [
          [],
          [128],
          [256, 128],
          [256, 128, 64],
          [512, 256],
          ],
          
      "separ" : [
          True,
          False,
          ],
      
      "layers" : [
          [256, 128, 64],
          [128, 32],
          [512, 256],
          ],
          
      "inp_drp" : [
          0,
          0.3
          ],
  
      "o_drp" : [
          0,
          0.2,
          0.5,
          ],
          
      "bn" : [
          True,
          False
          ],
          
      "bn_inp" : [
          True,
          False
          ],
          
      "activ" : [
          'tanh',
          'relu',
          'selu',
          'sigmoid',
          ],
          
      "lr"  :[
          0.01,
          0.005,
          ],
          
      "s_comb" : [
          'sub',
          'add',
          'cat',
          'abs',
          'sqr',
          'eucl',
          ],
      
      's_l2' : [
          True,
          False,
          ],
          
      's_bn' :[
          True,
          False
          ],
          
      'rev' :[
          True,
          False,
          ],
          
      'loss' : [
          'bce',
          'fl',
          ],

      'c_act' : [
          None
          ],
          
          
      'vector_func':[
          l_glv,
          l_glv_rep
          ],
          
      'bal' : [
          True,
          False,
          ],
          
          
                
        
      }

  grid_CL= {
      "siam_lyrs" : [
#          [128],
          [256, 128],
#          [256, 128, 64],
          [512, 256],
          ],
          
      "separ" : [
          False,
          ],
      
      "layers" : [
          [],
          ],
          
      "inp_drp" : [
          0,
          0.3
          ],
  
      "o_drp" : [
          0,
          0.2,
          0.5,
          ],
          
      "bn" : [
          True,
          False
          ],
          
      "bn_inp" : [
          True,
          False
          ],
          
      "activ" : [
#          'tanh',
          'relu',
#          'selu',
#          'sigmoid',
          ],
          
      "lr"  :[
          0.0001,
          ],
          
      'c_act' : [
          'tanh',
          'relu',
          ],
          
      "s_comb" : [
          'eucl',
          ],
      
      's_l2' : [
          True,
          False,
          ],

      's_bn' : [
          True,
          False
          ],
          
      'rev' :[
          True,
          ],
          
      'loss' : [
          'cl',
          ],
          
      'vector_func':[
          l_glv,
          l_glv_rep
          ],
          
      'bal' : [
          True,
          False,
          ]
      
        
      }

        
  def filter_func(grid_iter):
    test_contrastive_loss = (
        grid_iter['separ'] or 
        grid_iter['rev'] == False or 
        grid_iter['layers'] != [] or
        grid_iter['siam_lyrs'] == [] or
        grid_iter['s_comb'] != 'eucl'
        )
    if grid_iter['loss'] == 'cl' and test_contrastive_loss:
      return False
    if grid_iter['layers'] == [] and grid_iter['siam_lyrs'] == [] and grid_iter['other_drop'] != 0 :
      return False
    if grid_iter['layers'] != [] and grid_iter['s_comb'] == 'eucl':
      return False
    if not grid_iter['separ'] and grid_iter['siam_lyrs'] == []:
      return False
    return True
  
  options1 = prepare_grid_search(grid_non_CL, valid_fn=filter_func, nr_trials=350)
  options2 = prepare_grid_search(grid_CL, valid_fn=filter_func, nr_trials=250)
  options = options1 + options2
  options = [options[x] for x in np.random.choice(len(options), size=len(options), replace=False)]
  timings = []
  t_left = np.inf
  top_models = []
  k=3
  last_ensemble = ''
  for grid_iter, option in enumerate(options):    
    model_name = 'H3v2_{:03d}'.format(grid_iter+1)
    P("\n\n" + "=" * 70)
    P("Running grid search iteration {}/{} '{}': {}".format(
        grid_iter+1, len(options), model_name, option))
    P("  Time left for grid search completion: {:.1f} hrs".format(t_left / 3600))
  
    vector_func = option.pop('vector_func')
    _t_start = time()
    #### we need this ...
    x_dev, y_dev = nli.word_entail_featurize(
        data=dev, 
        vector_func=vector_func, 
        vector_combo_func=arr
        )
    model = WordEntailClassifier(
        model_name=model_name,
        x_dev=x_dev,
        y_dev=y_dev,
        **option)
    res = nli.wordentail_experiment(
            train_data=trn,
            assess_data=dev,
            vector_func=vector_func,
            vector_combo_func=arr,
            model=model,
            )
    score = res['macro-F1']
    top_models = maybe_add_top_model(
        top_models=top_models,
        model=model,
        score=score,
        k=k
        )
    y_pred = model.predict(x_dev)
    report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
    assert score == report['macro avg']['f1-score']
    P_REC = report['1']['recall']
    ####
    t_res = time() - _t_start
    timings.append(t_res)
    t_left = (len(options) - grid_iter - 1) * np.mean(timings)
    dct_res = add_res(
        dct=dct_res, 
        model_name=model_name, 
        score=score, 
        P_REC=P_REC,
        VF=vector_func.__name__,
        **option)
    if len(top_models) >= k:
      P("Testing ensemble so far...")
      ensemble = EnsembleWrapper([x[0] for x in top_models])
      if last_ensemble != ensemble.model_name:
        last_ensemble = ensemble.model_name
        res = nli.wordentail_experiment(
                train_data=trn,
                assess_data=dev,
                vector_func=vector_func,
                vector_combo_func=arr,
                model=ensemble,
                )
        score = res['macro-F1']
        y_pred = ensemble.predict(x_dev)
        report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
        assert score == report['macro avg']['f1-score']
        P_REC = report['1']['recall']
        dct_res = add_res(
          dct=dct_res, 
          model_name=ensemble.model_name, 
          score=score, 
          P_REC=P_REC,
          VF=vector_func.__name__,
          )      
    df = pd.DataFrame(dct_res).sort_values('SCORE')
    P("Results so far:\n{}".format(df.iloc[-50:]))
    df.to_csv("models/"+_date+"_results.csv")
  # end grid
  return df
      
def vect_neighbors(v, df):
  import scipy
  distfunc = scipy.spatial.distance.cosine
  dists = df.apply(lambda x: distfunc(v, x), axis=1)
  return dists.sort_values().head()


def ensemble_train_test(lst_models_params, trn, dev, vect_func):
  x_trn, y_trn = nli.word_entail_featurize(
      data=trn, 
      vector_func=vect_func, 
      vector_combo_func=arr
      )
  x_dev, y_dev = nli.word_entail_featurize(
      data=dev, 
      vector_func=vect_func, 
      vector_combo_func=arr
      )
  clfs = []
  res = {'MODEL':[],'SCORE':[]}
  for i, model_params in enumerate(lst_models_params):
    model_name = 'M_{}'.format(i+1)
    clf = WordEntailClassifier(model_name=model_name, 
                               x_dev=x_dev,
                               y_dev=y_dev,
                               **model_params)
    clf.fit(x_trn, y_trn)
    clfs.append(clf)
    y_pred = clf.predict(x_dev)
    report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
    res = add_res(
        res, 
        model_name, 
        score=report['macro avg']['f1-score'] * 100,
        pos_f1=report['1']['f1-score'] * 100,
        pos_rc=report['1']['recall'] * 100,
        pos_pr=report['1']['precision'] * 100,
        )
    df = pd.DataFrame(res).sort_values('SCORE')
    P("Results:\n{}".format(df))
    
  ens = EnsembleWrapper(
      clfs, 
      use_proba=True
      )
  y_pred = ens.predict(x_dev)
  report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
  res = add_res(
      res, 
      ens.model_name, 
      score=report['macro avg']['f1-score'] * 100,
      pos_f1=report['1']['f1-score'] * 100,
      pos_rc=report['1']['recall'] * 100,
      pos_pr=report['1']['precision'] * 100,
      )
  df = pd.DataFrame(res).sort_values('SCORE')
  P("Results:\n{}".format(df))
  return ens
  

###############################################################################
###############################################################################
###############################################################################
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)  
  
CALC_BASELINES = False
RUN_GRID = True
TEST_RUN = False
TEST_ENSEMBLE = False


utils.fix_random_seeds()
GLOVE_DIM = 300
if "GLOVE" not in globals():
  P("Loading GloVe-{}...".format(GLOVE_DIM))
  GLOVE = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(GLOVE_DIM)))  
  P("GloVe-{} loaded.".format(GLOVE_DIM))

with open(wordentail_filename) as f:
  wordentail_data = json.load(f)  
  
train_data = wordentail_data['word_disjoint']['train']
dev_data = wordentail_data['word_disjoint']['dev']
dct_results = OrderedDict({'MODEL':[], 'SCORE':[], 'P_REC': []})


maybe_find_glove_replacement('aromatization')
glv_analysis = test_glove_vs_data(train_data, dev_data)
dct_data = glv_analysis[0]
miss_train, miss_dev = glv_analysis[1]

if False:
  if 'df_GLOVE' not in globals():
    df_GLOVE = pd.DataFrame.from_dict(GLOVE, orient='index')
  P("Train replacements:")
  for w in miss_train:
    repl = l_glv_rep(w)
    P("replacement for '{}':".format(w))
    P(vect_neighbors(repl, df_GLOVE))
    P("")
  
  P("Dev replacements:")
  for w in miss_dev:
    repl = l_glv_rep(w)
    P("replacement for '{}':".format(w))
    P(vect_neighbors(repl, df_GLOVE))
    P("")



if CALC_BASELINES:
  dct_results = get_baselines(dct_results, train_data, dev_data)
      

if RUN_GRID:    
  dct_results = run_grid_search(dct_results, train_data, dev_data)
    

##############
if TEST_RUN:
  dct_res = {
      "DATA" : [],
      "VF" : [],
      "Macro-F1" : [],
      "Pos F1" : [],
      "Pos Recall" : [],
      "Pos Precis" : [],
      }
  def _log_data_result(dn, mf1, pf1, rec, prec, vf):
    dct_res['DATA'].append(dn)
    dct_res['VF'].append(vf.__name__)
    dct_res['Macro-F1'].append(mf1)
    dct_res['Pos F1'].append(pf1)
    dct_res['Pos Recall'].append(rec)
    dct_res['Pos Precis'].append(prec)

  for VECT_FUNC in [l_glv]: #, l_glv_rep]:
    _x_trn, _y_trn = nli.word_entail_featurize(
        data=train_data, 
        vector_func=VECT_FUNC, 
        vector_combo_func=arr
        )
    _x_dev, _y_dev = nli.word_entail_featurize(
        data=dev_data, 
        vector_func=VECT_FUNC, 
        vector_combo_func=arr
        )
  
      
    test_model = WordEntailClassifier(
      siam_lyrs=[512, 256],
      s_l2=False,
      s_bn=True,
      bn=True,
      bn_inp=True,
      c_act=None,
      separ=True,
      layers=[128, 32],
      inp_drp=0.3,
      o_drp=0.5,
      activ='sigmoid',
      s_comb='abs',
      loss='bce',
      rev=True,
      lr=0.005, 
      batch=256,
      
      bal=True,

      cl_m=1,
      
      x_dev=_x_dev,
      y_dev=_y_dev,
      model_name='test',
      optim=th.optim.Adam  
      )
    
    res = nli.wordentail_experiment(
          train_data=train_data,
          assess_data=dev_data,
          vector_func=VECT_FUNC,
          vector_combo_func=arr,
          model=test_model,    
        )
    
    
    
    
    y_pred = test_model.predict(_x_dev)
    report = classification_report(_y_dev, y_pred, digits=3, output_dict=True)
    _log_data_result(
        'dev_full', 
        report['macro avg']['f1-score'] * 100,
        report['1']['f1-score'] * 100,
        report['1']['recall'] * 100,
        report['1']['precision'] * 100,
        VECT_FUNC,
        )
    y_pred = test_model.predict(_x_trn)
    report = classification_report(_y_trn, y_pred, digits=3, output_dict=True)
    _log_data_result(
        'train_full', 
        report['macro avg']['f1-score'] * 100,
        report['1']['f1-score'] * 100,
        report['1']['recall'] * 100,
        report['1']['precision'] * 100,
        VECT_FUNC,
        )
    
    for test_name in dct_data:
      P("\n\nTesting on '{}':".format(test_name))
      _x, _y = nli.word_entail_featurize(
          data=dct_data[test_name], 
          vector_func=VECT_FUNC, 
          vector_combo_func=arr
          )
      _yh = test_model.predict(_x)
      P(classification_report(_y, _yh, digits=3))
      report = classification_report(_y, _yh, digits=3, output_dict=True)
      _log_data_result(
          test_name, 
          report['macro avg']['f1-score'] * 100,
          report['1']['f1-score'] * 100,
          report['1']['recall'] * 100,
          report['1']['precision'] * 100,
          VECT_FUNC,
          )
    df = pd.DataFrame(dct_res).sort_values('Macro-F1')
    P(df)
    
  
if TEST_ENSEMBLE:
  lst_models =[
      {'siam_lyrs': [512, 256],
       's_l2': False,
       's_bn': True,
       'bn': True,
       'bn_inp': True,
       'c_act': None,
       'separ': True,
       'layers': [128, 32],
       'inp_drp': 0.3,
       'o_drp': 0.5,
       'activ': 'sigmoid',
       's_comb': 'abs',
       'loss': 'bce',
       'bal' : xxx,
       'rev': True,
       'lr': 0.005,
       'batch': 256
       },
       
      {'siam_lyrs': [512, 256],
       's_l2': False,
       's_bn': True,
       'bn': True,
       'bn_inp': True,
       'c_act': None,
       'separ': True,
       'layers': [128, 32],
       'inp_drp': 0.3,
       'o_drp': 0.5,
       'activ': 'sigmoid',
       's_comb': 'abs',
       'loss': 'fl',
       'bal' : xxx,
       'rev': True,
       'lr': 0.005,
       'batch': 256},
       
      {'siam_lyrs': [128],
       's_l2': False,
       's_bn': False,
       'bn': True,
       'bn_inp': False,
       'c_act': None,
       'separ': True,
       'layers': [256, 128, 64],
       'inp_drp': 0.3,
       'o_drp': 0.3,
       'activ': 'relu',
       's_comb': 'cat',
       'loss': 'bce',
       'bal' : xxx,
       'rev': False,
       'lr': 0.005,
       'batch': 256},
      ]
  
  ens_model = ensemble_train_test(
      lst_models_params=lst_models,
      trn=train_data, 
      dev=dev_data, 
      vect_func=l_glv,
      )