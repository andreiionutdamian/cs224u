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

def labels_to_class_weights(labels, nc=2): 
   # Get class weights (inverse frequency) from training labels 
   classes = labels.astype(np.int)  # labels 
   weights = np.bincount(classes, minlength=nc)  # occurences per class 
   weights[weights == 0] = 1  # replace empty bins with 1 
   weights = 1 / weights  # number of targets per class 
   weights /= weights.sum()  # normalize 
   return weights

def binary_focal_loss(y_pred , y_true,gamma=2.0 , alpha=0.25 ,reduction="mean"): #,function=th.sigmoid,**kwargs):
    """
    Binary Version of Focal Loss
    :args
    
    y_pred : prediction
    
    y_true : true target labels
    
    gamma: dampeing factor default value 2 works well according to reasearch paper
    
    alpha : postive to negative ratio default value 0.25 means 1 positive and 3 negative can be tuple ,list ,int and float
    
    reduction = mean,sum,none

    function = can be sigmoid or softmax or None
    
    **kwargs: parameters to pass in activation function like dim in softmax
    
    """
    if isinstance(alpha,(list,tuple)):
        pos_alpha = alpha[0] # postive sample ratio in the entire dataset
        neg_alpha = alpha[1] #(1-alpha) # negative ratio in the entire dataset
    elif isinstance(alpha ,(int,float)):
        pos_alpha = alpha
        neg_alpha = (1-alpha)
        
    # if else in function can be simplified be removing setting to default to sigmoid  for educational purpose
    if function is not None:
        y_pred = th.sigmoid(y_pred , **kwargs) #apply activation function
    else :
        assert ((y_pred <= 1) & (y_pred >= 0)).all().item() , "negative value in y_pred value should be in the range of 0 to 1 inclusive"
    
    pos_pt = th.where(y_true==1 , y_pred , th.ones_like(y_pred)) # positive pt (fill all the 0 place in y_true with 1 so (1-pt)=0 and log(pt)=0.0) where pt is 1
    neg_pt = th.where(y_true==0 , y_pred , th.zeros_like(y_pred)) # negative pt
    
    pos_modulating = (1-pos_pt)**gamma # compute postive modulating factor for correct classification the value approaches to zero
    neg_modulating = (neg_pt)**gamma # compute negative modulating factor
    
    
    pos = -pos_alpha* pos_modulating*th.log(pos_pt) #pos part
    neg = -neg_alpha* neg_modulating*th.log(1-neg_pt) # neg part
    
    loss = pos+neg  # this is final loss to be returned with some reduction
    
    # apply reduction
    if reduction =="mean":
        return loss.mean()
    elif reduction =="sum":
        return loss.sum()
    elif reduction =="none":
        return loss # reduction mean
    else:
        raise f"Wrong reduction {reduction} is choosen \n choose one among [mean,sum,none]  "

DATA_HOME = 'data'

NLIDATA_HOME = os.path.join(DATA_HOME, 'nlidata')

wordentail_filename = os.path.join(
    NLIDATA_HOME, 'nli_wordentail_bakeoff_data.json')

GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')

def calc_dist(data_split):
    return pd.DataFrame(data_split)[1].value_counts()    



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
from itertools import combinations
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
  if 'MODEL' not in dct:
    dct['MODEL'] = []
  if 'SCORE' not in dct:
    dct['SCORE'] = []
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
          top_models[jj] = top_models[jj-1].copy()
        top_models[i][0] = model
        top_models[i][1] = score
        break          
  return sorted(top_models, key=lambda x: x[1], reverse=True)
       

###############################################################################
###############################################################################
###############################################################################
####                                                                       ####
####                      END utility code section                         ####
####                                                                       ####
###############################################################################
###############################################################################
###############################################################################


def calc_label_distrib(data_split):
  info = pd.DataFrame(data_split)[1].value_counts()
  P(info)
  return info 


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
    v = np.random.uniform(low=-1e-7, high=1e-7, size=GLOVE_DIM)
    if nw is not None:
      v = v + GLOVE[nw]
    return v
    
def l_glv(w):    
  """Return lower `w`'s GloVe representation if available, else return 
  a zeros vector."""
  return GLOVE.get(w.lower(), np.zeros(GLOVE_DIM))

def glv(w):    
  return GLOVE.get(w, np.zeros(GLOVE_DIM))

  
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


class FocalLossWithLogits_B(th.nn.Module):
  def __init__(self, alpha=0.3, gamma=2):
    super().__init__()
    assert alpha <= 0.9 and alpha >= 0.1
    self.alpha = alpha
    self.gamma = gamma

  def forward(self, inputs, targets):
    pos_alpha = self.alpha
    neg_alpha = 1 - self.alpha
    eps = 1e-14
    
    y_pred = th.sigmoid(inputs)
    
    pos_pt = th.where(targets==1 , y_pred , th.ones_like(y_pred)) # positive pt (fill all the 0 place in y_true with 1 so (1-pt)=0 and log(pt)=0.0) where pt is 1
    neg_pt = th.where(targets==0 , y_pred , th.zeros_like(y_pred)) # negative pt
    
    pos_pt = th.clamp(pos_pt, eps, 1 - eps)
    neg_pt = th.clamp(neg_pt, eps, 1 - eps)
    
    pos_modulating = th.pow(1-pos_pt, self.gamma) # compute postive modulating factor for correct classification the value approaches to zero
    neg_modulating = th.pow(neg_pt, self.gamma) # compute negative modulating factor
    
    
    pos = - pos_alpha * pos_modulating * th.log(pos_pt) #pos part
    neg = - neg_alpha * neg_modulating * th.log(1 - neg_pt) # neg part
    
    loss = pos + neg  # this is final loss to be returned with some reduction
    
    return th.mean(loss)
      
  def __repr__(self):
    s = self.__class__.__name__ + "(alpha={}, gamma={})".format(
        self.alpha,
        self.gamma,
        )
    return s




class FocalLossWithLogits_A(th.nn.Module):
  def __init__(self, alpha=4, gamma=2):
    super().__init__()
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
               model_name,  # model name
               siam_lyrs,   # layers of the siamese or the individual word encoders
               s_l2,        # applycation of l2 on siamese/paths
               s_bn,        # BN in siams/paths
               c_act,       # apply activation on siam/paths combiner
               separ,       # use separate paths for each word
               layers,      # layers of the final classifier dnn
               inp_drp,     # apply drop on inputs
               o_drp,       # apply drop on each fc
               bn,          # apply BN on each liniar in final dnn
               bn_inp,      # apply BN on inputs
               activ,       # activation (all)
               x_dev,       # x_dev for early stop w. lr decay
               y_dev,       # y_dev for early stop w. lr decay
               s_comb,      # method for combining siams/paths
               rev,         # reverse targets during training/predict
               lr,          # starting lr
               loss,        # loss function name for CL/BCE/FL
               bal,         # apply sample balancing during training
               VF,
               
               cl_m=1,        # if using CL this is margin
               fl_g=2,        # focal loss discount exponent
               lr_decay=0.5,  # lr decay factor
               batch=256,     # batch size
               l2_strength=0, # l2 weight decay
               max_epochs=10000,  # not really used
               max_patience=10,   # maximum patience before reload & lr decay  
               max_fails=40,      # max consecutive fails before stop
               optim=th.optim.Adam,
               no_remove=False,
               device=th.device("cuda" if th.cuda.is_available() else "cpu"),
               ):
    self.model = None
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
    self.no_remove = no_remove
    self.focal_loss_alpha = 0.3 if loss == 'flb' else 4
    self.focan_loss_gamma = fl_g
    self.vector_func_name = VF if type(VF) == str else VF.__name__
    if loss == 'cl' and not rev:
      raise ValueError("CL must receive reversed targets")
    return
  
  
  def define_graph(self):
    if not hasattr(self, 'input_dim'):
      self.input_dim = GLOVE_DIM
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
    elif 'fl' in self.loss_type:
      if self.loss_type == 'flb':
        loss = FocalLossWithLogits_B(
            alpha=self.focal_loss_alpha,
            gamma=self.focan_loss_gamma,
            )
      else:
        loss = FocalLossWithLogits_A(
            alpha=self.focal_loss_alpha,
            gamma=self.focan_loss_gamma,
            )        
    else:
      raise ValueError('unknown loss {}'.format(self.loss_type))
    if self.model is None:
      self.model = self.define_graph()
      P("Initialized model:")
      self.print_model()
      P("  Loss: {}\n\n".format(str(loss) if loss.__class__.__name__ != 'method' else loss.__name__))

    else:
      P("\rFitting already loaded model...\t\t\t")
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
    not_del_fns = []
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
            not_del_fns.append(last_best_fn)
            not_del_fns.append(last_best_fn + '.optim')
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
      not_del_fns.append(best_fn + '.optim')
      if (best_f1 < 0.67) and (not self.no_remove):
        P("  Removing '{}'".format(best_fn))        
        not_del_fns.append(best_fn)
      else:
        os.rename(best_fn, "models/{}.th".format(self.model_name))
    P("  Cleaning after fit...")
    atmps = 0
    while atmps < 10 and len(not_del_fns) > 0:   
      removed = []
      for fn in not_del_fns:
        if os.path.isfile(fn):
          try:
            os.remove(fn)
            removed.append(fn)
            P("  Removed '{}'".format(fn))
          except:
            pass
        else:
          removed.append(fn)            
      not_del_fns = [x for x in not_del_fns if x not in removed]
    return self
  

  def predict_proba(self, X):
    self.model.eval()
    with th.no_grad():
      self.model.to(self.device)
      X = th.tensor(X, dtype=th.float).to(self.device)
      preds = self.model(X)
      if self.loss_type in ['bce', 'fl', 'fla', 'flb']:
        result = th.sigmoid(preds).cpu().numpy().ravel()
        if self.reverse_target:
          result = 1 - result
      else:
        dists = preds.cpu().numpy().ravel()
        result = self._dist_to_proba(dists, eps=0.52)
        
      return result


  def predict(self, X):
    probs = self.predict_proba(X)
    classes = (probs >= 0.5).astype(np.int8)

    return classes
  
  def print_model(self, indent=2):
    P("{}Model name: {}".format(indent * " ",self.model_name))
    P(textwrap.indent(str(self.model), " " * indent))
    P("{}Vector func: {}".format(indent * " ", self.vector_func_name))
    P("{}Trained on {} data.".format(indent * " ", "BALANCED" if self.use_balancing else "raw unbalanced"))
    return
  
  
  def _dist_to_proba(self, y_pred, eps):
    s = -0.75 * y_pred / eps + 1.75
    d = -0.25 * y_pred / eps + 0.25
    sgn = ((eps - y_pred) > 0) + 0 - ((eps - y_pred) < 0)
    
    yproba = ((s + sgn * d) / 2).clip(0)
    return yproba
  
  def save(self, label=None):
    if label is None:
      label = self.model_name    
    fn = 'models/{}.th'.format(label)
    th.save(self.model.state_dict(), fn)
    P("Saved '{}'".format(fn))
    return
  
  def load(self, label=None):
    if label is None:
      label = self.model_name    
    fn = 'models/{}.th'.format(label)
    if not os.path.isfile(fn):
      raise ValueError("Model file '{}' not found!".format(fn))
    if self.model is None:
      self.model = self.define_graph()
    self.model.load_state_dict(th.load(fn))
    P("Loaded model from '{}'".format(fn))
    return
  

  def has_saved(self, label=None):
    if label is None:
      label = self.model_name    
    fn = 'models/{}.th'.format(label)
    return os.path.isfile(fn)
    

  
  
  
  
class EnsembleWrapper():
  def __init__(self, models, vector_func_name, verbose=False):
    self.models = models
    self.vector_func_name = vector_func_name if type(vector_func_name) == str else vector_func_name.__name__
    names = [x.model_name.split('_')[1] for x in self.models]
    self.model_name = "E_" + "_".join(names)
    if verbose:
      P("Initialized ensemble '{}' with {} models using vectorizer '{}':".format(
          self.model_name, len(self.models), self.vector_func_name))
      self.print_models()
            
  def print_models(self):
    P("Ensemble '{}' architecture:".format(self.model_name))
    P("  " + "-"* 80)
    for i,model in enumerate(self.models):
      P("  Model {}/{}".format(i+1, len(self.models)))
      model.print_model()
      P("  " + "-"* 80)
  
  def predict(self, x):
    preds = []
    for model in self.models:
      model_preds = model.predict_proba(x)
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
  for vf in [l_glv, l_glv_rep, glv]:
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
        dct_res = add_res(
            dct_res, 
            model_name, 
            score=round(score * 100,2), 
            P_REC=round(P_REC*100,2), 
            P_PRE=round( report['1']['precision'] * 100, 2),
            VF=vf.__name__,
            )
        df = pd.DataFrame(dct_results).sort_values('SCORE')
        P("\nResults so far:\n{}\n".format(df))
  return dct_res


def run_grid_search(dct_res, trn, dev, non_cl_runs=350, cl_runs=100):
  grids = {
      300: {
          'non_cl' : {
                "siam_lyrs" : [[],[600, 300],[600, 300, 150],],          
                "separ" : [True,False,],      
                "layers" : [[512, 256],[1024, 384, 128]],          
                "inp_drp" : [0.3],  
                "o_drp" : [0.5,],          
                "bn" : [True,False],          
                "bn_inp" : [True,False],          
                "activ" : ['relu',],          
                "lr"  :[0.005,],          
                "s_comb" : ['sqr',],      
                's_l2' : [True,False,],          
                's_bn' :[True,False],          
                'rev' :[True,False,],          
                'loss' : ['bce','fla','flb'],
                'c_act' : [None],                    
                'VF':['l_glv','l_glv_rep',],          
                'bal' : [True,False,],          
                'cl_m' : [None,]        
              },
          'cl' : {
                "siam_lyrs" : [[256, 128],[512, 256],[512, 256, 128],[1024, 512],],          
                "separ" : [False,],      
                "layers" : [[],],          
                "inp_drp" : [0,0.3],  
                "o_drp" : [0.5,],          
                "bn" : [None],          
                "bn_inp" : [True,False],          
                "activ" : ['relu',],          
                "lr"  :[0.0001,],          
                'c_act' : [None,],
                "s_comb" : ['eucl',],      
                's_l2' : [True,],
                's_bn' : [False],          
                'rev' :[True,],          
                'loss' : ['cl',],          
                'VF':['l_glv','l_glv_rep',],          
                'bal' : [True,False,],
              }
          },
      100: {
          'non_cl' : {
                "siam_lyrs" : [[],[200, 100],[200, 100, 50],],          
                "separ" : [True,False,],      
                "layers" : [[368, 64],[512, 256, 64]],          
                "inp_drp" : [0.3],  
                "o_drp" : [0.5,],          
                "bn" : [True,False],          
                "bn_inp" : [True,False],          
                "activ" : ['relu',],          
                "lr"  :[0.005,],          
                "s_comb" : ['sqr',],      
                's_l2' : [True,False,],          
                's_bn' :[True,False],          
                'rev' :[True,False,],          
                'loss' : ['bce','fla','flb'],
                'c_act' : [None],                    
                'VF':['l_glv','l_glv_rep',],          
                'bal' : [True,False,],          
                'cl_m' : [None,]        
              },
          'cl' : {
                "siam_lyrs" : [[128, 64],[256, 128],[256, 128, 64],[512, 256],],          
                "separ" : [False,],      
                "layers" : [[],],          
                "inp_drp" : [0,0.3],  
                "o_drp" : [0.5,],          
                "bn" : [None],          
                "bn_inp" : [True,False],          
                "activ" : ['relu',],          
                "lr"  :[0.0001,],          
                'c_act' : [None,],
                "s_comb" : ['eucl',],      
                's_l2' : [True,],
                's_bn' : [False],          
                'rev' :[True,],          
                'loss' : ['cl',],          
                'VF':['l_glv','l_glv_rep',],          
                'bal' : [True,False,],
              }
          }
      
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
  dct_main_grid = grids[GLOVE_DIM]
  options1 = prepare_grid_search(dct_main_grid['non_cl'], valid_fn=filter_func, nr_trials=non_cl_runs)
  options2 = prepare_grid_search(dct_main_grid['cl'], valid_fn=filter_func, nr_trials=cl_runs)
  options = options1 + options2
  options = [options[x] for x in np.random.choice(len(options), size=len(options), replace=False)]
  timings = []
  t_left = np.inf
  top_models = []
  k=3
  last_ensemble = ''
  ver = '4'
  GD = str(GLOVE_DIM)[0]
  for grid_iter, option in enumerate(options):    
    g_type = 'G' if option['VF'] == 'l_glv' else 'R'
    model_name = 'H3_{}{}{}D{:03d}'.format(
        ver, g_type, GD, grid_iter+1)
    P("\n\n" + "=" * 70)
    P("Running grid search iteration {}/{}\n '{}' : {}".format(
        grid_iter+1, len(options), model_name, option))
#    for k in option:
#      P("  {}={},".format(k,option[k] if type(option[k]) != str else "'" + option[k] + "'"))
    P("  Time left for grid search completion: {:.1f} hrs".format(t_left / 3600))
    VF = option['VF']
    vector_func = globals()[VF]
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
    assert round(score,3) == round(report['macro avg']['f1-score'],3), "ERROR:  score {} differs from report score {}".format(
        round(score,3), round(report['macro avg']['f1-score'],3))
    P_REC = report['1']['recall']
    ####
    t_res = time() - _t_start
    timings.append(t_res)
    t_left = (len(options) - grid_iter - 1) * np.mean(timings)
    dct_res = add_res(
        dct=dct_res, 
        model_name=model_name, 
        score=round(score * 100, 2), 
        P_REC=round(P_REC * 100, 2),
        **option)
    if len(top_models) >= k:
      P("Testing ensemble so far...")
      ensemble = EnsembleWrapper(
          [x[0] for x in top_models], 
          vector_func_name=VF)
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
          VF=VF,
          )      
    df = pd.DataFrame(dct_res).sort_values('SCORE')
    P("Results so far:\n{}".format(df.iloc[-100:]))
    df.to_csv("models/"+_date+"_results.csv")
  # end grid
  return df
      
def vect_neighbors(v, df):
  import scipy
  distfunc = scipy.spatial.distance.cosine
  dists = df.apply(lambda x: distfunc(v, x), axis=1)
  return dists.sort_values().head()


def ensemble_test(clfs, trn, dev, vect_func):
  res = {'MODEL':[],'SCORE':[]}

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
  
  for model in clfs:
    P("Testing model '{}' on data vectorized with '{}'".format(
        model.model_name, vect_func.__name__))
    y_pred = model.predict(x_dev)
    report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
    res = add_res(
        res, 
        model.model_name, 
        score=round(report['macro avg']['f1-score'] * 100,2),
        pos_f1=report['1']['f1-score'] * 100,
        pos_rc=report['1']['recall'] * 100,
        pos_pr=report['1']['precision'] * 100,
        )
    
  
  ens = EnsembleWrapper(
      clfs, 
      vector_func_name=vect_func.__name__,
      )
  y_pred = ens.predict(x_dev)
  report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
  res = add_res(
      res, 
      ens.model_name, 
      score=round(report['macro avg']['f1-score'] * 100, 2),
      pos_f1=report['1']['f1-score'] * 100,
      pos_rc=report['1']['recall'] * 100,
      pos_pr=report['1']['precision'] * 100,
      )
  df = pd.DataFrame(res).sort_values('SCORE')
  P("Results:\n{}".format(df.iloc[-50:]))
  return df



def ensemble_train_test(dct_models_params, trn, dev, vect_funcs, n_models=None):
  clfs = []
  res = {'MODEL':[],'SCORE':[]}
  for ii, model_name in enumerate(dct_models_params):
    model_params = dct_models_params[model_name]
    c_vect_func = globals()[model_params['VF']]
    P("\n" + "-"*80)
    P("Loading or training ensemble component {}/{}: '{}' with vect_func: '{}'...".format(
        ii+1, len(dct_models_params), model_name, c_vect_func.__name__))
    x_dev, y_dev = nli.word_entail_featurize(
        data=dev, 
        vector_func=c_vect_func, 
        vector_combo_func=arr
        )
      
    clf = WordEntailClassifier(model_name=model_name, 
                               x_dev=x_dev,
                               y_dev=y_dev,
                               **model_params)
    if clf.has_saved():
      clf.load()
    else:
      x_trn, y_trn = nli.word_entail_featurize(
          data=trn, 
          vector_func=c_vect_func, 
          vector_combo_func=arr
          )  
      clf.fit(x_trn, y_trn)
      
    clfs.append(clf)
    y_pred = clf.predict(x_dev)
    report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
    res = add_res(
        res, 
        model_name,         
        score=round(report['macro avg']['f1-score'] * 100,2),
        pos_F1=report['1']['f1-score'] * 100,
        pos_Rec=report['1']['recall'] * 100,
        pos_Pre=report['1']['precision'] * 100,
        vf=c_vect_func.__name__,
        )
    df = pd.DataFrame(res).sort_values('SCORE')
    P("Results:\n{}".format(df))
  
  best_ens = None
  best_mf1 = 0
  if n_models is None:
    lst_n_models = list(range(2,6))
  elif type(n_models) == int:
    lst_n_models = [n_models]
  else:
    lst_n_models = n_models

  all_clfs_combs = []
  for n_clfs in lst_n_models:
    all_clfs_combs = all_clfs_combs + list(combinations(clfs, n_clfs))
  P("Testing/searching for best ensemble...")
  for i, selected_clfs in enumerate(all_clfs_combs):    
    for vect_func in vect_funcs:
      Pr(" Testing ensmble {}/{} ({:.2f}%) with {} models and {} vector func\t".format(
          i+1, len(all_clfs_combs), (i+1)/len(all_clfs_combs) * 100, 
          len(selected_clfs), vect_func.__name__))
      x_trn, y_trn = nli.word_entail_featurize(
          data=trn, 
          vector_func=vect_func, 
          vector_combo_func=arr,
          )
      x_dev, y_dev = nli.word_entail_featurize(
          data=dev, 
          vector_func=vect_func, 
          vector_combo_func=arr,
          )
        
      ens = EnsembleWrapper(
          selected_clfs, 
          vector_func_name=vect_func.__name__,
          verbose=False,
          )
      y_pred = ens.predict(x_dev)
      report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
      score = round(report['macro avg']['f1-score'] * 100,2)
      if score > best_mf1:
        best_mf1 = score
        best_ens = ens
      res = add_res(
          res, 
          ens.model_name, 
          score=score,
          pos_F1=report['1']['f1-score'] * 100,
          pos_Rec=report['1']['recall'] * 100,
          pos_Pre=report['1']['precision'] * 100,
          vf=vect_func.__name__,        
          )
  df = pd.DataFrame(res).sort_values('SCORE')
  df.to_csv('models/{}_ensembles.csv'.format(_date))
  P("Results:\n{}".format(df.iloc[-50:]))
  return best_ens, df


def test_model(model, trn, dev, dct_res, model_sufix=''):
  model_name = model.model_name
  VECT_FUNC = globals()[model.vector_func_name]
  P("Testing model '{}' with vector func '{} on train: {},  dev: {}".format(
      model_name, VECT_FUNC.__name__, len(trn), len(dev)))
  dct_res_extra = {
      "DATA" : [],
      "VF" : [],
      "Macro-F1" : [],
      "Pos F1" : [],
      "Pos Recall" : [],
      "Pos Precis" : [],
      }
  def _log_data_result(dn, mf1, pf1, rec, prec, vf):
    dct_res_extra['DATA'].append(dn)
    dct_res_extra['VF'].append(vf.__name__)
    dct_res_extra['Macro-F1'].append(mf1)
    dct_res_extra['Pos F1'].append(pf1)
    dct_res_extra['Pos Recall'].append(rec)
    dct_res_extra['Pos Precis'].append(prec)

  x_trn, y_trn = nli.word_entail_featurize(
      data=trn, 
      vector_func=VECT_FUNC, 
      vector_combo_func=arr
      )
    
  x_dev, y_dev = nli.word_entail_featurize(
      data=dev, 
      vector_func=VECT_FUNC, 
      vector_combo_func=arr
      )
  y_pred = model.predict(x_dev)
  report = classification_report(y_dev, y_pred, digits=3, output_dict=True)
  mf1 = round(report['macro avg']['f1-score'] * 100,2)
  pf1 = round(report['1']['f1-score'] * 100,2)
  prc = round(report['1']['recall'] * 100,2)
  ppr = round(report['1']['precision'] * 100,2)
  dct_res = add_res(
      dct=dct_res,
      model_name=model_name + model_sufix,
      score=mf1,
      P_REC=prc,
      P_PRE=ppr,
      VF=model.vector_func_name,
      )
  P("\n{}\n  MF1: {:.2f}, 1F1: {:.2f}, 1R: {:.2f}, 1P: {:.2f}\n".format(model_name, mf1, pf1, prc, ppr))
  _log_data_result(
      'dev_full', 
      mf1,
      pf1,
      prc,
      ppr,
      VECT_FUNC,
      )
  y_pred = model.predict(x_trn)
  report = classification_report(y_trn, y_pred, digits=3, output_dict=True)
  _log_data_result(
      'train_full', 
      report['macro avg']['f1-score'] * 100,
      report['1']['f1-score'] * 100,
      report['1']['recall'] * 100,
      report['1']['precision'] * 100,
      VECT_FUNC,
      )
  
  for test_name in dct_data_GLOBAL:
    P("\n\nTesting on '{}':".format(test_name))
    x, y = nli.word_entail_featurize(
        data=dct_data_GLOBAL[test_name], 
        vector_func=VECT_FUNC, 
        vector_combo_func=arr
        )
    yh = model.predict(x)
    P(classification_report(y, yh, digits=3))
    report = classification_report(y, yh, digits=3, output_dict=True)
    _log_data_result(
        test_name, 
        report['macro avg']['f1-score'] * 100,
        report['1']['f1-score'] * 100,
        report['1']['recall'] * 100,
        report['1']['precision'] * 100,
        VECT_FUNC,
        )
  df = pd.DataFrame(dct_res_extra).sort_values('Macro-F1')
  P(df)
  return dct_res


def train_test_config(model_name, model_config, trn, dev, dct_res):
    
  VECT_FUNC = globals()[model_config['VF']]
  

  x_dev, y_dev = nli.word_entail_featurize(
      data=dev, 
      vector_func=VECT_FUNC, 
      vector_combo_func=arr
      )

  model = WordEntailClassifier(      
    batch=256,
    x_dev=x_dev,
    y_dev=y_dev,
    model_name=model_name,
    optim=th.optim.Adam,
    **model_config      
    )
  
  if not model.has_saved():
    _ = nli.wordentail_experiment(
          train_data=trn,
          assess_data=dev,
          vector_func=VECT_FUNC,
          vector_combo_func=arr,
          model=model,    
        )
  else:
    model.load()
  
  dct_res = test_model(model, trn, dev, dct_res)
  
  return dct_res


def test_model_configs(dct_test_models, dct_res):
  for _model_name, _model_params in dct_test_models.items():
    VECT_FUNC = globals()[_model_params['VF']]
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
    _model = WordEntailClassifier(      
      batch=256,
      x_dev=_x_dev,
      y_dev=_y_dev,
      model_name=_model_name,
      optim=th.optim.Adam,
      **_model_params      
      )  
    _model.load()
    _y_pred = _model.predict(_x_dev)
    report = classification_report(_y_dev, _y_pred, digits=3, output_dict=True)
    mf1 = round(report['macro avg']['f1-score'] * 100,2)
    pf1 = round(report['1']['f1-score'] * 100,2)
    prc = round(report['1']['recall'] * 100,2)
    ppr = round(report['1']['precision'] * 100,2)
    dct_res = add_res(
        dct_res,
        model_name=_model_name,
        score=mf1,
        P_REC=prc,
        P_PRE=ppr,
        VF=_model_params['VF'],
        pf1=pf1,
        )
  df_scores = pd.DataFrame(dct_res).sort_values('SCORE')
  P("-" * 80 + "\nResults:\n")
  P(df_scores)
  return dct_res
  
      

def train_models(dct_models, trn, dev, no_remove=False):
  all_models = []
  for i, (model_name, model_params) in enumerate(dct_models.items()):
    P("\nLoading or training/saving model {}/{}".format(i+1, len(dct_models)))
    VECT_FUNC = globals()[model_params['VF']]
    x_trn, y_trn = nli.word_entail_featurize(
        data=trn, 
        vector_func=VECT_FUNC, 
        vector_combo_func=arr
        )
    x_dev, y_dev = nli.word_entail_featurize(
        data=dev, 
        vector_func=VECT_FUNC, 
        vector_combo_func=arr
        )
    model = WordEntailClassifier(      
      batch=256,
      x_dev=x_dev,
      y_dev=y_dev,
      model_name=model_name,
      optim=th.optim.Adam,
      no_remove=no_remove,
      **model_params      
      )  
    if model.has_saved():
      model.load()
      all_models.append(model)
      continue
    _ = nli.wordentail_experiment(
          train_data=train_data,
          assess_data=dev_data,
          vector_func=VECT_FUNC,
          vector_combo_func=arr,
          model=model,    
        )
    all_models.append(model)
  return all_models
    

###############################################################################
###############################################################################
###############################################################################
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)  
  
CALC_BASELINES  = False

RUN_GRID        = False
RUN_ALL_TRAIN   = False
RUN_ALL_TEST    = False
RUN_SINGLE      = False
RUN_ENS_SEARCH  = False


RUN_ORIGINAL_SYSTEM = False

RUN_BAKE_ENS = True

utils.fix_random_seeds()
GLOVE_DIM = 300


if "GLOVE" not in globals() or len(next(iter(GLOVE.values()))) != GLOVE_DIM:
  P("Loading GloVe-{}...".format(GLOVE_DIM))
  GLOVE = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(GLOVE_DIM)))  
  P("GloVe-{} loaded.".format(GLOVE_DIM))

with open(wordentail_filename) as f:
  wordentail_data = json.load(f)  

  
train_data = wordentail_data['word_disjoint']['train']
dev_data = wordentail_data['word_disjoint']['dev']
dct_results = OrderedDict({'MODEL':[], 'SCORE':[], 'P_REC': []})

test_data_filename = os.path.join(
  NLIDATA_HOME,
  "bakeoff-wordentail-data",
  "nli_wordentail_bakeoff_data-test.json")

with open(test_data_filename, encoding='utf8') as f:
    dct_bake_data = json.load(f)
    
bake_dev = dct_bake_data['word_disjoint']['test']

dev_data = bake_dev



maybe_find_glove_replacement('aromatization')
glv_analysis = test_glove_vs_data(train_data, dev_data)
dct_data_GLOBAL = glv_analysis[0]
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
  dct_results = run_grid_search(
      train_data, 
      dev_data,
      non_cl_runs=200,
      cl_runs=50)
    

#"H3_3G017"
#"H3_3G171"
#"H3_2G197"
#"H3_3G093"
#"H3_2R321"
#"H3_3R020"
#'H3_3G283"
  
# E_2G424_3G017_3G015_3G093_2R321
  


__dct_all_models = {
  # l_glv      
  300 : {
#  'B3_3G017' : {'siam_lyrs': [], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.005, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': False},  
#  'B3_3G396' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sub', 's_l2': False, 's_bn': True, 'rev': True, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},
#  'B3_3G197' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sub', 's_l2': False, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},  
#  'B3_3G015' : {'siam_lyrs': [256, 128, 64], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': False, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sqr', 's_l2': True, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},
#  'B3_2G424': {'siam_lyrs': [256, 128], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sub', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'fl', 'c_act': None, 'VF': 'l_glv', 'bal': True},
#  'B3_3G093' : {'siam_lyrs': [], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'tanh', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv', 'bal': False},
#  'B3_3G171' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': False, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},  
#

  'H3_2G424': {'siam_lyrs': [256, 128], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sub', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'fl', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  'H3_3G017' : {'siam_lyrs': [], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.005, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': False},  
  'H3_3G015' : {'siam_lyrs': [256, 128, 64], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': False, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sqr', 's_l2': True, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  'H3_3G093' : {'siam_lyrs': [], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'tanh', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv', 'bal': False},
  "H3_2G197" : {'siam_lyrs': [512, 256], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': False, 'activ': 'tanh', 'lr': 0.01, 's_comb': 'abs', 's_l2': False, 's_bn': False, 'rev': True, 'loss': 'fl', 'c_act': None, 'bal': False, 'VF':'l_glv'},      
  'H3_2G500': {'siam_lyrs': [], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'bce', 'c_act': None, 'VF' : 'l_glv', 'bal': False},
  'H3_3G171' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': False, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},  
  'H3_3G197' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sub', 's_l2': False, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},  
  'H3_3G396' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sub', 's_l2': False, 's_bn': True, 'rev': True, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  
  # l_glv_rep
  "H3_2R321" : {'siam_lyrs': [256, 128, 64], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': False, 'activ': 'sigmoid', 'lr': 0.01, 's_comb': 'sqr', 's_l2': False, 's_bn': True, 'rev': True, 'loss': 'bce', 'c_act': None, 'bal': True, 'VF' : 'l_glv_rep'},
  "H3_2R335" : {'siam_lyrs': [256, 128], 'separ': False, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': True, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sub', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'fl', 'c_act': None, 'VF':'l_glv_rep', 'bal': False},  
  'H3_3R020' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv_rep', 'bal': False},

  'H3_4R3D1' : {'siam_lyrs': [600, 300], 'separ': True, 'layers': [512, 128], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': True, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sqr', 's_l2': True, 's_bn': False, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv_rep', 'bal': False, 'cl_m': None},
  
  # CL l_glv  
  'H3_3G283' : {'siam_lyrs': [512, 256], 'separ': False, 'layers': [], 'inp_drp': 0, 'o_drp': 0.5, 'bn': None, 'bn_inp': True, 'activ': 'relu', 'lr': 0.0001, 'c_act': None, 's_comb': 'eucl', 's_l2': True, 's_bn': False, 'rev': True, 'loss': 'cl', 'VF': 'l_glv', 'bal': False, 'cl_m' : 1},

  'H3_4G1' : {'siam_lyrs': [512, 256], 'separ': False, 'layers': [], 'inp_drp': 0, 'o_drp': 0.5, 'bn': None, 'bn_inp': True, 'activ': 'relu', 'lr': 0.0001, 'c_act': None, 's_comb': 'eucl', 's_l2': True, 's_bn': False, 'rev': True, 'loss': 'cl', 'VF': 'l_glv', 'bal': False, 'cl_m' : 0.2},
  },
      
  100 : {
  'H3_4R1D236' : {'siam_lyrs': [200, 100], 'separ': True, 'layers': [368, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': True, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sqr', 's_l2': True, 's_bn': False, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv_rep', 'bal': False, 'cl_m': None}
      },
  }
  
__dct_models = __dct_all_models[GLOVE_DIM]

##############################################################################
if RUN_ALL_TRAIN:
  _ = train_models(__dct_models, trn=train_data, dev=dev_data, no_remove=True)
      
##############################################################################
if RUN_ALL_TEST:
  dct_results = test_model_configs(
      __dct_models, 
      dct_res=dct_results)
     

##############################################################################
if RUN_SINGLE:  
  __model_name = "B3_3G017"
  __model_params = __dct_models[__model_name]
  dct_results = train_test_config(
      __model_name, 
      __model_params,
      trn=train_data,
      dev=dev_data,
      dct_res=dct_results)
  
    
  
if RUN_ENS_SEARCH:
  VECT_FUNCS = [l_glv, l_glv_rep]
    
      
  
  ens_model, df_results = ensemble_train_test(
      dct_models_params=__dct_models,
      trn=train_data, 
      dev=dev_data, 
      vect_funcs=VECT_FUNCS,
      n_models=[3,4,5],
      )

  dct_results = test_model(
      ens_model,
      train_data,
      dev_data,
      dct_res=dct_results,
      )

  P("Best ensemble:")  
  res = nli.wordentail_experiment(
      train_data=train_data,
      assess_data=dev_data,
      vector_func=globals()[ens_model.vector_func_name],
      vector_combo_func=arr,
      model=ens_model,
      )

##############################################################################

sol_models_params = {
  'H3_2G424': {'siam_lyrs': [256, 128], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sub', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'fl', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  'H3_3G017' : {'siam_lyrs': [], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.005, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': False},  
  'H3_3G015' : {'siam_lyrs': [256, 128, 64], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': False, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sqr', 's_l2': True, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  'H3_3G093' : {'siam_lyrs': [], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'tanh', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv', 'bal': False},
  "H3_2R321" : {'siam_lyrs': [256, 128, 64], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': False, 'activ': 'sigmoid', 'lr': 0.01, 's_comb': 'sqr', 's_l2': False, 's_bn': True, 'rev': True, 'loss': 'bce', 'c_act': None, 'bal': True, 'VF' : 'l_glv_rep'},
    }

best_sol_models_params = {
  'H3_2G424': {'siam_lyrs': [256, 128], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': False, 'activ': 'relu', 'lr': 0.005, 's_comb': 'sub', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'fl', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  'H3_3G093' : {'siam_lyrs': [], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'tanh', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv', 'bal': False},
  'H3_3R020' : {'siam_lyrs': [256, 128], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv_rep', 'bal': False},
  'H3_2G500': {'siam_lyrs': [], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': False, 'rev': False, 'loss': 'bce', 'c_act': None, 'VF' : 'l_glv', 'bal': False},
    }

bakeoff_sol_models_params = {
  'H3_3G017' : {'siam_lyrs': [], 'separ': True, 'layers': [128, 32], 'inp_drp': 0.3, 'o_drp': 0.2, 'bn': True, 'bn_inp': True, 'activ': 'sigmoid', 'lr': 0.005, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': False},  
  'H3_3G015' : {'siam_lyrs': [256, 128, 64], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0.5, 'bn': False, 'bn_inp': False, 'activ': 'relu', 'lr': 0.01, 's_comb': 'sqr', 's_l2': True, 's_bn': True, 'rev': False, 'loss': 'fla', 'c_act': None, 'VF': 'l_glv', 'bal': True},
  'H3_3G093' : {'siam_lyrs': [], 'separ': True, 'layers': [256, 128, 64], 'inp_drp': 0.3, 'o_drp': 0, 'bn': True, 'bn_inp': True, 'activ': 'tanh', 'lr': 0.01, 's_comb': 'abs', 's_l2': True, 's_bn': True, 'rev': True, 'loss': 'flb', 'c_act': None, 'VF': 'l_glv', 'bal': False},
    }



if RUN_ORIGINAL_SYSTEM:  
  
  
  USE_NEW_SPLIT = True
  
  
    
  
  for SOL_VECT_FUNC in [l_glv_rep, l_glv]:
    SUFIX = '_R' if SOL_VECT_FUNC == l_glv_rep else '_g'
    
    P("Train label dist:")
    calc_label_distrib(train_data)
    P("Dev label dist:")
    calc_label_distrib(dev_data)
    # reduce DEV and add to train
    new_dev_size = 500
    train_added = len(dev_data) - new_dev_size
    dev_to_train_idxs = np.random.choice(len(dev_data), size=train_added, replace=False)
    added_train_data = [dev_data[x] for x in dev_to_train_idxs]
    
    
    base_sol_models = train_models(
        sol_models_params, 
        trn=train_data, 
        dev=dev_data,
        )
  
    base_solution_ens = EnsembleWrapper(
        models=base_sol_models,
        vector_func_name=SOL_VECT_FUNC.__name__,
        verbose=True
        )  
    P("Ensemble results with standard train/dev distrib:")
    dct_results = test_model(
        base_solution_ens,
        trn=train_data,
        dev=bake_dev, ####### !!!!!!!!!!!!!!!!!!!!!!
        dct_res=dct_results,
        model_sufix=SUFIX,
        )
  
    if USE_NEW_SPLIT:  
      new_train_data = train_data + added_train_data
      new_dev_data = [dev_data[x] for x in range(len(dev_data)) if x not in dev_to_train_idxs]
      new_model_dict = {k+'XT':v for k,v in sol_models_params.items()}
    
      P("NEW Train label distrib:")
      calc_label_distrib(new_train_data)
      P("NEW Dev label distrib:")
      calc_label_distrib(new_dev_data)
    
      sol_models = train_models(
          new_model_dict,
          trn=new_train_data,
          dev=new_dev_data) 
    
      
      solution_ensemble = EnsembleWrapper(
          models=sol_models,
          vector_func_name=SOL_VECT_FUNC.__name__,
          verbose=True
          )  
      
      P("\nTEST on original train/dev splits")
      dct_results = test_model(
          solution_ensemble, 
          trn=train_data,
          dev=bake_dev, ####### !!!!!!!!!!!!!!!!!!!!!!
          dct_res=dct_results,
          model_sufix='_S' + SUFIX)
      P("\nTEST on NEW train/dev splits")
      dct_results = test_model(
          solution_ensemble, 
          trn=new_train_data,
          dev=bake_dev, ####### !!!!!!!!!!!!!!!!!!!!!!
          dct_res=dct_results,
          model_sufix=SUFIX)
        
    else:
      solution_ensemble = base_solution_ens
  
#    solution_result = nli.wordentail_experiment(
#        train_data=train_data,
#        assess_data=dev_data,
#        vector_func=globals()[solution_ensemble.vector_func_name],
#        vector_combo_func=arr,
#        model=solution_ensemble,
#        )
  
#  nli.bake_off_evaluation(solution_result)

if RUN_BAKE_ENS:

  base_sol_models = train_models(
      sol_models_params, 
      trn=train_data, 
      dev=dev_data,
      )

  base_solution_ens = EnsembleWrapper(
      models=base_sol_models,
      vector_func_name=l_glv,
      verbose=True
      )  
  
  dct_results = test_model(
      base_solution_ens,
      train_data,
      dev_data,
      dct_res=dct_results,
      )    
  
  ############

  bakeoff_sol_models = train_models(
      bakeoff_sol_models_params, 
      trn=train_data, 
      dev=dev_data,
      )

  bakeoff_solution_ens = EnsembleWrapper(
      models=bakeoff_sol_models,
      vector_func_name=l_glv,
      verbose=True
      )  
  
  dct_results = test_model(
      bakeoff_solution_ens,
      train_data,
      dev_data,
      dct_res=dct_results,
      )    

  ############


  best_sol_models = train_models(
      best_sol_models_params, 
      trn=train_data, 
      dev=dev_data,
      )

  best_solution_ens = EnsembleWrapper(
      models=best_sol_models,
      vector_func_name=l_glv,
      verbose=True
      )  
  
  dct_results = test_model(
      best_solution_ens,
      train_data,
      dev_data,
      dct_res=dct_results,
      )    


df = pd.DataFrame(dct_results).sort_values('SCORE')
P(df)