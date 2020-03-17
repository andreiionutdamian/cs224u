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
        vector_func=glove_vec, 
        vector_combo_func=vcf
        )
    X_dev, y_dev = nli.word_entail_featurize(
        data=dev_data,  
        vector_func=glove_vec, 
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

  
def prepare_grid_search(params_grid, valid_fn, nr_trials):
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
  cleaned_iters = [x for x in grid_iterations if valid_fn(x)]
  n_options = len(cleaned_iters)
  idxs = np.arange(n_options)
  np.random.shuffle(idxs)
  idxs = idxs[:nr_trials]
  P("Generated {} random grid-search iters out of a total of {} iters".format(
      len(idxs), n_options))
  return [cleaned_iters[i] for i in idxs]

        
  
###############################################################################
###############################################################################
###############################################################################
####                                                                       ####
####                      END utility code section                         ####
####                                                                       ####
###############################################################################
###############################################################################
###############################################################################

def lower_glove_vec(w):    
  """Return `w`'s GloVe representation if available, else return 
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
  return miss_train, miss_dev


class ThSiameseCombine(th.nn.Module):
  def __init__(self, input_dim, method, norm_each):
    super().__init__()
    self.method = method
    self.input_dim = input_dim
    self.norm_each = norm_each
    if method in ['sub','abs','sqr']:
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
    if self.norm_each:
      path1 = th.nn.functional.normalize(path1, p=2, dim=1)
      path2 = th.nn.functional.normalize(path2, p=2, dim=1)
      
    if self.method == 'sub':
      th_x = path1 - path2
    elif self.method == 'cat':
      th_x = th.cat((path1, path2), dim=1)
    elif self.method == 'abs':
      th_x = (path1 - path2).abs()
    elif self.method == 'sqr':
      th_x = th.pow(path1 - path2, 2)
    elif self.method == 'eucl':
      th_x = th.pairwise_distance(path1, path2, keepdim=True)
    
    return th_x
  
  def __repr__(self):
    s = self.__class__.__name__ + "(input_dim={}, output_dim={}, method='{}', norm_each={})".format(
        self.input_dim,
        self.output_dim,
        self.method,
        self.norm_each,
        )
    return s

class ThWordEntailModel(th.nn.Module):
  def __init__(self,
               input_dim,
               siam_lyrs,
               siam_norm,
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
      if input_drop > 0:
        paths[path_no].append(th.nn.Dropout(input_drop))
      if bn_inputs:
        paths[path_no].append(th.nn.BatchNorm1d(last_output))
      if len(siam_lyrs) > 0:
        for i, layer in enumerate(siam_lyrs):
          paths[path_no].append(th.nn.Linear(last_output, layer, bias=not bn))
          if bn:
            paths[path_no].append(th.nn.BatchNorm1d(layer))
          paths[path_no].append(self.get_activation(activ))
          if other_drop > 0:
            paths[path_no].append(th.nn.Dropout(other_drop))
          last_output = layer
    if self.separate :
      self.path1_layers = th.nn.ModuleList(paths[0])
      self.path2_layers = th.nn.ModuleList(paths[1])
    else:
      self.siam_layers = th.nn.ModuleList(paths[0])
      
    
    siam_combine = ThSiameseCombine(last_output, self.smethod, norm_each=self.siam_norm)
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
               siam_lyrs,
               s_l2,
               separ,
               layers,                 
               input_drop,
               other_drop,
               bn,
               bn_inputs,
               activ,
               x_dev,
               y_dev,
               s_comb,
               rev,
               lr,
               loss,
               lr_decay=0.5,
               batch=256,
               l2_strength=0,
               max_epochs=10000,
               max_patience=10,
               max_fails=40,
               optim=th.optim.Adam,
               device=th.device("cuda" if th.cuda.is_available() else "cpu"),
               ):
    self.input_drop = input_drop
    self.layers = layers
    self.siam_lyrs = siam_lyrs
    self.input_drop = input_drop
    self.other_drop = other_drop
    self.bn = bn
    self.bn_inputs = bn_inputs
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
        )
    return model    
      
  
  def fit(self, X, y):
    utils.fix_random_seeds()
    # Data prep:
    X = np.array(X).astype(np.float32)
    n_obs = X.shape[0]
    # here is a trick: we consider the words that entail those that have
    # minimal distance if using siamese
    y = np.array(y).reshape(-1,1).astype(np.float32)
    if self.reverse_target:
      y = 1 - y
    self.input_dim = X.shape[-1]
    X = th.tensor(X, dtype=th.float32)
    y = th.tensor(y, dtype=th.float32)
    dataset = th.utils.data.TensorDataset(X, y)
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=self.batch_size, shuffle=True,
        pin_memory=True)
    # Optimization:
    if self.loss_type == 'bce':
      loss = th.nn.BCEWithLogitsLoss()
    elif self.loss_type == 'cl':
      self.margin = 1.0
      loss = self._constrastive_loss
    else:
      raise ValueError('unknown loss {}'.format(self.loss_type))
    if not hasattr(self, "model"):
      self.model = self.define_graph()
      print("Initialized model {}:\n{}\n{}".format(
          self.model.__class__.__name__,
          textwrap.indent(str(self.model), " " * 2),
          textwrap.indent("Loss: " + (str(loss) if loss.__class__.__name__ != 'method' else loss.__name__), " " * 2),
          ))

    else:
      print("\rFitting already loaded model...\t\t\t", end='', flush=True)
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
        best_fn = "models/_tmp_we_ep_{:03}_f1_{:.4f}.th".format(epoch, macrof1)
        best_epoch = epoch
        P("\rFound new best macro-f1 {:.4f} > {:.4f} at epoch {}. \t\t\t".format(macrof1, best_f1, epoch))
        best_f1 = macrof1
        th.save(self.model.state_dict(), best_fn)
        th.save(optimizer.state_dict(), best_fn + '.optim')
        if last_best_fn != '':
          os.remove(last_best_fn)
          os.remove(last_best_fn + '.optim')
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
          P("\nPatience reached {}/{} - reloaded from ep {} reduced lr from {:.1e} to {:.1e}".format(
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
      if self.loss_type == 'bce':
        result = th.sigmoid(preds).cpu().numpy()
      else:
        result = preds.cpu().numpy()
      return result


  def predict(self, X):
    probs = self.predict_proba(X)
    if self.loss_type == 'bce':
      classes = (probs >= 0.5).astype(np.int8).ravel()
      if self.reverse_target:
        # we have to reverse
        classes = 1 - classes
    else:
      thr = self.margin / 2
      classes = (probs >= thr).astype(np.int8).ravel()
      if self.reverse_target:
        # we have to reverse
        classes = 1 - classes
    return classes
  
  
  def _constrastive_loss(self, dist, gold):
    th_d_sq = th.pow(dist, 2)
    th_d_sqm = th.pow(th.clamp(self.margin - dist, 0), 2)
    loss = (1 - gold) * th_d_sq + gold * th_d_sqm
    return loss.mean()
    

def add_res(dct, model_name, score, **kwargs):
  n_existing = len(dct['MODEL'])
  dct['MODEL'].append(model_name)
  dct['SCORE'].append(score)
  for key in kwargs:
    if key not in dct:
      dct[key] = [' ' ] * n_existing
    dct[key].append(kwargs[key])
  for k in dct:
    if len(dct[k]) < (n_existing + 1):
      dct[k] = dct[k] + [' '] * ((n_existing + 1) - len(dct[k]))
  return dct


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
  
  for vcf in baseline_vector_combo_funcs:
    for model_name in baseline_model_facts:
      P("=" * 70)
      P("Running baseline model '{}' with '{}'".format(
          model_name, vcf.__name__))
      model = baseline_model_facts[model_name]()
      res = nli.wordentail_experiment(
          train_data=trn,
          assess_data=dev,
          vector_func=lower_glove_vec,
          vector_combo_func=vcf,
          model=model,
          )
      score = res['macro-F1']
      add_res(dct_res, model_name, score, VECT=vcf.__name__)
      df = pd.DataFrame(dct_results).sort_values('SCORE')
      P("\nResults so far:\n{}\n".format(df))
  return dct_res


def run_grid_search(dct_res, trn, dev):
  grid = {
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
          [],
          [256, 128, 64],
          [128, 32],
          ],
          
      "input_drop" : [
          0,
          0.3
          ],
  
      "other_drop" : [
          0,
          0.2,
          0.5,
          ],
          
      "bn" : [
          True,
          False
          ],
          
      "bn_inputs" : [
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
          0.005,
          0.0005,
          0.00005,
          ],
          
      "s_comb" : [
          'sub',
          'cat',
          'abs',
          'sqr',
          'eucl',
          ],
      
      's_l2' : [
          True,
          False,
          ],
          
      'rev' :[
          True,
          False,
          ],
          
      'loss' : [
          'bce',
          'cl',
          ],
        
      
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
    if grid_iter['layers'] == [] and grid_iter['other_drop'] != 0 :
      return False
    return True
  
  options = prepare_grid_search(grid, valid_fn=filter_func, nr_trials=100)
  timings = []
  t_left = np.inf
  for grid_iter, option in enumerate(options):
    model_name = 'H3_v1_{:02}'.format(grid_iter+1)
    P("\n\n" + "=" * 70)
    P("Running grid search iteration {}/{} '{}': {}".format(
        grid_iter+1, len(options), model_name, option))
    P("  Time left for grid search completion: {:.1f} hrs".format(t_left / 3600))
  
    vcf = concat
    _t_start = time()
    #### we need this ...
    x_dev, y_dev = nli.word_entail_featurize(
        data=dev, 
        vector_func=lower_glove_vec, 
        vector_combo_func=arr
        )
    model = WordEntailClassifier(
        x_dev=x_dev,
        y_dev=y_dev,
        **option)
    res = nli.wordentail_experiment(
            train_data=trn,
            assess_data=dev,
            vector_func=lower_glove_vec,
            vector_combo_func=arr,
            model=model,
            max_patience=20,
            )
    score = res['macro-F1']
    ####
    t_res = time() - _t_start
    timings.append(t_res)
    t_left = (len(options) - grid_iter - 1) * np.mean(timings)
    dct_results = add_res(
        dct=dct_res, 
        model_name=model_name, 
        score=score, 
        VECT=vcf.__name__,
        **option)
    df = pd.DataFrame(dct_results).sort_values('SCORE')
    P("Results so far:\n{}".format(df))

###############################################################################
###############################################################################
###############################################################################
TEST_GLOVE_VS_TRAIN_DEV = True
CALC_BASELINES = True
RUN_GRID = True
TEST_RUN = False
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
dct_results = OrderedDict({'MODEL':[], 'SCORE':[]})


if TEST_GLOVE_VS_TRAIN_DEV:
  test_glove_vs_data(train_data, dev_data)


if CALC_BASELINES:
  dct_results = get_baselines(dct_results, train_data, dev_data)
      

if RUN_GRID:    
  dct_results = run_grid_search(dct_results, train_data, dev_data)
    

##############
if TEST_RUN:
  x_dev, y_dev = nli.word_entail_featurize(
      data=dev_data, 
      vector_func=lower_glove_vec, 
      vector_combo_func=arr
      )
  
  test_model = WordEntailClassifier(
      siam_lyrs=[256,256],
      s_l2=False,
      separ=False,
      layers=[], 
      input_drop=0.2,
      other_drop=0.5,
      bn=False,
      bn_inputs=False,
      activ='tanh',
      x_dev=x_dev,
      y_dev=y_dev,
      s_comb='eucl',
      loss='cl',
      rev=True,
      lr=0.0001,      
      batch=256,
      optim=th.optim.Adam
      )
  
  res = nli.wordentail_experiment(
        train_data=train_data,
        assess_data=dev_data,
        vector_func=lower_glove_vec,
        vector_combo_func=arr,
        model=test_model,    
      )
  print(res['macro-F1'])