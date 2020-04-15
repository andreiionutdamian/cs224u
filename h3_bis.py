# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 08:18:31 2020

@author: Andrei
"""
import numpy as np
import pandas as pd
import os
import utils
import torch as th
from datetime import datetime as dt
import json
import scipy
from torch_utils import InputPlaceholder, L2_Normalizer, TripletLoss, get_activation, ModelTrainer
from torch_utils import GatedDense, MultiGatedDense
import nli
import textwrap
from time import time
from collections import OrderedDict

from sklearn.metrics import classification_report

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


def add_res(dct, model_name, **kwargs):
  if 'MODEL' not in dct:
    dct['MODEL'] = []
  n_existing = len(dct['MODEL'])
  dct['MODEL'].append(model_name)
  for key in kwargs:
    if key not in dct:
      dct[key] = ['-' ] * n_existing
    dct[key].append(kwargs[key])
  for k in dct:
    if len(dct[k]) < (n_existing + 1):
      dct[k] = dct[k] + ['-'] * ((n_existing + 1) - len(dct[k]))
  return dct  
  
def l_glv(w):    
  """Return lower `w`'s GloVe representation if available, else return 
  a zeros vector."""
  return GLOVE.get(w.lower(), np.zeros(GLOVE_DIM))

def arr(u,v):
  return np.array((u,v)).astype(np.float32)

def _dist(w1,w2, embeds=None):
  if embeds is None:
    embeds = GLOVE
  w1_emb = embeds.get(w1.lower())
  w3_emb = embeds.get(w2.lower())
  if w1_emb is not None and w3_emb is not None:
    dist = scipy.spatial.distance.cosine(w1_emb, w3_emb)
  else:
    dist = 2.0
  return round(dist, 2)
  

def _get_triplets_candidates(data, embeds, thr=0.2):
  pos_examples = [x for x in data if x[1] == 1]
  neg_examples = [x for x in data if x[1] == 0]
  all_candidates = []
  pos_dists = []
  neg_dists = []
  for i, obs in enumerate(pos_examples):
    w1 = obs[0][0] 
    w2 = obs[0][1]
    pos_dists.append(_dist(w1, w2))
    for neg_obs in neg_examples:
      if neg_obs[0][0] == w1:
        w3 = neg_obs[0][1]
        d = _dist(w1, w3)
        if d < 2:
          neg_dists.append(d)
        all_candidates.append([w1,w2,w3, d])
    Pr("Processed {:.1f}% based on thr of {:.2f}".format((i+1)/len(pos_examples)*100, thr))
  candidates = []
  for trip in all_candidates:
    if trip[3] <= thr:
      candidates.append(trip[:-1])
  P("Total {} triplets".format(len(all_candidates)))
  P("Pos dist stats:\n {}".format(
      pd.Series(pos_dists).describe()))
  P("Neg dist stats:\n {}".format(
      pd.Series(neg_dists).describe()))
  return candidates, all_candidates

def get_train_dev(trn, dev, thr=0.8, return_triplets=False):
  train_triplets, all_triplets = _get_triplets_candidates(data=trn, thr=thr, embeds=GLOVE)
  lst_x_train = [[l_glv(t[0]), l_glv(t[1]), l_glv(t[2])] for t in train_triplets]
  x_train = np.array(lst_x_train).astype(np.float32)
  
  x_dev, y_dev = nli.word_entail_featurize(
      data=dev, 
      vector_func=l_glv, 
      vector_combo_func=arr,
      )  
  if return_triplets:
    return x_train, x_dev, y_dev, train_triplets, all_triplets
  else:
    return x_train, x_dev, y_dev


class ThSiamTrainer(th.nn.Module):
  def __init__(self, 
               input_dim=50, 
               layers=[250,200],
               input_drop=0,
               bn_inputs=False,
               use_gated=None,
               bn=False,
               activ='relu',
               drop=0.5,
               siam_norm=True,):
    super().__init__()
    
    utils.fix_random_seeds()
    
    siam_layers = [InputPlaceholder(input_dim)]
    last_output = input_dim
    if input_drop > 0:
      siam_layers.append(th.nn.Dropout(input_drop))
    if bn_inputs:
      siam_layers.append(th.nn.BatchNorm1d(last_output))
    for i in range(len(layers) - 1):
      layer = layers[i]
      if use_gated is None:
        has_bias = (not bn) and (i < (len(layers) - 1))
        siam_layers.append(th.nn.Linear(last_output, layer, bias=has_bias))
        if bn:
          siam_layers.append(th.nn.BatchNorm1d(layer))
        if i < (len(layers) - 1):
         siam_layers.append(get_activation(activ))
      else:   
        if use_gated == 'no_bn':
          gated_bn = None
          siam_layers.append(GatedDense(last_output, layer, activ=activ, bn=gated_bn))
        elif use_gated == 'bn_pr':
          gated_bn = 'pre'
          siam_layers.append(GatedDense(last_output, layer, activ=activ, bn=gated_bn))
        elif  use_gated == 'bn_po':
          gated_bn = 'post'          
          siam_layers.append(GatedDense(last_output, layer, activ=activ, bn=gated_bn))
        elif use_gated == 'bn_ln':
          gated_bn = 'lnorm'
          siam_layers.append(GatedDense(last_output, layer, activ=activ, bn=gated_bn))
        elif use_gated == 'multi':
          siam_layers.append(MultiGatedDense(last_output, layer, activ=activ))
        else:
          raise ValueError("Unknown use_gated='{}'".format(use_gated))
      if drop > 0:
         siam_layers.append(th.nn.Dropout(drop))
      last_output = layer
      
    siam_layers.append(th.nn.Linear(last_output, layers[-1]))
    if siam_norm:
      siam_layers.append(L2_Normalizer())
    
    self.siamese_net = th.nn.ModuleList(siam_layers)
    return
  
  def forward(self, triplet):
    th_anchor = triplet[:,0]
    th_positive = triplet[:,1]
    th_negative = triplet[:,2]
    th_org_embed = self.get_embeds(th_anchor)
    th_pos_embed = self.get_embeds(th_positive)
    th_neg_embed = self.get_embeds(th_negative)    
    return th_org_embed, th_pos_embed, th_neg_embed
    
  
  def get_embeds(self, th_x):
    for layer in self.siamese_net:
      th_x = layer(th_x)
    return th_x
    


class TripletBasedClassifier(ModelTrainer):
  def __init__(self,
               gated=None,
               beta=0.2,
               thrs=[0.3],
               layers=None,
               input_dim=None,
               bn_inp=False,
               drp_inp=False,
               bn=False,
               act='relu',
               drp=0.5,
               model_name='H3TB',
               **kwargs,
               ):
    super().__init__(model_name=model_name,**kwargs)
    self.thrs = thrs
    self.X_dev = None
    self.y_dev = None
    if self.validation_data is not None:
      self.X_dev, self.y_dev = nli.word_entail_featurize(
          data=self.validation_data, 
          vector_func=l_glv, 
          vector_combo_func=arr,
          )  
    self.last_thr = None
    self.gated = gated
    self.layers = layers
    self.beta = beta
    self.bn_inputs = bn_inp
    self.bn = bn
    self.activ = act
    self.drop = drp
    self.input_drop = drp_inp
    self.input_dim = input_dim
    self.best_thrs = []
    self.P("Initialized {} with siamese embeds thrs={}...".format(
        self.__class__.__name__, self.thrs[:5]))
    return
  

  def define_graph(self):
    if self.input_dim is None:
      raise ValueError("Unknown model input dimension")
    self.model = ThSiamTrainer(
        input_dim=self.input_dim,
        bn_inputs=self.bn_inputs,
        layers=self.layers,
        input_drop=self.input_drop,
        bn=self.bn,
        activ=self.activ,
        drop=self.drop,
        use_gated=self.gated,
        )
    
    
  def define_loss(self):
    self.loss = TripletLoss(beta=self.beta, device=self.device)
  
  
  def predict_dists(self, X):
    self.model.eval()
    with th.no_grad():
      th_X = th.tensor(X, device=self.device, dtype=th.float32)
      th_w1 = th_X[:,0]
      th_w2 = th_X[:,1]
      th_w1_e = self.model.get_embeds(th_w1)
      th_w2_e = self.model.get_embeds(th_w2)
      th_dist = th.sqrt(th.pow(th_w1_e - th_w2_e, 2)).mean(1)
      np_dists = th_dist.cpu().numpy()
    self.model.train()
    return np_dists
  
  
  def recalculate_best_thr(self, np_dists, verbose=False):
    assert self.X_dev != None
    if verbose:
      self.P("")
      self.P("  Calculating best threshold...")      
    best_f1 = 0
    best_thr = None
    for thr in self.thrs:
      np_preds = np_dists < thr
      macrof1 = utils.safe_macro_f1(self.y_dev, np_preds)
      if macrof1 > best_f1:
        best_f1 = macrof1
        best_thr = thr
    self.best_thr = best_thr
    if verbose:
      self.P("  Best threshold settled at {} based on macro-F1: {:.3f}".format(
          self.best_thr,
          best_f1))
    self.best_thrs.append(self.best_thr)
    return best_f1
    
  
  def predict(self, X, return_dist=False, recalc_thr=False, verbose=False):
    np_dists = self.predict_dists(X)
    if recalc_thr or not hasattr(self, 'best_thr') or self.best_thr is None:
      self.recalculate_best_thr(np_dists=np_dists, verbose=verbose)
    if verbose:
      best_epoch = self.training_status.get('best_epoch')
      self.P("  Predict using thr={}, best ep: {}".format(
          self.best_thr,
          dict(self.epochs_data[best_epoch]) if best_epoch is not None else None))
    np_preds = (np_dists < self.best_thr).astype(np.uint8)
    if return_dist:
      return np_preds, np_dists
    else:
      return np_preds
    
    
  def reload_init(self, dct_epoch_data):
    self.best_thr = dct_epoch_data['thr']
    return
    
  
  def evaluate(self, data=None, full_report=False, verbose=True):
    if data is None and self.validation_data is None:
      raise ValueError('Validation data not defined!')
    if data is not None:
      self.P("  Featurizing input data...")
      X_dev, y_dev = nli.word_entail_featurize(
          data=data, 
          vector_func=l_glv, 
          vector_combo_func=arr,
          ) 
    else:
      X_dev = self.X_dev
      y_dev = self.y_dev
      data = self.validation_data
      
    preds, dists = self.predict(
        X_dev, 
        return_dist=True, 
        recalc_thr=self.in_training,
        verbose=verbose
        )
    
    self._debug_data = [[data[i][0][0], data[i][0][1], data[i][1], dists[i]] for i in range(18)]
    rep = classification_report(y_dev, preds, output_dict=True)
    macrof1 = round(rep['macro avg']['f1-score'] * 100,2)
    dct_results = OrderedDict()
    dct_results[self.score_key] = macrof1
    dct_results['p_rec'] = round(rep['1']['recall'] * 100,2)
    dct_results['p_pre'] = round(rep['1']['precision'] * 100,2)
    dct_results['ep'] = self.in_training if self.in_training else self.training_status['best_epoch']
    dct_results['thr'] = self.best_thr
    dct_results['min_t'] = min(self.best_thrs)
    dct_results['max_t'] = max(self.best_thrs)

    if full_report:
      pos_dists = [dists[i] for i in range(len(dists)) if y_dev[i] == 1]
      neg_dists = [dists[i] for i in range(len(dists)) if y_dev[i] == 0]
      self.P("Positive dev obs stats:\n{}".format(
          textwrap.indent(str(pd.Series(pos_dists).describe()), "  ")))
      self.P("Negative dev obs stats:\n{}".format(
          textwrap.indent(str(pd.Series(neg_dists).describe()), "  ")))
      self.P("Best thresholds distrib:\n{}".format(
          textwrap.indent(str(pd.Series(self.best_thrs).describe()),"  ")))
      self.P(classification_report(y_dev, preds))
    return dct_results

  

if __name__ == '__main__':
  
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.max_colwidth', 500)
  pd.set_option('display.width', 1000)
  pd.set_option('precision', 4)  
  
  ver = 'X3'
  
  DATA_HOME = 'data'
  GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')
  
  NLIDATA_HOME = os.path.join(DATA_HOME, 'nlidata')
  
  wordentail_filename = os.path.join(
      NLIDATA_HOME, 'nli_wordentail_bakeoff_data.json')  
  
  utils.fix_random_seeds()
  GLOVE_DIM = 100
    
  if "GLOVE" not in globals() or len(next(iter(GLOVE.values()))) != GLOVE_DIM:
    P("Loading GloVe-{}...".format(GLOVE_DIM))
    GLOVE = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(GLOVE_DIM)))  
    P("GloVe-{} loaded.".format(GLOVE_DIM))  
    np_words = np.array([x for x in GLOVE])
    np_embds = np.array([GLOVE[x] for x in np_words])
    np.savez(
        os.path.join(GLOVE_HOME, 'glove_words_and_embeds_{}d'.format(GLOVE_DIM)),
        np_words,
        np_embds,
        )
    
    
  
  with open(wordentail_filename) as f:
    wordentail_data = json.load(f)      
    
  train_data = wordentail_data['word_disjoint']['train']
  dev_data = wordentail_data['word_disjoint']['dev']  

  test_data_filename = os.path.join(
    NLIDATA_HOME,
    "bakeoff-wordentail-data",
    "nli_wordentail_bakeoff_data-test.json")
  
  with open(test_data_filename, encoding='utf8') as f:
      dct_bake_data = json.load(f)
      
  bake_dev = dct_bake_data['word_disjoint']['test']

  TRAIN = train_data # train_data + dev_data
  DEV = dev_data # bake_data
  x_trn, _, _, trips, allt = get_train_dev(TRAIN, dev_data, thr=0.8, return_triplets=True)
  
  dev_data = DEV

    
  grid  = {
      'gated' : [
          'bn_pr',
          'bn_po',
          'bn_ln',
          'no_bn',
          None,
          'multi',
          ],
          
      'beta' : [
#          2.0,
#          1.0,
#          0.4,
          0.2,
#          0.1,
          ],
          
      'bn_inp' : [
#          True,  # BAD
          False
          ],
          
      'drp' : [
#          0.5,
#          0.7,
          0.3,
#          0.1,
#          0.0,
          ],
          
      'layers' : [
#          [512, 256],
#          [1024, 512],
#          [300, 200, 100],
#          [128, 64],
#          [2048, 768, 256],
          [2048, 1024, 512],
          ],
          
      'drp_inp' : [
          0.0,
#          0.1,
#          0.3,
#          0.5,
          ],
          
      'act' : [
          'relu',
#          'tanh'
          ],
      
      'bn' : [
          True,
          False,
          ],
          
      
      }  
  
  def filter_func(grid_options):
    if not grid_options['bn'] and grid_options['gated'] not in ['no_bn', None]:
      return False
    if grid_options['bn'] and grid_options['gated'] == 'no_bn':
      return False
    return True
  
  options = prepare_grid_search(grid, valid_fn=filter_func, nr_trials=300)
  dct_res = {}
  timings = []
  t_left = np.inf
  score_key = 'dev_f1'
  for grid_iter, option in enumerate(options):    
    model_name = 'H3{}_{:03d}'.format(
        ver,  grid_iter+1)
    P("\n\n" + "=" * 70)
    P("Running grid search iteration {}/{}\n '{}' : {}".format(
        grid_iter+1, len(options), model_name, option))
    P("  Time left for grid search completion: {:.1f} hrs".format(t_left / 3600))
    _t_start = time()
    
    clf = TripletBasedClassifier(
        model_name=model_name,
        score_key=score_key,
        lr=0.0001,
        max_patience=7,
        max_fails=7 * 2 + 1,
        cooldown=2,
        input_dim=GLOVE_DIM,
        batch_size=256,
        thrs=np.linspace(0.01,0.2,num=200).round(4),
        validation_data=dev_data,
        min_score_thr=68,
        **option
        )
    
    clf.fit(
        x_train=x_trn,
        verbose=False,
        )
    
    
    dct_eval = clf.evaluate(
        dev_data, 
        full_report=True, 
        verbose=True
        )
    dct_res = add_res(
        dct=dct_res, 
        model_name=model_name, 
        **dct_eval, 
        **option)
    df = pd.DataFrame(dct_res).sort_values(score_key)
    P("Results so far:\n{}".format(df))
    
    t_res = time() - _t_start
    timings.append(t_res)
    t_left = (len(options) - grid_iter - 1) * np.mean(timings)    
  
