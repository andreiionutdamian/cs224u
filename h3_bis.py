# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 08:18:31 2020

@author: Andrei
"""
import numpy as np
import os
import utils
import torch as th
from datetime import datetime as dt
import json
import scipy
from torch_utils import InputPlaceholder, L2_Normalizer, TripletLoss, get_activation, ModelTrainer
import nli

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
  
  
def l_glv(w):    
  """Return lower `w`'s GloVe representation if available, else return 
  a zeros vector."""
  return GLOVE.get(w.lower(), np.zeros(GLOVE_DIM))

def arr(u,v):
  return np.array((u,v)).astype(np.float32)

def dist(w1,w2):
  w1_emb = GLOVE[w1.lower()]
  w3_emb = GLOVE[w2.lower()]
  return round(scipy.spatial.distance.cosine(w1_emb, w3_emb),3)
  

def _get_triplets_candidates(data, embeds, thr=0.2):
  pos_examples = [x for x in data if x[1] == 1]
  neg_examples = [x for x in data if x[1] == 0]
  candidates = []
  for i, obs in enumerate(pos_examples):
    w1 = obs[0][0] 
    w2 = obs[0][1]
    for neg_obs in neg_examples:
      if neg_obs[0][0] == w1:
        w3 = neg_obs[0][1]
        w1_emb = embeds.get(w1)
        w3_emb = embeds.get(w3)
        if w1_emb is not None and w3_emb is not None and scipy.spatial.distance.cosine(w1_emb, w3_emb) < thr:
          candidates.append([w1,w2,w3])
    Pr("Processed {:.1f}%".format((i+1)/len(pos_examples)*100))
  return candidates

def get_train_dev(trn, dev, return_triplets=False):
  train_trip = _get_triplets_candidates(data=trn, embeds=GLOVE)
  lst_x_train = [[l_glv(t[0]), l_glv(t[1]), l_glv(t[2])] for t in train_trip]
  x_train = np.array(lst_x_train).astype(np.float32)
  
  x_dev, y_dev = nli.word_entail_featurize(
      data=dev, 
      vector_func=l_glv, 
      vector_combo_func=arr,
      )  
  if return_triplets:
    return x_train, x_dev, y_dev, train_trip
  else:
    return x_train, x_dev, y_dev


class ThSiamTrainer(th.nn.Module):
  def __init__(self, 
               input_dim=50, 
               layers=[128,256],
               input_drop=0,
               bn_inputs=False,
               bn=False,
               activ='relu',
               drop=0.5,
               siam_norm=True,):
    super().__init__()
    siam_layers = [InputPlaceholder(input_dim)]
    last_output = input_dim
    if input_drop > 0:
      siam_layers.append(th.nn.Dropout(input_drop))
    if bn_inputs:
      siam_layers.append(th.nn.BatchNorm1d(last_output))
    for i, layer in enumerate(layers):
      siam_layers.append(th.nn.Linear(last_output, layer, bias=not bn))
      if bn:
        siam_layers.append(th.nn.BatchNorm1d(layer))
      if i < (len(layers) - 1):
       siam_layers.append(get_activation(activ))
       if drop > 0:
         siam_layers.append(th.nn.Dropout(drop))
      last_output = layer
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
               beta=0.2,
               thr=0.5,
               input_dim=None,
               model_name='H3TB',
               **kwargs,
               ):
    super().__init__(model_name=model_name,**kwargs)
    self.thr = thr
    self.beta = beta
    self.input_dim = input_dim
    return
  

  def define_graph(self):
    if self.input_dim is None:
      raise ValueError("Unknown model input dimension")
    self.model = ThSiamTrainer(input_dim=self.input_dim)
    
    
  def define_loss(self):
    self.loss = TripletLoss(beta=self.beta, device=self.device)
  
  
  def predict(self, X, return_dist=False):
    self.model.eval()
    with th.no_grad():
      th_X = th.tensor(X, device=self.device, dtype=th.float32)
      th_w1 = th_X[:,0]
      th_w2 = th_X[:,1]
      th_w1_e = self.model.get_embeds(th_w1)
      th_w2_e = self.model.get_embeds(th_w2)
      th_dist = th.sqrt(th.pow(th_w1_e - th_w2_e, 2)).mean(1)
      th_preds = th_dist < self.thr
      dists = th_dist.cpu().numpy()
      preds = th_preds.cpu().numpy()    
    self.model.train()
    if return_dist:
      return preds, dists
    else:
      return preds

  
  def evaluate(self, data=None, verbose=False):
    if data is None and self.validation_data is None:
      raise ValueError('Validation data not defined!')
    data = self.validation_data if data is None else data
    X_dev, y_dev = nli.word_entail_featurize(
        data=data, 
        vector_func=l_glv, 
        vector_combo_func=arr,
        )  
    preds, dists = self.predict(X_dev, return_dist=True)
    self._debug_data = [[data[i][0][0], data[i][0][1], data[i][1], dists[i]] for i in range(18)]
    macrof1 = utils.safe_macro_f1(y_dev, preds)
    if verbose:
      self.P(classification_report(y_dev, preds))
    return macrof1

  

if __name__ == '__main__':
  
  DATA_HOME = 'data'
  GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')
  
  NLIDATA_HOME = os.path.join(DATA_HOME, 'nlidata')
  
  wordentail_filename = os.path.join(
      NLIDATA_HOME, 'nli_wordentail_bakeoff_data.json')  
  
  utils.fix_random_seeds()
  GLOVE_DIM = 300
    
  if "GLOVE" not in globals() or len(next(iter(GLOVE.values()))) != GLOVE_DIM:
    P("Loading GloVe-{}...".format(GLOVE_DIM))
    GLOVE = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(GLOVE_DIM)))  
    P("GloVe-{} loaded.".format(GLOVE_DIM))  
    
  
  if 'x_trn' not in globals():
    with open(wordentail_filename) as f:
      wordentail_data = json.load(f)      
      
    train_data = wordentail_data['word_disjoint']['train']
    dev_data = wordentail_data['word_disjoint']['dev']
    x_trn, x_dev, y_dev, lst_trip = get_train_dev(train_data, dev_data, return_triplets=True)
  
  clf = TripletBasedClassifier(
      beta=1,
      lr=0.0001,
      thr=0.037,
      input_dim=GLOVE_DIM,
      validation_data=dev_data,
      )
  clf.fit(
      x_train=x_trn,
      )

  clf.evaluate(dev_data, verbose=True)
