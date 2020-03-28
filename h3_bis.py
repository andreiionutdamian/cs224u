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
from torch_utils import InputPlaceholder, L2_Normalizer, TripletLoss, get_activation
import nli

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

def get_train_dev(trn, dev):
  train_trip = _get_triplets_candidates(data=trn, embeds=GLOVE)
  lst_x_train = [[l_glv(t[0]), l_glv(t[1]), l_glv(t[2])] for t in train_trip]
  x_train = np.array(lst_x_train).astype(np.float32)
  
  x_dev, y_dev = nli.word_entail_featurize(
      data=dev, 
      vector_func=l_glv, 
      vector_combo_func=arr,
      )  
  return x_train, x_dev, y_dev


class ThSiamTrainer(th.nn.Module):
  def __init__(self, 
               input_dim=50, 
               layers=[128,256],
               input_drop=0,
               bn_inputs=False,
               bn=False,
               activ='relu',
               drop=0.3,
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
    


class TripletBasedClassifier():
  def __init__(self,
               lr=0.001,
               max_patience=10,
               max_fails=30,
               lr_decay=0.5,
               device=th.device("cuda" if th.cuda.is_available() else "cpu"),
               model_name='H3TB'
               ):
    self.batch_size=32
    self.device = device
    self.lr = lr
    self.model_name = model_name
    self.max_patience = max_patience
    self.max_fails = max_fails
    self.model = None
    self.lr_decay = lr_decay
    self.no_remove=False
    self.errors= []
    return
  
  def fit(self, X, y):
    # does nothing
    return
  
  def fit_triplets(self, x_train_triplets, x_dev, y_dev, epochs=1000, batch_size=32, dev_data=None):
    self.epochs = epochs
    self.batch_size = batch_size
    n_obs = x_train_triplets.shape[0]
    embed_size = x_train_triplets.shape[-1]
    if self.model is None:
      self.model = ThSiamTrainer(input_dim=embed_size).to(self.device)
      
    th_dl = th.utils.data.DataLoader(
        th.utils.data.TensorDataset(th.tensor(x_train_triplets, dtype=th.float32, device=self.device)),
        batch_size=self.batch_size
        )
    self.model.train()
    loss = TripletLoss(device=self.device, beta=0.2)
    optimizer = th.optim.Adam(self.model.parameters(), lr=self.lr)
    patience = 0
    fails = 0
    best_f1 = 0
    best_fn = ''
    not_del_fns = []
    for epoch in range(1, self.epochs + 1):
      epoch_losses = []
      for batch_iter, (X_batch,) in enumerate(th_dl):
        batch_embeds = self.model(X_batch)
        th_loss = loss(batch_embeds)
        optimizer.zero_grad()
        th_loss.backward()
        optimizer.step()
        epoch_losses.append(th_loss.detach().cpu().numpy())
        Pr("Training epoch {} - {:.1f}% - avg loss: {:.3f},  Patience {}/{},  Fails {}/{}\t\t\t\t\t".format(
            epoch, 
            (batch_iter + 1) / (n_obs // self.batch_size + 1) * 100,
            np.mean(epoch_losses),
            patience, self.max_patience,
            fails, self.max_fails))
      # end epoch
      self.errors.append(np.mean(epoch_losses))
      if dev_data is not None:
        predictions = self.predict_on_data(dev_data)
      else:
        predictions = self.predict(x_dev)
      macrof1 = utils.safe_macro_f1(y_dev, predictions)         
         
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
      
    return
  
  def predict(self, X):
    self.model.eval()
    with th.no_grad():
      th_X = th.tensor(X, device=self.device, dtype=th.float32)
      th_w1 = th_X[:,0]
      th_w2 = th_X[:,1]
      th_w1_e = self.model.get_embeds(th_w1)
      th_w2_e = self.model.get_embeds(th_w2)
      th_dist = (th_w1_e - th_w2_e).abs().sum(1)
      th_preds = th_dist > 0.5
      preds = th_preds.cpu().numpy()    
    self.model.train()
    return preds

  def predict_on_data(self, data):
    X, y = nli.word_entail_featurize(
        data=data, 
        vector_func=l_glv, 
        vector_combo_func=arr,
        )  
    self.model.eval()
    with th.no_grad():
      th_X = th.tensor(X, device=self.device, dtype=th.float32)
      th_w1 = th_X[:,0]
      th_w2 = th_X[:,1]
      th_w1_e = self.model.get_embeds(th_w1)
      th_w2_e = self.model.get_embeds(th_w2)
      th_dist = (th_w1_e - th_w2_e).abs().sum(1)
      th_preds = th_dist > 0.5
      preds = th_preds.cpu().numpy()    
    self.model.train()
    return preds
  

if __name__ == '__main__':
  
  DATA_HOME = 'data'
  GLOVE_HOME = os.path.join(DATA_HOME, 'glove.6B')
  
  NLIDATA_HOME = os.path.join(DATA_HOME, 'nlidata')
  
  wordentail_filename = os.path.join(
      NLIDATA_HOME, 'nli_wordentail_bakeoff_data.json')  
  
  utils.fix_random_seeds()
  GLOVE_DIM = 50
    
  if "GLOVE" not in globals() or len(next(iter(GLOVE.values()))) != GLOVE_DIM:
    P("Loading GloVe-{}...".format(GLOVE_DIM))
    GLOVE = utils.glove2dict(os.path.join(GLOVE_HOME, 'glove.6B.{}d.txt'.format(GLOVE_DIM)))  
    P("GloVe-{} loaded.".format(GLOVE_DIM))  
    
  with open(wordentail_filename) as f:
    wordentail_data = json.load(f)      
    
  train_data = wordentail_data['word_disjoint']['train']
  dev_data = wordentail_data['word_disjoint']['dev']
  
  if 'x_trn' not in globals():
    x_trn, x_dev, y_dev = get_train_dev(train_data, dev_data)
  
  clf = TripletBasedClassifier(lr=0.0001)
  clf.fit_triplets(
      x_train_triplets=x_trn,
      x_dev=x_dev,
      y_dev=y_dev,
      dev_data=dev_data,
      )
