# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:21:23 2020

@author: Andrei
"""
import numpy as np
import torch as th
import os
import textwrap

def Pr(s=''):
  print('\r' + str(s), end='', flush=True)


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
  
  
class TripletLoss(th.nn.Module):
  def __init__(self, device, beta=0.2):
    super().__init__()
    self.beta = th.tensor(beta, device=device)
    self.offset = th.tensor(0.0, device=device)
    return
    
    
  def forward(self, triplet):
    th_anchor = triplet[0]
    th_positive = triplet[1]
    th_negative = triplet[2]
    th_similar_dist = th.pow(th_anchor - th_positive, 2).sum(1)
    th_diff_dist = th.pow(th_anchor - th_negative, 2).sum(1)
    th_batch_pre_loss = th_similar_dist - th_diff_dist + self.beta
    th_batch_loss = th.max(input=th_batch_pre_loss, other=self.offset)
    th_loss = th_batch_loss.mean()
    return th_loss
  
  def __repr__(self):
    s = self.__class__.__name__ + "(beta={})".format(
        self.beta,
        )
    return s  
  

class ModelTrainer():
  def __init__(self,
               opt=th.optim.Adam,
               lr=0.001,
               max_patience=10,
               max_fails=30,
               cooldown=2,
               lr_decay=0.5,
               batch_size=32,
               score_mode='max',
               device=th.device("cuda" if th.cuda.is_available() else "cpu"),
               model_name='',
               validation_data=None,
               base_folder='models',
               ):
    self.score_mode = score_mode
    self.opt = opt
    self.batch_size=batch_size
    self.device = device
    self.lr = lr    
    self.model_name = model_name
    self.max_patience = max_patience
    self.max_fails = max_fails
    self.cooldown = cooldown
    self.model = None
    self.loss = None
    self.optimizer = None
    self.lr_decay = lr_decay
    self.lr_decay_iters = 0
    self.no_remove=False
    self.errors= []
    self._last_best_fn = ''
    self._not_del_fns = []
    self.base_folder = base_folder
    self.validation_data = validation_data
    self._debug_data = None
    if not hasattr(self, 'P'):
      def _P(s):
        print(s, flush=True)
      setattr(self, 'P', _P)
    return
  
  
  def __repr__(self):
    _s  = "  Model: '{}'+\n".format(self.model_name)
    _s += textwrap.indent(str(self.model), " " * 4)
    _s += textwrap.indent("Loss: {}\n".format(self.loss), " " * 4)
    return _s
  
  def define_graph(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `define_graph`")

  def define_loss(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `define_loss`")
    
  def evaluate(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `evaluate`")
    
  def predict(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `predict`")
  
  def train_on_batch(self, batch):    
    if len(batch) == 2:
      th_x, th_y = batch
    elif len(batch) == 1:
      th_x = batch[0]
      th_y = None
    th_yh = self.model(th_x)
    if th_y is not None:
      th_loss = self.loss(th_yh, th_y)
    else:
      th_loss = self.loss(th_yh)
    self.optimizer.zero_grad()
    th_loss.backward()
    self.optimizer.step()
    err = th_loss.detach().cpu().numpy()
    return err
    
  
  
  def fit(self, x_train, y_train=None, epochs=10000):
    if self.model is None:
      self.define_graph()    
    if self.loss is None:
      self.define_loss()
    if self.optimizer is None:
      self.optimizer = self.opt(self.model.parameters(), lr=self.lr)
    self.model.to(self.device)
    tensors = [th.tensor(x, device=self.device) for x in [x_train,y_train] if x is not None]
    th_ds = th.utils.data.TensorDataset(*tensors)
    th_dl = th.utils.data.DataLoader(
        th_ds, 
        batch_size=self.batch_size,
        shuffle=True)
    n_obs = len(th_dl)
    self.P("\nTraining model:\n  {}\n  Training on {} observations.\n".format(
      self,
      tensors[0].shape[0]))
    patience = 0
    fails = 0
    best_epoch = -1
    best_score = -np.inf if self.score_mode == 'max' else np.inf
    eval_func = max if self.score_mode == 'max' else min
    for epoch in range(1, epochs + 1):
      epoch_errors = []
      for batch_iter, batch_data in enumerate(th_dl):
        err = self.train_on_batch(batch_data)
        epoch_errors.append(err)
        Pr("Training epoch {} - {:.1f}% - avg loss: {:.3f},  Patience {}/{},  Fails {}/{}\t\t\t\t\t".format(
            epoch, 
            (batch_iter + 1) / (n_obs // self.batch_size + 1) * 100,
            np.mean(epoch_errors),
            patience, self.max_patience,
            fails, self.max_fails,
            ))
      # end epoch
      score = self.evaluate()
      if eval_func(score, best_score) != best_score:
        self.P("\rFound new best score {:.4f} better than {:.4f} at epoch {}. \t\t\t".format(score, best_score, epoch))
        self.save_model(epoch, score)
        best_score = score
        best_epoch = epoch
      else:
        patience += 1
        if patience > 0:
          fails += 1
        self.P("\rFinished epoch {}, loss: {:.4f}. Current score {:.3f} < {:.3f}. Patience {}/{},  Fails {}/{}".format(
            epoch, np.mean(epoch_errors), score, best_score, patience, self.max_patience, fails, self.max_fails))
        if patience >= self.max_patience:
          self.P("Patience reached {}/{} - reloading from epoch and reducting lr".format(
              patience, self.max_patience, best_epoch))
          self.reload_best()
          self.reduce_lr()
          patience = -self.cooldown
        if fails >= self.max_fails:
          self.P("\nMax fails {}/{} reached!".format(fails, self.max_fails))
          break     
    self.restore_best_and_cleanup()
    return self
          
    
  
  
  def save_model(self, epoch, score, cleanup=True, verbose=True):
    if not os.path.isdir(self.base_folder):
      os.mkdir(self.base_folder)
    best_fn = self.base_folder + "/{}_e{:03}_F{:.4f}.th".format(
            self.model_name, epoch, score)
    th.save(self.model.state_dict(), best_fn)
    th.save(self.optimizer.state_dict(), best_fn + '.optim')
    if verbose:
      self.P("  Saved: '{}'".format(best_fn))
      self.P("  Saved: '{}'".format(best_fn + '.optim'))
    if self._last_best_fn != '':
      self._not_del_fns.append(self._last_best_fn)
      self._not_del_fns.append(self._last_best_fn + '.optim')
    self._last_best_fn = best_fn
    if cleanup:
      self.cleanup_files()
    return    
  
    
  
  def reload_best(self, verbose=True):
    self.model.load_state_dict(th.load(self._last_best_fn))
    self.optimizer.load_state_dict(th.load(self._last_best_fn + '.optim'))
    if verbose:
      self.P("  Reloaded model & optimizer '{}'".format(self._last_best_fn))
    return
  
  
  def reduce_lr(self, verbose=True):
    self.lr_decay_iters += 1
    factor = self.lr_decay ** self.lr_decay_iters
    for i, param_group in enumerate(self.optimizer.param_groups):
      lr_old = param_group['lr'] 
      param_group['lr'] = param_group['lr'] * factor
      lr_new = param_group['lr']
    if verbose:
      self.P("  Reduced lr from {:.1e} to {:.1e}".format(lr_old, lr_new))
    return
    


  def cleanup_files(self):
    atmps = 0
    while atmps < 10 and len(self._not_del_fns) > 0:   
      removed = []
      for fn in self._not_del_fns:
        if os.path.isfile(fn):
          try:
            os.remove(fn)
            removed.append(fn)
            self.P("  Removed '{}'".format(fn))
          except:
            pass
        else:
          removed.append(fn)            
      self._not_del_fns = [x for x in self._not_del_fns if x not in removed]
    return
  
  
  def restore_best_and_cleanup(self):
    self.reload_best()
    self._not_del_fns.append(self._last_best_fn + '.optim')
    self.cleanup_files()
    return
    
  
  
def get_activation(act):
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
