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
  
class GateLayer(th.nn.Module):
  def __init__(self, input_dim, output_dim, bias=-2):
    super().__init__()
    self.gate_lin = th.nn.Linear(input_dim, output_dim)
    self.gate_act = th.nn.Sigmoid()
    if bias is not None:
      self.gate_lin.bias.data.fill_(bias) # negative fc => zero sigmoid => (1 - 0) * bypass
    return
    
  def forward(self, inputs, value1, value2):
    th_gate = self.gate_lin(inputs)
    th_gate = self.gate_act(th_gate)
    th_out = th_gate * value1 + (1 - th_gate) * value2
    return th_out
  
  
class MultiGatedDense(th.nn.Module):
  """
  TODO: 
    - experiment with various biases for gates
  """
  def __init__(self, input_dim, output_dim, activ):
    super().__init__()
    self.bypass = th.nn.Linear(input_dim, output_dim, bias=False)

    self.fc = th.nn.Linear(input_dim, output_dim)
    self.fc_act = get_activation(activ)
    
    self.bn_pre = th.nn.BatchNorm1d(output_dim)
    self.bn_post = th.nn.BatchNorm1d(output_dim)

    self.bn_pre_vs_post_gate = GateLayer(input_dim, output_dim, bias=0) 

    self.lnorm_post = th.nn.LayerNorm(output_dim)

    self.bn_vs_lnorm = GateLayer(input_dim, output_dim, bias=0)
    
    self.has_bn_gate = GateLayer(input_dim, output_dim, bias=-1)
    
    self.final_gate = GateLayer(input_dim, output_dim, bias=-2)
    return

  
  def forward(self, inputs):
    # bypass
    th_bypass = self.bypass(inputs)
    
    # FC unit
    th_fc = self.fc(inputs)
    th_fc_act = self.fc_act(th_fc)
    
    # apply post layer norm
    th_fc_act_lnorm = self.lnorm_post(th_fc_act)    

    # FC with pre activ bn
    th_bn_pre = self.fc_act(self.bn_pre(th_fc))
    # FC with post activ bn
    th_bn_post = self.bn_post(th_fc_act)
    
    # select between bn pre or post
    th_bn_out = self.bn_pre_vs_post_gate(inputs, th_bn_pre, th_bn_post)
    
    # select between bn or layer norm
    th_bn_vs_lnorm = self.bn_vs_lnorm(inputs, th_bn_out, th_fc_act_lnorm)
    
    # select between normed or FC-activ
    th_norm_vs_simple = self.has_bn_gate(inputs, th_bn_vs_lnorm, th_fc_act)
    
    # finally select between processed or bypass
    th_out = self.final_gate(inputs, th_norm_vs_simple, th_bypass)
  
    return th_out
    
    
  

class GatedDense(th.nn.Module):
  def __init__(self, input_dim, output_dim, activ, bn):
    super().__init__()
    self.bn = bn
    self.linearity = th.nn.Linear(input_dim, output_dim, bias=not bn)
    if self.bn == 'pre':
      self.pre_bn = th.nn.BatchNorm1d(output_dim)
    elif self.bn == 'post':
      self.post_bn = th.nn.BatchNorm1d(output_dim)
    elif self.bn == 'lnorm':
      self.post_lnorm = th.nn.LayerNorm(output_dim)
    self.activ = get_activation(activ)
    self.gate = GateLayer(input_dim, output_dim)
    self.bypass_layer = th.nn.Linear(input_dim, output_dim, bias=False)
  
  def forward(self, inputs):
    # the skip/residual
    th_bypass = self.bypass_layer(inputs)
    # the normal linear-bn-activ
    th_x = self.linearity(inputs)
    if self.bn == 'pre':
      th_x = self.pre_bn(th_x)
    th_x = self.activ(th_x)
    if self.bn == 'post':
      th_x = self.post_bn(th_x)
    elif self.bn == 'lnorm':
      th_x = self.post_lnorm(th_x)
    # the gate
    th_x = self.gate(inputs, th_x, th_bypass)
    return th_x
  
  


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
    s = self.__class__.__name__ + "(beta={:.3f})".format(
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
               score_key=None,
               device=th.device("cuda" if th.cuda.is_available() else "cpu"),
               model_name='',
               validation_data=None,
               base_folder='models',
               min_score_thr=None,
               ):
    self.score_mode = score_mode
    self.score_key = score_key
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
    self.training_status = {}
    self._files_for_removal = []
    self.base_folder = base_folder
    self.validation_data = validation_data
    self._debug_data = None
    self.epochs_data = {}
    self.in_training = False
    self.score_eval_func = max if self.score_mode == 'max' else min
    if min_score_thr is None:
      self.min_score_thr =  -np.inf if self.score_mode == 'max' else np.inf
    else:
      self.min_score_thr = min_score_thr
    if not hasattr(self, 'P'):
      def _P(s):
        print(s, flush=True)
      setattr(self, 'P', _P)
    
    return
  
  
  def __repr__(self):
    _s  = "  Model: '{}'+\n".format(self.model_name)
    _s += textwrap.indent(str(self.model), " " * 4) + '\n'
    _s += textwrap.indent("Loss: {}".format(self.loss), " " * 4)
    return _s
  
  def define_graph(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `define_graph`")

  def define_loss(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `define_loss`")
    
  def evaluate(self, verbose=False):
    raise ValueError("You must subclass ModelTrainer and overwrite `evaluate`")
    
  def predict(self):
    raise ValueError("You must subclass ModelTrainer and overwrite `predict`")    
    
  def reload_init(self, dct_epoch_data):
    raise ValueError("You must subclass ModelTrainer and overwrite `reload_init`")
  
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
    
  
  
  def fit(self, x_train, y_train=None, epochs=10000, verbose=False):
    if self.model is None:
      self.define_graph()    
    if self.loss is None:
      self.define_loss()
    if self.score_key is None:
      raise ValueError("Scoring config is incomplete!")
    if self.optimizer is None:
      self.optimizer = self.opt(self.model.parameters(), lr=self.lr)
    self.model.to(self.device)
    tensors = [th.tensor(x, device=self.device) for x in [x_train,y_train] if x is not None]
    th_ds = th.utils.data.TensorDataset(*tensors)
    th_dl = th.utils.data.DataLoader(
        th_ds, 
        batch_size=self.batch_size,
        shuffle=True)
    n_batches = len(th_dl)
    self.P("\nTraining model:\n  {}\n    Training on {} observations with batch size {}.\n".format(
      self,
      tensors[0].shape[0],
      self.batch_size,
      ))
    patience = 0
    fails = 0
    self.training_status['best_score'] = -np.inf if self.score_mode == 'max' else np.inf
    
    for epoch in range(1, epochs + 1): 
      epoch_errors = []
      self.in_training = epoch
      for batch_iter, batch_data in enumerate(th_dl):
        err = self.train_on_batch(batch_data)
        epoch_errors.append(err)
        Pr("Training epoch {:03d} - {:.1f}% - avg loss: {:.3f},  Patience {:02d}/{:02d},  Fails {:02d}/{:02d}\t\t\t\t\t".format(
            epoch, 
            (batch_iter + 1) / n_batches * 100,
            np.mean(epoch_errors),
            patience, self.max_patience,
            fails, self.max_fails,
            ))
      # end epoch
      dct_score = self.evaluate(verbose=verbose)
      dct_score['ep'] = epoch
      self.epochs_data[epoch] = dct_score
      score = dct_score[self.score_key]
      if self.score_eval_func(score, self.training_status['best_score']) != self.training_status['best_score']:
        self.P("\rFound new best score {:.3f} better than {:.3f} at epoch {}. \t\t\t".format(
            score, self.training_status['best_score'], epoch))
        self.save_best_model_and_track(epoch, score)
        fails = 0
        patience = 0
      else:
        patience += 1
        if patience > 0:
          fails += 1
        self.P("\rFinished epoch {:03d}, loss: {:.4f}. Current score {:.2f} < {:.2f}. Patience {:02d}/{:02d},  Fails {:02d}/{:02d}".format(
            epoch, np.mean(epoch_errors), score, self.training_status['best_score'], patience, self.max_patience, fails, self.max_fails))
        if patience >= self.max_patience:
          self.P("Patience reached {}/{} - reloading from epoch and reducting lr".format(
              patience, self.max_patience, self.training_status['best_epoch']))
          self.reload_best()
          self.reduce_lr()
          patience = -self.cooldown
        if fails >= self.max_fails:
          self.P("\nMax fails {}/{} reached!".format(fails, self.max_fails))
          break     
    self.restore_best_and_cleanup()
    self.in_training = None
    return self
          
    
  
  
  def save_best_model_and_track(self, epoch, score, cleanup=True, verbose=True):
    if not os.path.isdir(self.base_folder):
      os.mkdir(self.base_folder)
    best_fn = self.base_folder + "/{}_e{:03}_F{:.2f}.th".format(
            self.model_name, epoch, score)
    self.save_model(best_fn)
    if verbose:
      self.P("  Saved: '{}'".format(best_fn))
      self.P("  Saved: '{}'".format(best_fn + '.optim'))
    last_best =  self.training_status.get('best_file')
    if last_best != None:
      self._files_for_removal.append(last_best)
      self._files_for_removal.append(last_best + '.optim')
    self.training_status['best_file'] = best_fn
    self.training_status['best_epoch'] = epoch
    self.training_status['best_score'] = score
    if cleanup:
      self.cleanup_files()
    return    
  
  
  def save_model(self, fn):
    th.save(self.model.state_dict(), fn)
    th.save(self.optimizer.state_dict(), fn + '.optim')
    return
   
  def load_model(self, fn):
    self.model.load_state_dict(th.load(fn))
    self.optimizer.load_state_dict(th.load(fn + '.optim'))
    return
    
  
  def reload_best(self, verbose=True):
    self.load_model(self.training_status['best_file'])
    self.reload_init(self.epochs_data[self.training_status['best_epoch']])
    if verbose:
      self.P("  Reloaded '{}' data: {}".format(
          self.training_status['best_file'], 
          dict(self.epochs_data[self.training_status['best_epoch']])))
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
    while atmps < 10 and len(self._files_for_removal) > 0:   
      removed = []
      for fn in self._files_for_removal:
        if os.path.isfile(fn):
          try:
            os.remove(fn)
            removed.append(fn)
            self.P("  Removed '{}'".format(fn))
          except:
            pass
        else:
          removed.append(fn)            
      self._files_for_removal = [x for x in self._files_for_removal if x not in removed]
    return
  
  
  def restore_best_and_cleanup(self):
    self.reload_best()
    if self.score_eval_func(self.min_score_thr, self.training_status['best_score']) != self.training_status['best_score']:
      self.P("Best score {:.4f} did not pass minimal threshold of {} - model file will be deleted".format(
          self.training_status['best_score'], self.min_score_thr))
      self._files_for_removal.append(self.training_status['best_file'])
    else:
      self.P("Best score {:.4f} passed minimal threshold of {}".format(
          self.training_status['best_score'], self.min_score_thr))
    self._files_for_removal.append(self.training_status['best_file'] + '.optim')
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
