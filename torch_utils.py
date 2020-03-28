# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:21:23 2020

@author: Andrei
"""

import torch as th


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
        self.margin,
        )
    return s  
  
  
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
