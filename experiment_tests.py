# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 19:57:23 2020

@author: Andrei
"""

import numpy as np
import torch as th

if __name__ == '__main__':
  d = {x:list(np.random.choice(100, size=np.random.randint(2, x), replace=False)) for x in range(3, 10)}

  dev = th.device('cuda')
  
  th_e = th.tensor(
          [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4],[5,5,5,5]], 
          dtype=th.float32, 
          device=dev)
  
  param = th.nn.Parameter(
      data=th_e+0.1)
  
  pad = th_e.shape[0] 
  emb = th.nn.Embedding(th_e.shape[0]+1, th_e.shape[1], padding_idx=pad).to(dev)
  emb.weight[:-1] = param
  
  t = th.tensor([[1,0,1,pad,pad]], device=dev)
  t_ids = th.tensor([[1],[0]], device=dev)
  t_rel = th.tensor([[1,2,5],[2,5,5]], device=dev)
  t_e_old = th_e[t_ids]
  t_e_new = emb(t_ids)
  t_e_rel = emb(t_rel)
  
  t_pre = (t_e_old - t_e_new).abs().sum(-1)
  
  t_e_rel_m = (t_e_rel.sum(dim=-1) > 0).float()
  
  t_rel_v = (t_e_rel - t_e_new).abs().sum(-1)
  
  t_rel = t_rel_v * t_e_rel_m
  
  