# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:51:07 2020

@author: Andrei
"""

from scipy import sparse
import numpy as np
import pandas as pd
from time import time
from itertools import  permutations
import matplotlib.pyplot as plt
import textwrap
import os
from datetime import datetime as dt


class Log():
  def __init__(self):      
    self.lst_log = []
    self._date = dt.now().strftime("%Y%m%d_%H%M")
    self.log_fn = dt.now().strftime("logs/"+self._date+"_log.txt")

  def P(self, s=''):
    self.lst_log.append(s)
    print(s, flush=True)
    try:
      with open(self.log_fn, 'w') as f:
          for item in self.lst_log:
              f.write("{}\n".format(item))
    except:
      pass
    return
  
  def Pr(self, s=''):
      print('\r' + str(s), end='', flush=True)

def add_to_mco(df_chunk, dct_mco, basket_id_field, item_id_field, log):
  log.Pr("  Grouping the transactions...")
  t1 = time()
  transactions = df_chunk.groupby(basket_id_field)[item_id_field].apply(list)
  nr_trans = df_chunk[basket_id_field].unique().shape[0]
  t_trans = time() - t1
  log.Pr("  Transacts:   {} (grouped in {:.2f} min)".format(
      nr_trans, t_trans / 60))
  log.log.P("")
  times = []
  counts = []
  time_delta = 1000
  last_time = time()
  for i, (index, l) in enumerate(transactions.items()):
    t1 = time()
    market_basket = np.unique(l)  # keep only unique elements
    counts.append(market_basket.shape[0])
    if market_basket.shape[0] == 1:
      continue
    perm_market_basket = list(permutations(market_basket, 2))
    for pair in perm_market_basket:
      if pair not in dct_mco: 
        dct_mco[pair] = 0
      dct_mco[pair] += 1
    if (i % time_delta) == 0:
      elapsed = time() - last_time
      last_time = time()
      times.append(elapsed)
      mean_time = np.mean(times) / time_delta
      remain_time = nr_trans * mean_time - (i + 1) * mean_time
      log.Pr("  Processed transactions {:.1f}% - {:.2f} min remaning...".format(
          (i + 1) / nr_trans * 100,
         remain_time / 60))
    # endfor
  # endfor
  log.log.P("")
  return dct_mco, counts


def show_distrib(data, log, cutoff=np.inf, plot=False, return_data=False):
  d = data.astype(int)
  d[d>cutoff] = cutoff
  _h = np.bincount(d).tolist()
  _x = np.arange(len(_h))
  s_counts = ["{:02d}".format(x) for x in _h][:10]
  s_nitems = ["{:>"+str(len(x))+"}" for x in s_counts]# ["{:02d}".format(x) for x in range(max(all_counts)+1)]
  s_nitems = [x.format(i) for i,x in enumerate(s_nitems)]
  log.P("    Counts:   " + ' '.join(s_counts))
  log.P("    Nr items: " + ' '.join(s_nitems))
  if plot:
    plt.figure(figsize=(10,6))
    plt.bar(x=_x, height=_h, log=True)
    plt.ylabel("Count")
    plt.xlabel("Values")
    plt.title("Distribution (logscale)")
    plt.show()  
  if return_data:
    return _x, _h
  else:
    return

_MAX_COOC_VAL = 250



def generate_sparse_mco(file_name, 
                        mco_out_file,
                        log,
                        chunk_size=100 * 1024 ** 2, 
                        basket_id_field='BasketId', 
                        item_id_field='IDE',
                        date_field='TimeStamp',
                        plot=False,
                        return_counts=False,
                        DEBUG=False,
                        mco_file=None,
                        ):
  t1 = time()
  t2 = time()
  min_date = None
  max_date = None
  if DEBUG:    
    log.P("Loading MCO...")
    csr_mco = sparse.load_npz(mco_file)
    log.P("Loading raw data...")
    df = pd.read_csv(file_name)
    all_counts = df.groupby(basket_id_field)['IDE'].count()
  else:      
    data_size = os.path.getsize(file_name)
    log.P("Reading transactional data file '{}' of size {:.2f} GB...".format(file_name, data_size / 1024**3))
    chunk_generator = pd.read_csv(file_name, chunksize=chunk_size)  
    all_counts = []
    dct_mco = {}
    n_rows = 0
    for i, df in enumerate(chunk_generator):
      if date_field in df.columns:
        _max_date = df[date_field].max()
        _min_date = df[date_field].min()
        if min_date is None or _min_date < min_date:
          min_date = _min_date
        if max_date is None or _max_date > max_date:
          max_date = _max_date
          
      n_rows += df.shape[0]
      log.P("Processing chunk {} of data:".format(i+1))
      log.P("  Start date:  {}".format(_min_date))
      log.P("  End date:    {}".format(_max_date))
      log.P("  Chunk/sofar: {}/{}".format(df.shape[0], n_rows))
      dct_mco, counts = add_to_mco(
          df, dct_mco,  basket_id_field, item_id_field,
          log=log)
      all_counts += counts
    # end for
    all_counts = np.array(all_counts)
    log.P("  Converting dict to sparse matrix...")
    t2 = time()
    csr_mco = sparse.csr_matrix((
            list(dct_mco.values()),
            [list(x) for x in zip(*list(dct_mco.keys()))],
        ))
  t3 = time()
  t_full = t3 - t1
  t_csr = t3 - t2
  log.P("  MCO Processing done in {:.2f} min (sparse mat creation: {:.2f} min):".format(
      t_full / 60, t_csr / 60))
  log.P("  Start date: {}".format(min_date))
  log.P("  End date:   {}".format(max_date))
  log.P("  Max co-occ: {}".format(csr_mco.max()))
  log.P("")
  log.P("  Co-occurence distribution:")
  _xc, _hc = show_distrib(csr_mco.data, cutoff=_MAX_COOC_VAL, 
                          return_data=True, log=log)
  log.P("")
  log.P("  Transactions size distrib:")
  _x, _h = show_distrib(all_counts, cutoff=25, 
                        return_data=True, log=log)
  log.P("")
  log.P("  Transactional data:")
  log.P(textwrap.indent(str(df.iloc[:5, :6]), " " * 4))
  mco_cut = 7
  log.P("")
  log.P("  MCO data: (only {}x{} values displayed)".format(mco_cut, mco_cut))
  log.P(textwrap.indent(str(csr_mco[:mco_cut,:mco_cut].toarray()), " " * 4))
  if plot:
    plt.figure(figsize=(8,5))
    plt.bar(x=_x, height=_h, log=True)
    plt.ylabel("Number of transactions")
    plt.xlabel("Number of items per transaction")
    plt.title("Distribution of products per transaction (logscale)")
    plt.show()
    plt.figure(figsize=(11,7))
    plt.bar(_xc, _hc, log=True)
    plt.title("Distribution of co-occurence counts (logscale)")
    plt.ylabel("Number of co-occurences")
    plt.xlabel("Co-occurence count")    
    plt.show()
  log.P("Saving '{}'".format(mco_out_file))
  sparse.save_npz(mco_out_file, csr_mco)
  if return_counts:
    return csr_mco, _h
  else:
    return csr_mco
  

def load_categs_from_json(df_meta, mapping_file):
  import json
  with open(mapping_file, 'rb') as fh:
    dct_mapping = json.load(fh)
  for field in dct_mapping:
    dct_rev = {v:k for k,v in dct_mapping[field].items()}
    hn = df_meta[field].apply(lambda x: dct_rev[x])
    df_meta[field+'_name'] = hn
  return df_meta, dct_mapping


def show_categs(df_meta, dct_categories, log, k=np.inf):
  for level in dct_categories:
    log.P("Hierarchy level {} '{}' contains {} categories".format
     (level[-1], level, len(dct_categories[level])))
    cnt = 0
    categs = [x for x in dct_categories[level]][:k]
    max_len = max([len(x) for x in categs])
    for categ in categs:
      df_slice = df_meta[df_meta[level] == dct_categories[level][categ]]
      if df_slice.shape[0] == 0:
        continue
      log.P(("  Category name: {:<"+str(max_len)+"}  Id: {:>3}  Prods: {:>4}").format(
          categ,dct_categories[level][categ], df_slice.shape[0]))
      cnt += 1
      if cnt > k:
        break
    if cnt < len(categs):
      log.P("  ...")  
  return


def filter_categs(proposed_categs, df_meta, dct_categories, max_n_top_products):  
  prod_categs = {}
  for level in proposed_categs:
    prod_categs[level] = {}
    for categ in proposed_categs[level]:
      df_slice = df_meta[
          (df_meta[level] == dct_categories[level][categ]) &
          (df_meta.IDE < max_n_top_products)]  
      if df_slice.shape[0] > 2:
        prod_categs[level][categ] = dct_categories[level][categ]
  return prod_categs
    

def neighbors_by_idx(idx, embeds, k=None):
  v = embeds[idx]
  dists = np.maximum(0, 1 - embeds.dot(v) / (np.linalg.norm(v) * np.linalg.norm(embeds, axis=1)))
  idxs = np.argsort(dists)
  return idxs[:k], dists[idxs][:k]

def show_neighbors(idx, embeds, dct_i2n, log,
                   k=10, df=None, field='IDE', h1fld='Ierarhie1_name', h2fld='Ierarhie2_name'):
  if type(dct_i2n) == np.ndarray:
    dct_i2n = {i:dct_i2n[i] for i in range(dct_i2n.shape[0])}
  dct_rev = {name:idx for idx, name in dct_i2n.items()}
  if type(idx) != int:
    idx = dct_rev[idx]
  idxs, dists = neighbors_by_idx(idx, embeds, k=k)
  max_len = max([len(str(dct_i2n[ii])) for ii in idxs]) + 1
  log.P("Top neighbors for '{}'".format(dct_i2n[idx]))
  if df is None:
    for i,ii in enumerate(idxs):
      log.P(("  {:<" + str(max_len) + "} {:.3f}").format(str(dct_i2n[ii]) + ':', dists[i]))
  else:
    names = [dct_i2n[ii] for ii in idxs]
    h1s = [df[df[field]==ii][h1fld].iloc[0] for ii in idxs]
    h2s = [df[df[field]==ii][h2fld].iloc[0] for ii in idxs]
    d = {
        'ID'    : idxs,
        'DIST'  : dists,
        'NAME'  : [x[:40] for x in names],
        'H1'    : [x[:15] for x in h1s],
        'H2'    : [x[:20] for x in h2s],
        }
    return pd.DataFrame(d).sort_values('DIST')
  
  
    

###############################################################################
###############################################################################

