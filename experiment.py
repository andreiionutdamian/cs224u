# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:29:46 2020

@author: Andrei
"""

import numpy as np
from mittens import GloVe
from scipy import sparse
import itertools
import os
import pandas as pd
from datetime import datetime as dt
from time import time
import textwrap
from itertools import combinations
import matplotlib.pyplot as plt

DEBUG = True # in-development flag

DATA_HOME = 'exp_data'
MODEL_HOME = 'exp_models'
DATA_FILE = os.path.join(DATA_HOME, 'df_tran_proc_top_13.5k.csv')
DATA_SLICE_FILE = os.path.join(DATA_HOME, 'df_tran_proc_top_13.5k_slice.csv')
META_FILE = os.path.join(DATA_HOME, 'df_items_top_13.5k.csv')
META_INFO = os.path.join(DATA_HOME, 'obfuscated_keys.txt')

MCO_OUT_FILE = os.path.join(MODEL_HOME, 'exp_mco.npz')


CHUNK_SIZE = 100 * 1024 ** 2 # read 100MB chunks

MAX_N_TOP_PRODUCTS = 13000  # top sold products

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)  
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=500)

plt.style.use('ggplot')


###############################################################################

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


###############################################################################
###############################################################################
###############################################################################
###############################################################################

def add_to_mco(df_chunk, dct_mco, basket_id_field, item_id_field):
  Pr("  Grouping the transactions...")
  t1 = time()
  transactions = df_chunk.groupby(basket_id_field)[item_id_field].apply(list)
  nr_trans = df_chunk[basket_id_field].unique().shape[0]
  t_trans = time() - t1
  Pr("  Transacts:   {} (grouped in {:.2f} min)".format(
      nr_trans, t_trans / 60))
  P("")
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
    perm_market_basket = list(itertools.permutations(market_basket, 2))
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
      Pr("  Processed transactions {:.1f}% - {:.2f} min remaning...".format(
          (i + 1) / nr_trans * 100,
         remain_time / 60))
    # endfor
  # endfor
  P("")
  return dct_mco, counts


def show_distrib(data, cutoff=np.inf, plot=False, return_data=False):
  d = data.astype(int)
  d[d>cutoff] = cutoff
  _h = np.bincount(d).tolist()
  _x = np.arange(len(_h))
  s_counts = ["{:02d}".format(x) for x in _h][:15]
  s_nitems = ["{:>"+str(len(x))+"}" for x in s_counts]# ["{:02d}".format(x) for x in range(max(all_counts)+1)]
  s_nitems = [x.format(i) for i,x in enumerate(s_nitems)]
  P("    Counts:   " + ' '.join(s_counts))
  P("    Nr items: " + ' '.join(s_nitems))
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
_MCO_FILE  = os.path.join(DATA_HOME, 'mco_top_13.5k.npz')


def generate_sparse_mco(file_name, 
                        chunk_size=CHUNK_SIZE, 
                        basket_id_field='BasketId', 
                        item_id_field='IDE',
                        date_field='TimeStamp',
                        plot=False,
                        return_counts=False,
                        ):
  t1 = time()
  t2 = time()
  min_date = None
  max_date = None
  if DEBUG:
    P("Loading MCO...")
    csr_mco = sparse.load_npz(_MCO_FILE)
    P("Loading raw data...")
    df = pd.read_csv(DATA_FILE)
    all_counts = df.groupby(basket_id_field)['IDE'].count()
  else:      
    data_size = os.path.getsize(file_name)
    P("Reading transactional data file '{}' of size {:.2f} GB...".format(file_name, data_size / 1024**3))
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
      P("Processing chunk {} of data:".format(i+1))
      P("  Start date:  {}".format(_min_date))
      P("  End date:    {}".format(_max_date))
      P("  Chunk/sofar: {}/{}".format(df.shape[0], n_rows))
      dct_mco, counts = add_to_mco(df, dct_mco,  basket_id_field, item_id_field)
      all_counts += counts
    # end for
    all_counts = np.array(all_counts)
    P("  Converting dict to sparse matrix...")
    t2 = time()
    csr_mco = sparse.csr_matrix((
            list(dct_mco.values()),
            [list(x) for x in zip(*list(dct_mco.keys()))],
        ))
  t3 = time()
  t_full = t3 - t1
  t_csr = t3 - t2
  P("  MCO Processing done in {:.2f} min (sparse mat creation: {:.2f} min):".format(
      t_full / 60, t_csr / 60))
  P("  Start date: {}".format(min_date))
  P("  End date:   {}".format(max_date))
  P("  Max co-occ: {}".format(csr_mco.max()))
  P("  Co-occurence distribution:")
  _xc, _hc = show_distrib(csr_mco.data, cutoff=_MAX_COOC_VAL, return_data=True)
  P("  Transactions size distrib:")
  _x, _h = show_distrib(all_counts, cutoff=25, return_data=True)
  P("  Transactional data:")
  P(textwrap.indent(str(df.iloc[:15]), " " * 4))
  P("  MCO data:")
  P(textwrap.indent(str(csr_mco[:15,:15].toarray()), " " * 4))
  if plot:
    plt.figure(figsize=(15,9))
    plt.bar(x=_x, height=_h, log=True)
    plt.ylabel("Number of transactions")
    plt.xlabel("Number of items per transaction")
    plt.title("Distribution of products per transaction (logscale)")
    plt.show()
    plt.figure(figsize=(15,9))
    plt.bar(_xc, _hc, log=True)
    plt.title("Distribution of co-occurence counts (logscale)")
    plt.ylabel("Number of co-occurences")
    plt.xlabel("Co-occurence count")    
    plt.show()
  sparse.save_npz(csr_mco, MCO_OUT_FILE)
  if return_counts:
    return csr_mco, _h
  else:
    return csr_mco
  

def load_categories(df_meta, mapping_file):
  import json
  with open(mapping_file, 'rb') as fh:
    dct_mapping = json.load(fh)
  for field in dct_mapping:
    dct_rev = {v:k for k,v in dct_mapping[field].items()}
    hn = df_meta[field].apply(lambda x: dct_rev[x])
    df_meta[field+'_name'] = hn
  return df_meta, dct_mapping
    
  
    

###############################################################################
###############################################################################

  
  
if __name__ == '__main__':  
  basket_id_field='BasketId'
  item_id_field='IDE'
  
  if 'df' not in globals():
    df = pd.read_csv(DATA_FILE)
  
  if 'df_meta' not in globals():
    df_meta = pd.read_csv(META_FILE)

#  P("METHOD #1")
#  csr_mco1 = generate_sparse_mco1(DATA_FILE)
#  assert (csr_mco1 != loaded_mco).nnz == 0
#  P("Data sanity check ok")
#
#  P("METHOD #2")
#  csr_mco2 = generate_sparse_mco2(DATA_FILE)
#  assert (csr_mco2 != loaded_mco).nnz == 0
#  P("Data sanity check ok")
  
  P("METHOD #3")
  csr_mco, counts = generate_sparse_mco(DATA_FILE, plot=False, return_counts=True)
  
  
  ####################
  MAX_FREQ = _MAX_COOC_VAL
  SLICE = MAX_N_TOP_PRODUCTS 
  
  np_data = csr_mco.toarray().astype(np.float32)[:SLICE,:SLICE]

  np_15_slice = sparse.load_npz(os.path.join(DATA_HOME, 'mco.npz')).toarray()[:SLICE,:SLICE]
  assert np.all(np_15_slice == np_data)
  P("Data sanity check ok")

  glove_model = GloVe(n=128, 
                      xmax=MAX_FREQ, 
                      max_iter=250000,
                      save_folder=MODEL_HOME,
                      save_iters=1000,
                      name='exp_v1')
  embeds = glove_model.fit(np_data)
  
  
  