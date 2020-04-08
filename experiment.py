# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:29:46 2020

@author: Andrei
"""

import numpy as np
from scipy import sparse
import itertools
import os
import pandas as pd
from datetime import datetime as dt
from time import time
import textwrap
from itertools import combinations


DATA_HOME = 'experiment_data'
DATA_FILE = os.path.join(DATA_HOME, 'df_tran_proc_top_15k.csv')
META_FILE = os.path.join(DATA_HOME, 'df_items.csv')
MCO_FILE = os.path.join(DATA_HOME, 'mco.npz')

CHUNK_SIZE = 2**24 # read chunks of CHUNK_SIZE rows

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

def add_to_mco1(df_chunk, dct_mco, basket_id_field, item_id_field):
  Pr("  Grouping the transactions...")
  t1 = time()
  transactions = df_chunk.groupby(basket_id_field)[item_id_field].apply(list)
  nr_trans = df_chunk[basket_id_field].unique().shape[0]
  t_trans = time() - t1
  Pr("  {} transactions grouped in {:.2f}s".format(
      nr_trans, t_trans))
  P("")
  times = []
  time_delta = 1000
  last_time = time()
  for i, (index, l) in enumerate(transactions.items()):
    t1 = time()
    market_basket = np.unique(l)  # keep only unique elements
    if market_basket.shape[0] == 1:
      continue
    perm_market_basket = list(zip(*list(itertools.permutations(market_basket, 2))))
    for j in range(len(perm_market_basket[0])):
      p1, p2 = perm_market_basket[0][j], perm_market_basket[1][j]
      if p1 not in dct_mco: dct_mco[p1] = {}
      if p2 not in dct_mco[p1]: dct_mco[p1][p2] = 0
      dct_mco[p1][p2] += 1

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
  return dct_mco
  

def generate_sparse_mco1(file_name, chunk_size=CHUNK_SIZE, basket_id_field='BasketId', item_id_field='IDE'):
    data_size = os.path.getsize(file_name)
    P("Reading transactional data file '{}' of size {:.2f} GB".format(file_name, data_size / 1024 ** 3))
    t1 = time()
    chunk_generator = pd.read_csv(file_name, chunksize=chunk_size)

    dct_mco = {}
    n_rows = 0
    for i, df in enumerate(chunk_generator):
      n_rows += df.shape[0]
      P("Processing chunk {} of {} rows - ({} rows loaded so far) ...".format(
          i+1, 
          df.shape[0],
          n_rows
          ))
      dct_mco = add_to_mco1(df, dct_mco, basket_id_field, item_id_field)
    P("  Converting dict to sparse matrix...")
    t2 = time()
    dok_mco = sparse.dok_matrix((len(dct_mco), len(dct_mco)))
    for k1, d in dct_mco.items():
        for k2, v in d.items():
            dok_mco[k1, k2] = v
        
    csr_mco = dok_mco.tocsr()    
    t3 = time()
    t_full = t3 - t1
    t_csr = t3 - t2
    P("MCO Processing done in {:.2f} min (sparse mat creation: {:.2f} min):".format(
        t_full / 60, t_csr / 60))
    P("  Transactional data:")
    P(textwrap.indent(str(df.iloc[:15]), " " * 4))
    P("  MCO data:")
    P(textwrap.indent(str(csr_mco[:15,:15].toarray()), " " * 4))
    return csr_mco
  
  
################################################################  
  

  
def add_to_mco2(df_chunk, dct_mco, basket_id_field, item_id_field):
  Pr("  Grouping the transactions...")
  t1 = time()
  transactions = df_chunk[basket_id_field].unique()
  transaction_groups = df_chunk.groupby(basket_id_field)
  nr_trans = len(transactions)
  t_trans = time() - t1
  Pr("  {} transactions grouped in {:.2f}s".format(
      nr_trans, t_trans))
  P("")
  times = []
  time_delta = 1000
  last_time = time()
  for i, (tran_id, df_tran) in enumerate(transaction_groups):
    market_basket = df_tran[item_id_field].unique()
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
  return dct_mco



def generate_sparse_mco2(file_name, chunk_size=CHUNK_SIZE, basket_id_field='BasketId', item_id_field='IDE'):
  data_size = os.path.getsize(file_name)
  P("Reading transactional data file '{}' of size {:.2f} GB...".format(file_name, data_size / 1024**3))
  t1 = time()
  chunk_generator = pd.read_csv(file_name, chunksize=chunk_size)

  dct_mco = {}
  n_rows = 0
  for i, df in enumerate(chunk_generator):
    n_rows += df.shape[0]
    P("Processing chunk {} of data - ({} rows so far) ...".format(i+1, n_rows))
    dct_mco = add_to_mco2(df, dct_mco, basket_id_field, item_id_field)
  t2 = time()
  P("  Converting dict to sparse matrix...")
  csr_mco = sparse.csr_matrix((
          list(dct_mco.values()),
          [list(x) for x in zip(*list(dct_mco.keys()))],
      ))
  t3 = time()
  t_full = t3 - t1
  t_csr = t3 - t2
  P("MCO Processing done in {:.2f} min (sparse mat creation: {:.2f} min):".format(
      t_full / 60, t_csr / 60))
  P("  Transactional data:")
  P(textwrap.indent(str(df.iloc[:15]), " " * 4))
  P("  MCO data:")
  P(textwrap.indent(str(csr_mco[:15,:15].toarray()), " " * 4))
  return csr_mco


################################################################


def add_to_mco(df_chunk, dct_mco, basket_id_field, item_id_field):
  Pr("  Grouping the transactions...")
  t1 = time()
  transactions = df_chunk.groupby(basket_id_field)[item_id_field].apply(list)
  nr_trans = df_chunk[basket_id_field].unique().shape[0]
  t_trans = time() - t1
  Pr("  {} transactions grouped in {:.2f}s".format(
      nr_trans, t_trans))
  P("")
  times = []
  time_delta = 1000
  last_time = time()
  for i, (index, l) in enumerate(transactions.items()):
    t1 = time()
    market_basket = np.unique(l)  # keep only unique elements
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
  return dct_mco


def generate_sparse_mco(file_name, 
                        chunk_size=CHUNK_SIZE, 
                        basket_id_field='BasketId', 
                        item_id_field='IDE',
                        date_field='TimeStamp'):
  data_size = os.path.getsize(file_name)
  P("Reading transactional data file '{}' of size {:.2f} GB...".format(file_name, data_size / 1024**3))
  t1 = time()
  chunk_generator = pd.read_csv(file_name, chunksize=chunk_size)

  dct_mco = {}
  n_rows = 0
  min_date = None
  max_date = None
  for i, df in enumerate(chunk_generator):
    _s = ''
    if date_field in df.columns:
      _max_date = df[date_field].max()
      _min_date = df[date_field].min()
      if min_date is None or _min_date < min_date:
        min_date = _min_date
      if max_date is None or _max_date > max_date:
        max_date = _max_date
      _s = " with dates between {} and {} ".format(
          _min_date, _max_date)
        
    n_rows += df.shape[0]
    P("Processing chunk {} of data {}- ({} rows so far) ...".format(i+1, _s, n_rows))
    dct_mco = add_to_mco(df, dct_mco, basket_id_field, item_id_field)
    
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
  P("  Transactional data:")
  P(textwrap.indent(str(df.iloc[:15]), " " * 4))
  P("  MCO data:")
  P(textwrap.indent(str(csr_mco[:15,:15].toarray()), " " * 4))
  return csr_mco

  
  
if __name__ == '__main__':  
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.max_colwidth', 500)
  pd.set_option('display.width', 1000)
  pd.set_option('precision', 4)  
  np.set_printoptions(precision=3)
  np.set_printoptions(suppress=True)
  np.set_printoptions(linewidth=500)

  if 'loaded_mco' not in globals():
    loaded_mco = sparse.load_npz(MCO_FILE)  
  basket_id_field='BasketId'
  item_id_field='IDE'
  if 'df' not in globals():
    df = pd.read_csv(DATA_FILE)

  P("METHOD #1")
  csr_mco1 = generate_sparse_mco1(DATA_FILE)
  assert (csr_mco1 != loaded_mco).nnz == 0
  P("Data sanity check ok")

  P("METHOD #2")
  csr_mco2 = generate_sparse_mco2(DATA_FILE)
  assert (csr_mco2 != loaded_mco).nnz == 0
  P("Data sanity check ok")
  
  P("METHOD #3")
  csr_mco = generate_sparse_mco(DATA_FILE)
  assert (csr_mco != loaded_mco).nnz == 0
  P("Data sanity check ok")
  
  