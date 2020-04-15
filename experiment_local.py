# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:29:46 2020

@author: Andrei
"""

import numpy as np
from mittens import GloVe
from scipy import sparse
import os
import pandas as pd

import matplotlib.pyplot as plt
print_df = True




DEBUG = True # in-development flag

MODEL_NAME = 'exp_v1'

DATA_HOME = 'exp_data'
MODEL_HOME = 'exp_models'
DATA_FILE = os.path.join(DATA_HOME, 'df_tran_proc_top_13.5k.csv')
DATA_SLICE_FILE = os.path.join(DATA_HOME, 'df_tran_proc_top_13.5k_slice.csv')
META_FILE = os.path.join(DATA_HOME, 'df_items_top_13.5k.csv')
META_INFO = os.path.join(DATA_HOME, 'obfuscated_keys.txt')

MCO_OUT_FILE = os.path.join(MODEL_HOME, 'exp_mco.npz')

EMB_OUT_FILE = os.path.join(MODEL_HOME, MODEL_NAME + '_embeds.npy')

_MCO_FILE  = os.path.join(MODEL_HOME, 'mco_top_13.5k.npz')

CHUNK_SIZE = 100 * 1024 ** 2 # read 100MB chunks

MAX_N_TOP_PRODUCTS = 13000  # top sold products

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)  
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=500)

plt.style.use('ggplot')


from prove import Log
from prove import load_categs_from_json
from prove import generate_sparse_mco
from prove import show_neighbors
from prove import filter_categs
from prove import show_categs
  

log = Log()


## Understanding the data

def show_product(IDE, df):
    rec = df[df.IDE==IDE].iloc[0]
    log.P("   Name:  {}".format(rec.ItemName))
    log.P("   Id:    {}".format(rec.ItemId))
    log.P("   Freq:  {}".format(rec.Freq))
    log.P("   Hrchy: {}/{}".format(rec.Ierarhie1, rec.Ierarhie2))
    return
    
log.P("Metatada information:")    
df_meta = pd.read_csv(META_FILE)
df_meta = df_meta[df_meta.IDE < MAX_N_TOP_PRODUCTS]
log.P("  Total no of products: {}".format(df_meta.ItemId.unique().shape[0]))
log.P("  Total no of level 1 hierarchies: {}".format(df_meta.Ierarhie1.unique().shape[0]))
log.P("  Total no of level 2 hierarchies: {}".format(df_meta.Ierarhie2.unique().shape[0]))
log.P("  Most sold individual product over period:")
show_product(df_meta[df_meta.Freq == df_meta.Freq.max()].iloc[0]['IDE'], df_meta)
log.P("  Least sold individual product over period:")
show_product(df_meta[df_meta.Freq == df_meta.Freq.min()].iloc[0]['IDE'], df_meta)
df_meta[df_meta.ItemName.str.len()<30].iloc[:15,:-1]

df_meta, dct_categories = load_categs_from_json(df_meta, META_INFO)
df_meta[df_meta.ItemName.str.len()<30].iloc[:15,[3,7,8]]

chunk_reader = pd.read_csv(DATA_SLICE_FILE, iterator=True) 
chunk_reader.get_chunk(15).iloc[:,:-2]


#################
chunk_reader.close()


## NLU meets BPA

csr_mco, tran_sizes = generate_sparse_mco(
    DATA_FILE, mco_out_file=MCO_OUT_FILE,
    plot=True, return_counts=True, log=log,
    DEBUG=DEBUG, mco_file=_MCO_FILE)


### Meta-information & categories

dct_i2n = {k:v for k,v in zip(df_meta.IDE, df_meta.ItemName)}
    
# we slice and convert the sparse matrix
np_mco = csr_mco.toarray()[:MAX_N_TOP_PRODUCTS,:MAX_N_TOP_PRODUCTS]
df_meta = df_meta.iloc[:MAX_N_TOP_PRODUCTS]
# raw sanity-check
for i in range(np_mco.shape[0]):
    assert i in df_meta.IDE

show_categs(df_meta, dct_categories, k=5, log=log)


### Analogy with word-vectors

if 'wemb' not in globals():
    log.P("Loading GloVe word embeddings")
    glove_words = os.path.join(DATA_HOME, 'glove_words_and_embeds_100d.npz')
    data = np.load(glove_words)
    np_words = data['arr_0']
    wemb = data['arr_1']

def show_word(word):
    show_neighbors(word, wemb, np_words, log=log)
    
show_word('beatles')


## Self-supervised training and supervised testing

proposed_test_categs = {
  'Ierarhie1': {
      'COMICS&MANGA': 21, 'GASTRONOMIE': 20, 'HOME&DECO AA': 10, 'LIFESTYLE': 12, 'MULTIMEDIA': 8,
      'MUZICA': 2, 'NOVELTY': 23,'VINURI': 14,
  },
  'Ierarhie2': {
      'AUDIO & VIDEOBOOK': 259, 'Accesorii apple': 193, 'Activity': 294, 'Audiobook CD': 111,
      'BLU-RAY': 36, 'BLU-RAY 3D': 224, 'BOARD GAMES': 140, 'Backtoschool': 108, 'Bakery': 269,
      'Bath': 262, 'Blu-Ray Audio': 165, 'Blu-Ray Video': 65, 'Body Care': 330, 'CD Clasica/Opera': 83,
      'COMICS': 87, 'Comics': 120, 'DRAW MANGA/COMICS/VIDEOGAMES': 72, 'FILOSOFIE': 43, 
      'Film Blu-Ray': 136, 'Film DVD': 52, 'Film VHS': 343, 'GASTRO/BAKERY': 314, 'Giftware': 107,
      'Gourmet': 192, 'Jazz': 308, 'Kitchen': 118, 'LIFESTYLE': 283, 'MANGA': 63, 'ROBOTICA': 300, 
      'SelfHelp': 198, 'TRAVELING COLLECTION': 139, 'Travel': 243, 'Travel Mug': 144, 
      'Travelling collection': 75, 'UHD': 183, 'Vinyl': 10,
   }
}      
dct_cat = filter_categs(
    proposed_test_categs, df_meta, dct_categories,
    max_n_top_products=MAX_N_TOP_PRODUCTS)


show_categs(df_meta, dct_cat, k=1000, log=log)
df_tpr = df_meta[
      df_meta.Ierarhie1.isin(list(dct_cat['Ierarhie1'].values())) |
      df_meta.Ierarhie2.isin(list(dct_cat['Ierarhie2'].values()))
      ]


# Metrics and overall evaluation

def mrr_k(np_cands_batch, np_gold_batch, k=5):
  """
  will compute Mean Reciprocal Rank for a batch of predictions
  inputs:
    np_cands_batch: [batch, no_products] - the candidates for each experiment in batch
    np_gold_batch: [batch,] or [batch, 1] - the actual gold target product replacement
    k: int (default 5) the max number of candidates evaluated
  """
  assert len(np_cands_batch.shape) == 2
  np_gold_batch  = np_gold_batch.reshape(-1,1)
  np_cands_batch = np_cands_batch[:,:k]
  n_obs, n_cands = np_cands_batch.shape
  matches = np_cands_batch == np_gold_batch  
  _x = np.repeat(np.arange(1,n_cands+1).reshape(1,-1),n_obs, axis=0) 
  _y = np.ones((n_obs,n_cands)) * np.inf
  np_res = 1 / np.where(matches, _x, _y).min(axis=1)
  return np_res.mean()
  


## Hyperparameters

EMBED_SIZE = 128
MAX_FREQ = 250
MAX_ITERS = 250000
SAVE_ITERS = 1000

## ProVe implementation

if os.path.isfile(EMB_OUT_FILE):
    embeds = np.load(EMB_OUT_FILE)
else:
    np_data = csr_mco.toarray()
    glove_model = GloVe(
        n=EMBED_SIZE, 
        xmax=MAX_FREQ, 
        max_iter=MAX_ITERS,
        save_folder=MODEL_HOME,
        save_iters=SAVE_ITERS,
        name='exp_v1',
    )
    embeds = glove_model.fit(np_data)
log.P("ProVe embeds {} loaded.".format(embeds.shape))

def show_prod(pid ,k=10):
    show_neighbors(pid, embeds, dct_i2n, k=k, log=log)

def show_prod_df(pid,k=10):
    df = show_neighbors(pid, embeds, dct_i2n, df=df_meta, k=k, log=log)
    if print_df:
      log.P(df)
    else:
      return df
    
############################################    
      
      
    
