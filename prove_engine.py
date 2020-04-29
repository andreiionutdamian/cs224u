# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:51:07 2020

@author: Andrei
"""

import numpy as np
import pandas as pd
from time import time
import textwrap
import os
from collections import OrderedDict
import pickle


import prove_utils
          
__EMBENG_VER__ = '0.1.0.1'

###############################################################################
###############################################################################
# BEGIN EmbedsEngine
###############################################################################
###############################################################################


class EmbedsEngine():
  """
  This is the main workhorse for the product replacement and new product creating 
  process.
  """
  def __init__(self,
             np_embeddings,
             df_metadata,
             name_field,
             id_field,
             categ_fields,
             log,
             strict_relations=True,
             save_folder=None,
             dct_categ_names=None,
             name='emb_eng',
             ):
    """
    Inputs:
       `np_embeddings`    : the embeddings matrix `(N, D)`
       
       `df_metadata`      : the metadata dataframe with `N` products
       
       `name_field`       : field name where item names are stored in metadata 
       
       `id_field`         : field name for the item ids
       
       `categ_fields`     : list of category fields in order of hierarchy levels
       
       `log` : Log object
       
       `strict_relations` : (boolean, default True) flag that decides whether positive
                            relations are based on any category or only on category intersection
                            between products
       
       `save_folder`      : where to save the outputs - if `None` and `log` has `get_data_folder` will use it
       
       `dct_categ_names`  : dictionary that maps each each of `categ_fields` to ids and names
      
    """
    assert type(np_embeddings) == np.ndarray
    assert len(np_embeddings.shape) == 2
    assert type(df_metadata) == pd.DataFrame
    assert df_metadata.shape[0] > 1
    assert id_field in df_metadata.columns, "Field {} not in meta-data {}".format(
        id_field, list(df_metadata.columns))
    for fld in categ_fields:
      assert fld in df_metadata.columns, "Field {} not in meta-data {}".format(
        fld, list(df_metadata.columns))
      
    self.version = __EMBENG_VER__
    
    self.dct_pos_edges = None
    self.dct_neg_edges = None
    self.strict_relations = strict_relations
    
    self.embeds = np_embeddings    
    self.df_meta = df_metadata
    self.name_fld = name_field
    self.id_fld = id_field
    self.categ_fields = categ_fields
    self.log = log
    self.name = name
    self.is_categ_field_str = True
    for categ_field in self.categ_fields:
      if type(self.df_meta[categ_field].iloc[0]) != str:
        self.is_categ_field_str = False
    self.dct_categ_names = dct_categ_names
    
    if save_folder is None and not hasattr(log, 'get_data_folder'):
      raise ValueError("Either Logger must provide `get_data_folder` or `save_folder` must be supplied")
    else:
      if save_folder is not None:
        self._save_folder = save_folder
      else:
        self._save_folder = log.get_data_folder()

    if not hasattr(self, 'P'):
      def _P(s):
        self.log.P(s)
      setattr(self, 'P', _P)

    self.P("Initializing Embeddings processing Engine v{}.".format(self.version))
    self._setup_meta_data()
    return
  
  def _maybe_load_graph(self):
    fn = os.path.join(self._save_folder, self.name + '.pkl')
    data = None, None, None
    if os.path.isfile(fn):
      try:
        print("\rLoading item knowledge graph...", end='\r',flush=True)
        with open(fn, 'rb') as fh:
          data = pickle.load(fh)
        self.P("Item knowledge graph loaded.\t\t")
      except:
        pass
    
    self.dct_edges, self.dct_categ_prods, self.dct_prods_categs = data
    if self.dct_edges is not None and type(self.dct_edges[0]) != list:
      self.P("Re-arange edges...")
      self.dct_edges = {k:list(v) for k,v in self.dct_edges.items()}
      self._save_graph()
    return True if self.dct_edges is not None else False
      

  def _save_graph(self):
    fn = os.path.join(self._save_folder, self.name + '.pkl')
    data = self.dct_pos_edges, self.dct_neg_edges
    try:
      with open(fn, 'wb') as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)
      self.P("Item knowledge graph save.")
      return True
    except:
      self.P("Item knowledge graph save failed!")
      return False
        
  
  def _setup_meta_data(self):
    self.categs_names = []
    for categ_fld in self.categ_fields:
      name_field = categ_fld + '_name'      
      dct_rev = {v:k for k, v in self.dct_categ_names[categ_fld].items()} if self.dct_categ_names else None
      if name_field not in self.df_meta.columns:
        if self.df_meta[categ_fld].dtype != object:
          if dct_rev is None:
            raise ValueError("Please provide `dct_categ_names` to translate from '{}' to actual names".format(
                categ_fld))
          hn = self.df_meta[categ_fld].apply(lambda x: dct_rev[x])
          self.df_meta[name_field] = hn
          self.categs_names.append(name_field)
          self.P("Created category string in '{}'".format(name_field))
        else:
          self.categs_names.append(categ_fld)
          self.P("Found category string in '{}'".format(categ_fld))
      else:
        self.categs_names.append(name_field)
        self.P("Found category string in '{}'".format(name_field))
        
    if not self._maybe_load_graph():
      self._construct_graph_from_meta()
      self._save_graph()
    return


  def _get_item_negatives(self, item_id, k=100, max_dist=0.75):
    idxs, dists = prove_utils.neighbors_by_idx(item_id, self.embeds, k=k)    
    filtered = []
    for i, idx in enumerate(idxs):
      if dists[i] < max_dist:
        filtered.append(idx)
    if len(filtered) > 0:
      df = self._items_to_df(filtered)
      for categ_field in self.categ_fields:
        df = df[df[categ_field] != self.dct_prods_categs[item_id][categ_field]]
    if df.shape[0] > 0:
      return df[self.id_fld].values
    else:
      return []
    
      
  
  def _construct_graph_from_meta(self, MAX_PROD_NEIGH=3000):
    dct_pos_edges = {}    
    dct_neg_edges = {}
    dct_categ_prods = {}
    dct_prods_categs = {x : {} for x in range(self.embeds.shape[0])}
    self.P("Constructing items knowledge graph")    
    for categ_field in self.categ_fields:
      self.P("  Retrieving products for category '{}'".format(categ_field))
      dct_categ_prods[categ_field] = {}
      for categ_id in self.df_meta[categ_field].unique():
        dct_categ_prods[categ_field][categ_id] = self.df_meta[
            self.df_meta[categ_field] == categ_id][self.id_fld].unique().tolist()
        for prod_id in dct_categ_prods[categ_field][categ_id]:
          dct_prods_categs[prod_id][categ_field] = categ_id
    
    for emb_idx in range(self.embeds.shape[0]):
      prod_neighbors = []
      for categ_field in dct_categ_prods:
        prod_categ = dct_prods_categs[emb_idx][categ_field]
        prods = dct_categ_prods[categ_field][prod_categ]
        if self.strict_relations:
          if len(prod_neighbors) > 0:
            prod_neighbors = set(prod_neighbors) & set(prods)
          else:
            prod_neighbors = prods
        else:
          prod_neighbors += prods
        if len(prod_neighbors) > MAX_PROD_NEIGH:
          raise ValueError('Looks like item {} has over {} products!'.format(
              emb_idx, MAX_PROD_NEIGH))            
      dct_pos_edges[emb_idx] = set(prod_neighbors)
      if (emb_idx % 1000) == 0:
        self.log.Pr("  Creating positive relations graph {:.1f}%".format(
            (emb_idx + 1) / self.embeds.shape[0] * 100), end='', flush=True)    
    self.log.P("  Created positive relations graph.\t\t\t")


    for emb_idx in range(self.embeds.shape[0]):
      prod_neg_neigh = self._get_item_negatives(emb_idx)
      dct_neg_edges[emb_idx] = prod_neg_neigh
      if (emb_idx % 1000) == 0:
        self.log.Pr("  Creating negative relations graph {:.1f}%".format(
            (emb_idx + 1) / self.embeds.shape[0] * 100), end='', flush=True)    
    self.log.P("  Created negative relations graph.\t\t\t")

    self.dct_categ_prods = dct_categ_prods
    self.dct_prods_categs = dct_prods_categs
    self.dct_pos_edges = {k:list(v) for k,v in dct_pos_edges.items()}
    self.dct_neg_edges = {k:list(v) for k,v in dct_neg_edges.items()}
    return
  
  def _items_to_df(self, items):
    fields = [self.id_fld, self.name_fld] + self.categs_names
    df = self.df_meta[self.df_meta[self.id_fld].isin(items)][fields]
    return df
        
      
  def get_item_info(self, item_id, verbose=False, show_relations=False):
    predefined_names = ['ID','NAME', 'POS_EDGES', 'NEG_EDGES']
    dct_info = OrderedDict({})
    dct_info['ID'] = item_id
    dct_info['NAME'] = self.df_meta[self.df_meta[self.id_fld] == item_id][[self.name_fld]].iloc[0,0]
    for i, categ_field in enumerate(self.categ_fields):
      dct_info[categ_field] = self.dct_prod_categs[item_id][categ_field]
      if self.categs_names[i] != categ_field:
        dct_info[self.categs_names[i]] = self.df_meta[self.df_meta[self.id_fld] == item_id][[self.categs_names[i]]].iloc[0,0]
    dct_info['POS_EDGES'] = self.dct_pos_edges[item_id]
    dct_info['NEG_EDGES'] = self.dct_neg_edges[item_id]
    if verbose:
      self.P("")
      self.P("Product '{}' info:".format(dct_info['ID']))
      for k in dct_info:
        if k not in predefined_names:
          _s = "  {}: {}".format(k, dct_info[k])
          self.P(_s)
        elif k != 'ID':
          val = dct_info[k]
          if type(val) != list:
            self.P("  {}: {}".format(k, val))
          else:
            # assume products
            self.P("  {}:".format(k))
            self.P(textwrap.indent(str(self._items_to_df(dct_info[k])),' ' * 4))
            
    return dct_info
  
  def analize_item(self, 
                   item_id,
                   positive_id,
                   negative_id,
                   embeds=None,
                   show_df=False,
                   embeds_name=None,
                   ):
    if embeds is None:
      embeds = self.embeds      
    self.log.Pr("Performing analysis of item {}...".format(item_id))
    d_i = self.get_item_info(item_id)
#    d_p = self.get_item_info(positive_id)
#    d_n = self.get_item_info(negative_id)
    idxs, dists = prove_utils.neighbors_by_idx(item_id, embeds, k=None)
    p_dist = dists[np.where(idxs==positive_id)[0][0]]
    n_dist = dists[np.where(idxs==negative_id)[0][0]]
    df_f = self.get_similar_items(item_id, embeds, filtered=True)
    df_n = self.get_similar_items(item_id, embeds, filtered=False)
    self.P("Analysis of {}: '{}'  {}".format(
        item_id, d_i['NAME'],
        "using model {}".format(embeds_name) if embeds_name else '')
        )
    self.P("  Distance from positive {:<7} {:.3f}".format(str(positive_id)+':', p_dist))
    self.P("  Distance from negative {:<7} {:.3f}".format(str(negative_id)+':', n_dist))
    if show_df:
      self.P("  Non-filtered neighbors:")
      self.P(textwrap.indent(str(df_n), "    "))
      self.P("  Filtered neighbors:")
      self.P(textwrap.indent(str(df_f), "    "))
    return
    

  def get_similar_items(self, item_id, 
                        embeds=None, 
                        filtered=False, 
                        k=10, 
                        show=False,
                        name=None
                        ):
    if embeds is None:
      embeds = self.embeds
    if filtered:
      _k = max(5000, k)
    else:
      _k = max(100, k)
      
    dct_info = self.get_item_info(item_id, verbose=False)
    c1 = dct_info[self.categs_names[0]]
    c2 = dct_info[self.categs_names[1]]
    
    df_res = prove_utils.show_neighbors(
        idx=item_id, 
        embeds=embeds, 
        log=self.log,
        k=_k, 
        df=self.df_meta, 
        id_field=self.id_fld, 
        name_field=self.name_fld,
        h1fld=self.categs_names[0], 
        h2fld=self.categs_names[1]
        )
    if filtered:
      df_res = df_res[
          (df_res['H1'] == c1) |
          (df_res['H2'] == c2)
          ]
    df = df_res.iloc[:k,:]
    if show:
      if name is not None:
        self.P("")
        self.P("  Table: {}".format(name))
      else:
        self.P("")
        self.P("  Top neighbors for product {}:".format(item_id))
      self.P(textwrap.indent(str(df), "  "))
    else:
      if name is not None:
        self.log.Pmdc("Table: {}".format(name))
    return df

  
  
  def get_retrofitted_embeds(self, 
                             prod_ids=None, 
                             method='v1', 
                             dct_negative=None, 
                             skip_negative=False,
                             full_edges=True, 
                             **kwargs):
    self.P("")
    self.P("Performing retrofitting on {} embedding matrix...".format(self.embeds.shape))
    self.P("  Product(s):  {}".format(prod_ids))
    self.P("  Full edges:  {}".format(full_edges))
    if dct_negative is not None:
      self.P("  Negatives:   {}".format(len(dct_negative)))
    _dct = self.dct_pos_edges
    _dct_neg = self.dct_neg_edges if not skip_negative else {}
    if prod_ids is not None:
      _dct = {}
      _dct_neg = {}
      if type(prod_ids) == int:
        prod_ids = [prod_ids]
      for prod_id in prod_ids:
        related_prods = self.dct_pos_edges[prod_id]
        _dct[prod_id] = related_prods
        if full_edges:
          for related_id in related_prods:
            if related_id not in _dct:
              _dct[related_id] = self.dct_edges[related_id]
            elif prod_id not in _dct[related_id]:
              _dct[related_id].append(prod_id)
        if dct_negative is None and not skip_negative: 
          # unless we give specific negative dict or no negative
          negative_prods = self.dct_neg_edges[prod_id]
          _dct_neg[prod_id] = negative_prods
          if full_edges:
            for neg_id in negative_prods:
              if neg_id not in _dct_neg:
                _dct_neg[neg_id] = prod_id
            
    if dct_negative is not None and not skip_negative:
      _dct_neg = dct_negative.copy()
      if full_edges:
        for neg_id in dct_negative:
          neg_neigh = _dct_neg[neg_id]
          for nn in neg_neigh:
            if nn not in _dct_neg:
              _dct_neg[nn] = [neg_id]
            elif neg_id not in _dct_neg[nn]:
              _dct_neg[nn].append(neg_id)
            
    method_name = '_get_retrofitted_embeds_' + method    
    func = getattr(self, method_name)
    self.P("  Method:      {}".format(func.__name__))
    self.P("  Pos edges: {}".format(len(_dct)))
    self.P("  Neg edges: {}".format(len(_dct_neg)))
    t1 = time()
    embeds = func(dct_edges=_dct, dct_negative=_dct_neg, **kwargs)
    t2 = time()
    self.P("Retrofit with '{}' took {:.1f}s".format(func.__name__, t2-t1))
    return embeds
  
  
  def _get_retrofitted_embeds_v1(self, dct_edges, **kwargs):
    embeds = self._retrofit_faruqui_fast(
        np_X=self.embeds,
        dct_edges=dct_edges,
        )
    return embeds
    
  
  
  @staticmethod
  def _measure_changes(Y, Y_prev):
    """
    this helper function measures changes in the embedding matrix 
    between previously step of the retrofiting loop and the current one
    """
    return np.abs(np.mean(np.linalg.norm(
                              np.squeeze(Y_prev) - np.squeeze(Y),
                              ord=2)))



  def _retrofit_faruqui_fast(
      self, 
      np_X, 
      dct_edges, 
      max_iters=100, 
      tol=5e-3, 
      alpha=None, 
      beta=None):
    """
    Implements retrofitting method of Faruqui et al. https://arxiv.org/abs/1411.4166
    however using the implementation from Potts et at / Dingwell et al
    
    Inputs:
    ======
    np_X : np.ndarray
      This is the input embedding matrix
    
    dct_edges: dict
      This is the dict that maps a certain vector to all its relatives 
      
    max_iters: int (default=100)
    
    alpha, beta: callbacks that return floats as per paper alpha/beta

    tol : float (default=1e-2)
      If the average distance change between two rounds is at or
      below this value, we stop. Default to 10^-2 as suggested
      in the paper.
      
    
    Outputs: 
    ======
      np.ndarray: the retrofitted version of np_X
      
    Original code by Dingwell et all:
      ```
        for iteration in range(1, max_iters+1):
            t1 = time()
            for i in dct_edges:
              neighbors = dct_edges[i]
              n_neighbors = len(neighbors)
              if n_neighbors > 0:
                a = alpha(i)
                b = beta(i)
                retro = np.array([b * np_Y[j] for j in neighbors])
                retro = retro.sum(axis=0) + (a * np_X[i])
                norm = np.array([b for j in neighbors])
                norm = norm.sum(axis=0) + a
                np_Y[i] = retro / norm
      ```
      
    """

    if alpha is None:
      alpha = lambda x: 1.0
    if beta is None:
      beta = lambda x: 1.0 / len(dct_edges[x])

    np_Y = np_X.copy()
    np_Y_prev = np_Y.copy()
    self.P("  Stop tol:    {:.1e}".format(tol))
    self.P("  Training log:")   
    for iteration in range(1, max_iters+1):
      t1 = time()
      for i in dct_edges:
        neighbors = dct_edges[i]
        n_neighbors = len(neighbors)
        if n_neighbors > 0:
          a = alpha(i)
          b = beta(i)
          retro = b * np_Y[neighbors]
          retro = retro.sum(axis=0) + a * np_X[i] # - b_neg * neg_retro...
          norm = n_neighbors * b + a
          np_Y[i] = retro / norm
        if (i % 1000) == 0:
          self.log.Pr("    Iteration {:02d} - {:.1f}%".format(
              iteration, (i+1)/np_X.shape[0]*100))
        # end if
      # end for matrix rows
      changes = self._measure_changes(np_Y, np_Y_prev)
      t2 = time()
      if changes <= tol:
        self.P("    Retrofiting converged at iteration {} - change was {:.1e} ".format(
                iteration, changes))
        break
      else:
        np_Y_prev = np_Y.copy()
        self.P("    Iteration {:02d} - change was {:.1e}, iter time: {:.2f}s".format(
            iteration, changes, t2 - t1))
    # end for iterations
    return np_Y

  
  
  def _retrofit_vector_to_embeddigs(self, np_start_vect, np_similar_vectors):
    """
    this function will retrofit a single vector (can be even a random vector but 
    preferably a centroid from the original latent space) to a pre-prepared matrix 
    of similar embeddings using the basic Faruqui et al approach
    """
    self.P("Creating new item embedding starting from a {} vector and {} similar items".format(
        np_start_vect.shape, np_similar_vectors.shape))
    self.P("  Current distance: {:.2f}".format(
        self._measure_changes(np_start_vect, np_similar_vectors)))
    n_simils = np_similar_vectors.shape[0]
    np_full = np.concatenate((np_start_vect.reshape(-1,1),
                              np_similar_vectors))
    dct_edges = {0:[x for x in range(1, n_simils)]}
#    for x in range(1, n_simils):
#      dct_edges[x] = 0
    np_new_embeds = self._retrofit_faruqui(np_full, dct_edges)
    np_new_embed = np_new_embeds[0]
    self.P("  New distance: {:.2f}".format(
        self._measure_changes(np_new_embed, np_similar_vectors)))
    return np_new_embed
  
  
  def cold_start_item(self, dct_categs, need_items=None, similar_items=None):
    """
    Steps:
      1. USER: categ1, categ2...
      2. SYS: embed
      3. 
      
      
      1. USER: categ1, categ2, interests categ
    """
    pass
  
  def get_product_replacement(self, prod_id):
    pass
  

  def _prepare_retrofit_data(self, dct_positive, 
                             dct_negative=None, 
                             split=False, 
                             pad_id=-1,
                             fix_weights=None):
    self.P("  Preparing retrofit data based on dict({})...".format(len(dct_positive)))
    if len(dct_positive) <= 1:
      raise ValueError("`dct_positive` must have more than 1 item")

    if split:
      # split in ids(N,) positives(N, var) negatives(N, var), pos_w(N), neg_w(N)
      dct_negative = {} if dct_negative is None else dct_negative
      start_ids = list(set(
              list(dct_positive.keys()) + list(dct_negative.keys())
              ))
      self.P("    Positive edges: {}".format(len(dct_positive)))
      self.P("    Negative edges: {}".format(len(dct_negative)))
      pos_lists = [dct_positive.get(x,[]) for x in start_ids]
      pos_lens = [len(x) for x in pos_lists]
      
      neg_lists = [dct_negative.get(x,[]) for x in start_ids]
      neg_lens = [len(x) for x in neg_lists]
      
      max_pos_len = max([1] + pos_lens)
      max_neg_len = max([1] + neg_lens)
      
      if fix_weights is not None:
        pos_w = [fix_weights for _ in pos_lists]
        neg_w = [fix_weights for _ in neg_lists]
      else:
        pos_w = [1/len(x) if x!=[] else 0 for x in pos_lists]
        neg_w = [1/len(x) if x!=[] else 0 for x in neg_lists]   
      
      for i,seq in enumerate(pos_lists):
        nn = max_pos_len - len(seq)
        pos_lists[i] = seq + [pad_id] * nn
      for i,seq in enumerate(neg_lists):
        nn = max_neg_len - len(seq)
        neg_lists[i] = seq + [pad_id] * nn
      return start_ids, pos_lists, pos_w, neg_lists, neg_w    
    else:
      all_data = []  
      n_pos = len(dct_positive)
      view_step = n_pos / 100
      for i, idx in enumerate(dct_positive):
        neibs = dct_positive[idx]
        pre_weight = 1 / len(neibs)
        rel_weight = 1 / len(neibs)
        pairs = [[idx, x, pre_weight, rel_weight] for x in neibs]
        all_data += pairs
        if (i % view_step) == 0:
          self.log.Pr("    Preparing positive edges {:.1f}%".format((i+1)/n_pos * 100))
      self.P("    Prepared positive edges - total {:,} relations".format(len(all_data)))
      if dct_negative is not None:
        all_negative = []
        n_neg = len(dct_negative)
        view_step = n_neg / 100
        for i, idx in enumerate(dct_negative):
          neg_neibs = dct_negative[idx]
          pre_weight = 1 / len(neg_neibs) 
          neg_weight = -1 / len(neg_neibs)
          neg_pairs = [[idx, x, pre_weight, neg_weight] for x in neg_neibs]
          all_negative += neg_pairs
          if (i % view_step) == 0:
            self.log.Pr("    Preparing negative edges {:.1f}%".format((i+1)/n_neg * 100))
        all_data += all_negative
        self.P("    Prepared negative edges - total {:,} relations".format(len(all_negative)))
      np_all_data = np.array(all_data, dtype='float32')
      self.P("    Dataset: {}".format(np_all_data.shape))
      return np_all_data
      
      
  
  def _get_retrofitted_embeds_v2_tf(self, 
                                    dct_edges, 
                                    dct_negative=None,
                                    eager=False, 
                                    use_fit=False,
                                    epochs=99, 
                                    batch_size=16384,
                                    gpu_optim=True,
                                    lr=0.1,
                                    patience=2,
                                    tol=1e-1,
                                    **kwargs):
    """
    this method implements a similar approach to Dingwell et al
    """
    self.P("Starting `_get_retrofitted_embeds_v2_tf`...")
    import tensorflow as tf

    
    vocab_size = self.embeds.shape[0]
    embedding_dim = self.embeds.shape[1]

    data = self._prepare_retrofit_data(
        dct_positive=dct_edges,
        dct_negative=dct_negative,
        split=False,
        )
    
    self.P("  Preparing model...")
    
    nr_inputs = 4
    assert data.shape[-1] == nr_inputs

    embeds_old = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(self.embeds),
        trainable=False,
        dtype=tf.float32,
        name='org_emb')
    embeds_new = tf.keras.layers.Embedding(
        vocab_size, embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(self.embeds),
        trainable=True,
        dtype=tf.float32,
        name='new_emb')
    
    src_sel = tf.keras.layers.Lambda(lambda x: x[:,0], name='inp_src')
    dst_sel = tf.keras.layers.Lambda(lambda x: x[:,1], name='inp_dst')
    wgh_p_sel = tf.keras.layers.Lambda(lambda x: x[:,2], name='inp_pres_weight')
    wgh_r_sel = tf.keras.layers.Lambda(lambda x: x[:,3], name='inp_rela_weight')
    
    p_diff = tf.keras.layers.Subtract(name='preserve_diff')
    r_diff = tf.keras.layers.Subtract(name='relation_diff')
    
    p_norm = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.pow(x, 2), axis=1), name='preserve_dst')
    r_norm = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.pow(x, 2), axis=1), name='relation_dst')
    
    p_weighting = tf.keras.layers.Multiply(name='preserve_weight')
    r_weighting = tf.keras.layers.Multiply(name='relation_weight')
    
    final_add = tf.keras.layers.Add(name='preserve_and_relation')
    
        
    def identity_loss(y_true, y_pred):
      return tf.math.maximum(0.0, tf.reduce_sum(y_pred))
      
       
    tf_input = tf.keras.layers.Input((nr_inputs,))
    tf_src = src_sel(tf_input)
    tf_dst = dst_sel(tf_input)
    tf_weight_p = wgh_p_sel(tf_input)
    tf_weight_r = wgh_r_sel(tf_input)
    
    tf_src_orig = embeds_old(tf_src)
    tf_src_new = embeds_new(tf_src)
    tf_dst_new = embeds_new(tf_dst)
    
    tf_preserve_diff = p_diff([tf_src_orig, tf_src_new])
    tf_relation_diff = r_diff([tf_src_new, tf_dst_new])
    
    tf_preserve_nw = p_norm(tf_preserve_diff)
    tf_relate_nw = r_norm(tf_relation_diff)
    
    tf_preserve = p_weighting([tf_preserve_nw, tf_weight_p])
    tf_relate = r_weighting([tf_relate_nw, tf_weight_r])
    
    tf_retro_loss_batch = final_add([tf_preserve, tf_relate])
    
    model = tf.keras.models.Model(tf_input, tf_retro_loss_batch)
    self.P("  Training model for {} epochs, batch={}, lr={:.1e}, tol={:.1e}".format(
        epochs, batch_size, lr, tol))
    opt = tf.keras.optimizers.SGD(lr=lr)
    losses = []
    best_loss = np.inf
    fails = 0
    last_embeds = self.embeds
    best_embeds = None
    if eager:
      def _convert(idx_slices):
        return tf.scatter_nd(tf.expand_dims(idx_slices.indices, 1),
                         idx_slices.values, idx_slices.dense_shape)
      self.P("    Starting eager training loop")
      ds = tf.data.Dataset.from_tensor_slices(data)
      n_batches = data.shape[0] // batch_size + 1
      ds = ds.batch(batch_size).prefetch(1)
      for epoch in range(1, epochs+1):
        epoch_losses = []
        for i, tf_batch in enumerate(ds):
          with tf.GradientTape() as tape:
            tf_s = src_sel(tf_batch)
            tf_d = dst_sel(tf_batch)
            tf_w_p = wgh_p_sel(tf_batch)
            tf_w_r = wgh_r_sel(tf_batch)
            
            tf_s_orig = embeds_old(tf_s)
            tf_s_new = embeds_new(tf_s)
            tf_d_new = embeds_new(tf_d)
            
            tf_p_diff = p_diff([tf_s_orig, tf_s_new])
            tf_r_diff = r_diff([tf_s_new, tf_d_new])    
            
            tf_p_nw = p_norm(tf_p_diff)
            tf_r_nw = r_norm(tf_r_diff)
            
            tf_p = p_weighting([tf_p_nw, tf_w_p]) 
            tf_r = r_weighting([tf_r_nw, tf_w_r]) 
            
            tf_retro = final_add([tf_p, tf_r])
            
            tf_loss = identity_loss(None, tf_retro)
          epoch_losses.append(tf_loss.numpy())
          grads = tape.gradient(tf_loss, model.trainable_weights)
          test = _convert(grads[0])
          opt.apply_gradients(zip(grads, model.trainable_weights))
          self.log.Pr("    Epoch {:03d} - {:.1f}% - loss: {:.2f}".format(
              epoch, i / n_batches * 100, np.mean(epoch_losses)))
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)          
        self.P("    Epoch {:03d} - loss: {:.2f}".format(epoch, epoch_loss))
        if epoch_loss < best_loss:
          best_loss = epoch_loss
          fails = 0
        else:
          fails += 1
        if fails >= patience or epoch_loss <= tol:
          self.P("    Stopping traing at epoch {}".format(epoch))
          break          
      self.P("  End eager dubug training")
      # end EAGER DEBUG training        
    else:              
      model.compile(optimizer=opt, loss=identity_loss)      
      tf.keras.utils.plot_model(
          model,
          to_file=os.path.join(self._save_folder,'model.png'),
          show_shapes=True,
          show_layer_names=True,
          expand_nested=True,
          )
      if use_fit:
        model.fit(x=data, y=data, epochs=epochs, batch_size=batch_size)
        best_embeds = embeds_new.get_weights()[0]  
      else:
        ds = tf.data.Dataset.from_tensor_slices(data)
        n_batches = data.shape[0] // batch_size + 1
        ds = ds.batch(batch_size)
        if gpu_optim:
          ds = ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))
          ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
          # ds = ds.
        else:
          ds = ds.prefetch(1)
        for epoch in range(1, epochs+1):
          epoch_losses = []
          t1 = time()
          b_shape = None          
          for i, tf_batch in enumerate(ds):
            if i == 0:
              b_shape = tf_batch.shape
            loss = model.train_on_batch(x=tf_batch, y=tf_batch)
            epoch_losses.append(loss)            
            self.log.Pr("    Epoch {:02d} - {:.1f}% - loss: {:.2f}".format(
                epoch, i / n_batches * 100, np.mean(epoch_losses)))
          t2 = time()
          epoch_loss = np.mean(epoch_losses)
          losses.append(epoch_loss)
          new_embeds = embeds_new.get_weights()[0]          
          if epoch_loss < best_loss:
            best_embeds = new_embeds
            best_loss = epoch_loss
            fails = 0
          else:
            fails += 1
          diff = self._measure_changes(last_embeds, new_embeds)
          self.P("    Epoch {:02d}/{} - loss: {:.2f}, change:{:.3f}, time: {:.1f}s, batch: {}, fails: {}".format(
              epoch, epochs, epoch_loss, diff, t2 - t1, b_shape, fails))
          if fails >= patience or diff <= tol:
            self.P("    Stopping traing at epoch {}".format(epoch))
            break
          last_embeds = new_embeds
        # end batch
      # end epoch
    # end else eager
    
    return best_embeds
  
  def _get_retrofitted_embeds_v2_th(self, 
                                    dct_edges, 
                                    dct_negative=None,
                                    eager=False, 
                                    use_fit=False,
                                    epochs=99, 
                                    batch_size=16384,
                                    gpu_optim=True,
                                    lr=0.05,
                                    tol=1e-3,
                                    patience=2,
                                    DEBUG=False,
                                    dist='l2',
                                    **kwargs):
    """
    this method implements a similar approach to Dingwell et al
    """
    self.P("Starting `_get_retrofitted_embeds_v2_th`...")
    import torch as th

    vocab_size = self.embeds.shape[0]
    embedding_dim = self.embeds.shape[1]

    data = self._prepare_retrofit_data(
        dct_positive=dct_edges,
        dct_negative=dct_negative,
        split=True,
        pad_id = vocab_size
        )
        
    self.P("  Preparing torch model...")
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
      th.cuda.empty_cache()
    
    tensors = [th.tensor(x, requires_grad=False, device=device) for x in data]

    th_embeds = th.tensor(self.embeds, dtype=th.float32, requires_grad=False, device=device)
    th_embeds_pad = th.cat((th_embeds, th.zeros((1,embedding_dim), device=device)))
    
    if DEBUG:
      if dct_negative is not None and len(dct_negative) > 0:
        for i, v in enumerate(data[3]):
          if v[0] != vocab_size:
            break
        batch_size = i + 1
    
    ds = th.utils.data.TensorDataset(*tensors)
    dl = th.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size, 
        shuffle=not DEBUG,
        )
    
    emb_new = th.nn.Embedding(
        vocab_size + 1, 
        embedding_dim, 
        padding_idx=vocab_size).to(device)
    
    emb_new.weight.data.copy_(th_embeds_pad)
    opt = th.optim.SGD(params=emb_new.parameters(), lr=lr) 
    n_batches = len(data[0]) // batch_size + 1
    losses = []
    best_loss = np.inf
    fails = 0
    last_embeds = self.embeds
    if dct_negative is not None and len(dct_negative) > 0:
      negative_margin = 128 
    else:
      negative_margin = 0
    self.P("  Training model for {} epochs, batch={}, lr={:.1e}, tol={:.1e}{}, dist={}".format(
        epochs, batch_size, lr, tol, 
        ", negative margin: {}".format(negative_margin) if negative_margin>0 else "",
        dist))
    for epoch in range(1, epochs + 1):
      epoch_losses = []
      t1 = time()
      for i, batch in enumerate(dl):
        th_ids, th_pos, th_pos_w, th_neg, th_neg_w = batch
        b_shape = th_pos.shape
        th_pos_w = th_pos_w.unsqueeze(-1)
        th_neg_w = th_neg_w.unsqueeze(-1)
        
        th_org_embs_raw = th_embeds[th_ids].unsqueeze(1)
        th_new_embs_raw = emb_new(th_ids).unsqueeze(1)
        th_pos_embs_raw = emb_new(th_pos)
        th_neg_embs_raw = emb_new(th_neg)
        
        th_org_embs = th_org_embs_raw
        th_new_embs = th_new_embs_raw
        th_pos_embs = th_pos_embs_raw
        th_neg_embs = th_neg_embs_raw
        
        
#        th_org_embs = th.nn.functional.normalize(th_org_embs, p=2, dim=-1)
#        th_new_embs = th.nn.functional.normalize(th_new_embs, p=2, dim=-1)
#        th_pos_embs = th.nn.functional.normalize(th_pos_embs, p=2, dim=-1)
#        th_neg_embs = th.nn.functional.normalize(th_neg_embs, p=2, dim=-1)
        if dist == 'l2':
          th_preserve_loss = (th_org_embs - th_new_embs).pow(2).sum(-1)
        else:
          th_preserve_loss = (th_org_embs - th_new_embs).abs().sum(-1)
        
        if dist == 'l2':
          th_relate_nm = (th_pos_embs - th_new_embs).pow(2).sum(-1)
        else:
          th_relate_nm = (th_pos_embs - th_new_embs).abs().sum(-1)
        th_relate_mask = ((th_pos_embs == 0).sum(-1) < 128).float()
        th_relate_masked = th_relate_nm * th_relate_mask        
        th_relate_w = th_relate_masked * th_pos_w
        th_relate_loss = th_relate_w.sum(-1, keepdims=True)
        
        if dist == 'l2':
          th_neg_nm = (th_new_embs - th_neg_embs).pow(2).sum(-1)
        else:
          th_neg_nm = (th_new_embs - th_neg_embs).abs().sum(-1)          
        th_neg_nm_d = th.clamp(negative_margin - th_neg_nm, min=0)
        th_neg_mask = (th_neg_embs.sum(-1) > 0).float()
        th_neg_masked = th_neg_mask * th_neg_nm_d        
        th_neg = th_neg_masked * th_neg_w
        th_neg_loss = th_neg.sum(-1, keepdims=True)
        
        th_loss = th_preserve_loss.sum() + th_relate_loss.sum() + th_neg_loss.sum()
        
        opt.zero_grad()
        th_loss.backward()
        opt.step()
        epoch_losses.append(th_loss.detach().cpu().numpy())
        self.log.Pr("    Epoch {:02d} - {:.1f}% - loss: {:.2f}".format(
            epoch, i / n_batches * 100, np.mean(epoch_losses)))
      # end batch
      t2 = time()
      epoch_loss = np.mean(epoch_losses)
      losses.append(epoch_loss)
      new_embeds = emb_new.weight.data.detach().cpu().numpy()[:-1]         
      if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_embeds = new_embeds
        fails = 0
      else:
        fails += 1
      diff = self._measure_changes(last_embeds, new_embeds)
      self.P("    Epoch {:02d}/{} - loss: {:.2f}, change:{:.3f}, time: {:.1f}s,  batch: {}, fails: {}".format(
          epoch, epochs, epoch_loss, diff, t2 - t1, b_shape, fails))
      if fails >= patience or diff <= tol:
        self.P("    Stopping traing at epoch {}".format(epoch))
        break
      last_embeds = new_embeds
    # end epoch
    return best_embeds
        
    

  def _get_retrofitted_embeds_v3_th(self, 
                                    dct_edges, 
                                    dct_negative=None,
                                    eager=False, 
                                    use_fit=False,
                                    epochs=99, 
                                    batch_size=16384,
                                    gpu_optim=True,
                                    lr=0.05,
                                    tol=1e-4,
                                    patience=2,
                                    DEBUG=False,
                                    **kwargs):
    """
    this method implements a similar approach to Dingwell et al
    """
    self.P("Starting `_get_retrofitted_embeds_v3_th`...")
    import torch as th

    vocab_size = self.embeds.shape[0]
    embedding_dim = self.embeds.shape[1]

    data = self._prepare_retrofit_data(
        dct_positive=dct_edges,
        dct_negative=dct_negative,
        split=True,
        pad_id = vocab_size,
#        fix_weights=1,
        )
    
    if DEBUG:
      batch_size = 8
      if dct_negative is not None and len(dct_negative) > 0:
        for i, v in enumerate(data[3]):
          if v[0] != vocab_size:
            break
        batch_size = i + 1
    
        
    self.P("  Preparing torch model...")
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
      th.cuda.empty_cache()
    
    tensors = [th.tensor(x, requires_grad=False, device=device) for x in data]

    th_embeds = th.tensor(self.embeds, dtype=th.float32, requires_grad=False, device=device)
    th_embeds_pad = th.cat((th_embeds, th.zeros((1,embedding_dim), device=device)))
    
    ds = th.utils.data.TensorDataset(*tensors)
    dl = th.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size, 
        shuffle=not DEBUG,
        )
    
    emb_new = th.nn.Embedding(
        vocab_size + 1, 
        embedding_dim, 
        padding_idx=vocab_size).to(device)
    
    emb_new.weight.data.copy_(th_embeds_pad)
    opt = th.optim.SGD(params=emb_new.parameters(), lr=lr) 
    n_batches = len(data[0]) // batch_size + 1
    losses = []
    best_loss = np.inf
    fails = 0
    last_embeds = self.embeds
    self.P("  Training model for {} epochs, batch={}, lr={:.1e}, tol={:.1e}".format(
        epochs, batch_size, lr, tol))
    for epoch in range(1, epochs + 1):
      epoch_losses = []
      t1 = time()
      for i, batch in enumerate(dl):
        th_ids, th_pos, th_pos_w, th_neg, th_neg_w = batch
        b_shape = th_pos.shape
        th_pos_w = th_pos_w.unsqueeze(-1)
        th_neg_w = th_neg_w.unsqueeze(-1)
        
        th_org_embs = th_embeds[th_ids].unsqueeze(1)
        th_new_embs = emb_new(th_ids).unsqueeze(1)
        th_pos_embs = emb_new(th_pos)
        th_neg_embs = emb_new(th_neg)
        
        
        th_preserve_loss = (th_org_embs - th_new_embs).pow(2).sum(-1)
        
        
        th_relate_nm = 1 - th.nn.functional.cosine_similarity(th_new_embs, th_pos_embs, dim=-1) 
        th_relate_mask = ((th_pos_embs == 0).sum(-1) < 128).float()
        th_relate_masked = th_relate_nm * th_relate_mask        
        th_relate_w = th_relate_masked * th_pos_w
        th_relate_loss = th_relate_w.sum(-1, keepdims=True)
        
        th_neg_nm = th.nn.functional.cosine_similarity(th_new_embs, th_neg_embs, dim=-1)#(th_new_embs - th_neg_embs).abs().sum(-1)
        th_neg_v = th.clamp(th_neg_nm, min=0)
        th_neg_mask = (th_neg_embs.sum(-1) > 0).float()
        th_neg_masked = th_neg_mask * th_neg_v        
        th_neg_weighted = th_neg_masked * th_neg_w
        th_neg_loss = th_neg_weighted.sum(-1, keepdims=True)
        
        th_loss = th_preserve_loss.sum() + th_relate_loss.sum() + th_neg_loss.sum()
        
        opt.zero_grad()
        th_loss.backward()
        opt.step()
        epoch_losses.append(th_loss.detach().cpu().numpy())
        self.log.Pr("    Epoch {:02d} - {:.1f}% - loss: {:.2f}".format(
            epoch, i / n_batches * 100, np.mean(epoch_losses)))
      # end batch
      t2 = time()
      epoch_loss = np.mean(epoch_losses)
      losses.append(epoch_loss)
      new_embeds = emb_new.weight.data.detach().cpu().numpy()[:-1]         
      if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_embeds = new_embeds
        fails = 0
      else:
        fails += 1
      diff = self._measure_changes(last_embeds, new_embeds)
      self.P("    Epoch {:02d}/{} - loss: {:.2f}, change:{:.3f}, time: {:.1f}s,  batch: {}, fails: {}".format(
          epoch, epochs, epoch_loss, diff, t2 - t1, b_shape, fails))
      if fails >= patience or diff <= tol:
        self.P("    Stopping traing at epoch {}".format(epoch))
        break
      last_embeds = new_embeds
    # end epoch
    return best_embeds
    
    
  
  

###############################################################################
# END EmbedsEngine
###############################################################################

