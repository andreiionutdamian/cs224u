# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:24:36 2020

@author: damia
"""
import numpy as np
from collections import Counter
import os
import rel_ext
import utils

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from functools import partial

def P(s=''):
  print(s, flush=True)

def Pr(s=''):
  print('\r' + str(s), end='', flush=True)
  
def get_POS(str_POS):
  base = [x for x in str_POS.strip().split(' ') if x]
  parts = [x.strip().split('/', 1)[1] for x in base]
  return ['<s>'] + parts + ['</s>']

def middle_bigram_pos_tag_featurizer(kbt, corpus, feature_counter):      
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        parts = get_POS(ex.middle_POS)
        for i in range(len(parts)-1):
          tag = parts[i] + ' ' + parts[i+1]
          feature_counter[tag] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        parts = get_POS(ex.middle_POS)
        for i in range(len(parts)-1):
          tag = parts[i] + ' ' + parts[i+1]
          feature_counter[tag] += 1
    return feature_counter    
  
def glove_middle_featurizer(dct_glove, kbt, corpus, np_func=np.sum):
    reps = []
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split():
            rep = dct_glove.get(word)
            if rep is not None:
                reps.append(rep)
    # A random representation of the right dimensionality if the
    # example happens not to overlap with GloVe's vocabulary:
    if len(reps) == 0:
        dim = len(next(iter(dct_glove.values())))                
        return utils.randvec(n=dim)
    else:
        return np_func(reps, axis=0)    

  
def simple_bag_of_words_featurizer(kbt, corpus, feature_counter):
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split(' '):
            feature_counter[word] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for word in ex.middle.split(' '):
            feature_counter[word] += 1
    return feature_counter  
  
def directional_bag_of_words_featurizer(kbt, corpus, feature_counter): 
    # Append these to the end of the keys you add/access in 
    # `feature_counter` to distinguish the two orders. You'll
    # need to use exactly these strings in order to pass 
    # `test_directional_bag_of_words_featurizer`.
    subject_object_suffix = "_SO"
    object_subject_suffix = "_OS"
    
    ##### YOUR CODE HERE
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for word in ex.middle.split(' '):
            feature_counter[word + subject_object_suffix] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for word in ex.middle.split(' '):
            feature_counter[word + object_subject_suffix] += 1
    return feature_counter    
  
  

if __name__ == '__main__':
  
  RUN_BASELINE = False
  RUN_GLOVE_LIN = False
  RUN_SVC = False
  RUN_DIRECTION = False
  RUN_POS = True
  
  
  utils.fix_random_seeds()
  P("Loading data...")
  rel_ext_data_home = os.path.join('data', 'rel_ext_data')
  P("  Loading corpus...")
  corpus = rel_ext.Corpus(os.path.join(rel_ext_data_home, 'corpus.tsv.gz'))
  P("  Loading kb...")
  kb = rel_ext.KB(os.path.join(rel_ext_data_home, 'kb.tsv.gz'))
  dataset = rel_ext.Dataset(corpus, kb)
  splits = dataset.build_splits(
    split_names=['tiny', 'train', 'dev'],
    split_fracs=[0.01, 0.79, 0.20],
    seed=1)
  P("Splits:")
  for split in splits:
    P("  {:<7} {}".format(split+':', splits[split]))
    
  baseline_model_factory = lambda: LogisticRegression(fit_intercept=True, solver='liblinear', verbose=True)
  svc_model_factory = lambda: SVC(kernel='linear', verbose=True)
    
  if RUN_BASELINE:   
    featurizers = [simple_bag_of_words_featurizer]
    
    P("Training LogReg on hand-made features...")
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        featurizers=featurizers,
        model_factory=baseline_model_factory,
        vectorize=True,
        verbose=True,
        return_macro=True)  
    baseline_results, baseline_f1 = res
    
    if False:
      rel_ext.examine_model_weights(baseline_results)

  
  GLOVE_HOME = os.path.join('data', 'glove.6B')

  P("Loading Glove...")    
  glove_lookup = utils.glove2dict(
    os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))  

  if RUN_GLOVE_LIN:
    P("Training LogReg on glove vectors...")
    glove_middle_featurizer_func = partial(glove_middle_featurizer, dct_glove=glove_lookup)
    
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        featurizers=[glove_middle_featurizer_func], 
        model_factory=baseline_model_factory,
        vectorize=False, # Crucial for this featurizer!
        verbose=True,
        return_macro=True)     
    glove_results, glove_f1 = res
    
  if RUN_SVC:
    res = rel_ext.experiment(
                splits,
                train_split='train',
                test_split='dev',
                featurizers=[glove_middle_featurizer], 
                model_factory=svc_model_factory,
                vectorize=False, # we are using Glove
                verbose=True,
                return_macro=True)  
    svc_results, svc_f1 = res
    
       
    
  if RUN_DIRECTION:
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        featurizers=[directional_bag_of_words_featurizer],
        model_factory=baseline_model_factory,
        verbose=True,
        return_macro=True)
    dir_results, dir_f1 = res
    if RUN_BASELINE:
      print("Previous features: {}".format(len(baseline_results['vectorizer'].feature_names_)))
      print("Current features: {}".format(len(dir_results['vectorizer'].feature_names_)))
  # end glove  
    
  if RUN_POS:
    from collections import defaultdict
    kbt = rel_ext.KBTriple(rel='worked_at', sbj='Randall_Munroe', obj='xkcd')
    feature_counter = defaultdict(int)
    # Make sure `feature_counter` is being updated, not reinitialized:
    feature_counter['<s> VBZ'] += 5
    feature_counter = middle_bigram_pos_tag_featurizer(kbt, corpus, feature_counter)
    expected = defaultdict(
        int, {'<s> VBZ':6,'VBZ DT':1,'DT JJ':1,'JJ VBN':1,'VBN IN':1,'IN </s>':1})
    assert feature_counter == expected, \
        "Expected:\n{}\nGot:\n{}".format(expected, feature_counter)      
    P("middle_bigram_pos_tag_featurizer PASSED")
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        model_factory=baseline_model_factory,
        featurizers=[middle_bigram_pos_tag_featurizer],
        verbose=True,
        return_macro=True)
    pos_results, pos_f1 = res          
