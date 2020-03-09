# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:24:36 2020

@author: damia
"""
import numpy as np
import os
import rel_ext
import utils
if "wn" not in globals():
  print("Loading WordNet...", flush=True)
  from nltk.corpus import wordnet as wn
  print("Done loading WordNet.", flush=True)
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from time import time
from collections import OrderedDict

from torch_rnn_classifier import TorchRNNClassifier

RUN_BASELINE    = False # 0.56
RUN_GLOVE_LIN   = False # 0.53
RUN_SVC         = False # 0.10
RUN_DIRECTION   = False # 0.61
RUN_POS         = False # 0.45
RUN_SYN         = False # 0.53

RUN_RNN1        = True  #

SVC_MAX_ITER    = 4



def P(s=''):
  print(s, flush=True)

def Pr(s=''):
  print('\r' + str(s), end='', flush=True)
  
def get_POS(str_POS):
  base = [x for x in str_POS.strip().split(' ') if x]
  parts = [x.strip().split('/', 1)[1] for x in base]
  return ['<s>'] + parts + ['</s>']

def middle_bigram_pos_tag_featurizer2(kbt, corpus, feature_counter):      
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
  

def get_tag_bigrams(s):
    """Suggested helper method for `middle_bigram_pos_tag_featurizer`.
    This should be defined so that it returns a list of str, where each 
    element is a POS bigram."""
    # The values of `start_symbol` and `end_symbol` are defined
    # here so that you can use `test_middle_bigram_pos_tag_featurizer`.
    start_symbol = "<s>"
    end_symbol = "</s>"    
    ##### YOUR CODE HERE
    parts = [start_symbol] + get_tags(s) + [end_symbol]
    res = []
    for i in range(len(parts)-1):
        res.append(parts[i]+ ' ' + parts[i+1])
    return res

    
def get_tags(s): 
    """Given a sequence of word/POS elements (lemmas), this function
    returns a list containing just the POS elements, in order.    
    """
    return [parse_lem(lem)[1] for lem in s.strip().split(' ') if lem]


def parse_lem(lem):
    """Helper method for parsing word/POS elements. It just splits
    on the rightmost / and returns (word, POS) as a tuple of str."""
    return lem.strip().rsplit('/', 1)  
  
def get_synsets(s):
    """Suggested helper method for `synset_featurizer`. This should
    be completed so that it returns a list of stringified Synsets 
    associated with elements of `s`.
    """   
    # Use `parse_lem` from the previous question to get a list of
    # (word, POS) pairs. Remember to convert the POS strings.
    wt = [parse_lem(lem) for lem in s.strip().split(' ') if lem]
    
    ##### YOUR CODE HERE
    res = []
    for word, tag in wt:
        t = convert_tag(tag)
        syns = wn.synsets(word, pos=t)
        for ss in syns:
            res.append(str(ss))
    return res

    
    
def convert_tag(t):
    """Converts tags so that they can be used by WordNet:
    
    | Tag begins with | WordNet tag |
    |-----------------|-------------|
    | `N`             | `n`         |
    | `V`             | `v`         |
    | `J`             | `a`         |
    | `R`             | `r`         |
    | Otherwise       | `None`      |
    """        
    if t[0].lower() in {'n', 'v', 'r'}:
        return t[0].lower()
    elif t[0].lower() == 'j':
        return 'a'
    else:
        return None    
      
      
def middle_bigram_pos_tag_featurizer(kbt, corpus, feature_counter):
    
    ##### YOUR CODE HERE
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for bigram in get_tag_bigrams(ex.middle_POS):
          feature_counter[bigram] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for bigram in get_tag_bigrams(ex.middle_POS):
          feature_counter[bigram] += 1
    return feature_counter    
  
  
def synset_featurizer(kbt, corpus, feature_counter):
    
    ##### YOUR CODE HERE
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
        for ss in get_synsets(ex.middle_POS):
          feature_counter[ss] += 1
    for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        for ss in get_synsets(ex.middle_POS):
          feature_counter[ss] += 1
    return feature_counter      

    return feature_counter  



###############################################################################
###############################################################################
###############################################################################


if __name__ == '__main__':

  ################# timing and results centralization ##################
  dct_timers = OrderedDict()
  def start_timer(s):
    _t_start = time()
    dct_timers[s + '_S'] = _t_start
    return _t_start
  
  def end_timer(s, result):
    _t_end = time()
    dct_timers[s + '_E'] =_t_end
    _t_full = _t_end - dct_timers[s + '_S']
    dct_timers[s] = _t_full
    dct_timers[s + '_R'] = result
    return _t_full
  
  def print_results():
    all_models = [x for x in dct_timers if x[-2:] not in ['_S','_E','_R']]
    dct_res = {x:dct_timers[x+'_R'] for x in all_models}
    P("\nFinal results:")
    l = sorted([(k,v) for k,v in dct_res.items()], key=lambda x:x[1])
    for n,v in l:
      P("  {:<17} {:.4f} - time: {:>5.1f}s".format(n+':',v, dct_timers[n]))
    
  
  
  utils.fix_random_seeds()
  P("Loading data...")
  rel_ext_data_home = os.path.join('data', 'rel_ext_data')
  if 'corpus' not in globals():
    P("  Loading corpus...")
    corpus = rel_ext.Corpus(os.path.join(rel_ext_data_home, 'corpus.tsv.gz'))
  if 'kb' not in globals():
    P("  Loading kb...")
    kb = rel_ext.KB(os.path.join(rel_ext_data_home, 'kb.tsv.gz'))
  if 'dataset' not in globals():
    dataset = rel_ext.Dataset(corpus, kb)
    splits = dataset.build_splits(
      split_names=['tiny', 'train', 'dev'],
      split_fracs=[0.01, 0.79, 0.20],
      seed=1)
  P("  Splits:")
  for split in splits:
    P("    {:<7} {}".format(split+':', splits[split]))
  GLOVE_HOME = os.path.join('data', 'glove.6B')

  if 'glove_lookup' not in globals():
    P("  Loading Glove...")    
    glove_lookup = utils.glove2dict(
      os.path.join(GLOVE_HOME, 'glove.6B.300d.txt'))  
  glv_emb_size = list(glove_lookup.values())[0].shape[0]
  P("Done loading data.")
    
######################
#  Models factory
######################
  baseline_model_factory = lambda: LogisticRegression(fit_intercept=True, 
                                                      solver='liblinear', 
                                                      verbose=True,
                                                      random_state=42)
  svc_model_factory = lambda: SVC(kernel='linear', 
                                  verbose=True, 
                                  max_iter=SVC_MAX_ITER,
                                  random_state=42)
  
######################
#  END Models factory
######################  
  
######### glove feature generators


  def glove_middle_featurizer(kbt, corpus, np_func=np.sum):
      reps = []
      for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
          for word in ex.middle.split():
              rep = glove_lookup.get(word)
              if rep is not None:
                  reps.append(rep)
      # A random representation of the right dimensionality if the
      # example happens not to overlap with GloVe's vocabulary:
      if len(reps) == 0:
          dim = len(next(iter(glove_lookup.values())))                
          return utils.randvec(n=dim)
      else:
          return np_func(reps, axis=0)    
  
  
  def get_pair_feats(kbt, corpus, how='middle', max_words=1000):
    reps = []
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
      str_ex = ''
      if how == 'full':
        str_ex = ' '.join((ex.left, ex.mention_1, ex.middle, ex.mention_2, ex.right))
      else:
        if 'left' in how:
          str_ex += ex.left
        if 'middle' in how:
          str_ex += ex.middle
        if 'right' in how:
          str_ex += ex.right
      for word in str_ex.split():
        rep = glove_lookup.get(word)
        if rep is not None:
          reps.append(rep.astype(np.float32))
    if len(reps) == 0:
      reps = [np.zeros(glv_emb_size, dtype=np.float32)]
    if len(reps) > max_words:
      P("kbt: {} has {} obs, reducing to {}".format(
          kbt, len(reps), max_words))
      reps = reps[:max_words]
    return np.array(reps)


#######################################################       
#######################################################       
    
  if RUN_BASELINE:
    exp_name = 'baseline'
    start_timer(exp_name)
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
    end_timer(exp_name, baseline_f1)
    
    if False:
      rel_ext.examine_model_weights(baseline_results)


#######################################################       
#######################################################       
  
  if RUN_GLOVE_LIN:
    exp_name = 'glove_lin'
    start_timer(exp_name)
    P("Training LogReg on glove vectors...")
    
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        featurizers=[glove_middle_featurizer], 
        model_factory=baseline_model_factory,
        vectorize=False, # Crucial for this featurizer!
        verbose=True,
        return_macro=True)     
    glove_results, glove_f1 = res
    end_timer(exp_name,glove_f1)

#######################################################       
#######################################################       
    
  if RUN_SVC:
    start_timer('glove_svc')
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
    end_timer('glove_svc',svc_f1)
    
#######################################################       
#######################################################       
    
  if RUN_DIRECTION:
    start_timer('directional_lin')
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        featurizers=[directional_bag_of_words_featurizer],
        model_factory=baseline_model_factory,
        verbose=True,
        return_macro=True)
    dir_results, dir_f1 = res
    end_timer('directional_lin', dir_f1)
    if RUN_BASELINE:
      P("Previous features: {}".format(len(baseline_results['vectorizer'].feature_names_)))
      P("Current features: {}".format(len(dir_results['vectorizer'].feature_names_)))
  # end glove  

#######################################################       
#######################################################       
    
  if RUN_POS:
    start_timer('pos_lin')
    P("POS tests")
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
    end_timer('pos_lin',pos_f1)

#######################################################       
#######################################################       

  if RUN_SYN:
    start_timer('syn_lin')
    P("Syns test")
    from collections import defaultdict
    kbt = rel_ext.KBTriple(rel='worked_at', sbj='Randall_Munroe', obj='xkcd')
    feature_counter = defaultdict(int)
    # Make sure `feature_counter` is being updated, not reinitialized:
    feature_counter["Synset('be.v.01')"] += 5
    feature_counter = synset_featurizer(kbt, corpus, feature_counter)
    # The full return values for this tend to be long, so we just
    # test a few examples to avoid cluttering up this notebook.
    test_cases = {
        "Synset('be.v.01')": 6,
        "Synset('embody.v.02')": 1
    }
    for ss, expected in test_cases.items():   
        result = feature_counter[ss]
        assert result == expected, \
            "Incorrect count for {}: Expected {}; Got {}".format(ss, expected, result)
        
    # Call to `rel_ext.experiment`:
    ##### YOUR CODE HERE    
    P("synset_featurizer passed")
    res = rel_ext.experiment(
        splits,
        train_split='train',
        test_split='dev',
        model_factory=baseline_model_factory,
        featurizers=[synset_featurizer],
        verbose=True,
        return_macro=True)    
    syn_results, syn_f1 = res
    end_timer('syn_lin',syn_f1)
    
  if RUN_RNN1:
    rnn1_model_factory = lambda: TorchRNNClassifier(vocab={}, 
                                                    use_embedding=False,
                                                    batch_size=64,
                                                    max_iter=25)
    featurizer_func = partial(get_pair_feats, how='middle')
    start_timer('rnn1_middle')
    res = rel_ext.experiment(
            splits,
            train_split='train',
            test_split='dev',
            model_factory=rnn1_model_factory,
            featurizers=[featurizer_func],
            vectorize=False,
            verbose=True,
            return_macro=True)   
    
    rnn1_results, rnn1_f1 = res
    end_timer('rnn1_middle', rnn1_f1)

  print_results()