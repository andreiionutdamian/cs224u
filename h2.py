# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 17:24:36 2020

@author: damia
"""
import numpy as np
import pandas as pd
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

RUN_RNN1        = False # 0.58
RUN_GRID        = True  # 0.69
RUN_FINAL       = False

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
    
#######################################################       
#######################################################       
#######################################################       
#######################################################     
    
  """

  """
  
  def prepare_grid_search(params_grid, nr_trials):
    import itertools
  
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
  
    params = []
    values = []
    for k in params_grid:
      params.append(k)
      assert type(params_grid[k]) is list, 'All grid-search params must be lists. Error: {}'.format(k)
      values.append(params_grid[k])
    combs = list(itertools.product(*values))
    n_options = len(combs)
    grid_iterations = []
    for i in range(n_options):
      comb = combs[i]
      func_kwargs = {}
      for j,k in enumerate(params):
        func_kwargs[k] = comb[j]
      grid_iterations.append(func_kwargs)
    idxs = np.arange(n_options)
    if nr_trials < n_options:
      idxs = np.random.choice(idxs, size=nr_trials, replace=False)
    return [grid_iterations[i] for i in idxs]
  
  
  def get_seq_feats(kbt, corpus, how='midl', two_dir=True, max_words=50):
    assert np.all([x[:4] in "left-rght-men1-midl-men2-full".split('-') for x in how.split('-')])
    reps = []
    so_sents = []
    how = how.lower()
    def extract_text(exmpl):
      str_ex = ''
      if how == 'full':
        str_ex = ' '.join((exmpl.left, exmpl.mention_1, exmpl.middle, exmpl.mention_2, exmpl.right))
      else:
        if 'left' in how:
          snr = how[how.index('left')+4:how.index('left')+6]
          nr = int(snr) if snr.isalnum() else 0          
          str_ex += ' ' + exmpl.left[-nr:]
        if 'men1' in how:
          str_ex += ' ' + exmpl.mention_1
        if 'midl' in how:
          str_ex += ' ' + exmpl.middle
        if 'men2' in how:
          str_ex += ' ' + exmpl.mention_2
        if 'rght' in how:
          snr = how[how.index('rght')+4:how.index('rght')+6]
          nr = int(snr) if snr.isalnum() else 0          
          str_ex += ' ' + exmpl.right[:nr]
      return str_ex
    
    for ex in corpus.get_examples_for_entities(kbt.sbj, kbt.obj):
      str_ex = extract_text(ex)
      so_sents.append(str_ex)
    so_sents = sorted(so_sents, key=lambda x: len(x))
    so_best = so_sents[-1] if len(so_sents) > 0 else ''
    
    os_best = ''
    if two_dir:
      os_sents = []
      for ex in corpus.get_examples_for_entities(kbt.obj, kbt.sbj):
        str_ex = extract_text(ex)
        os_sents.append(str_ex)
      os_sents = sorted(os_sents, key=lambda x:len(x))
      os_best = os_sents[-1] if len(os_sents) > 0 else ''
  
    str_text = so_best + ' ' + os_best
      
    str_rep = ''
    for word in str_text.split():
      w = word.lower()
      rep = glove_lookup.get(w)
      if rep is not None:
        reps.append(rep.astype(np.float32))
        str_rep += ' ' + w
  
  
    if len(reps) == 0:
      reps = [np.zeros(glv_emb_size, dtype=np.float32)]
      
    return np.array(reps[:max_words], dtype=np.float32)

    
          
    
  if RUN_RNN1:
    rnn1_model_factory = lambda: TorchRNNClassifier(vocab={}, 
                                                    use_embedding=False,
                                                    batch_size=128,
                                                    bidirectional=True,
                                                    eta=0.001,
                                                    max_iter=50)
    featurizer_func = partial(get_seq_feats, how='middle')
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

  if RUN_GRID:
    from sklearn.metrics import precision_recall_fscore_support
    import torch as th
  #### GRID
    grid = {
        'eta' : [
            0.005,
            0.0005,
            ],
            
        'l2_strength' : [
            0,
            0.0001
            ],
            
        'how' : [
#            'full',
#            'midl',
#            'left-right',
            'men1-midl-men2',
            'left15-men1-midl-men2-rght15',
            ],
            
        'batch_size' : [
#            64,
#            128,
            256,
            512,
            ],
        'bidirectional' : [
             True,
             False,
            ],
            
        'max_words' : [
            50,
            100,
            ],
            
        'two_dir' : [
#            True,
            False
            ],
            
        'hidden_dim' : [
            128,
            256,
            512
            ],
            
        }
    
    dct_results = OrderedDict({'MODEL':[], 'F05': [], 'HRS': []})
    for k in grid: 
      dct_results[k] = []

    max_epochs = 1000
    early_stop_steps = 5
    epochs_per_fit = 1
    
    options = prepare_grid_search(grid, nr_trials=40)
    timings = []
    t_left = np.inf
    max_epochs = 1000
    for grid_iter, option in enumerate(options):
      model_name = 'rnn_v1_{:02}'.format(grid_iter+1)
      P("\n\n" + "=" * 70)
      P("Running grid search iteration {}/{} '{}': {}".format(
          grid_iter+1, len(options), model_name, option))
      P("  Time left for grid search completion: {:.1f} hrs".format(t_left / 3600))
      dct_results['MODEL'].append(model_name)
      for k in option:
        dct_results[k].append(option[k])
      max_words = option.pop('max_words')
      how = option.pop('how')
      two_dir = option.pop('two_dir')
      rnn1_model_factory = lambda: TorchRNNClassifier(vocab={}, 
                                                      use_embedding=False,
                                                      max_iter=epochs_per_fit,
                                                      **option)
      featurizer_func = partial(get_seq_feats, how=how, max_words=max_words, two_dir=two_dir)
      start_timer(model_name)
      ####
      ### now we prepare the dataset
      # first the train
      train_dataset = splits['train']
      train_o, train_y = train_dataset.build_dataset()
      P("Featurizing train dataset...")
      train_X, vectorizer = train_dataset.featurize(
          train_o, [featurizer_func], vectorize=False)
      # now train_X, train_y holds the train data
      
      # now the dev
      assess_dataset = splits['dev']
      assess_o, assess_y = assess_dataset.build_dataset()
      P("Featurizing dev dataset...")
      featurizers = [featurizer_func]
      test_X, _ = assess_dataset.featurize(
          assess_o,
          featurizers=featurizers,
          vectorizer=None,
          vectorize=False)
      # now test_X and assess_y holds the dev data
      

      models = {}
      early_stops = {}
      bests = {}
      n_rels = len(splits['all'].kb.all_relations)
      P("Training {} {} classifiers on {} relations".format(
          n_rels, rnn1_model_factory().__class__.__name__, n_rels))
      
      for i_rel, rel in enumerate(splits['all'].kb.all_relations):
        models[rel] = rnn1_model_factory()
        P("Training {}/{}: Running {}.fit() for rel={} for max {} epochs with early stop...".format(
                i_rel + 1, n_rels, models[rel].__class__.__name__, rel, max_epochs))
        best_rel_f1 = 0
        best_rel_model = ''
        patience = 0
        max_patience = 10
        for ep in range(1, max_epochs + 1):
          models[rel].fit(train_X[rel], train_y[rel])
          # finished fit stage now lets evaluate
          predictions =  models[rel].predict(test_X[rel], verbose=False)
          stats = precision_recall_fscore_support(assess_y[rel], predictions, beta=0.5)
          stats = [stat[1] for stat in stats]     
          rel_f1 = stats[2]
          if best_rel_f1 < rel_f1:
            patience = 0
            if best_rel_model != '':
              try:
                os.remove(best_rel_model)
                P("  Old model file '{}' removed".format(best_rel_model))
              except:
                P("FAILED to remove old model file")
            best_rel_f1 = rel_f1
            best_rel_model = 'model_rel_{}_{}.th'.format(rel, ep)
            th.save(models[rel].model.state_dict(), best_rel_model)
            P("  Found new best for rel={} with f05={:.4f} @ ep {}".format(
                rel, best_rel_f1, ep * epochs_per_fit))
            early_stops[rel] = ep * epochs_per_fit
            bests[rel] = raound(best_rel_f1,3)
          else:
            patience += 1
            P("  Model did not improve {:.4f} < {:.4f}. Patience {}/{}".format(
                rel_f1, best_rel_f1, patience, max_patience))
          if patience >= max_patience:
            P("  Stopping trainn for rel '{}' after {} epochs".format(
                rel, ep * epochs_per_fit))
            break
        if best_rel_model != '':
          P("  Loading best model '{}' for rel='{}'".format(best_rel_model, rel))
          models[rel].model.load_state_dict(th.load(best_rel_model))
          predictions =  models[rel].predict(test_X[rel], verbose=False)
          stats = precision_recall_fscore_support(assess_y[rel], predictions, beta=0.5)
          stats = [stat[1] for stat in stats]     
          rel_f1 = stats[2]   
          P("  Final model for rel '{}' has a F0.5 of {:.4f}".format(rel,
            rel_f1))
          assert rel_f1 == best_rel_f1, "Results can not be replicated {} vs {} ".format(best_rel_f1, rel_f1)
          try:
            new_fn = "best_rel_ext_{}_F05_{:.4f}_ep_{:03}.th".format(
                rel, rel_f1,  early_stops[rel])
            os.rename(best_rel_model, new_fn)
            P("Model '{}' renamed to '{}'".format(best_rel_model, new_fn))
          except:
            P("  Model could not be renamed")
      # now we have trained one model for each realtion with independent early stopping                        
      train_result = {
          'featurizers': featurizers,
          'vectorizer': vectorizer,
          'models': models,
          'all_relations': splits['all'].kb.all_relations,
          'vectorize': False}
      predictions, test_y = rel_ext.predict(
          splits,
          train_result,
          split_name='dev',
          vectorize=False)
      eval_res = rel_ext.evaluate_predictions(
                    predictions,
                    test_y,
                    verbose=True)
      P("\nDouble check: {}\n".format(bests))

      
      rnn_f1 = eval_res
      ####
      t_res = end_timer(model_name, rnn_f1)  
      timings.append(t_res)
      t_left = (len(options) - grid_iter - 1) * np.mean(timings)
      dct_results['F05'].append(rnn_f1)
      dct_results['HRS'].append(round(t_res/3600,2))
      df = pd.DataFrame(dct_results).sort_values('F05')
      P("Results so far:\n{}".format(df))


  if RUN_FINAL:
    from sklearn.metrics import precision_recall_fscore_support
    import torch as th
    
    utils.fix_random_seeds()
      
    max_epochs = 1000
    early_stop_steps = 5
    epochs_per_fit = 1
    fit_iters = max_epochs // epochs_per_fit
    final_model_factory  = lambda: TorchRNNClassifier(vocab={}, 
                                                      use_embedding=False,
                                                      warm_start=True,
                                                      max_iter=epochs_per_fit,
                                                      eta=0.001,
                                                      bidirectional=True,
                                                      batch_size=512,
                                                      l2_strength=0.001,
                                                      hidden_dim=128)    
    final_featurizer = partial(get_seq_feats, 
                               how='left15-men1-midl-men2-rght15', two_dir=False, max_words=100)
    ### now we prepare the dataset
    # first the train
    train_dataset = splits['train']
    train_o, train_y = train_dataset.build_dataset()
    P("Featurizing train dataset...")
    train_X, vectorizer = train_dataset.featurize(
        train_o, [final_featurizer], vectorize=False)
    # now train_X, train_y holds the train data
    
    # now the dev
    assess_dataset = splits['dev']
    assess_o, assess_y = assess_dataset.build_dataset()
    P("Featurizing dev dataset...")
    featurizers=[final_featurizer]
    test_X, _ = assess_dataset.featurize(
        assess_o,
        featurizers=featurizers,
        vectorizer=None,
        vectorize=False)
    # now test_X and assess_y holds the dev data
    
    # lets train all the models
    start_timer('final_model')
    models = {}
    early_stops = {}
    n_rels = len(splits['all'].kb.all_relations)
    P("Training {} {} classifiers on {} relations".format(
        n_rels, final_model_factory().__class__.__name__, n_rels))
    for i_rel, rel in enumerate(splits['all'].kb.all_relations):
      models[rel] = final_model_factory()
      P("Training {}/{}: Running {}.fit() for rel={} for max {} epochs with early stop...".format(
              i_rel + 1, n_rels, models[rel].__class__.__name__, rel, max_epochs))
      best_rel_f1 = 0
      best_rel_model = ''
      patience = 0
      max_patience = 10
      for ep in range(1, max_epochs + 1):
        models[rel].fit(train_X[rel], train_y[rel])
        # finished fit stage now lets evaluate
        predictions =  models[rel].predict(test_X[rel], verbose=False)
        stats = precision_recall_fscore_support(assess_y[rel], predictions, beta=0.5)
        stats = [stat[1] for stat in stats]     
        rel_f1 = stats[2]
        if best_rel_f1 < rel_f1:
          patience = 0
          if best_rel_model != '':
            try:
              os.remove(best_rel_model)
              P("  Old model file '{}' removed".format(best_rel_model))
            except:
              P("FAILED to remove old model file")
          best_rel_f1 = rel_f1
          best_rel_model = 'model_rel_{}_{}.th'.format(rel, ep)
          th.save(models[rel].model.state_dict(), best_rel_model)
          P("  Found new best for rel={} with f05={:.4f} @ ep {}".format(
              rel, best_rel_f1, ep * epochs_per_fit))
          early_stops[rel] = ep * epochs_per_fit
        else:
          patience += 1
          P("  Model did not improve {:.4f} < {:.4f}. Patience {}/{}".format(
              rel_f1, best_rel_f1, patience, max_patience))
        if patience >= max_patience:
          P("  Stopping trainn for rel '{}' after {} epochs".format(
              rel, ep * epochs_per_fit))
          break
      if best_rel_model != '':
        P("  Loading best model '{}' for rel='{}'".format(best_rel_model, rel))
        models[rel].model.load_state_dict(th.load(best_rel_model))
        predictions =  models[rel].predict(test_X[rel], verbose=False)
        stats = precision_recall_fscore_support(assess_y[rel], predictions, beta=0.5)
        stats = [stat[1] for stat in stats]     
        rel_f1 = stats[2]   
        P("  Final model for rel '{}' has a F0.5 of {:.4f}".format(rel,
          rel_f1))
        assert rel_f1 == best_rel_f1, "Results can not be replicated {} vs {} ".format(best_rel_f1, rel_f1)
        try:
          new_fn = "best_rel_ext_{}_F05_{:.4f}_ep_{:03}.th".format(
              rel, rel_f1,  early_stops[rel])
          os.rename(best_rel_model, new_fn)
          P("Model '{}' renamed to '{}'".format(best_rel_model, new_fn))
        except:
          P("  Model could not be renamed")
    # now we have trained one model for each realtion with independent early stopping                        
    train_result = {
        'featurizers': featurizers,
        'vectorizer': vectorizer,
        'models': models,
        'all_relations': splits['all'].kb.all_relations,
        'vectorize': False}
    predictions, test_y = rel_ext.predict(
        splits,
        train_result,
        split_name='dev',
        vectorize=False)
    eval_res = rel_ext.evaluate_predictions(
                  predictions,
                  test_y,
                  verbose=True)
    end_timer('final_model', eval_res)
    
    

  print_results()