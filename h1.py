# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 07:47:06 2020

@author: Andrei
"""


from collections import defaultdict
import csv
import itertools
import numpy as np
import os
import pandas as pd
from scipy.stats import spearmanr
import vsm
from IPython.display import display

from retrofitting import Retrofitter, retrofit

from torch_autoencoder import TorchAutoencoder

import torch as th

class L2_Normalizer(th.nn.Module):
  def __init__(self,):
    super().__init__()
    
  def forward(self, inputs):
    return th.nn.functional.normalize(inputs, p=2, dim=1)


class TorchDeepAutoencoder(TorchAutoencoder):
    def define_graph(self):
      self.out_layer = 5
      return th.nn.Sequential(
          th.nn.Dropout(0.1),
          th.nn.Linear(self.input_dim_, self.hidden_dim),
          th.nn.Tanh(),
          th.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
          L2_Normalizer(),
          th.nn.Linear(self.hidden_dim // 2, self.hidden_dim),
          th.nn.Tanh(),
          th.nn.Linear(self.hidden_dim, self.output_dim_)
          )
        
    def get_repr(self, X_tensor, X_df):
      with th.no_grad():
        H = X_tensor
        self.model.to('cpu')
        for i in range(self.out_layer):
          H = self.model[i](H)
        print("Out layer: {}".format(self.model[i]))
        return self.convert_output(H, X_df)
  


VSM_HOME = os.path.join('data', 'vsmdata')

WORDSIM_HOME = os.path.join('data', 'wordsim')

def wordsim_dataset_reader(
        src_filename, 
        header=False, 
        delimiter=',', 
        score_col_index=2):
    """Basic reader that works for all similarity datasets. They are 
    all tabular-style releases where the first two columns give the 
    word and a later column (`score_col_index`) gives the score.

    Parameters
    ----------
    src_filename : str
        Full path to the source file.
    header : bool
        Whether `src_filename` has a header. Default: False
    delimiter : str
        Field delimiter in `src_filename`. Default: ','
    score_col_index : int
        Column containing the similarity scores Default: 2

    Yields
    ------
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity
       score in the file so that we are intuitively aligned with our
       distance-based code. To align with our VSMs, all the words are 
       downcased.

    """
    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if header:
            next(reader)
        for row in reader:
            w1 = row[0].strip().lower()
            w2 = row[1].strip().lower()
            score = row[score_col_index]
            # Negative of scores to align intuitively with distance functions:
            score = -float(score)
            yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'wordsim353', 'combined.csv')
    return wordsim_dataset_reader(
        src_filename, header=True)

def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'MTURK-771.csv')
    return wordsim_dataset_reader(
        src_filename, header=False)

def simverb3500dev_reader():
    """SimVerb-3500: http://people.ds.cam.ac.uk/dsg40/simverb.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'SimVerb-3500', 'SimVerb-500-dev.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=True, score_col_index=3)

def simverb3500test_reader():
    """SimVerb-3500: http://people.ds.cam.ac.uk/dsg40/simverb.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'SimVerb-3500', 'SimVerb-3000-test.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=True, score_col_index=3)

def men_reader():
    """MEN: http://clic.cimec.unitn.it/~elia.bruni/MEN"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'MEN', 'MEN_dataset_natural_form_full')
    return wordsim_dataset_reader(
        src_filename, header=False, delimiter=' ') 
    
    


def get_reader_name(reader):
    """Return a cleaned-up name for the similarity dataset 
    iterator `reader`
    """
    return reader.__name__.replace("_reader", "")
  
  
def get_reader_vocab(reader):
    """Return the set of words (str) in `reader`."""
    vocab = set()
    for w1, w2, _ in reader():
        vocab.add(w1)
        vocab.add(w2)
    return vocab  
  
READERS = (wordsim353_reader, mturk771_reader, simverb3500dev_reader, 
         simverb3500test_reader, men_reader)    
  

def mturk287_reader():
    """MTurk-287: http://tx.technion.ac.il/~kirar/Datasets.html"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'bakeoff-wordsim-test-data', 'MTurk-287.csv')
    return wordsim_dataset_reader(
        src_filename, header=False)

def simlex999_reader(wordsim_test_home=WORDSIM_HOME):
    """SimLex999: https://www.cl.cam.ac.uk/~fh295/SimLex-999.zip"""
    src_filename = os.path.join(
        WORDSIM_HOME, 'bakeoff-wordsim-test-data', 'SimLex-999', 'SimLex-999.txt')
    return wordsim_dataset_reader(
        src_filename, delimiter="\t", header=True, score_col_index=3)

BAKEOFF = (simlex999_reader, mturk287_reader)


  
def get_reader_vocab_overlap(readers=READERS):
    """Get data on the vocab-level relationships between pairs of 
    readers. Returns a a pd.DataFrame containing this information.
    """
    data = []
    for r1, r2 in itertools.product(readers, repeat=2):       
        v1 = get_reader_vocab(r1)
        v2 = get_reader_vocab(r2)
        d = {
            'd1': get_reader_name(r1),
            'd2': get_reader_name(r2),
            'overlap': len(v1 & v2), 
            'union': len(v1 | v2),
            'd1_size': len(v1),
            'd2_size': len(v2)}
        data.append(d)
    return pd.DataFrame(data)
  
def vocab_overlap_crosstab(vocab_overlap):
    """Return an intuitively formatted `pd.DataFrame` giving 
    vocab-overlap counts for all the datasets represented in 
    `vocab_overlap`, the output of `get_reader_vocab_overlap`.
    """        
    xtab = pd.crosstab(
        vocab_overlap['d1'], 
        vocab_overlap['d2'], 
        values=vocab_overlap['overlap'], 
        aggfunc=np.mean)
    # Blank out the upper right to reduce visual clutter:
    for i in range(0, xtab.shape[0]):
        for j in range(i+1, xtab.shape[1]):
            xtab.iloc[i, j] = ''        
    return xtab        

def get_reader_pairs(reader):
    """Return the set of alphabetically-sorted word (str) tuples 
    in `reader`
    """
    return {tuple(sorted([w1, w2])): score for w1, w2, score in reader()}  
  
def get_reader_pair_overlap(readers=READERS):
    """Return a `pd.DataFrame` giving the number of overlapping 
    word-pairs in pairs of readers, along with the Spearman 
    correlations.
    """    
    data = []
    for r1, r2 in itertools.product(READERS, repeat=2):
        if r1.__name__ != r2.__name__:
            d1 = get_reader_pairs(r1)
            d2 = get_reader_pairs(r2)
            overlap = []
            for p, s in d1.items():
                if p in d2:
                    overlap.append([s, d2[p]])
            if overlap:
                s1, s2 = zip(*overlap)
                rho = spearmanr(s1, s2)[0]
            else:
                rho = None
            # Canonical order for the pair:
            n1, n2 = sorted([get_reader_name(r1), get_reader_name(r2)])
            d = {
                'd1': n1,
                'd2': n2,
                'pair_overlap': len(overlap),
                'rho': rho}
            data.append(d)
    df = pd.DataFrame(data)
    df = df.sort_values(['pair_overlap','d1','d2'], ascending=False)
    # Return only every other row to avoid repeats:
    return df[::2].reset_index(drop=True)  


def word_similarity_evaluation(reader, df, distfunc=vsm.cosine):
    """Word-similarity evalution framework.
    
    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).    
    df : pd.DataFrame
        The VSM being evaluated.        
    distfunc : function mapping vector pairs to floats.
        The measure of distance between vectors. Can also be 
        `vsm.euclidean`, `vsm.matching`, `vsm.jaccard`, as well as 
        any other float-valued function on pairs of vectors.    
        
    Raises
    ------
    ValueError
        If `df.index` is not a subset of the words in `reader`.
    
    Returns
    -------
    float, data
        `float` is the Spearman rank correlation coefficient between 
        the dataset scores and the similarity values obtained from 
        `df` using  `distfunc`. This evaluation is sensitive only to 
        rankings, not to absolute values.  `data` is a `pd.DataFrame` 
        with columns['word1', 'word2', 'score', 'distance'].
        
    """
    data = []
    for w1, w2, score in reader():
        d = {'word1': w1, 'word2': w2, 'score': score}
        for w in [w1, w2]:
            if w not in df.index:
                raise ValueError(
                    "Word '{}' is in the similarity dataset {} but not in the "
                    "DataFrame, making this evaluation ill-defined. Please "
                    "switch to a DataFrame with an appropriate vocabulary.".
                    format(w, get_reader_name(reader))) 
        d['distance'] = distfunc(df.loc[w1], df.loc[w2])
        data.append(d)
    data = pd.DataFrame(data)
    rho, pvalue = spearmanr(data['score'].values, data['distance'].values)
    return rho, data
  
def word_similarity_error_analysis(eval_df):    
    eval_df['distance_rank'] = _normalized_ranking(eval_df['distance'])
    eval_df['score_rank'] = _normalized_ranking(eval_df['score'])
    eval_df['error'] =  abs(eval_df['distance_rank'] - eval_df['score_rank'])
    return eval_df.sort_values('error')
    
    
def _normalized_ranking(series):
    ranks = series.rank(method='dense')
    return ranks / ranks.sum()      
  
def full_word_similarity_evaluation(df, readers=READERS, distfunc=vsm.cosine, verbose=True):
    """Evaluate a VSM against all datasets in `readers`.
    
    Parameters
    ----------
    df : pd.DataFrame
    readers : tuple 
        The similarity dataset readers on which to evaluate.
    distfunc : function mapping vector pairs to floats.
        The measure of distance between vectors. Can also be 
        `vsm.euclidean`, `vsm.matching`, `vsm.jaccard`, as well as 
        any other float-valued function on pairs of vectors.    
    
    Returns
    -------
    pd.Series
        Mapping dataset names to Spearman r values.
        
    """        
    print("* Computing evaluation with {} *".format(distfunc.__name__), flush=True)
    scores = {}     
    for reader in readers:
        score, data_df = word_similarity_evaluation(reader, df, distfunc=distfunc)
        scores[get_reader_name(reader)] = score
    series = pd.Series(scores, name='Spearman r')
    series['Macro-average'] = series.mean()
    if verbose:
      for key in dict(series):
        print("  {:<16} {:.4f}".format(key+':', series[key]))
    return series  
  

  
def test_run_giga_ppmi_baseline(run_giga_ppmi_baseline):
    result = run_giga_ppmi_baseline()
    ws_result = result.loc['wordsim353'].round(2)
    ws_expected = 0.58
    assert ws_result == ws_expected, \
        "Expected wordsim353 value of {}; got {}".format(ws_expected, ws_result)  
  
def run_giga_ppmi_baseline():    
    giga20 = pd.read_csv(os.path.join(VSM_HOME, 'giga_window20-flat.csv.gz'), index_col=0)
    giga20_pmi = vsm.pmi(giga20, positive=True)
    giga20_pmid1p = pmid(giga20, positive=True, delta_on_pmi=True)
    giga20_pmid1n = pmid(giga20, positive=False, delta_on_pmi=True)
    giga20_pmid2p = pmid(giga20, positive=True, delta_on_pmi=False)
    giga20_pmid2n = pmid(giga20, positive=False, delta_on_pmi=False)
    res1 = full_word_similarity_evaluation(giga20_pmi)
    res2 = full_word_similarity_evaluation(giga20_pmid1p)
    res3 = full_word_similarity_evaluation(giga20_pmid1n)
    res4 = full_word_similarity_evaluation(giga20_pmid2p)
    res5 = full_word_similarity_evaluation(giga20_pmid2n)
    print("\ngiga20 with pmip\n", res1)
    print("\ngiga20 with pmid1p \n", res2)
    print("\ngiga20 with pmid1n \n", res3)
    print("\ngiga20 with pmid2p \n", res4)
    print("\ngiga20 with pmid2n \n", res5)
    return res1  

def run_giga5_ppmi_baseline():    
    giga20 = pd.read_csv(os.path.join(VSM_HOME, 'giga_window5-scaled.csv.gz'), index_col=0)
    giga20_pmi = vsm.pmi(giga20, positive=True)
    giga20_pmid1p = pmid(giga20, positive=True, delta_on_pmi=True)
    giga20_pmid1n = pmid(giga20, positive=False, delta_on_pmi=True)
    giga20_pmid2p = pmid(giga20, positive=True, delta_on_pmi=False)
    giga20_pmid2n = pmid(giga20, positive=False, delta_on_pmi=False)
    res1 = full_word_similarity_evaluation(giga20_pmi)
    res2 = full_word_similarity_evaluation(giga20_pmid1p)
    res3 = full_word_similarity_evaluation(giga20_pmid1n)
    res4 = full_word_similarity_evaluation(giga20_pmid2p)
    res5 = full_word_similarity_evaluation(giga20_pmid2n)
    print("\ngiga5 with pmip\n", res1)
    print("\ngiga5 with pmip delta-pmi \n", res2)
    print("\ngiga5 with pmin delta-pmi \n", res3)
    print("\ngiga5 with pmip delta-df\n", res4)
    print("\ngiga5 with pmin delta-df\n", res5)
    return res1  

  
def run_ppmi_lsa_pipeline(count_df, k):
    pmip_df = vsm.pmi(count_df, positive=True)
    #pmin_df = vsm.pmi(count_df, positive=False)
    pmid1_df = pmid(count_df, positive=True) 
    #pmid2_df = pmid(count_df, positive=False) 
    lsa1_df = vsm.lsa(pmip_df, k=k)
    #lsa2_df = vsm.lsa(pmin_df, k=k)
    lsa3_df = vsm.lsa(pmid1_df, k=k)
    #lsa4_df = vsm.lsa(pmid2_df, k=k)
    res1 = full_word_similarity_evaluation(lsa1_df)
    #res2 = full_word_similarity_evaluation(lsa2_df)
    res3 = full_word_similarity_evaluation(lsa3_df)
    #res4 = full_word_similarity_evaluation(lsa4_df)
    print("\nLSA {} w PMI pos\n{}".format(k, res1))
    #print("\nLSA w PMI neg\n", res2)
    print("\nLSA {} w PMID pos\n{}".format(k, res3))
    #print("\nLSA w PMID neg\n", res4)
    return res1
    
  

def test_run_ppmi_lsa_pipeline(run_ppmi_lsa_pipeline):
    giga20 = pd.read_csv(
        os.path.join(VSM_HOME, "giga_window20-flat.csv.gz"), index_col=0)
    for k in [10, 50, 100, 200]:
      res = run_ppmi_lsa_pipeline(giga20, k=k)
      if k==10:
        results = res
    men_expected = 0.57
    men_result = results.loc['men'].round(2)
    assert men_result == men_expected,\
        "Expected men value of {}; got {}".format(men_expected, men_result)  

  
def run_small_glove_evals():
  from mittens import GloVe
  all_res = {}
  giga20 = pd.read_csv(
      os.path.join(VSM_HOME, "giga_window20-flat.csv.gz"), index_col=0)
  for max_iter in [10, 100, 200]:
    glove_model = GloVe(max_iter=max_iter)
    np_giga20 = glove_model.fit(giga20.values)
    glove_model.sess.close()
    giga20_glove = pd.DataFrame(np_giga20, index=giga20.index)
    res = full_word_similarity_evaluation(giga20_glove)
    all_res[max_iter] = res.loc['Macro-average'].round(2)    
  return all_res
    
  
def test_run_small_glove_evals(run_small_glove_evals):
    data = run_small_glove_evals()
    for max_iter in (10, 100, 200):
        assert max_iter in data
        assert isinstance(data[max_iter], float)  
        

def dice(u, v):
    n = 2 * np.minimum(u,v).sum()
    d = (u+v).sum()
    return 1 - n/d
  
def ttest(df):
    X = df.values
    P_X_i_j = X / X.sum()
    col = (X.sum(axis=1) / X.sum()).reshape(-1,1)
    row = X.sum(axis=0) / X.sum()
    P_X_i_s = np.hstack([col for _ in range(X.shape[1])])
    P_X_j_s = np.vstack([row for _ in range(X.shape[0])])
    d = np.sqrt(P_X_i_s * P_X_j_s)
    n = P_X_i_j - (P_X_i_s * P_X_j_s)    
    res = pd.DataFrame( n / d, index=df.index, columns=df.columns)
    return res
    
    ##### YOUR CODE HERE  
        
    
def test_ttest_implementation(func):
    """`func` should be an implementation of t-test reweighting as 
    defined above.
    """
    X = pd.DataFrame(np.array([
        [  4.,   4.,   2.,   0.],
        [  4.,  61.,   8.,  18.],
        [  2.,   8.,  10.,   0.],
        [  0.,  18.,   0.,   5.]]))    
    actual = np.array([
        [ 0.33056, -0.07689,  0.04321, -0.10532],
        [-0.07689,  0.03839, -0.10874,  0.07574],
        [ 0.04321, -0.10874,  0.36111, -0.14894],
        [-0.10532,  0.07574, -0.14894,  0.05767]])    
    predicted = func(X)
    assert np.array_equal(predicted.round(5), actual)
    
def subword_enrichment(df, n=4):
    # 1. Use `vsm.ngram_vsm` to create a character-level 
    # VSM from `df`, using the above parameter `n` to 
    # set the size of the ngrams.
    
    print("Generating ngram={} MCO".format(n), flush=True)
    df_ngram = vsm.ngram_vsm(df, n=n)

        
    # 2. Use `vsm.character_level_rep` to get the representation
    # for every word in `df` according to the character-level
    # VSM you created above.
    print("  Summarising ngrams reps to the each word", flush=True)    
    df_new_vsm = df.apply(func=lambda x: pd.Series(vsm.character_level_rep(x.name, cf=df_ngram, n=n),
                                                   index=df.columns), 
                          axis=1)
    
    # 3. For each representation created at step 2, add in its
    # original representation from `df`. (This should use
    # element-wise addition; the dimensionality of the vectors
    # will be unchanged.)
    print("  Combinind final MCO", flush=True)
    df_final_vsm = df + df_new_vsm

    
    # 4. Return a `pd.DataFrame` with the same index and column
    # values as `df`, but filled with the new representations
    # created at step 3.
                            
    return df_final_vsm
      
    
def test_subword_enrichment(func):
    """`func` should be an implementation of subword_enrichment as 
    defined above.
    """
    vocab = ["ABCD", "BCDA", "CDAB", "DABC"]
    df = pd.DataFrame([
        [1, 1, 2, 1],
        [3, 4, 2, 4],
        [0, 0, 1, 0],
        [1, 0, 0, 0]], index=vocab,columns=vocab)
    expected = pd.DataFrame([
        [14, 14, 18, 14],
        [22, 26, 18, 26],
        [10, 10, 14, 10],
        [14, 10, 10, 10]], index=vocab, columns=vocab)
    new_df = func(df, n=2)
    assert np.array_equal(expected.columns, new_df.columns), \
        "Columns are not the same"
    assert np.array_equal(expected.index, new_df.index), \
        "Indices are not the same"
    assert np.array_equal(expected.values, new_df.values), \
        "Co-occurrence values aren't the same"        

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
        


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    
    
    adapted from:
      https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """
    for word in word_vectors:
        word_vectors[word] /= np.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors

def distance(v1, v2, normalised_vectors=True):
    """
    Returns the cosine distance between two vectors. 
    If the vectors are normalised, there is no need for the denominator, which is always one. 

    adapted from:
      https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """
    if normalised_vectors:
        return 1 - np.dot(v1, v2)
    else:
        return 1 - np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2) )


def vector_partial_gradient(u, v, normalised_vectors=True):
    """
    This function returns the gradient of cosine distance: \frac{ \partial dist(u,v)}{ \partial u}
    If they are both of norm 1 (we do full batch and we renormalise at every step), we can save some time.
    
    adapted from:
      https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """

    if normalised_vectors:
        gradient = u * np.dot(u,v)  - v 
    else:       
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        nominator = u * np.dot(u,v) - v * np.power(norm_u, 2)
        denominator = norm_v * np.power(norm_u, 3)
        gradient = nominator / denominator

    return gradient


def one_step_SGD(word_vectors, antonym_pairs,
                 delta=1.0, lr=0.1, gamma=0):
    """
    This method performs a step of SGD to optimise the counterfitting cost function.

    adapted from:
      https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
    """
    from copy import deepcopy
    new_word_vectors = deepcopy(word_vectors)

    gradient_updates = {}
    update_count = {}

    # AR term:
    for i, (word_i, word_j) in enumerate(antonym_pairs):
        print("\r    Processing antonym pairs {:.1f}%...".format(i/len(antonym_pairs)*100), flush=True, end='')

        current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

        if current_distance < delta:
    
            gradient = vector_partial_gradient( new_word_vectors[word_i], new_word_vectors[word_j])
            gradient = gradient * lr 

            if word_i in gradient_updates:
                gradient_updates[word_i] += gradient
                update_count[word_i] += 1
            else:
                gradient_updates[word_i] = gradient
                update_count[word_i] = 1

   
    print("\r    Applying gradients...", flush=True, end='')
    for word in gradient_updates:
        # we've found that scaling the update term for each word helps with convergence speed. 
        update_term = gradient_updates[word] / (update_count[word]) 
        new_word_vectors[word] += update_term 
    print("\r    Done Applying gradients.", flush=True, end='')
        
    return normalise_word_vectors(new_word_vectors)
  
  
def counter_fit_antonyms(dct_word_vectors,  antonyms, epochs=20, lr=0.1):
  """
  This method repeatedly applies SGD steps to counter-fit word vectors to linguistic constraints. 
  
  https://raw.githubusercontent.com/nmrksic/counter-fitting/master/counterfitting.py
  """
  word_vectors = normalise_word_vectors(dct_word_vectors)
  
  current_iteration = 0
  
  
  max_iter = epochs
  print("Antonym pairs:", len(antonyms), flush=True)
  print("Running the optimisation procedure for", max_iter, "SGD steps...", flush=True)
  
  while current_iteration < max_iter:
    current_iteration += 1
    print("\r  Counter-fitting SGD step {}...".format(current_iteration), flush=True, end='')
    word_vectors = one_step_SGD(word_vectors, antonyms, lr=lr)
  print("")
  return word_vectors  

##################################################        
##################################################  
  

def eval_model(dct, dataset_name, model_name, df, distfunc=vsm.cosine):
  print("\nEvaluation of model '{}' {} based on '{}' MCO".format(
        model_name, df.shape, dataset_name),
        flush=True)
  if 'MODEL' not in dct:
    dct['MODEL'] = []
  if 'DATA' not in dct:
    dct['DATA'] = []

  if 'DST' not in dct:
    dct['DST'] = []
  
  dct['DATA'].append(dataset_name)
  dct['MODEL'].append(model_name)
  dct['DST'].append(distfunc.__name__)
  
  res = full_word_similarity_evaluation(df, distfunc=distfunc)
  print("Results for '{}' on data '{}' with distfunc={}:".format(
      model_name, dataset_name, distfunc.__name__), flush=True)
  
  for key in dict(res):
    if key not in dct:
      dct[key] = []
    dct[key].append(res[key])
    
  return dct, res['Macro-average']

def grid_search_vsm(files, dct_model_funcs):
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)
  
  best_model = None
  best_macro = 0
  best_model_name = ''
  dct_res = {}
  dist_funcs = [vsm.cosine] # [dice, vsm.cosine]
  for distfunc in dist_funcs:
    print("Using distfunc={}".format(distfunc.__name__))
    for fn in files:
      print("Loading '{}'".format(fn), flush=True)
      data = pd.read_csv(os.path.join(VSM_HOME, fn), index_col=0)  
      print("  Loaded {}".format(data.shape))
      data_name = fn[:-7]
      for model_name, model_func in dct_model_funcs.items():
        print("=" * 70)
        print("Running '{}' on '{}'".format(model_name, data_name), flush=True)
        df = model_func(data)
        if df is None:
          print("{} returned None".format(model_func.__name__))
          continue
        print("Done running {}. Obtained df: {}".format(
            model_func.__name__, df.shape), flush=True)
        dct_res, macro = eval_model(dct_res, 
                                    dataset_name=data_name, 
                                    model_name=model_name, 
                                    df=df,
                                    distfunc=distfunc
                                    )
        if best_macro < macro:
          old_best_file = best_model_name
          best_macro = macro
          best_model = df
          best_model_name = 'best_model_{:.4f}_'.format(macro).replace('.','') + model_name + '_' + fn
          print("Found new best macro-average: {:.4f}".format(best_macro), flush=True)
          if old_best_file != '':
            try:
              old_best_file = old_best_file + '.csv.gz'
              os.remove(old_best_file)
              print("Old best '{}' deleted.".format(old_best_file))
            except:
              print("ERROR: Cound not remove file '{}' !!!".format(old_best_file))            
          best_model.to_csv(best_model_name, compression='gzip')
          
        df_res = pd.DataFrame(dct_res).sort_values('Macro-average')
        df_res.to_csv("20200303_test.csv")
        print("\nResults so far:\n{}".format(df_res), flush=True)
  
  return best_model


def glove_model(df, n_embeds=75, max_iters=10000, retrofit=False):
  from mittens import GloVe
  print("Computing GloVe model with {} embeds for {} iters".format(
      n_embeds, max_iters), flush=True)
  glove_model = GloVe(n=n_embeds, max_iter=max_iters)
  np_res = glove_model.fit(df.values)
  glove_model.sess.close()
  print("", flush=True)
  df_res = pd.DataFrame(np_res, index=df.index)
  if retrofit:
    df_out = retrofit_model(df_res)
  else:
    df_out = df_res
  return df_out



def calc_delta(mco):
  col_totals = np.array(mco).sum(axis=0)
  row_totals = np.array(mco).sum(axis=1)
  cm = [col_totals for _ in range(mco.shape[0])]
  col_mat = np.vstack(cm)
  row_mat = row_totals.reshape((-1,1))
  rm = [row_mat for _ in range(mco.shape[1])]
  row_mat = np.hstack(rm)
  d1 = mco / (mco + 1)
  mins = np.minimum(col_mat, row_mat)
  d2 = mins / (mins + 1)
  delta = d1 * d2
  return delta

def pmid(m, positive=True, delta_on_pmi=True, before=True):
  df = vsm.observed_over_expected(m)
  # Silence distracting warnings about log(0):
  with np.errstate(divide='ignore'):
    pmi = np.log(df)
  pmi[np.isinf(pmi)] = 0.0  # log(0) = 0
  if positive and before:
      pmi[pmi < 0] = 0.0
  delta = calc_delta(pmi if delta_on_pmi else m)
  pmi = pmi * delta
  if positive and not before:
    pmi[pmi < 0] = 0.0
  return pmi


def counterfit_model(data):
  print("Counter-fitting on {} embeds".format( data.shape), flush=True)


  from nltk.corpus import wordnet as wn
  print("  Preparing antonyms", flush=True)
  ant_set = set()
  for ss in wn.all_synsets():
    lema = ss.lemmas()[0]      
    w1 = lema.name()
    ants = [lem.name() for lem in lema.antonyms()]
    if len(ants)>0:
      for w2 in ants:
        if w1 in data.index and w2 in data.index:
          ant_set.add((w1, w2))
  
    
  dct_word_vectors = {k:v for k,v in zip(data.index, data.values)}
  for word in ['expensive','east','smart','adult']:
    if word in data:
      break
  neibs1 = vsm.neighbors(word, data)
  dct_new_embeds = counter_fit_antonyms(dct_word_vectors, ant_set, lr=0.1, epochs=30)
  df = pd.DataFrame.from_dict(dct_new_embeds, orient='index')
  neibs = vsm.neighbors(word, df)

  print("  Status before counter-fitting for word: {}".format(word), flush=True)
  for w in dict(neibs1.iloc[:5]):
    print("  {:<20} {:.3f}".format(w+':', neibs1[w]))
  print("  Status AFTER counter-fitting for word: {}".format(word), flush=True)
  for w in dict(neibs.iloc[:5]):
    print("  {:<20} {:.3f}".format(w+':', neibs[w]))
  return df
  

def retrofit_model(data, name='', direct=True):
  if direct:
    print("Direct retrofit...", flush=True)
    retro_result = retrofit(data)
  else:
    print("Retrofitting model '{}' with {} embeds".format(name, data.shape[1]), flush=True)
    from nltk.corpus import wordnet as wn
    print("  Constructing edges for words similarity", flush=True)
    edges = defaultdict(set)
    for ss in wn.all_synsets():
      lem_names = {lem.name() for lem in ss.lemmas()}
      for lem in lem_names:
        edges[lem] |= lem_names            
    print("  Preparing indices...",flush=True)
    lookup = dict(zip(data.index, range(data.shape[0])))
    index_edges = defaultdict(set)
    for start, finish_nodes in edges.items():
        s = lookup.get(start)
        if s:
            f = {lookup[n] for n in finish_nodes if n in lookup}
            if f:
                index_edges[s] = f  
    
    wn_retro = Retrofitter(verbose=True,
                           max_iter=1000,
                           tol=1e-4,
                           )
    print("  Running retrofitter ...", flush=True)
    retro_result = wn_retro.fit(data, index_edges)
    print("")
  return retro_result



def lsa_model(data, k=100, use_ttest=False, disc_pmi=True, retrofit=False, delta=True):
  if use_ttest:
    print("Computing ttest reweighting for LSA {}".format(k), flush=True)
    lsa_input = ttest(data)
  else:
    if disc_pmi:
      print("Computing discounted positive PMI reweighting for LSA {} (delta:{})".format(
          k, delta), flush=True)
      lsa_input = pmid(data, delta_on_pmi=delta)
    else:
      print("Computing positive PMI reweighting for LSA {}".format(k), flush=True)
      lsa_input = vsm.pmi(data, positive=True)    
  
  print("Pre-LSA bakeoff")
  bakeoff_eval(lsa_input)
  print("Computing LSA k={}...".format(k))
  lsa_output = vsm.lsa(lsa_input, k=k)
  if retrofit:
    df_out = retrofit_model(lsa_output)
  else:
    df_out = lsa_output
  return df_out

def ae_model(data, n_embeds, 
             distfunc=vsm.cosine,
             epochs=100000, 
             retrofit=True, 
             delta=True, 
             max_patience=5, 
             lr=1e-4,
             lsa_factor=2,
             end_counterfit=False,
             disc_pmi=True,
             ):
  
  
  bakes = []
  print("Generating autoencoder based model with {} embeds...".format(n_embeds), flush=True)
  lsa_output = lsa_model(data, k=int(n_embeds * lsa_factor), disc_pmi=disc_pmi, use_ttest=False, delta=delta)
  print("Post-LSA bake-off eval...")
  bakeoff_eval(lsa_output)
  n_step_epochs = 10
  steps = epochs // n_step_epochs
    
  ae_model = TorchAutoencoder(max_iter=n_step_epochs, 
                              hidden_dim=n_embeds, 
                              eta=lr,
                              warm_start=True)
  best_macro = 0
  best_model = None
  patience = 0
  print("Performing autoencoder training for {} steps of {} epochs with early stopping on {}...".format(
      steps, n_step_epochs, lsa_output.shape), flush=True)
  for step in range(1, steps+1):
    print("Fitting step {}/{} for {} step-epochs".format(step, steps, n_step_epochs), flush=True)
    ae_output = ae_model.fit(lsa_output)
    print("\nCalculating step {} results...".format(step))
    print("  Before retrofit ...")
    res = full_word_similarity_evaluation(ae_output, distfunc=distfunc)
    bakeoff_eval(ae_output)
    mb = res['Macro-average']
    if retrofit:
      df_out = retrofit_model(ae_output)
    else:
      df_out = ae_output
    res = full_word_similarity_evaluation(df_out, distfunc=distfunc)
    macro = res['Macro-average']
    print("  Before retro: {:.4f}".format(mb))
    print("  After retro:  {:.4f}".format(macro))
    print("  {}".format("Good!" if macro>mb else "WORSE!!!"), flush=True)
    res = round(bakeoff_eval(df_out),4)
    bakes.append(res)
    print(bakes)
    
    if macro > best_macro:
      patience = 0
      best_macro = macro
      best_model = df_out
      print("Found best model at epoch {}".format(step * n_step_epochs), flush=True)
    else:
      patience += 1
      print("Macro {:.4f} < {:.4f} best. Patience {}/{}".format(
          macro, best_macro, patience, max_patience))
      
    if patience >= max_patience:
      print("Early stopping training loop at step {}".format(step))
      break      
  if end_counterfit:
    best_model = counterfit_model(best_model)
  print('\n\n' + '=' * 40)
  print("Max bake-off score {} at iter {}".format(max(bakes), np.argmax(bakes)))
  print('=' * 40 + '\n')
  return best_model

##########################################################
##########################################################
##########################################################
##########################################################
##########################################################
##########################################################

def get_final_model_embeds(distfunc=vsm.cosine):
  imdb5 = pd.read_csv(os.path.join(VSM_HOME, "imdb_window5-scaled.csv.gz"), index_col=0)
  ae_embeds = ae_model(data=imdb5, 
                       n_embeds=200, 
                       epochs=100000, 
                       retrofit=True, 
                       delta=False,
                       max_patience=5,
                       lsa_factor=2)
  res = full_word_similarity_evaluation(ae_embeds, distfunc=distfunc)
  macro = res['Macro-average']
  fn = 'best_model_{:.4f}.csv.gz'.format(macro).replace('.','')
  ae_embeds.to_csv(fn,  compression='gzip')
  return ae_embeds



def comb_1(data, n_embed=100, reduce=True):
  print("Computing comb_1 model", flush=True)
  lsa1 = lsa_model(data, k=n_embed, disc_pmi=False, retrofit=False)
  lsa2 = lsa_model(data, k=n_embed, disc_pmi=True, retrofit=True, delta=True)
  ae = ae_model(data, n_embeds=n_embed, epochs=1500, retrofit=True, delta=False)
  np_embeds = np.concatenate((lsa1.values, lsa2.values, ae.values), axis=1)
  df_res = pd.DataFrame(np_embeds, index=data.index)
  if reduce:
    print("Computing final LSA..", flush=True)
    return vsm.lsa(df_res, k=n_embed + n_embed // 2)
  else:
    return df_res
  

def comb_2(data, n_embed=100, reduce=True):
  print("Computing comb_2 model", flush=True)
  ae = ae_model(data, n_embeds=n_embed, epochs=2000, retrofit=True, delta=False)
  glv = glove_model(data, n_embeds=n_embed, max_iters=2000, retrofit=True)
  
  np_embeds = np.concatenate((glv.values, ae.values), axis=1)
  df_res = pd.DataFrame(np_embeds, index=data.index)
  if reduce:
    print("Computing final LSA..", flush=True)
    return vsm.lsa(df_res, k=n_embed )
  else:
    return df_res
  
def comb_3(data, ngram=3):
  df = data
  df = subword_enrichment(df, n=ngram)
  df = ae_model(df, n_embeds=300, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.01)
  return df

def comb_4(data, ngram=3):
  df = data
  df = ae_model(df, n_embeds=300, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.01)
  df = subword_enrichment(df, n=ngram)
  return df


MODEL_FUNCTIONS = {
#    "RAW" :  lambda x : x,
#
#    "C2_150" : lambda x: comb_2(x, n_embed=150, reduce=False),
#    "C2_150r" : lambda x: comb_2(x, n_embed=150, reduce=True),
#
#    
#    "C1_150" : lambda x: comb_1(x, n_embed=150, reduce=False),
#    "C1_150r" : lambda x: comb_1(x, n_embed=150, reduce=True),

#    "C3_3" : lambda x : comb_3(x, ngram=3),
#    "C4_3" : lambda x : comb_4(x, ngram=3),
#
#    "C3_5" : lambda x : comb_3(x, ngram=5),
#    "C4_5" : lambda x : comb_4(x, ngram=5),
#
#
#
#    "AE200_1Kd1" : lambda x: ae_model(x, n_embeds=200, epochs=2000, delta=True),
#    "AE100_1Kd1" : lambda x: ae_model(x, n_embeds=100, epochs=2000, delta=True),
#    "AE200_1Kd2" : lambda x: ae_model(x, n_embeds=200, epochs=2000, delta=False),
#    "AE100_1Kd2"  : lambda x: ae_model(x, n_embeds=100, epochs=2000, delta=False),

#    "AE200_f2d1_RETR1" : lambda x: ae_model(x, n_embeds=200, epochs=100000, retrofit=True, delta=True, lsa_factor=2, lr=0.01),
#    "AE300_f2d1_RETR1" : lambda x: ae_model(x, n_embeds=300, epochs=100000, retrofit=True, delta=True, lsa_factor=2, lr=0.01),

#    "C_AE200_f2d2_R1" : lambda x: ae_model(x, distfunc=vsm.cosine, n_embeds=200, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.01),
    "C_AE300_f2d2_RC1" : lambda x: ae_model(x, end_counterfit=True, distfunc=vsm.cosine, n_embeds=300, retrofit=True, delta=False, lsa_factor=2, lr=0.01),
    "C_AE300_f2d2_R1"  : lambda x: ae_model(x, end_counterfit=False, distfunc=vsm.cosine, n_embeds=300, retrofit=True, delta=False, lsa_factor=2, lr=0.01),

#    "D_AE200_f2d2_R1" : lambda x: ae_model(x, distfunc=dice, n_embeds=200, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.01),
#    "D_AE300_f2d2_R1" : lambda x: ae_model(x, distfunc=dice, n_embeds=300, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.01),

#
#    "AE200_f2d1_RETR2" : lambda x: ae_model(x, n_embeds=200, epochs=100000, retrofit=True, delta=True, lsa_factor=2, lr=0.001),
#    "AE300_f2d1_RETR2" : lambda x: ae_model(x, n_embeds=300, epochs=100000, retrofit=True, delta=True, lsa_factor=2, lr=0.001),
#    "AE200_f2d2_RETR2" : lambda x: ae_model(x, n_embeds=200, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.001),
#    "AE300_f2d2_RETR2" : lambda x: ae_model(x, n_embeds=300, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.001),
#
#    "AE200_f2d1_RETR3" : lambda x: ae_model(x, n_embeds=200, epochs=100000, retrofit=True, delta=True, lsa_factor=2, lr=0.0001),
#    "AE300_f2d1_RETR3" : lambda x: ae_model(x, n_embeds=300, epochs=100000, retrofit=True, delta=True, lsa_factor=2, lr=0.0001),
#    "AE200_f2d2_RETR3" : lambda x: ae_model(x, n_embeds=200, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.0001),
#    "AE300_f2d2_RETR3" : lambda x: ae_model(x, n_embeds=300, epochs=100000, retrofit=True, delta=False, lsa_factor=2, lr=0.0001),
#
#    "AE100_f2d2_RETR" : lambda x: ae_model(x, n_embeds=100, epochs=10000, retrofit=True, delta=False, lsa_factor=2),

#    "AE200_f3d1_RETR" : lambda x: ae_model(x, n_embeds=200, epochs=10000, retrofit=True, delta=True, lsa_factor=3),
#    "AE100_f3d1_RETR" : lambda x: ae_model(x, n_embeds=100, epochs=10000, retrofit=True, delta=True, lsa_factor=3),
#    "AE200_f3d2_RETR" : lambda x: ae_model(x, n_embeds=200, epochs=100000, retrofit=True, delta=False, lsa_factor=3),
#    "AE100_f3d2_RETR" : lambda x: ae_model(x, n_embeds=100, epochs=10000, retrofit=True, delta=False, lsa_factor=3),


#
#
#    "G75_2K" : lambda x : glove_model(x, n_embeds=75, max_iters=2000),
#    "G200_2K" : lambda x : glove_model(x, n_embeds=200, max_iters=2000),
#    "G75_2K_RETR" : lambda x : glove_model(x, n_embeds=75, max_iters=2000, retrofit=True),
#    "G200_2K_RETR" : lambda x : glove_model(x, n_embeds=200, max_iters=2000, retrofit=True),
#    
#
#
#        
#    "PPMI" : lambda x : vsm.pmi(x, positive=True),
#    
#    "PPMID1" : lambda x : pmid(x, positive=True, delta_on_pmi=True),
#    "PPMID2" : lambda x : pmid(x, positive=True, delta_on_pmi=False),
#
#
#    "LSA200_TTEST" : lambda x : lsa_model(x, k=200, use_ttest=True, retrofit=False),
#
#    "LSA200_TTEST_RETR" : lambda x : lsa_model(x, k=200, use_ttest=True, retrofit=True),
#    
#    "LSA200_PPMI" : lambda x : lsa_model(x, k=200, disc_pmi=False, retrofit=False),
#
#    "LSA200_PPMID1" : lambda x : lsa_model(x, k=200, disc_pmi=True, retrofit=False, delta=True),
#
#    "LSA100_PPMID1" : lambda x : lsa_model(x, k=100, disc_pmi=True, retrofit=False, delta=True),
#
#
#    "LSA200_PPMI_RETR"    : lambda x : lsa_model(x, k=200, disc_pmi=False, retrofit=True),
#
#    "LSA200_PPMID1_RETR"  : lambda x : lsa_model(x, k=200, disc_pmi=True, retrofit=True, delta=True),
    
#    "LSA75_PPMID1_RETR"   : lambda x : lsa_model(x, k=75, disc_pmi=True, retrofit=True, delta=True),
#
#
#    "LSA200_PPMID2"       : lambda x : lsa_model(x, k=200, disc_pmi=True, retrofit=False, delta=False),

#    "LSA75_PPMID2"        : lambda x : lsa_model(x, k=75, disc_pmi=True, retrofit=False, delta=False),
    "LSA200_PPMID2_RETR"  : lambda x : lsa_model(x, k=200, disc_pmi=True, retrofit=True, delta=False),

    }

ALL_FILES = [
    "giga_window5-scaled.csv.gz",
    "giga_window20-flat.csv.gz",
    "imdb_window5-scaled.csv.gz",
    "imdb_window20-flat.csv.gz",
    ]


def bakeoff_eval(title, embds, dct_res):
    distfunc = vsm.jaccard
    print("\n\n{} {} evaluation on the TRAINING DATA".format(
        title, embds.shape))
    trn_res = full_word_similarity_evaluation(embds)

    distfunc = vsm.cosine
    print("\n{} {} results on the TEST sets using distfunc '{}'".format(
        title, embds.shape, distfunc.__name__), flush=True)
    res = full_word_similarity_evaluation(embds, readers=BAKEOFF, distfunc=distfunc, verbose=True)

    distfunc = vsm.jaccard
    print("\n{} {} results on the TEST sets using distfunc '{}'".format(
        title, embds.shape, distfunc.__name__), flush=True)
    res = full_word_similarity_evaluation(embds, readers=BAKEOFF, distfunc=distfunc, verbose=True)
    
    print("----------------------------", flush=True)
    print("Bake-off score: {:.4}".format(res['Macro-average']), flush=True)
    print("----------------------------\n", flush=True)
    dct_res['MODEL'].append(title)
    dct_res['BAKE'].append(res['Macro-average'])
    dct_res['TRAIN'].append(trn_res['Macro-average'])
    return dct_res
  
  
  
if __name__ == '__main__':
  """
  TODO:
    - data
    - pmid baseline
    - lsa 300 on pmid and pmip
    - lsa 300 on ttest
    - glove 300 with 10000 iters
  """  

  def get_sim3500_data_frame():
    datas = [simverb3500dev_reader, simverb3500test_reader]
    df = pd.DataFrame()
    for data_func in datas:
      df = pd.concat([df, pd.DataFrame(data_func(), columns=['w1','w2','score'])])
    return df.sort_values('score')

  
#  df_s = get_sim3500_data_frame()
  
#  vocab_overlap = get_reader_vocab_overlap()
#  print(vocab_overlap_crosstab(vocab_overlap))
#  print(get_reader_pair_overlap())
#  giga5 = pd.read_csv(os.path.join(VSM_HOME, "giga_window5-scaled.csv.gz"), index_col=0)  
#  rho, eval_df = word_similarity_evaluation(men_reader, giga5)  
#  print('rho: {}\n{}'.format(rho, eval_df.head()))
#  
#  print('Best preds\n{}'.format(word_similarity_error_analysis(eval_df).head()))
#  print('Worst preds\n{}'.format(word_similarity_error_analysis(eval_df).tail()))
#  
#  print("\nBasic giga5\n{}".format(full_word_similarity_evaluation(giga5)))
#  print("="*70)
#  test_run_giga_ppmi_baseline(run_giga_ppmi_baseline)  
#  print("="*70)
#  test_run_ppmi_lsa_pipeline(run_ppmi_lsa_pipeline)  
  
#  test_run_small_glove_evals(run_small_glove_evals)  
#  test_ttest_implementation(ttest)  


#  test_subword_enrichment(subword_enrichment)
  
#  run_giga5_ppmi_baseline()
  
#  grid_search_vsm(ALL_FILES, MODEL_FUNCTIONS)
#
#  mco_data = pd.read_csv(os.path.join(VSM_HOME, "giga_window20-flat.csv.gz"), index_col=0)    
#  ae_embeds = ae_model(data=mco_data, 
#                       n_embeds=300, 
#                       epochs=100000, 
#                       retrofit=True, 
#                       delta=False,
#                       max_patience=5,
#                       end_counterfit=True,
#                       lsa_factor=2)
#  full_word_similarity_evaluation(ae_embeds)
#  get_final_model_embeds()
    
#  df = pd.read_csv("models/h1/_h1_best_model_05842.csv.gz", index_col=0)
#  full_word_similarity_evaluation(df)
#  bakeoff_eval(df)
  """
        MODEL      BAKE     TRAIN
6    LSA PPMI -0.041168  0.480857
7  LSA DPPMI1 -0.033820  0.483567
5  LSA Direct -0.006980  0.113213
0         MCO  0.066034  0.159749
1        PPMI  0.451753  0.476300
3      DPPMI2  0.458192  0.480277
4  RET_DPPMI1  0.465135  0.519761
2      DPPMI1  0.488581  0.479645

  """
  results = {'MODEL':[], 'BAKE':[], 'TRAIN':[]}
  if 'mco_data' not in globals():
    mco_data = pd.read_csv(os.path.join(VSM_HOME, "imdb_window5-scaled.csv.gz"), index_col=0)    
  results = bakeoff_eval('MCO', mco_data, results)
  
  pp = vsm.pmi(mco_data, positive=True)
  results = bakeoff_eval('PP', pp, results)

#  pp_tt = ttest(pp)
#  results = bakeoff_eval('PP_TT', pp_tt, results)
#
#  pp_tt_r = retrofit(pp_tt)
#  results = bakeoff_eval('PP_TT_R', pp_tt_r, results)

#  pp_tt_r_l = vsm.lsa(pp_tt_r, k=512)
#  results = bakeoff_eval('PP_TT_R_L', pp_tt_r_l, results)
#
#  pp_tt_r_l_n = pp_tt_r_l.apply(vsm.length_norm, axis=1)
#  results = bakeoff_eval('PP_TT_R_L_N', pp_tt_r_l_n, results)
#
#  ae = TorchAutoencoder(hidden_dim=256, max_iter=100)
#  pp_tt_r_l_n_ae = pd.DataFrame(ae.fit(pp_tt_r_l_n), index=pp_tt_r_l_n.index)
#  results = bakeoff_eval('PP_TT_R_L_N_AE', pp_tt_r_l_n_ae, results)

  ########################################
  
  pp_n = pp.apply(vsm.length_norm, axis=1)
  results = bakeoff_eval('PP_N', pp_n, results)

  pp_n_l = vsm.lsa(pp_n, k=1024)
  results = bakeoff_eval('PP_N_L', pp_n_l, results)

  pp_n_l_r = retrofit(pp_n_l)
  results = bakeoff_eval('PP_N_L_R', pp_n_l_r, results)

  ae = TorchAutoencoder(hidden_dim=392, max_iter=100)
  pp_n_l_r_ae = ae.fit(pp_n_l_r)
  results = bakeoff_eval('PP_N_L_R_AE', pp_n_l_r_ae, results)

  pp_n_l_r_ae_r = retrofit(pp_n_l_r_ae)
  results = bakeoff_eval('PP_N_L_R_AE_R', pp_n_l_r_ae_r, results)

  ########################################

  
#  dp1 = pmid(mco_data, positive=True, delta_on_pmi=False)
#  results = bakeoff_eval('DPP1', dp1, results)
#
#  dp1_n = dp1.apply(vsm.length_norm, axis=1)
#  results = bakeoff_eval('DPP1_N', dp1_n, results)
#
#  dp1_n_l = vsm.lsa(dp1_n, k=1024)
#  results = bakeoff_eval('DPP1_N_L', dp1_n_l, results)
#
#  dp1_n_l_n = dp1_n_l.apply(vsm.length_norm, axis=1)
#  results = bakeoff_eval('DPP1_N_L_N', dp1_n_l_n, results)
#
#  dae = TorchDeepAutoencoder(hidden_dim=512,max_iter=500, eta=0.001)
#  dp1_n_l_n_dae = dae.fit(dp1_n_l_n)
#  results = bakeoff_eval('DPP1_N_L_N_DAE', dp1_n_l_n_dae, results)
#
#  dp1_n_l_n_dae_r = retrofit(dp1_n_l_n_dae)
#  results = bakeoff_eval('DPP1_N_L_N_DAE_R', dp1_n_l_n_dae_r, results)

  ########################################
  
#  dp1_n_l_r = retrofit(dp1_n_l)
#  results = bakeoff_eval('DPP1_N_L_R', dp1_n_l_r, results)
#
#  ae = TorchAutoencoder(hidden_dim=512,max_iter=100)
#  dp1_n_l_r_ae = pd.DataFrame(ae.fit(dp1_n_l_r), index=dp1_n_l_r.index)
#  results = bakeoff_eval('DPP1_N_L_R_AE', dp1_n_l_r_ae, results)
#  
#  dp1_n_l_r_ae_r = retrofit(dp1_n_l_r_ae)
#  results = bakeoff_eval('DPP1_N_L_R_AE_R', dp1_n_l_r_ae_r, results)
#  
#  ########################################
#  
#  ae = TorchAutoencoder(hidden_dim=1500,max_iter=200, eta=0.00005, l2_strength=0.0001)
#  pp_ae = pd.DataFrame(ae.fit(pp), index=pp.index)
#  results = bakeoff_eval('PP_AE', pp_ae, results)
#  
#  pp_ae_r = retrofit(pp_ae)
#  results = bakeoff_eval('PP_AE_R', pp_ae_r, results)

  

  
  df_res = pd.DataFrame(results).sort_values('BAKE')
  print(df_res)
  
  
  df_orig = pd.read_csv("models/h1/_h1_best_model_05842.csv.gz", index_col=0)
  results = bakeoff_eval('ORIG', df_orig, results)