import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import utils

from nltk.corpus import wordnet as wn
from collections import defaultdict
import os

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Fall 2020"


class Retrofitter(object):
    """
    Implements the baseline retrofitting method of Faruqui et al.

    Parameters
    ----------
    max_iter : int indicating the maximum number of iterations to run.

    alpha : func from `edges.keys()` to floats or None

    beta : func from `edges.keys()` to floats or None

    tol : float
        If the average distance change between two rounds is at or
        below this value, we stop. Default to 10^-2 as suggested
        in the paper.

    verbose : bool
        Whether to print information about the optimization process.

    introspecting : bool
        Whether to accumulate a list of the retrofitting matrices
        at each step. This should be set to `True` only for small
        illustrative tasks. For large ones, it will impose huge
        memory demands.

    """
    def __init__(self, max_iter=100, alpha=None, beta=None, tol=1e-2,
            verbose=False, introspecting=False):
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.verbose = verbose
        self.introspecting = introspecting

    def fit(self, X, edges):
        """
        The core internal retrofitting method.

        Parameters
        ----------
        X : np.array (distributional embeddings)

        edges : dict
            Mapping indices into `X` into sets of indices into `X`.

        Attributes
        ----------
        self.Y : np.array, same dimensions and arrangement as `X`.
           The retrofitting matrix.

        self.all_Y : list
           Set only if `self.introspecting=True`.

        Returns
        -------
        self

        """
        index = None
        columns = None
        if isinstance(X, pd.DataFrame):
            index = X.index
            columns = X.columns
            X = X.values

        if self.alpha is None:
            self.alpha = lambda x: 1.0
        if self.beta is None:
            self.beta = lambda x: 1.0 / len(edges[x])

        if self.introspecting:
            self.all_Y = []

        Y = X.copy()
        Y_prev = Y.copy()
        for iteration in range(1, self.max_iter+1):
            for i, vec in enumerate(X):
                neighbors = edges[i]
                n_neighbors = len(neighbors)
                if n_neighbors:
                    a = self.alpha(i)
                    b = self.beta(i)
                    retro = np.array([b * Y[j] for j in neighbors])
                    retro = retro.sum(axis=0) + (a * X[i])
                    norm = np.array([b for j in neighbors])
                    norm = norm.sum(axis=0) + a
                    Y[i] = retro / norm
            changes = self._measure_changes(Y, Y_prev)
            if changes <= self.tol:
                self._progress_bar(
                    "Converged at iteration {}; change was {:.4f} ".format(
                        iteration, changes))
                break
            else:
                if self.introspecting:
                    self.all_Y.append(Y.copy())
                Y_prev = Y.copy()
                self._progress_bar(
                    "Iteration {:d}; change was {:.4f}".format(
                        iteration, changes))
        if index is not None:
            Y = pd.DataFrame(Y, index=index, columns=columns)
        self.Y = Y
        return self.Y

    @staticmethod
    def _measure_changes(Y, Y_prev):
        return np.abs(
            np.mean(
                np.linalg.norm(
                    np.squeeze(Y_prev) - np.squeeze(Y),
                    ord=2)))

    def _progress_bar(self, msg):
        if self.verbose:
            utils.progress_bar(msg)



def plot_retro_vsm(Q, edges, ax=None, lims=None):
    ax = Q.plot.scatter(x=0, y=1, ax=ax)
    if lims is not None:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    _ = Q.apply(lambda x: ax.text(x[0], x[1], x.name, fontsize=18), axis=1)
    for i, vals in edges.items():
        for j in vals:
            x0, y0 = Q.iloc[i].values
            x1, y1 = (Q.iloc[j] - Q.iloc[i]) * 0.9
            ax.arrow(x0, y0, x1, y1, head_width=0.05, head_length=0.05)
    return ax


def plot_retro_path(Q_hat, edges, retrofitter=None):
    if retrofitter is None:
        retrofitter = Retrofitter(introspecting=True)
    retrofitter.introspecting = True
    retrofitter.fit(Q_hat, edges)
    all_Y = retrofitter.all_Y
    lims = [Q_hat.values.min()-0.1, Q_hat.values.max()+0.1]
    n_steps = len(all_Y)
    fig, axes = plt.subplots(nrows=1, ncols=n_steps+1, figsize=(12, 4), squeeze=False)
    plot_retro_vsm(Q_hat, edges, axes[0][0], lims=lims)
    for Q, ax in zip(all_Y, axes[0][1: ]):
        Q = pd.DataFrame(Q, index=Q_hat.index, columns=Q_hat.columns)
        ax = plot_retro_vsm(Q, edges, ax=ax, lims=lims)
    plt.tight_layout()
    return retrofitter
  
def get_wordnet_edges(wn):
  edges = defaultdict(set)
  for ss in wn.all_synsets():
      lem_names = {lem.name() for lem in ss.lemmas()}
      for lem in lem_names:
          edges[lem] |= lem_names
  return edges  

def convert_edges_to_indices(edges, Q):
  lookup = dict(zip(Q.index, range(Q.shape[0])))
  index_edges = defaultdict(set)
  for start, finish_nodes in edges.items():
      s = lookup.get(start)
      if s:
          f = {lookup[n] for n in finish_nodes if n in lookup}
          if f:
              index_edges[s] = f
  return index_edges  

def retrofit(X):
  wn_edges = get_wordnet_edges(wn)
  wn_index_edges = convert_edges_to_indices(wn_edges, X)
  wn_retro = Retrofitter(verbose=True)  
  X_retro = wn_retro.fit(X, wn_index_edges)  
  return X_retro
  
if __name__ == '__main__':
  
  data_home = 'data'
  
  glove_dict = utils.glove2dict(
    os.path.join(data_home, 'glove.6B', 'glove.6B.300d.txt'))
  X_glove = pd.DataFrame(glove_dict).T  

  lems = wn.lemmas('crane', pos=None)  

  for lem in lems:
    ss = lem.synset()
    print("="*70)
    print("Lemma name: {}".format(lem.name()))
    print("Lemma Synset: {}".format(ss))
    print("Synset definition: {}".format(ss.definition()))   
    

  wn_edges = get_wordnet_edges(wn)
    
  wn_index_edges = convert_edges_to_indices(wn_edges, X_glove)
  
  wn_retro = Retrofitter(verbose=True)  
  X_retro = wn_retro.fit(X_glove, wn_index_edges)  
  X_retro.to_csv(os.path.join(data_home, 'glove6B300d-retrofit-wn.csv.gz'), compression='gzip')  
  
  # lets test some neighnors on X_glove and on X_retro
  
