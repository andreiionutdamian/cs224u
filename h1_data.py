# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 17:02:15 2020

@author: Andrei
"""

import pandas as pd
import os


DATA_HOME = os.path.join('data', 'vsmdata')

imdb5 = pd.read_csv(os.path.join(DATA_HOME, 'imdb_window5-scaled.csv.gz'), index_col=0)
