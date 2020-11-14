
from sklearn import datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import pyLDAvis
import pyLDAvis.sklearn
import logging
import pickle


### CV viz
with open('/Users/Noah/Github_repos/Project_4/lda_cv', 'rb') as file:
    lda_cv = pickle.load(file)

with open('/Users/Noah/Github_repos/Project_4/text_cv', 'rb') as file:
    text_cv = pickle.load(file)

with open('/Users/Noah/Github_repos/Project_4/cv', 'rb') as file:
    cv = pickle.load(file)

# logging (set to INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

varname = pyLDAvis.sklearn.prepare(lda_cv, text_cv, cv)
pyLDAvis.show(varname)




"""

### TFIDF viz
with open('/Users/Noah/Github_repos/Project_4/lda_tfidf', 'rb') as file:
    lda_tfidf = pickle.load(file)

with open('/Users/Noah/Github_repos/Project_4/text_tfidf2', 'rb') as file:
    text_tfidf2 = pickle.load(file)

with open('/Users/Noah/Github_repos/Project_4/tfidf2', 'rb') as file:
    tfidf2 = pickle.load(file)

# logging (set to INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

varname = pyLDAvis.sklearn.prepare(lda_tfidf, text_tfidf2, tfidf2)
pyLDAvis.show(varname)
"""
