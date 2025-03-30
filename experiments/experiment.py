from sklearn.linear_model import LogisticRegressionCV
import time
import logging
import os
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import scipy as sp


data_filepath = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/FastText/data_lemmatize_stopwords_spellcheckFalse.npz'
labels = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).label.values.tolist()

features = sp.sparse.load_npz(data_filepath)

X = features
y = labels
print('data loaded, beginning classification')
clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=2000, verbose=True, solver='saga', n_jobs=23).fit(X,y)
clf.predict(X)
clf.score(X,y)
print('the end')

#KFOLD-5
#FIT-TEST-STATS
#CM of best performing fold.
#log it
#time it
#track memory