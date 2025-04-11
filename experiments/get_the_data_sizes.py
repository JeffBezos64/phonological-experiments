import os
import pickle
import scipy as sp
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
language2label = {
    'English': 0,
    'German': 1,
    'Bulgarian': 2,
    'Croatian': 3,
    'Czech': 4,
    'Estonian': 5,
    'Finnish': 6,
    'French': 7,
    'Greek': 8,
    'Hungarian': 9,
    'Italian': 10,
    'Lithuanian': 11,
    'Dutch': 12,
    'Norwegian': 13,
    'Polish': 14,
    'Portuguese': 15,
    'Romanian': 16,
    'Russian': 17,
    'Serbian': 18,
    'Slovenian': 19,
    'Spanish': 20,
    'Swedish': 21,
    'Turkish': 22
    }
label2language = {v: k for k, v in language2label.items()}
BASE_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'
FEATURE_TYPES = ['Parrish', 'FastText', 'Zouhar', 'Glove', 'Sharma', 'tfidf']
VALUES = ['True', 'False']
FILENAMES = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
LABELS = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).label.values.tolist()
y = LABELS
data_filepath = BASE_DATA_DIR+'Parrish'+'/'+'data_tokenize'+'True'+'.npz'
X = sp.sparse.load_npz(data_filepath)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Z = skf.split(X,y)
Q = [z for z in Z]
counterd = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0
}
for label in Q[0][0]:
    counterd[y[label]] += 1
for key in counterd.keys():
    print(f'The total for in domain training split is {label2language[key]} is {counterd[key]}')
print(f'Length of in domain training split{len(Q[0][0])}')
counterd = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0
}
for label in Q[0][1]:
    counterd[y[label]] += 1
for key in counterd.keys():
    print(f'The total for in domain testing split is {label2language[key]} is {counterd[key]}')
print(f'Length of in domain testing split{len(Q[0][1])}')

print(f'Shape of entire in domain dataset {X.shape}')

BASE_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'
OOD_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'
FEATURE_TYPES = ['Parrish', 'FastText', 'Zouhar', 'Glove', 'Sharma', 'tfidf']
VALUES = ['True', 'False']
FILENAMES = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
LABELS = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).label.values.tolist()
ood_labels = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/y_europe_labels_out_of_domain_experiment_dataframe_chunks.pkl', 'rb')).label.values.tolist()
scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']
y = LABELS



data_filepath = BASE_DATA_DIR+'Parrish'+'/'+'data_tokenize'+'True'+'.npz'
X = sp.sparse.load_npz(data_filepath)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print('----------- 90 10 ---------')
counterd = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0
}
for label in y_train:
    counterd[label] += 1
for key in counterd.keys():
    print(f'The total for in domain testing 90 10 split is {label2language[key]} is {counterd[key]}')
print(f'Length of 90:10 split Train {X_train.shape}')
counterd = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0
}
for label in y_test:
    counterd[label] += 1
for key in counterd.keys():
    print(f'The total for in domain testing 90 10 split is {label2language[key]} is {counterd[key]}')
print(f'Length of 90:10 split test {X_test.shape}')

data_filepath = OOD_DATA_DIR+'Parrish'+'/'+'data_tokenize'+'True'+'.npz'
X = sp.sparse.load_npz(data_filepath)
ood_labels = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/y_europe_labels_out_of_domain_experiment_dataframe_chunks.pkl', 'rb')).label.values.tolist()
counterd = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 0,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0
}
for label in ood_labels:
    counterd[label] += 1
for key in counterd.keys():
    print(f'The total for out of sample experiment 90 10 split is {label2language[key]} is {counterd[key]}')
print(f'Size of the out of sample dataset {X.shape}')



