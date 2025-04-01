import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/csse/research/NativeLanguageID/mthesis-phonological/experiment/experiments/experiment.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
#default is experiment.log
logger.setLevel(logging.DEBUG)
logger.info('----NEW RUN----')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import make_scorer
import time
import logging
import os
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import scipy as sp

language2label = {
    'English': 0,
    'German': 1,
    'Bulgarian': 2,
    'Croatian': 3,
    'Czech': 4,
    'Estonian': 5,
    'Finish': 6,
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

SEED = 42

BASE_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'
FEATURE_TYPES = ['Parrish', 'FastText', 'Zouhar', 'Glove', 'Sharma']
VALUES = ['True', 'False']
FILENAMES = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
LABELS = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).label.values.tolist()

for feature in FEATURE_TYPES:
    for data_file in FILENAMES:
        if feature != 'Zouhar':
            for value in VALUES:
                logger.DEBUG(f'entering {feature} {data_file} {value}')
                data_filepath = BASE_DATA_DIR+'/'+feature+'/'+data_file+value+'.npz'
                X = sp.sparse.load_npz(data_filepath)
                y = LABELS
                logger.DEBUG(f'loaded: {data_filepath}')
                scaler = MaxAbsScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial')
                scores = cross_validate(estimator=clf, X=X, y=y, cv=skf, return_estimator=True, return_indices=True)
                estimator_filepath = BASE_DATA_DIR+'/'+feature+'/'+'estimator'+data_file+value+'.pkl'
                with open(estimator_filepath, 'wb') as f:
                    pickle.dump(scores, f)
                f.close()
                del skf
                del clf
                del scores
                del X
                logger.DEBUG(f'exiting {feature} {data_file} {value}') 
        else:
            logger.DEBUG(f'entering {feature} {data_file}')
            data_filepath = BASE_DATA_DIR+'/'+feature+'/'+data_file+'.npz'
            X = sp.sparse.load_npz(data_filepath)
            y = LABELS
            logger.DEBUG(f'loaded: {data_filepath}')
            scaler = MaxAbsScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial')
            scores = cross_validate(estimator=clf, X=X, y=y, cv=skf, return_estimator=True, return_indices=True)
            estimator_filepath = BASE_DATA_DIR+'/'+feature+'/'+'estimator'+data_file+'.pkl'
            with open(estimator_filepath, 'wb') as f:
                pickle.dump(scores, f)
            f.close()
            del skf
            del clf
            del scores
            del X
            logger.DEBUG(f'exiting {feature} {data_file}')

logger.INFO('-----END OF RUN ------') 







#feature path names
#save confusion matrices
#save estimators
#log acc, prec, recall, F1.

features = sp.sparse.load_npz(data_filepath)

X = features
y = labels
print(X[0])
print(y[0])
print(X.shape)
print(len(y))

scaler = MaxAbsScaler()
scaler.fit(X)
X = scaler.transform(X)






print(f"The test score for the 5 Stratified K Fold is: {scores['test_score']}")
with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/estimators/data_lemmatize_stopwords_spellcheckFalse_estimators.pkl', 'wb') as f:
    pickle.dump(scores, f)
    f.close()




# print('data loaded, beginning classification')
# clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial').fit(X,y)
# print('doing a prediction')
# clf.predict(X)
# print('doing a scoreing')
# clf.score(X,y)
# print('the end')

#KFOLD-5
#FIT-TEST-STATS
#CM of best performing fold.
#log it
#time it
#track memory

#k-fold
#MaxAbsScalerTest
#[0.58201493, 0.58708955, 0.58813433, 0.592238, 0.58597015]
#[0.05529851, 0.05529851, 0.05537313, 0.055373, 0.05529851]              