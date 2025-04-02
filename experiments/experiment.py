import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/csse/research/NativeLanguageID/mthesis-phonological/experiment/experiments/experiment.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')
#default is experiment.log
logger.setLevel(logging.DEBUG)
logger.info('----NEW RUN----')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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

scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']

for feature in FEATURE_TYPES:
    for data_file in FILENAMES:
        if feature != 'Zouhar':
            for value in VALUES:
                logger.info(f'entering {feature} {data_file} {value}')
                estimator_filepath = BASE_DATA_DIR+feature+'/'+'estimator'+data_file+value+'.pkl'
                if os.path.exists(estimator_filepath):
                    logger.info('File already exists. Not processing.')
                    logger.info(f'exiting {feature} {data_file} {value}')
                else:
                    data_filepath = BASE_DATA_DIR+feature+'/'+data_file+value+'.npz'
                    X = sp.sparse.load_npz(data_filepath)
                    y = LABELS
                    logger.info(f'loaded: {data_filepath}')
                    scaler = MaxAbsScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial')
                    scores = cross_validate(estimator=clf, X=X, y=y, cv=skf, scoring=scoring, return_estimator=True, return_indices=True)

                
                    with open(estimator_filepath, 'wb') as f:
                        pickle.dump(scores, f)
                        logger.info(f'saving estumator file with path: {estimator_filepath}')
                    f.close()
                    logger.info(f'results for: {feature} {data_file} {value}')
                    logger.info(f'f1 scores: {scores['test_f1_macro']}')
                    logger.info(f'precision: {scores['test_precision_macro']}')
                    logger.info(f'recall: {scores['test_recall_macro']}')
                    logger.info(f'accuracy: {scores['test_accuracy']}')
                    logger.info(f'fit time: {scores['fit_time']}')

                    del clf
                    for i in range(0,5):
                        logger.info(f'generating confusion matrix {i+1} of 5 (total:5)')
                        clf = scores['estimator'][i]
                        X_ind = scores['indices']['test'][i]
                        X_tmp = sp.sparse.csr_matrix(sp.sparse.vstack([X[j] for j in X_ind]))
                        y_tmp =  [y[j] for j in X_ind]
                        predictions = clf.predict(X_tmp)
                        disp = ConfusionMatrixDisplay.from_predictions(y_tmp, predictions, display_labels=[label2language[label] for label in clf.classes_], xticks_rotation=90, colorbar=False)
                        disp.plot()
                        disp.ax_.set_title(f'{feature} Confusion Matrix')
                        disp.ax_.tick_params(labelrotation=90, axis='x')
                        disp.figure_.set_figwidth(11)
                        disp.figure_.set_figheight(11)
                        plot_estimator_filepath = BASE_DATA_DIR+feature+'/'+'plot'+f'{i+1}'+'estimator'+data_file+value+'.svg'
                        disp.figure_.savefig(plot_estimator_filepath)
                        logger.info(f'saving confusion matrix to {plot_estimator_filepath}')
                        logger.info(f'generation of confusion matrix {i+1} of 5 (total:5) complete')
                    del skf
                    del clf
                    del scores
                    del X
                    logger.info(f'exiting {feature} {data_file} {value}') 
        else:
            logger.info(f'entering {feature} {data_file}')
            estimator_filepath = BASE_DATA_DIR+feature+'/'+'estimator'+data_file+'.pkl'
            if os.path.exists(estimator_filepath):
                logger.info('File already exists. Not processing.')
                logger.info(f'exiting {feature} {data_file}')
            else:
                data_filepath = BASE_DATA_DIR+feature+'/'+data_file+'.npz'
                X = sp.sparse.load_npz(data_filepath)
                y = LABELS
                logger.info(f'loaded: {data_filepath}')
                scaler = MaxAbsScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial')
                scores = cross_validate(estimator=clf, X=X, y=y, cv=skf, scoring=scoring, return_estimator=True, return_indices=True)

                with open(estimator_filepath, 'wb') as f:
                    pickle.dump(scores, f)
                    logger.info(f'saving estumator file with path: {estimator_filepath}')
                f.close()
                logger.info(f'results for: {feature} {data_file} ')
                logger.info(f'f1 scores: {scores['test_f1_macro']}')
                logger.info(f'precision: {scores['test_precision_macro']}')
                logger.info(f'recall: {scores['test_recall_macro']}')
                logger.info(f'accuracy: {scores['test_accuracy']}')
                logger.info(f'fit time: {scores['fit_time']}')

                del clf
                for i in range(0,4):
                    logger.info(f'generating confusion matrix {i+1} of 5 (total:5)')
                    clf = scores['estimator'][i]
                    X_ind = scores['indices']['test'][i]
                    X_tmp = sp.sparse.csr_matrix(sp.sparse.vstack([X[j] for j in X_ind]))
                    y_tmp =  [y[j] for j in X_ind]
                    y_tmp =  [y[j] for j in X_ind]
                    predictions = clf.predict(X_tmp)
                    disp = ConfusionMatrixDisplay.from_predictions(y_tmp, predictions, display_labels=[label2language[label] for label in clf.classes_],xticks_rotation=90, colorbar=False)
                    disp.ax_.set_title(f'{feature} Confusion Matrix')
                    disp.plot()
                    disp.ax_.tick_params(labelrotation=90, axis='x')
                    disp.figure_.set_figwidth(11)
                    disp.figure_.set_figheight(11)
                    plot_estimator_filepath = BASE_DATA_DIR+feature+'/'+'plot'+f'{i+1}'+'estimator'+data_file+'.svg'
                    disp.figure_.savefig(plot_estimator_filepath)
                    logger.info(f'saving confusion matrix to {plot_estimator_filepath}')
                    logger.info(f'generation of confusion matrix {i+1} of 5 (total:5) complete')
                del skf
                del clf
                del scores
                del X
                logger.info(f'exiting {feature} {data_file} {value}') 

logger.info('-----END OF RUN ------')           