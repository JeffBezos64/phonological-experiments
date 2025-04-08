#load usual shit
#create test and train split of main dataset.
#train
#test generate confusion
#test set run
#generate confusion
#save and log results

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/csse/research/NativeLanguageID/mthesis-phonological/experiment/experiments/OOD_experiment.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s %(message)s')
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
from sklearn.model_selection import train_test_split
from collections import defaultdict
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

SEED = 42

BASE_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'
OOD_DATA_DIR = '/csse/research/NativeLanguageId/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'
FEATURE_TYPES = ['Parrish', 'FastText', 'Zouhar', 'Glove', 'Sharma', 'tfidf']
VALUES = ['True', 'False']
FILENAMES = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
LABELS = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).label.values.tolist()
ood_labels = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/y_europe_labels_out_of_domain_experiment_dataframe_chunks.pkl', 'rb')).label.values.tolist()
scoring = ['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']

for feature in FEATURE_TYPES:
    for data_file in FILENAMES:
        if feature != 'Zouhar' and feature != 'tfidf':
            for value in VALUES:
                logger.info(f'entering {feature} {data_file} {value}')
                results_filepath = BASE_DATA_DIR+feature+'/'+'9010results'+data_file+value+'.pkl'
                if os.path.exists(results_filepath):
                    logger.info('File already exists. Not processing.')
                    logger.info(f'exiting {feature} {data_file} {value}')
                else:
                    logger.info(f'starting in sample work for {feature} {data_file} {value}')
                    data_filepath = BASE_DATA_DIR+feature+'/'+data_file+value+'.npz'
                    X = sp.sparse.load_npz(data_filepath)
                    y = LABELS
                    logger.info(f'loaded: {data_filepath}')
                    scaler = MaxAbsScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                    
                    clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial')
                    clf.fit(X_train, y_train)
                    predictions = clf.predict(X_test)
                    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=[label2language[label] for label in clf.classes_], xticks_rotation=90, colorbar=False)
                    disp.plot()
                    disp.ax_.set_title(f'{feature} Confusion Matrix')
                    disp.ax_.tick_params(labelrotation=90, axis='x')
                    disp.figure_.set_figwidth(11)
                    disp.figure_.set_figheight(11)
                    plot_estimator_filepath = BASE_DATA_DIR+feature+'/'+'plot'+'9010estimator'+data_file+value+'.svg'
                    disp.figure_.savefig(plot_estimator_filepath)
                    logger.info(f'saving confusion matrix to {plot_estimator_filepath}')
                    logger.info(f'generation of confusion matrix for in-domain complete')

                    accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
                    precision = precision_score(y_pred=predictions, y_true=y_test, average='macro')
                    recall = recall_score(y_pred=predictions, y_true=y_test, average='macro')
                    f1 = f1_score(y_pred=predictions, y_true=y_test, average='macro')

                    logger.info(f'{feature}{data_file}{value} accuracy - in domain sample: {accuracy}')
                    logger.info(f'{feature}{data_file}{value} precision - in domain sample: {precision}')
                    logger.info(f'{feature}{data_file}{value} recall - in domain sample: {recall}')
                    logger.info(f'{feature}{data_file}{value} f1 score - in domain sample: {f1}')

                    in_results_dict = defaultdict(dict)
                    in_results_dict[feature] = {}
                    in_results_dict[feature][data_file] = defaultdict(dict)
                    in_results_dict[feature][data_file][value] = defaultdict(dict)
                    in_results_dict[feature][data_file][value]['f1_macro'] = f1
                    in_results_dict[feature][data_file][value]['precision_macro'] = precision
                    in_results_dict[feature][data_file][value]['recall_macro'] = recall
                    in_results_dict[feature][data_file][value]['accuracy'] = accuracy
                    in_results_dict[feature][data_file][value]['fit_time'] = None
                    with open(results_filepath, 'wb') as f:
                        pickle.dump(in_results_dict, f)
                        logger.info(f'saving results file with path: {results_filepath}')
                    f.close()
                    del in_results_dict


                    logger.info(f'starting out of sample testing for {feature} {data_file} {value}')               
                    ood_data_filepath = OOD_DATA_DIR+feature+'/'+data_file+value+'.npz'
                    ood_x = sp.sparse.load_npz(data_filepath)
                    ood_y = ood_labels
                    predictions = clf.predict(ood_x)
                    logger.info(f'preparing OOS confusion matrix')
                    disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=[label2language[label] for label in clf.classes_], xticks_rotation=90, colorbar=False)
                    disp.plot()
                    disp.ax_.set_title(f'{feature} OOS Confusion Matrix')
                    disp.ax_.tick_params(labelrotation=90, axis='x')
                    disp.figure_.set_figwidth(11)
                    disp.figure_.set_figheight(11)
                    plot_estimator_filepath = OOD_DATA_DIR+feature+'/'+'plot'+'9010estimator'+data_file+value+'.svg'
                    disp.figure_.savefig(plot_estimator_filepath)
                    logger.info(f'saving confusion matrix to {plot_estimator_filepath}')
                    logger.info(f'generation of confusion matrix for out of sample complete')

                    accuracy = accuracy_score(y_pred=predictions, y_true=ood_y)
                    precision = precision_score(y_pred=predictions, y_true=ood_y, average='macro')
                    recall = recall_score(y_pred=predictions, y_true=ood_y, average='macro')
                    f1 = f1_score(y_pred=predictions, y_true=ood_y, average='macro')

                    logger.info(f'{feature}{data_file}{value} accuracy - out of sample: {accuracy}')
                    logger.info(f'{feature}{data_file}{value} precision - out of sample: {precision}')
                    logger.info(f'{feature}{data_file}{value} recall - out of sample: {recall}')
                    logger.info(f'{feature}{data_file}{value} f1 score - out of sample: {f1}')
 
                    OOD_results_dict = defaultdict(dict)
                    OOD_results_dict[feature] = {}
                    OOD_results_dict[feature][data_file] = defaultdict(dict)
                    OOD_results_dict[feature][data_file][value] = defaultdict(dict)
                    OOD_results_dict[feature][data_file][value]['f1_macro'] = f1
                    OOD_results_dict[feature][data_file][value]['precision_macro'] = precision
                    OOD_results_dict[feature][data_file][value]['recall_macro'] = recall
                    OOD_results_dict[feature][data_file][value]['accuracy'] = accuracy
                    OOD_results_dict[feature][data_file][value]['fit_time'] = None
                    OOD_results_filepath = OOD_DATA_DIR+feature+'/'+'results'+data_file+value+'.pkl'
                    with open(OOD_results_filepath, 'wb') as f:
                        pickle.dump(OOD_results_dict, f)
                        logger.info(f'saving results file with path: {results_filepath}')
                    f.close()
                    del OOD_results_dict
                    logger.info(f'exiting {feature} {data_file} {value}') 

        elif feature == 'Zouhar' or feature == 'tfidf':
            logger.info(f'entering {feature} {data_file}')
            results_filepath = BASE_DATA_DIR+feature+'/'+'9010results'+data_file+'.pkl'
            if os.path.exists(results_filepath):
                logger.info('File already exists. Not processing.')
                logger.info(f'exiting {feature} {data_file}')
            else:
                logger.info(f'starting in sample work for {feature} {data_file}')
                data_filepath = BASE_DATA_DIR+feature+'/'+data_file+'.npz'
                X = sp.sparse.load_npz(data_filepath)
                y = LABELS
                logger.info(f'loaded: {data_filepath}')
                scaler = MaxAbsScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                    
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                    
                clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='multinomial')
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                disp = ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=[label2language[label] for label in clf.classes_], xticks_rotation=90, colorbar=False)
                disp.plot()
                disp.ax_.set_title(f'{feature} Confusion Matrix')
                disp.ax_.tick_params(labelrotation=90, axis='x')
                disp.figure_.set_figwidth(11)
                disp.figure_.set_figheight(11)
                plot_estimator_filepath = BASE_DATA_DIR+feature+'/'+'plot'+'9010estimator'+data_file+'.svg'
                disp.figure_.savefig(plot_estimator_filepath)
                logger.info(f'saving confusion matrix to {plot_estimator_filepath}')
                logger.info(f'generation of confusion matrix for in-domain complete')

                accuracy = accuracy_score(y_pred=predictions, y_true=y_test)
                precision = precision_score(y_pred=predictions, y_true=y_test, average='macro')
                recall = recall_score(y_pred=predictions, y_true=y_test, average='macro')
                f1 = f1_score(y_pred=predictions, y_true=y_test, average='macro')

                logger.info(f'{feature}{data_file} accuracy - in domain sample: {accuracy}')
                logger.info(f'{feature}{data_file} precision - in domain sample: {precision}')
                logger.info(f'{feature}{data_file} recall - in domain sample: {recall}')
                logger.info(f'{feature}{data_file} f1 score - in domain sample: {f1}')

                in_results_dict = defaultdict(dict)
                in_results_dict[feature] = {}
                in_results_dict[feature][data_file] = defaultdict(dict)
                in_results_dict[feature][data_file] = defaultdict(dict)
                in_results_dict[feature][data_file]['f1_macro'] = f1
                in_results_dict[feature][data_file]['precision_macro'] = precision
                in_results_dict[feature][data_file]['recall_macro'] = recall
                in_results_dict[feature][data_file]['accuracy'] = accuracy
                in_results_dict[feature][data_file]['fit_time'] = None
                with open(results_filepath, 'wb') as f:
                    pickle.dump(in_results_dict, f)
                    logger.info(f'saving results file with path: {results_filepath}')
                    f.close()
                del in_results_dict


                logger.info(f'starting out of sample testing for {feature} {data_file}')               
                ood_data_filepath = OOD_DATA_DIR+feature+'/'+data_file+'.npz'
                ood_x = sp.sparse.load_npz(data_filepath)
                ood_y = ood_labels
                predictions = clf.predict(ood_x)
                logger.info(f'preparing OOS confusion matrix')
                disp = ConfusionMatrixDisplay.from_predictions(ood_y, predictions, display_labels=[label2language[label] for label in clf.classes_], xticks_rotation=90, colorbar=False)
                disp.plot()
                disp.ax_.set_title(f'{feature} OOS Confusion Matrix')
                disp.ax_.tick_params(labelrotation=90, axis='x')
                disp.figure_.set_figwidth(11)
                disp.figure_.set_figheight(11)
                plot_estimator_filepath = OOD_DATA_DIR+feature+'/'+'plot'+'9010estimator'+data_file+'.svg'
                disp.figure_.savefig(plot_estimator_filepath)
                logger.info(f'saving confusion matrix to {plot_estimator_filepath}')
                logger.info(f'generation of confusion matrix for out of sample complete')

                accuracy = accuracy_score(y_pred=predictions, y_true=ood_y)
                precision = precision_score(y_pred=predictions, y_true=ood_y, average='macro')
                recall = recall_score(y_pred=predictions, y_true=ood_y, average='macro')
                f1 = f1_score(y_pred=predictions, y_true=ood_y, average='macro')

                logger.info(f'{feature}{data_file}{value} accuracy - out of sample: {accuracy}')
                logger.info(f'{feature}{data_file}{value} precision - out of sample: {precision}')
                logger.info(f'{feature}{data_file}{value} recall - out of sample: {recall}')
                logger.info(f'{feature}{data_file}{value} f1 score - out of sample: {f1}')
 
                OOD_results_dict = defaultdict(dict)
                OOD_results_dict[feature] = {}
                OOD_results_dict[feature][data_file] = defaultdict(dict)
                OOD_results_dict[feature][data_file] = defaultdict(dict)
                OOD_results_dict[feature][data_file]['f1_macro'] = f1
                OOD_results_dict[feature][data_file]['precision_macro'] = precision
                OOD_results_dict[feature][data_file]['recall_macro'] = recall
                OOD_results_dict[feature][data_file]['accuracy'] = accuracy
                OOD_results_dict[feature][data_file]['fit_time'] = None
                OOD_results_filepath = OOD_DATA_DIR+feature+'/'+'results'+data_file+'.pkl'
                with open(OOD_results_filepath, 'wb') as f:
                    pickle.dump(OOD_results_dict, f)
                    logger.info(f'saving results file with path: {results_filepath}')
                    f.close()
                del OOD_results_dict
                logger.info(f'exiting {feature} {data_file} {value}')  
logger.info('-----END OF RUN ------') 