import pickle
import os
from collections import defaultdict
import numpy as np
import scipy as sp
import re
import pandas as pd

BASE_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'
OOD_DATA_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'
FEATURE_TYPES = ['Parrish', 'FastText', 'Zouhar', 'Glove', 'Sharma', 'tfidf']
VALUES = ['True', 'False']
FILENAMES = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
#results_filepath = BASE_DATA_DIR+feature+'/'+'9010results'+data_file+value+'.pkl'
#OOD_results_filepath = OOD_DATA_DIR+feature+'/'+'results'+data_file+value+'.pkl'
#datum = ['accuracy','precision_macro','recall_macro', 'f1_macro', 'fit_time']
# print('loading the in_domain cross-val accuracy results')
# name_list = []
# data_list = []
# avg_list = []
# max_list = []
# min_list = []
# for filename in FILENAMES:
#     for feature in FEATURE_TYPES:
#         if feature != 'Zouhar' and feature != 'tfidf':
#             for value in VALUES:
#                 with open(BASE_DATA_DIR+f'{feature}/'+f'results{filename}'+f'{value}'+'.pkl', 'rb') as f:
#                     results = pickle.load(f)
#                     f.close()
#                 for data in datum:
#                     x = results[feature][filename][value][data]
#                     x = x.tolist()
#                     data_list.append([f'{y:.4f}' for y in x])
#                     name_list.append(f'{feature} {filename} {value} {data}')
#                     avg_list.append(f'{sum(x)/len(x):.4f}')
#                     min_list.append(f'{min(x):.4f}')
#                     max_list.append(f'{max(x):.4f}')
#         else:
#             with open(BASE_DATA_DIR+f'{feature}/'+f'results{filename}'+'.pkl','rb') as f:
#                 results = pickle.load(f)
#                 f.close()
#             for data in datum:
#                 x = results[feature][filename][data]
#                 x = x.tolist()
#                 data_list.append([f'{y:.4f}' for y in x])
#                 name_list.append(f'{feature} {filename} {data}')
#                 avg_list.append(f'{sum(x)/len(x):.4f}')
#                 min_list.append(f'{min(x):.4f}')
#                 max_list.append(f'{max(x):.4f}')
# dict1={'Name': name_list, 'Fold data':data_list, 'AVG':avg_list,'MAX':max_list,'MIN':min_list}
# full_table_df= pd.DataFrame(dict1)
# print(full_table_df.to_latex(index=False,float_format="{:.8f}".format, longtable=True))

datum = ['accuracy','precision_macro','recall_macro', 'f1_macro', 'fit_time']
print('loading the in_domain cross-val accuracy results')
name_list = []
accuracy_list = []
f1_list = []
precision_list = []
recall_list = []
train_time_list = []
for filename in FILENAMES:
    for feature in FEATURE_TYPES:
        if feature != 'Zouhar' and feature != 'tfidf':
            for value in VALUES:
                with open(BASE_DATA_DIR+f'{feature}/'+f'9010results{filename}'+f'{value}'+'.pkl', 'rb') as f:
                    results = pickle.load(f)
                    f.close()
                x = results[feature][filename][value]
                name_list.append(f'{feature} {filename} {value}')
                accuracy_list.append(f'{x['accuracy']:.4f}')
                precision_list.append(f'{x['precision_macro']:.4f}')
                recall_list.append(f'{x['recall_macro']:.4f}')
                f1_list.append(f'{x['f1_macro']:.4f}')

        else:
            with open(BASE_DATA_DIR+f'{feature}/'+f'9010results{filename}'+'.pkl','rb') as f:
                results = pickle.load(f)
                f.close()
                x = results[feature][filename]
                name_list.append(f'{feature} {filename} ')
                accuracy_list.append(f'{x['accuracy']:.4f}')
                precision_list.append(f'{x['precision_macro']:.4f}')
                recall_list.append(f'{x['recall_macro']:.4f}')
                f1_list.append(f'{x['f1_macro']:.4f}')
dict1={'Name': name_list, 'Accuracy':accuracy_list, 'Precision':precision_list,'Recall':recall_list,'F1 Score':f1_list}
full_table_df = pd.DataFrame(dict1)
print(full_table_df.to_latex(index=False,longtable=True))

datum = ['accuracy','precision_macro','recall_macro', 'f1_macro', 'fit_time']
print('loading the in_domain cross-val accuracy results')
name_list = []
accuracy_list = []
f1_list = []
precision_list = []
recall_list = []
train_time_list = []
for filename in FILENAMES:
    for feature in FEATURE_TYPES:
        if feature != 'Zouhar' and feature != 'tfidf':
            for value in VALUES:
                with open(OOD_DATA_DIR+f'{feature}/'+f'results{filename}'+f'{value}'+'.pkl', 'rb') as f:
                    results = pickle.load(f)
                    f.close()
                x = results[feature][filename][value]
                name_list.append(f'{feature} {filename} {value}')
                accuracy_list.append(f'{x['accuracy']:.4f}')
                precision_list.append(f'{x['precision_macro']:.4f}')
                recall_list.append(f'{x['recall_macro']:.4f}')
                f1_list.append(f'{x['f1_macro']:.4f}')

        else:
            with open(OOD_DATA_DIR+f'{feature}/'+f'results{filename}'+'.pkl','rb') as f:
                results = pickle.load(f)
                f.close()
                x = results[feature][filename]
                name_list.append(f'{feature} {filename} ')
                accuracy_list.append(f'{x['accuracy']:.4f}')
                precision_list.append(f'{x['precision_macro']:.4f}')
                recall_list.append(f'{x['recall_macro']:.4f}')
                f1_list.append(f'{x['f1_macro']:.4f}')
dict1={'Name': name_list, 'Accuracy':accuracy_list, 'Precision':precision_list,'Recall':recall_list,'F1 Score':f1_list}
full_table_df = pd.DataFrame(dict1)
print(full_table_df.to_latex(index=False,longtable=True))

# print('loading the out of domain 90:10 accuracy results.')
# for filename in FILENAMES:
#     for feature in FEATURE_TYPES:
#         if feature != 'Zouhar' and feature != 'tfidf':
#             for value in VALUES:
#                 with open(OOD_DATA_DIR+f'{feature}/'+f'results{filename}'+f'{value}'+'.pkl', 'rb') as f:
#                     results = pickle.load(f)
#                     f.close()
#                 x = results[feature][filename][value]['accuracy']
#                 results_string = f'{feature} {filename} {value} & {x: .4f} \\'
#                 print(results_string)
#         else:
#             with open(OOD_DATA_DIR+f'{feature}/'+f'results{filename}'+'.pkl','rb') as f:
#                 results = pickle.load(f)
#                 f.close()
#             x = results[feature][filename]['accuracy']
#             results_string = f'{feature} {filename} & {x: .4f} \\'
#             print(results_string)
# print('end of out of domain')