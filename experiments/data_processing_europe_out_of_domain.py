import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/csse/research/NativeLanguageID/mthesis-phonological/experiment/experiments/OOD_ParrishSharma_data_processing.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
#default is data_processing.log
logger.setLevel(logging.DEBUG)
logger.info('----NEW RUN----')
import pickle
import os
import numpy as np
from tqdm import tqdm
import re
import logging
import time
from tqdm import tqdm 
import scipy as sp
import gensim
import gensim.downloader
import time

#import Embedders
from sharma_embedding.embedding import Dictionary as Sharma
from parrish_embedding.embedding import Dictionary as Parrish
from zouhar_embedding.embedding import ZouharEmbedder as Zouhar

#import vectorizers 
from kramp_feature.normal_feature_extractor import NormalFeatureExtractor
from kay_feature.feature_extractor import NonGenSimMeanTfidfEmbeddingVectorizer
from kay_feature.feature_extractor import GensimEmbed

#import stuff for preprocessing
import string
import difflib
from collections import Counter
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from language_checkers.spell_checker import SpellChecker
from language_checkers.grammar_checker import GrammarChecker
from language_checkers.pos_tagger import POSTagger

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tracemalloc



process_parrish = True
process_sharma = True
process_zouhar = False
process_fasttext = False
process_glove = False
record_performance_data = True
BASE_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'

if os.path.exists('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_tokenize.pkl'):
    logger.info('data is present not processing.')
else:
    logger.info('building OOD experiment dataset')
    data = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/X_europe_out_of_domain_experiment_dataframe_chunks.pkl', 'rb')).text.values.tolist()

    SpellChecker = SpellChecker(max_edit_distance=2)
    def preprocess(data, spell_check=False, lemmatize=False, remove_stopwords=False, SpellChecker=SpellChecker):
        data = [text.replace('\n', ' ') for text in tqdm(data, desc='replacing newlines')]
        data = [text.translate(str.maketrans('', '', string.punctuation)) for text in tqdm(data, desc='remove punctuation')]
        data = [re.sub(r'\d+', '', text) for text in tqdm(data, desc='removing numbers')]
        data = [nltk.word_tokenize(x) for x in tqdm(data, desc='tokenizing')]

        if spell_check == True:
            data = [[SpellChecker.get_closest_correction(z) for z in y] for y in tqdm(data, desc='spellchecking')]

        if remove_stopwords == True:
            stop_words = set(stopwords.words('english'))
            data = [[w for w in y if not w.lower() in stop_words] for y in tqdm(data, desc='removing stopwords')]

        if lemmatize ==  True:
            lemmatizer = WordNetLemmatizer()
            data = [[lemmatizer.lemmatize(w) for w in y] for y in tqdm(data, desc='lemmatizing')]

        return data

    data_tokenize = preprocess(data)
    data_tokenize_df = pd.DataFrame(np.array(data_tokenize, dtype='object').T, columns=['data_tokenize'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_tokenize.pkl', 'wb') as f:
        pickle.dump(data_tokenize_df, f)
        print('data_tokenize written to disk')
        f.close()
    del data_tokenize
    del data_tokenize_df

    data_spellcheck = preprocess(data, spell_check=True)
    data_spellcheck_df = pd.DataFrame(np.array(data_spellcheck, dtype='object').T, columns=['data_spellcheck'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_spellcheck.pkl', 'wb') as f:
        pickle.dump(data_spellcheck_df, f)
        print('data_spellcheck written to disk')
        f.close()
    del data_spellcheck
    del data_spellcheck_df


    data_spellcheck_lemmatize = preprocess(data, spell_check=True, lemmatize=True)
    data_spellcheck_lemmatize_df = pd.DataFrame(np.array(data_spellcheck_lemmatize, dtype='object').T, columns=['data_spellcheck_lemmatize'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_spellcheck_lemmatize.pkl', 'wb') as f:
        pickle.dump(data_spellcheck_lemmatize_df, f)
        print('data_spellcheck_lemmatize written to disk')
        f.close()
    del data_spellcheck_lemmatize
    del data_spellcheck_lemmatize_df


    data_spellcheck_stopwords = preprocess(data, spell_check=True, remove_stopwords=True)
    data_spellcheck_stopwords_df = pd.DataFrame(np.array(data_spellcheck_stopwords, dtype='object').T, columns=['data_spellcheck_stopwords'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_spellcheck_stopwords.pkl', 'wb') as f:
        pickle.dump(data_spellcheck_stopwords_df, f)
        print('data_spellcheck_stopwords written to disk')
        f.close()
    del data_spellcheck_stopwords
    del data_spellcheck_stopwords_df


    data_stopwords = preprocess(data, remove_stopwords=True)
    data_stopwords_df = pd.DataFrame(np.array(data_stopwords, dtype='object').T, columns=['data_stopwords'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_stopwords.pkl', 'wb') as f:
        pickle.dump(data_stopwords_df, f)
        print('data_stopwords written to disk')
        f.close()
    del data_stopwords
    del data_stopwords_df

    data_lemmatize = preprocess(data, lemmatize=True)
    data_lemmatize_df = pd.DataFrame(np.array(data_lemmatize, dtype='object').T, columns=['data_lemmatize'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_lemmatize.pkl', 'wb') as f:
        pickle.dump(data_lemmatize_df, f)
        print('data_lemmatize written to disk')
        f.close()
    del data_lemmatize
    del data_lemmatize_df  


    data_lemmatize_stopwords = preprocess(data, lemmatize=True, remove_stopwords=True)
    data_lemmatize_stopwords_df = pd.DataFrame(np.array(data_lemmatize_stopwords, dtype='object').T, columns=['data_lemmatize_stopwords'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_lemmatize_stopwords.pkl', 'wb') as f:
        pickle.dump(data_lemmatize_stopwords_df, f)
        print('data_lemmatize_stopwords written to disk')
        f.close()
    del data_lemmatize_stopwords
    del data_lemmatize_stopwords_df  



    data_lemmatize_stopwords_spellcheck = preprocess(data, lemmatize=True, remove_stopwords=True, spell_check=True)
    data_lemmatize_stopwords_spellcheck_df = pd.DataFrame(np.array(data_lemmatize_stopwords_spellcheck, dtype='object').T, columns=['data_lemmatize_stopwords_spellcheck'])
    with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/data_lemmatize_stopwords_spellcheck.pkl', 'wb') as f:
        pickle.dump(data_lemmatize_stopwords_spellcheck_df, f)
        print('data_lemmatize_stopwords_spellcheck written to disk')
        f.close()
    del data_lemmatize_stopwords_spellcheck
    del data_lemmatize_stopwords_spellcheck_df 



data_filenames = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
#labels is still called labels.
labels = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/y_europe_labels_out_of_domain_experiment_dataframe_chunks.pkl', 'rb')).label.values.tolist()


if process_parrish == True:
    ParrishEmbedderTrue = Parrish(filepath='/csse/research/NativeLanguageID/mthesis-phonological/parrish-embedding-project/models/cmudict-0.7b-simvecs', OOVRandom=True)
    
    for file in data_filenames:
            fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
            OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
            if record_performance_data == True:
                tracemalloc.start()
            t1 = time.perf_counter()
            ParrishFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(ParrishEmbedderTrue, vectorizer=None)
            t2 = time.perf_counter()
            logger.info(f'ParrishEmbedderTrue {file} init time {t2 - t1}')
            tfidf = ParrishFeatureExtractorTrue.fit(fit_data_df[file],labels)
            ParrishFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(ParrishEmbedderTrue, vectorizer=tfidf)
            transformed_data_matrix = ParrishFeatureExtractorTrue.fit_transform(OOD_data_df[file], labels)
            t3 = time.perf_counter()
            if record_performance_data == True:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                logger.info(f'ParrishEmbedderTrue {file} max memory: {peak}')
                del current
                del peak
            logger.info(f'ParrishEmbedderTrue {file} feature extraction time: {t3 - t2}')
            sp.sparse.save_npz(BASE_DIR+'Parrish/'+str(file)+'True'+'.npz', transformed_data_matrix)
            del ParrishFeatureExtractorTrue
            del transformed_data_matrix
            del fit_data_df
            del OOD_data_df
            del t1
            del t2
            del t3


    ParrishEmbedderFalse = Parrish(filepath='/csse/research/NativeLanguageID/mthesis-phonological/parrish-embedding-project/models/cmudict-0.7b-simvecs', OOVRandom=False)
    
    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
            t1 = time.perf_counter()
        ParrishFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(ParrishEmbedderFalse)
        t2 = time.perf_counter()
        logger.info(f'ParrishEmbedderFalse {file} init time {t2 - t1}')
        tfidf = ParrishFeatureExtractorFalse.fit(fit_data_df[file],labels)
        ParrishFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(ParrishEmbedderFalse, vectorizer=tfidf)
        transformed_data_matrix = ParrishFeatureExtractorFalse.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'ParrishEmbedderFalse {file} max memory: {peak}')
            del current
            del peak
        logger.info(f'ParrishEmbedderFalse {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'Parrish/'+str(file)+'False'+'.npz', transformed_data_matrix) 
        del ParrishFeatureExtractorFalse
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df
        del t1
        del t2
        del t3



if process_sharma == True:
    
    SharmaEmbedderTrue = Sharma(filepath='/csse/research/NativeLanguageID/mthesis-phonological/sharma-embedding-project/models/simvecs', OOVRandom=True)
        
    for file in data_filenames:  
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter()
        SharmaFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(SharmaEmbedderTrue, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'SharmaEmbedderTrue {file} init time {t2 - t1}')
        tfidf = SharmaFeatureExtractorTrue.fit(fit_data_df[file],labels)
        SharmaFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(SharmaEmbedderTrue, vectorizer=tfidf)
        transformed_data_matrix = SharmaFeatureExtractorTrue.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'SharmaEmbedderTrue {file} max memory: {peak}')
            del current
            del peak 
        logger.info(f'SharmaEmbedderTrue {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'Sharma/'+str(file)+'True'+'.npz', transformed_data_matrix)
        del SharmaFeatureExtractorTrue
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df
        del t1
        del t2
        del t3


    SharmaEmbedderFalse = Sharma(filepath='/csse/research/NativeLanguageID/mthesis-phonological/sharma-embedding-project/models/simvecs', OOVRandom=False)
    
    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter()
        SharmaFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(SharmaEmbedderFalse, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'SharmaEmbedderFalse {file} init time {t2 - t1}')
        tfidf = SharmaFeatureExtractorFalse.fit(fit_data_df[file],labels)
        SharmaFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(SharmaEmbedderFalse, vectorizer=tfidf)
        transformed_data_matrix = SharmaFeatureExtractorFalse.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'SharmaEmbedderFalse {file} max memory: {peak}')
            del current
            del peak  
        logger.info(f'SharmaEmbedderFalse {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'Sharma/'+str(file)+'False'+'.npz', transformed_data_matrix) 
        del SharmaFeatureExtractorFalse
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df        
        del t1
        del t2
        del t3



if process_zouhar == True:
    ZouharEmbedder = Zouhar()

    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter() 
        ZouharFeatureExtractor = NonGenSimMeanTfidfEmbeddingVectorizer(ZouharEmbedder, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'ZouharEmbedder {file} init time {t2 - t1}')
        tfidf = ZouharFeatureExtractor.fit(fit_data_df[file],labels)
        ZouharFeatureExtractor = NonGenSimMeanTfidfEmbeddingVectorizer(ZouharEmbedder, vectorizer=tfidf)
        transformed_data_matrix = ZouharFeatureExtractor.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'ZouharEmbedder {file} max memory: {peak}')
            del current
            del peak
        logger.info(f'ZouharEmbedder {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'Zouhar/'+str(file)+'.npz', transformed_data_matrix)
        del ZouharFeatureExtractor
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df 
        del t1
        del t2
        del t3


if process_glove == True:
    glove_vectors = gensim.downloader.load('glove-twitter-50')

    GloveEmbedderTrue = GensimEmbed(model=glove_vectors, OOVRandom=True)
    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter() 
        GloveFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(GloveEmbedderTrue, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'GloveEmbedderTrue {file} init time {t2 - t1}')
        tfidf = GloveFeatureExtractorTrue.fit(fit_data_df[file],labels)
        GloveFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(GloveEmbedderTrue, vectorizer=tfidf)
        transformed_data_matrix = GloveFeatureExtractorTrue.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'GloveEmbedderTrue {file} max memory: {peak}')
            del current
            del peak
        logger.info(f'GloveEmbedderTrue {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'Glove/'+str(file)+'True'+'.npz', transformed_data_matrix)
        del GloveFeatureExtractorTrue
        del transformed_data_matrix 
        del fit_data_df
        del OOD_data_df
        del t1
        del t2
        del t3


    GloveEmbedderFalse = GensimEmbed(model=glove_vectors, OOVRandom=False)
    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter() 
        GloveFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(GloveEmbedderFalse, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'GloveEmbedderFalse {file} init time {t2 - t1}')
        tfidf = GloveFeatureExtractorFalse.fit(fit_data_df[file],labels)
        GloveFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(GloveEmbedderFalse, vectorizer=tfidf)
        transformed_data_matrix = GloveFeatureExtractorFalse.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'GloveEmbedderFalse {file} max memory: {peak}')
            del current
            del peak
        logger.info(f'GloveEmbedderFalse {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'Glove/'+str(file)+'False'+'.npz', transformed_data_matrix) 
        del GloveFeatureExtractorFalse
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df
        del t1
        del t2
        del t3



if process_fasttext == True:
    fasttext_vectors = gensim.downloader.load('fasttext-wiki-news-subwords-300')

    FastTextEmbedderTrue = GensimEmbed(model=fasttext_vectors, OOVRandom=True)
    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter() 
        FastTextFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(FastTextEmbedderTrue, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'FastTextEmbedderTrue {file} init time {t2 - t1}')
        tfidf = FastTextFeatureExtractorTrue.fit(fit_data_df[file],labels)
        FastTextFeatureExtractorTrue = NonGenSimMeanTfidfEmbeddingVectorizer(FastTextEmbedderTrue, vectorizer=tfidf)
        transformed_data_matrix = FastTextFeatureExtractorTrue.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'FastTextEmbedderTrue {file} max memory: {peak}')
            del current
            del peak
        logger.info(f'FastTextEmbedderTrue {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'FastText/'+str(file)+'True'+'.npz', transformed_data_matrix)
        del FastTextFeatureExtractorTrue
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df 
        del t1
        del t2
        del t3
  

   
    FastTextEmbedderFalse = GensimEmbed(model=fasttext_vectors, OOVRandom=False)
    for file in data_filenames:
        fit_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'+ f'{file}' + ".pkl", 'rb'))
        OOD_data_df = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/OOD/'+ f'{file}' + ".pkl", 'rb'))
        if record_performance_data == True:
            tracemalloc.start()
        t1 = time.perf_counter() 
        FastTextFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(FastTextEmbedderFalse, vectorizer=None)
        t2 = time.perf_counter()
        logger.info(f'FastTextEmbedderFalse {file} init time {t2 - t1}')
        tfidf = FastTextFeatureExtractorFalse.fit_transform(fit_data_df[file],labels)
        FastTextFeatureExtractorFalse = NonGenSimMeanTfidfEmbeddingVectorizer(FastTextEmbedderFalse, vectorizer=tfidf)
        transformed_data_matrix = FastTextFeatureExtractorFalse.fit_transform(OOD_data_df[file], labels)
        t3 = time.perf_counter()
        if record_performance_data == True:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            logger.info(f'FastTextEmbedderFalse {file} max memory: {peak}')
            del current
            del peak
        logger.info(f'FastTextEmbedderFalse {file} feature extraction time: {t3 - t2}')
        sp.sparse.save_npz(BASE_DIR+'FastText/'+str(file)+'False'+'.npz', transformed_data_matrix) 
        del FastTextFeatureExtractorFalse
        del transformed_data_matrix
        del fit_data_df
        del OOD_data_df
        del t1
        del t2
        del t3

logger.info(f'----END OF RUN----')
