import pickle
import os
from kramp_feature.normal_feature_extractor import NormalFeatureExtractor
from kay_feature.feature_extractor import NonGenSimMeanTfidfEmbeddingVectorizer, EmbeddingFeatureExtractor
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
from tqdm import tqdm
from parrish_embedding.embedding import Dictionary as Parrish
import re

data = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).text.values.tolist()
Y_DF = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb'))

if os.path.exists('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_X_parrish_chunks.pkl'):
    X_parrish = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_X_parrish_chunks.pkl', 'rb'))
else:
    print('starting to load embedder')
    ParrishEmbedder = Parrish(filepath='/csse/research/NativeLanguageID/mthesis-phonological/parrish-embedding-project/models/cmudict-0.7b-simvecs', OOVRandom=True)
    print('embedder loaded, now loading feature extractor')
    ParrishFeatureExtractor = NonGenSimMeanTfidfEmbeddingVectorizer(ParrishEmbedder)
    print('feature extractor loaded, now loading embedding feature extractor')
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

    from tqdm import tqdm
    import nltk
    from nltk.corpus import stopwords
 
    data = [text.replace('\n', ' ') for text in tqdm(data, desc='replacing newlines')]
    data = [text.translate(str.maketrans('', '', string.punctuation)) for text in tqdm(data, desc='remove punctuation')]
    data = [re.sub(r'\d+', '', text) for text in tqdm(data, desc='removing numbers')]

    data = [nltk.word_tokenize(x) for x in tqdm(data, desc='tokenizing')]


    stop_words = set(stopwords.words('english'))
    data = [[w for w in y if not w.lower() in stop_words] for y in tqdm(data, desc='removing stopwords')]


    SpellChecker = SpellChecker(max_edit_distance=2)
    
    data = [[SpellChecker.get_closest_correction(z) for z in y] for y in tqdm(data, desc='spellchecking')]


    print('testing tfidf mode')
    test4 = ParrishFeatureExtractor.fit_transform(data, Y_DF)
    print('exiting tfidf test mode')
    print(test4.shape)
    # with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_X_parrish_chunks.pkl', 'wb') as f:
    #     pickle.dump(X_parrish, f)
    #     f.close()

    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegressionCV
    clf = LogisticRegressionCV(cv=5, random_state=42, max_iter=5000).fit(test4, Y_DF.values.tolist())
    clf.score(test4, Y_DF.values.tolist())