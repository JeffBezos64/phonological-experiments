#PCA
#128,256,512,1024,2048,4096
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='/csse/research/NativeLanguageID/mthesis-phonological/experiment/experiments/data_processing.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
logger.setLevel(logging.DEBUG)
logger.info('----START OF RUN----')

import numpy as np
from sklearn.decomposition import PCA

data_filenames = ['data_tokenize', 'data_spellcheck', 'data_spellcheck_lemmatize', 'data_spellcheck_stopwords', 'data_stopwords', 'data_lemmatize', 'data_lemmatize_stopwords', 'data_lemmatize_stopwords_spellcheck']
EmbeddingMethods = ['Parrish', 'Sharma', 'Zouhar', 'Glove', 'FastText']
feature_size_list = [128,256,512,1024,2048,4096]
bools = ['True', 'False']
BASE_DIR = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/'

import tracemalloc
import time
import scipy as sp

record_performance_data = False


for EmbeddingMethod in EmbeddingMethods:
    LOOP_DIR = BASE_DIR + str(EmbeddingMethod) + '/'
    for filename in data_filenames:
        FILE = LOOP_DIR + str(filename)
        for feature_size in feature_size_list:
            for boolean in bools:
                if EmbeddingMethod != 'Zouhar':
                    sample_file = FILE + boolean + '.npz'
                    sample_matrix = sp.sparse.load(sample_file)
                    t1 = time.perf_counter()
                    if record_performance_data == True:
                        tracemalloc.start()
                    pca = PCA(n_components=feature_size, n_samples=67000)
                    reduced_mat = pca.fit_transform(sample_matrix)
                    t2 = time.perf_counter()
                    if record_performance_data == True:
                        current, peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        logger.info(f'{EmbeddingMethod}-{boolean}-{filename}-{feature_size}: peak memory {peak}')
                        del peak
                        del current
                    logger.info(f'{EmbeddingMethod}-{boolean}-{filename}-{feature_size}: time taken {t2-t1}')
                    mat_filename = LOOP_DIR + 'PCA/' + f'data-{filename}-' + f'{boolean}' + f'-{feature_size}'  + '.pkl'
                    pca_filename = LOOP_DIR + 'PCA/' + f'pca_classifier-{filename}-' + f'{boolean}' + f'-{feature_size}'  + '.pkl'
                    with open(mat_filename, 'wb') as f:
                        np.save(f, reduced_mat)
                        logger.info('saved reduced matrix')
                        f.close()
                    with open(pca_filename, 'wb') as f:
                        pickle.dump(pca, f)
                        logger.info('saved pca classifer')
                        f.close()
                    del pca
                    del reduced_mat

            if EmbeddingMethod == 'Zouhar' and boolean == 'True':
                sample_file = FILE + '.npz'
                sample_matrix = sp.sparse.load(sample_file)
                t1 = time.perf_counter()
                if record_performance_data == True:
                    tracemalloc.start()
                pca = PCA(n_components=feature_size, n_samples=67000)
                reduced_mat = pca.fit(transform(sample_matrix)
                t2 = time.perf_counter()
                if record_performance_data == True:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    logger.info(f'{EmbeddingMethod}-{boolean}-{filename}-{feature_size}: peak memory {peak}')
                    del peak
                    del current
                logger.info(f'{EmbeddingMethod}-{boolean}-{filename}-{feature_size}: time taken {t2-t1}')
                mat_filename = LOOP_DIR + 'PCA/' + f{filename} + f{feature_size} + '.pkl'
                pca_filename = LOOP_DIR + 'PCA/' + f'pca_classifier-{filename}' + f'-{feature_size}'  + '.pkl'
                with open(mat_filename, 'wb') as f:
                    np.save(f, reduced_mat)
                    logger.info('saved reduced matrix')
                    f.close()
                with open(pca_filename, 'wb') as f:
                    pickle.dump(pca, f)
                    logger.info('saved pca classifer')
                    f.close()
                del pca
                del reduced_mat

logger.info('----END OF RUN----')

