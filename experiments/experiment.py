from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
import time
import logging
import os
from tqdm import tqdm
import pickle
import pandas as pd
import numpy as np
import scipy as sp

SEED = 42

data_filepath = '/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/FastText/data_lemmatize_stopwords_spellcheckFalse.npz'
labels = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/full_labels_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).label.values.tolist()

features = sp.sparse.load_npz(data_filepath)

X = features
y = labels

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

skf.get_n_splits(X, y)
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    print(f"Fold {i}:")
    print(f"  Train: length={len(train_index)}")
    print(f"  Train: index={train_index}")
    print(f"  Test: length={len(test_index)}")
    print(f"  Test:  index={test_index}")


clf = LogisticRegression(random_state=42, max_iter=5000, verbose=True, solver='saga', n_jobs=23, multi_class='ovr')

scores = cross_validate(estimator=clf, X=X, y=y, cv=skf, return_estimator=True, return_indices=True)
print(scores)
with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/estimators/data_lemmatize_stopwords_spellcheckFalse_estimators.pkl') as f:
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

