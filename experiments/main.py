import pickle
import os
from kramp_feature.normal_feature_extractor import NormalFeatureExtractor
from kay_feature..feature_extractor import EmbeddingFeatureExtractor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import random
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
#import seaborn as sns

seed = 42


# data = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb'))
# with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/actual_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'wb') as f:
#     pickle.dump(data.get_dataframe(), f)
#     f.close()


# X_DF = data.get_dataframe()
# Y_DF = X_DF['label'].to_frame()
# with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/X_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'wb') as f:
#     pickle.dump(X_DF, f)
#     f.close()

# with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/Y_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'wb') as f:
#     pickle.dump(Y_DF, f)
#     f.close()

# print(X_DF.columns.values)
# print(Y_DF.columns.values)

# data = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/actual_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).text.values.tolist()

# if os.path.exists('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_X_normal_chunks.pkl'):
#     X_normal = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_X_normal_chunks.pkl', 'rb'))
# else:
#     normal_feature_extractor = NormalFeatureExtractor()
#     normal_feature_extractor.fit(data)
#     X_normal = normal_feature_extractor.transform(data)
#     with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/out_of_domain_X_normal_chunks.pkl', 'wb') as f:
#         pickle.dump(X_normal, f)
#         f.close()

data = pickle.load(open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/actual_out_of_domain_experiment_dataframe_clean_chunks.pkl', 'rb')).text.values.tolist()

kay_feature_extractor = EmbeddingFeatureExtractor()
kay_feature_extractor.fit(data)
kay_feature
with open('/csse/research/NativeLanguageID/mthesis-phonological/experiment/pickles/pickled_datasets/seed_42/kay_feature_extractor_test.pkl', 'wb') as f:
    pickle.dump(kay_feature, f)
    f.close()



