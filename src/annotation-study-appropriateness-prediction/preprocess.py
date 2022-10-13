import os
import sys
import random
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

random_states = [random.randint(0, 2**32) for i in range(5)]

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

DIMS = [
    'Appropriateness',
    'Emotions',
    'Emotional Intensity',
    'Emotional Typology',
    'Commitment',
    'Committed Seriousness',
    'Committed Openness',
    'Clarity',
    'Clear Position',
    'Clear Relevance',
    'Clear Organization',
    'Other',
    'Orthography',
    'Not classified'
]

data_dir = '../../data/'
df = pd.read_csv(data_dir+'appropriateness-corpus/appropriateness_corpus_conservative.csv')

X = df.post_id.values
y = df[DIMS].values
for j, random_state in enumerate(random_states):
    i = 0
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, tmp_index in mskf.split(X, y):
        split_dict = {}
        X_tmp, X_test = X[train_index], X[tmp_index]
        y_tmp, y_test = y[train_index], y[tmp_index]
        for post_id in X_test:
            split_dict[post_id] = 'TEST'

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
        for valid_index, test_index in msss.split(X_tmp, y_tmp):
            X_train, X_valid = X_tmp[valid_index], X_tmp[test_index]
            for post_id in X_train:
                split_dict[post_id] = 'TRAIN'
            for post_id in X_valid:
                split_dict[post_id] = 'VALID'
        df['fold{}.{}'.format(j,i)] = df['post_id'].apply(lambda x: split_dict[x])
        i+=1
df['arg_issue'] = df[['issue','post_text']].apply(lambda x: ' '.join(x), axis = 1)
df.to_csv(data_dir+'appropriateness-corpus/appropriateness_corpus_conservative_w_folds.csv')
