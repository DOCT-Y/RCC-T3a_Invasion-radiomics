# date: 20241011
# author: DOCT-Y
# github repository: https://github.com/DOCT-Y/RCC-T3a_Invasion-radiomics
# This code is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# You may not use this code for commercial purposes, modify it, or use it in other research without explicit permission from the author.
# For more details, see https://creativecommons.org/licenses/by-nc-nd/4.0/

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

import os
import pickle


set_config(transform_output='pandas')


class PearsonSelector(TransformerMixin):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold
    
    def fit(self, X, y=None):
        features = X.columns
        ranker = SelectKBest(score_func=f_classif, k='all')

        ranker.fit(X, y)
        statistic_scores = ranker.scores_
        score_dict = dict(zip(features, statistic_scores))

        corr = X.corr(method='pearson')
        [xx, yy] = np.where((abs(corr) > self.threshold) & (abs(corr) < 1))

        del_ = []
        for i, j in zip(xx, yy):
            if i < j: # the matrix is symmetric and only half of the points are compared to increase code speed.
                var1 = corr.index[i]
                var2 = corr.columns[j]
                if score_dict[var1] < score_dict[var2]:
                    del_.append(var1)
                else:
                    del_.append(var2)
            
        self.feature_names_in_ = X.columns.values
        self.n_features_in_ = len(self.feature_names_in_)
        self.feature_names_out_ = [feature for feature in features if feature not in del_] # avoid duplicates in del_
        self.n_features_out_ = len(self.feature_names_out_)

        return self

    def transform(self, X):
        return X.loc[:, self.feature_names_out_]


def make_pipeline(random_state, k):
    pipeline = Pipeline([
        ('VarianceThreshold', VarianceThreshold(threshold=0)),
        ('RobustScaler', RobustScaler()), 
        ('SelectKBest', SelectKBest(score_func=f_classif, k=k)), 
        ('PearsonSelector', PearsonSelector(threshold=0.85)), 
        ('LASSOSelector', SelectFromModel(LassoCV(max_iter=2000, cv=RepeatedStratifiedKFold(n_repeats=20, n_splits=5, random_state=random_state), random_state=random_state, n_jobs=-1), threshold=1e-8)), 
        ('LogisticRegression', LogisticRegression(max_iter=1000))
        ])
    return pipeline

    
def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        RuntimeWarning(
            "cumsum was found to be unstable: its last element does not correspond to sum"
            )
    return out


def cutoff_analysis(y_true, y_prob, pos_label=1):
    y_true = np.ravel(y_true)
    y_prob = np.ravel(y_prob)
    y_true = y_true == pos_label

    desc_prob_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_prob = y_prob[desc_prob_indices]
    y_true = y_true[desc_prob_indices]

    distinct_value_indices = np.where(np.diff(y_prob))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_prob[threshold_idxs]

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[max(thresholds[0], 1-1e-10), thresholds]

    total_p = y_true.sum()
    total_n = y_true.size -total_p

    youden = tps/total_p + (total_n-fps)/total_n - 1
    max_cutoff_ind = np.argmax(youden)
    
    return thresholds[max_cutoff_ind]


def train(data_dir, model_dir, **kwargs):
    data_df = pd.read_csv(data_dir, index_col=0)
    label = data_df.pop('label')

    pipeline = make_pipeline(random_state=42, **kwargs) # 42 is a magic number in machine learning
    pipeline.fit(data_df, label)

    with open(model_dir, 'wb') as model_file:
        pickle.dump(pipeline, model_file)


def internal_validation(data_dir, output_dir, **kwargs):
    data_df = pd.read_csv(data_dir, index_col=0)
    label = data_df.pop('label')
    with tqdm(total=1000, desc=f'Experiment', unit='turn') as pbar:
        for turn_id in range(200):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=turn_id)
            for fold_id, (train_index, test_index) in enumerate(skf.split(data_df, label)):
                train_X = data_df.iloc[train_index].copy()
                test_X = data_df.iloc[test_index].copy()
                train_y = label.iloc[train_index].copy()
                test_y = label.iloc[test_index].copy()

                pipeline = make_pipeline(turn_id, **kwargs)
                pipeline.fit(train_X, train_y)

                train_prob = pipeline.predict_proba(train_X)[:, 1]
                cutoff = cutoff_analysis(train_y, train_prob)
                train_pred = (train_prob >= cutoff).astype(int)

                test_prob = pipeline.predict_proba(test_X)[:, 1]
                test_pred = (test_prob >= cutoff).astype(int)

                train_result = pd.DataFrame({'prob':train_prob, 'pred':train_pred}, index=train_X.index)
                test_result = pd.DataFrame({'prob':test_prob, 'pred':test_pred}, index=test_X.index)

                train_result.to_csv(os.path.join(output_dir, f'train_{turn_id}_{fold_id}.csv'))
                test_result.to_csv(os.path.join(output_dir, f'test_{turn_id}_{fold_id}.csv'))

                pbar.update(1)


def external_validation(train_data_dir, output_dir, external_data_dirs, **kwargs):
    train_data_df = pd.read_csv(train_data_dir, index_col=0)
    train_label = train_data_df.pop('label')
    for external_data_dir in external_data_dirs:
        dataset_name = os.path.basename(external_data_dir).split('.')[0]
        external_data_df = pd.read_csv(external_data_dir, index_col=0)
        external_label = external_data_df.pop('label')

        pipeline = make_pipeline(random_state=42, **kwargs) # 42 is a magic number in machine learning

        pipeline.fit(train_data_df, train_label)
        train_prob = pipeline.predict_proba(train_data_df)[:, 1]
        cutoff = cutoff_analysis(train_label, train_prob)

        external_prob = pipeline.predict_proba(external_data_df)[:, 1]
        external_pred = (external_prob >= cutoff).astype(int)

        result = pd.DataFrame({'prob':external_prob, 'pred':external_pred}, index=external_data_df.index)
        result.to_csv(os.path.join(output_dir, f'prediction_{dataset_name}.csv'))


if __name__ == '__main__':
    for feature_type in ['M', 'IT', 'M-IT', 'PIT', 'M-PIT', 'M-IT-PIT']:
        train_data_dir = f'.\\experiments\\{feature_type}\\data\\center-1_features.csv'
        external_data_dirs = [
            f'.\\experiments\\{feature_type}\\bi-center-validation_features.csv', 
            f'.\\experiments\\{feature_type}\\TCIA_features.csv'
        ]
        
        if feature_type == 'M':
            k = 'all'
        else:
            k = 100

        train(train_data_dir, f'.\\results\\train\\{feature_type}.pkl', k=k)
        internal_validation(train_data_dir, '.\\results\\internal validation', k=k)
        external_validation(train_data_dir, '.\\results\\external validation', external_data_dirs, k=k)