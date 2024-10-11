# date: 20241011
# author: DOCT-Y
# github repository: https://github.com/DOCT-Y/RCC-T3a_Invasion-radiomics
# This code is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# You may not use this code for commercial purposes, modify it, or use it in other research without explicit permission from the author.
# For more details, see https://creativecommons.org/licenses/by-nc-nd/4.0/


import pandas as pd
from radiomics import featureextractor
import sklearn

from itertools import accumulate
import json
import os
import pickle
import time
from tqdm import tqdm


sklearn.set_config(transform_output='pandas')


class SignatureExtractor:
    def __init__(self, param_dir):
        self._extractor = featureextractor.RadiomicsFeatureExtractor(param_dir)

    def extract(self, dataset_dir, image_name, mask_names):
        cases = os.listdir(dataset_dir)
        cases.sort()

        output = []
        for mask_prefix, mask_name in mask_names.items():
            features = []
            for case in tqdm(cases):
                image_dir = os.path.join(dataset_dir, case, image_name)
                mask_dir = os.path.join(dataset_dir, case, mask_name)
                case_data = {'case_id':case}
                signature = self._extractor.execute(image_dir, mask_dir)
                case_data.update(signature)
                features.append(case_data)
            
            features = pd.DataFrame(features)
            features = features.set_index('case_id')
            features = features.rename(columns={i:f'{mask_prefix}_{i}' for i in features.columns})

            output.append(features)
        
        if len(output) > 1:
            for i in accumulate(output, lambda x, y: x.merge(y, how='left', on='case_id')):
                results = i
        else:
            results = output[0]

        return results


class ModelInference:
    def __init__(self, config_dir):
        with open(config_dir, 'r') as config_file:
            self.config = json.load(config_file)

        self.signature_extractor = SignatureExtractor(self.config['extraction parameter'])
        self.load_pretrained(self.config['model'])

    def load_pretrained(self, pkl_dir):
        with open(pkl_dir, 'rb') as model_file:
            pipeline = pickle.load(model_file)

        self.pipeline = pipeline
        self.input_features = pipeline[0].feature_names_in_
        self.final_features = pipeline[-2].get_feature_names_out()

    def predict(self):
        features = self.signature_extractor.extract(self.config['dataset'], self.config['image'], self.config['masks'])
        features = features[self.input_features]
        prob = self.pipeline.predict_proba(features)[:, 1]
        output = features[self.final_features].copy()
        output['prob'] = prob
        cutoff = self.config.get('cutoff')
        if cutoff is not None:
            pred = (prob >= self.config['cutoff']).astype(int)
            output['pred'] = pred

        output.to_csv(os.path.join(self.config['output'], f'prediction_results-{time.time():.0f}.csv'))