# RCC-T3a_Invasion-radiomics

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)  
![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)

## License

This code is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## Introduction

This is the official repository for our recent work: [CT-based radiomic model for identifying pathological T3a upstaging in renal cell carcinoma: model development and multi-source validation]. This repository contains ready-to-use codes for inference of the proposed models.


## Key Features

- **Features 1**: The training dataset contains sufficient patients with renal cell carcinoma (n = 999) and covers a wide spectrum of disease phenotype （common and rare RCC subtypes, all ISUP/fuhrman grades, all pathological T stages）.
- **Features 2**: The predictive performance of the models were externally evaluated in data from two medical centers and four TCIA datasets.
- **Features 3**: The morphology model improved the performance of junior radiologists.

## Get Started

### Main Requirements    
> numpy  
> pandas  
> pyradiomics  
> SimpleITK  
> scikit-learn  
> tqdm  

### Data preparation
NIfTI-formatted image and mask files were used for feature extraction and model prediction. Please organize the image and mask files in the following structure.
```
test_data/
├── Dataset001
    │── patient_1
    │    ├── image.nii.gz
    │    ├── tumor_mask.nii.gz
    │    └── peritumor_mask.nii.gz
    │── patient_2
    │    ├── image.nii.gz
    │    ├── tumor_mask.nii.gz
    │    └── peritumor_mask.nii.gz
    └── patient_n
         ├── image.nii.gz
         ├── tumor_mask.nii.gz
         └── peritumor_mask.nii.gz
```
### Config preparation
During inference, the configurations are loaded from a `.json` file. The structure of the file is as follows:

```json
{
    "model":"your\\path\\to\\model.pkl", 
    "extraction parameter":"your\\path\\to\\extraction.yaml", 
    "dataset":"your\\path\\to\\dataset", 
    "image":"image.nii.gz", 
    "masks": {
        "tumor":"tumor.nii.gz", 
        "peritumor":"peritumor.nii.gz"
    }, 
    "cutoff":0.3582773836548992, 
    "output":"your\\path\\to\\results\\"
}
```

Each of the parameters is explained as follows:

- `model`: The path to the pickle file of trained model.
- `extraction parameter`: The path to the yaml file of radiomics feature extraction configurations.
- `dataset`: The path to the dataset folder.
- `image`: The name of the image file. Both `.nii` and `.nii.gz` are supported.
- `masks`: The names of the mask files. The key is the prefix added to the feature name during extraction, and the value is the mask file name.
- `cutoff`: The cutoff value that determines whether a case is classified as T3a invasion positive or negative.
- `output`: The output folder to hold the output `.csv` file.

### Inference
```python
from inference import ModelInference

if __name__ == '__main__':
    config_dir = '.\\inference_config.json'
    model = ModelInference(config_dir)
    model.predict()
```
1. import `ModelInference` class from `inference` module.
2. initialize an instance of the `ModelInference` class, passing the configuration file path (`inference_config.json`) as an argument.
3. call the `predict` method of the `ModelInference` class.

## Model Training and validation
1. use `SignatureExtractor` class in `inference` and `.yaml` configuration file to extract the radiomic features.
2. merge the tabular radiomics data with `label` column and save it as `.csv` file.
3. run `train` in `train_validation` module to train the model and save it as `.pkl` file.
4. run `internal validaion` in `train_validation` module to internal validate the model using nested-cross validation. The outer loop is 200 repeats 5 fold cross validation, and the inner loop is 20 repeats 5 fold cross validation. For a detailed explanation of nested-cross validation, please refer to [A Guide to Cross-Validation for Artificial Intelligence in Medical Imaging](https://doi.org/10.1148/ryai.220232).
5. run `external_validation` in `train_validation` module to external validate the model in holdout datasets.
6. during internal validation and external validation, the raw predictions were saved as `.csv` files.
