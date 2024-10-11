# date: 20241011
# author: DOCT-Y
# github repository: https://github.com/DOCT-Y/RCC-T3a_Invasion-radiomics
# This code is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.
# You may not use this code for commercial purposes, modify it, or use it in other research without explicit permission from the author.
# For more details, see https://creativecommons.org/licenses/by-nc-nd/4.0/


from inference import ModelInference


if __name__ == '__main__':
    config_dir = '.\\inference_config.json'
    model = ModelInference(config_dir)
    model.predict()