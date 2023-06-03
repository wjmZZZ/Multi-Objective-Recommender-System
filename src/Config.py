#=================参数设置=================
import os

class CFG:
   version = 'otto'    
   version_explanation = ''

   folds = [ ['valid3', 'valid4'], ['valid4', 'test'] ]   
   max_negative_candidates = 20
   top = 20 
   seed=42
   train_files = ['train12', 'train123']
   valid_files = ['valid3', 'valid4']

   #===== Path =====
   root_path = os.getcwd()
   output_path = f'{root_path}/outputs/{version}' 
   data_path = f'{root_path}/data'
   features_path = f'{root_path}/data/features'
   log_path = f'{root_path}/logs'


   model_param = {'objective': 'lambdarank',
                   'learning_rate': 0.1,
                   'lambdarank_truncation_level': 15,
                   'verbose': -1,
                   'n_jobs': -1,
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'seed': seed,
                   } 
    