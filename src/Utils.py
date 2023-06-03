import os
import time
import random
import psutil
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


from Config import CFG


# ====================================================
# log
# ====================================================
def get_logger(filename=CFG.output_path+'/train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

if not os.path.exists(CFG.output_path):
    os.makedirs(CFG.output_path)
####
LOGGER = get_logger()

def create_dir():
    if not os.path.exists(CFG.output_path):
        os.makedirs(CFG.output_path)
    if not os.path.exists(CFG.features_path):
        os.makedirs(CFG.features_path)
    if not os.path.exists(CFG.data_path + '/raw/'):
        os.makedirs(CFG.data_path + '/raw/')
    if not os.path.exists(CFG.data_path + '/cv/'):
        os.makedirs(CFG.data_path + '/cv/')
    if not os.path.exists(CFG.data_path + '/covisit/'):
        os.makedirs(CFG.data_path + '/covisit/')
        
   
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def print_line():
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit*50 + suffix)
def log_line():
    prefix, unit, suffix = "#", "--", "#"
    LOGGER.info(prefix + unit*50 + suffix)

def Print_Parameter(obj): 
  LOGGER.info('\n************************************\n')
  LOGGER.info('\n'.join(['%s : %s' % item for item in obj.__dict__.items() if '__' not in item[0]]))
  LOGGER.info('\n************************************\n')

def get_metric():
    p = psutil.Process(os.getpid())
    m: float = p.memory_info()[0] / 2.0**30
    per: float = psutil.virtual_memory().percent
    LOGGER.info(f"{m:.1f}GB({per:.1f}%)")





