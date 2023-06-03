import os
import gc
import math
import pickle
import random
import time
from Utils import get_metric
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from pathlib import Path
from time import sleep



try:
    from Config import CFG
    from Dataset import data_processing
    from Candidates import generate_candidates
    from Features import generate_features
    from Model import Model
    from Evaluate import evaluate
    from Recommend import generate_labels, recommendate_aid
    from Utils import LOGGER, Print_Parameter, log_line, create_dir, seed_everything

except ImportError as e:
    print(f'Error importing required modules: {e}')
    exit(1)
    
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = str(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)

# ====================================================
# train loop
# ====================================================
def train_loop(folds, cfg):
    get_metric()
    
    for stage, fold in enumerate(folds):
        sleep(2)

        log_line()
        LOGGER.info(f'\n{"="*22} Stage {stage+1}: {folds[stage]} {"="*22}\n')

        train_file = fold[0]
        valid_file = fold[1]

        # ----------------------------------------- 一. 训练阶段 ----------------------------------------- 
        # 1. 为当前训练集生成候选
        candidates_train = generate_candidates(train_file, cfg)
        gc.collect()

        # 2. 为当前训练集添加标签
        candidates_train = generate_labels(candidates_train, train_file, cfg)
        gc.collect()

        # 3. 为当前训练集做特征
        candidates_train = generate_features(candidates_train, train_file, cfg)
        gc.collect()

        # 4. 训练模型 LGB Reranker
            # 4.1 训练click
        click = candidates_train
        click_model = Model(click, 'y_clicks', train_file, cfg)
        gc.collect()

            # 4.2 训练cart
        cart = candidates_train
        cart_model = Model(cart, 'y_carts', train_file, cfg)
        gc.collect()

            # 4.3 训练order
        order = candidates_train
        order_model = Model(order, 'y_orders', train_file, cfg)
        gc.collect()

 

        # ----------------------------------------- 二. 验证阶段 ----------------------------------------- 
        # 我们需要分别对验证集（valid3，valid4）和测试集（test）各生成一个候选物品集
        candidates_valid = generate_candidates(valid_file, cfg)

        if valid_file != 'test':
            candidates_valid = generate_labels(candidates_valid, fold[1], cfg)
            gc.collect()

        candidates_valid = generate_features(candidates_valid, fold[1], cfg)
        gc.collect()

        reco_clicks = recommendate_aid(candidates_valid, 'clicks', click_model, cfg)
        gc.collect()
        reco_carts = recommendate_aid(candidates_valid, 'carts', cart_model, cfg)
        gc.collect()
        reco_orders = recommendate_aid(candidates_valid, 'orders', order_model, cfg)
        gc.collect()
        
        all_recommendations = pl.concat([reco_clicks, reco_carts, reco_orders])
        if valid_file == 'test':
            all_recommendations.write_csv(f'{cfg.output_path}/{cfg.version}_submission.csv') 
            assert len(all_recommendations) == 5015409, 'Wrong length of submission'
        else:
            all_recommendations.write_csv(f'{cfg.output_path}/{valid_file}.csv') 

        gc.collect()
        del reco_clicks, reco_carts, reco_orders

        # 验证
        if valid_file != 'test':
            log_line()
            LOGGER.info(f'\n===== start validation : {valid_file} =====')
            valid_gt   = f'{cfg.data_path}/cv/{valid_file}_labels.jsonl'
            valid_pred = f'{cfg.output_path}/{valid_file}.csv'
            score = evaluate(valid_gt, valid_pred) 
            LOGGER.info(f" Scores: {score}")
            scores.append(score)
            gc.collect()

    return scores





# ====================================================
# main
# ====================================================
if __name__ == '__main__':
    # 记录开始时间
    start_time = time.time()        
    
    create_dir()
    seed_everything(CFG.seed)
    
    # 记录当前实验的参数配置 
    Print_Parameter(CFG)

    log_line()
    data_processing(CFG)

    # ========
    log_line()
    LOGGER.info(f'\n{"="*22} {CFG.version} started training using cross-validation !!! {"="*22}')
    scores = []
    score_avg = train_loop(CFG.folds, CFG)
    if len(scores) > 0:
        score_avg = pl.DataFrame(scores).mean()
    
    LOGGER.info(f'\n{"="*25} Local CV  {"="*25}\n')
    LOGGER.info(f'{score_avg}\n')

    # 记录结束时间
    end_time = time.time()
    # 计算时间差
    elapsed_time = end_time - start_time 

    LOGGER.info(f'\n{"="*35} {CFG.version} Done!!! Spent time {elapsed_time/3600:.2f} h {"="*35}\n')

