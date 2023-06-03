import os
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from Utils import LOGGER, log_line


# ========================================
# 训练ranker model，对召回的aid排序
# ========================================
def Model(candidates, type, train_file, cfg):
    log_line()

    # 转换数据类型
    cols_to_float32 = []
    for i in zip(candidates.columns, candidates.dtypes):
        if i[1].__name__ == 'Float64':
            cols_to_float32.append(i[0])

    candidates = candidates.with_columns(pl.col(cols_to_float32).cast(pl.Float32))  # Float16替换Float32

    # 对用户排序
    candidates = candidates.sort(by='session')  

    # 'y_clicks', 'y_carts', 'y_orders' 标签列为0或1
    non_neg = candidates.groupby('session').agg(
        [pl.col(type).max().alias('is_positive'), pl.col(type).min().alias('is_negative')]) # 0 1标签
    non_neg = non_neg.filter(pl.col('is_positive') > 0).filter(pl.col('is_negative') == 0).select(pl.col('session'))  
    candidates = candidates.join(non_neg, on='session', how='inner')
    del non_neg

    candidates = candidates.sample(frac=1.0, shuffle=True, seed=42)

    candidates = candidates.with_columns(
        pl.col('session').cumcount().over(['session', type]).alias('rank'))
    candidates = candidates.filter((pl.col(type) == 1) | (pl.col('rank') <= cfg.max_negative_candidates)).drop('rank')

    candidates = candidates.sort(by='session')

    train_baskets = candidates.groupby(['session']).agg(pl.col('aid').count().alias('basket'))  
    train_baskets = train_baskets.select(pl.col('basket'))
    train_baskets = train_baskets.to_numpy().ravel() 


    y = candidates.select(pl.col(type)).to_numpy().ravel()
    candidates = candidates.select(
        pl.col(cfg.features)).to_numpy().astype(np.float32)

    LOGGER.info(f'===== start training lgb model for {type} =====')
    train_dataset = lgb.Dataset(data=candidates, label=y, group=train_baskets)
    model = lgb.train(train_set=train_dataset,
                      params=cfg.model_param,
                      feature_name=cfg.features)
    # 保存模型
    model.save_model(f'{cfg.output_path}/{train_file}_{type}_model.lgb')

    if not os.path.exists(f'{cfg.output_path}/feature_importance'):
        os.makedirs(f'{cfg.output_path}/feature_importance')
    # 导出特征重要性表
    feature_importance = pd.DataFrame({'feature': cfg.features, 'importance': model.feature_importance()})
    feature_importance.to_csv(f'{cfg.output_path}/feature_importance/{train_file}_{type}_feature_importance.csv', index=False)

    # 画出特征重要性图
    fig,ax = plt.subplots(figsize=(20,8))
    lgb.plot_importance(model, ax=ax) # importance_type='gain',
    plt.savefig(f'{cfg.output_path}/feature_importance/{train_file}_{type}_feature_importance.png', bbox_inches='tight')
    return model

