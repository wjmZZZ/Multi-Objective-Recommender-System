import numpy as np
import polars as pl
from tqdm import tqdm
from Utils import LOGGER, log_line

# ========================================
# 添加标签，'y_clicks', 'y_carts', 'y_orders'
# ========================================
def generate_labels(candidates, data_name, cfg):
    log_line()

    file = f'{cfg.data_path}/cv/{data_name}_labels.parquet'
    labels = pl.read_parquet(file)

    print(f'===== generate_labels {cfg.data_path}cv/{data_name}_labels.parquet =====')

    labels = labels.with_columns(pl.lit(1).alias('one'))
    labels = labels.pivot(values='one', columns='type',index=['session', 'aid'])
    labels.columns = ['session', 'aid', 'y_clicks', 'y_carts', 'y_orders']
    labels = labels.select([pl.col(['session', 'aid']).cast(pl.Int32), pl.col(['y_clicks', 'y_carts', 'y_orders']).cast(pl.UInt8)])
    candidates = candidates.join(labels, on=['session', 'aid'], how='left')
    candidates = candidates.with_columns(pl.col(['y_clicks', 'y_carts', 'y_orders']).fill_null(0))
    return candidates






'''
根据给定的候选、类型、模型，为每个session生成推荐商品。
将候选项按session分组，对于每个session，使用模型对候选项进行打分，根据得分对候选项排序，并选择前k个作为推荐项。
'''
def recommendate_aid(candidates, type, model, cfg, k=20):
    log_line()

    LOGGER.info(f'    ======= Predict the score of each candidate aid for {type} =======    ')

    batch_size = 100_000
    batch_num = 0
    sessions = candidates.select(pl.col('session').unique())
    sessions_num = sessions.shape[0]
    CHUNKS = 20
    chunk_size = int(np.ceil(sessions_num) / CHUNKS)
    
    recommendations = []

    for i in tqdm(range(CHUNKS), desc='Inference', total=CHUNKS):
        if i == CHUNKS - 1: # last chunk
            batch_sessions = sessions[i * chunk_size:]
        else:
            batch_sessions = sessions[i * chunk_size : (i + 1) * chunk_size]
            

        batch_candidates = candidates.join(batch_sessions, on='session')

        test_data = batch_candidates.select(pl.col(cfg.features)).to_numpy().astype(np.float32)
        scores = model.predict(test_data)
        batch_candidates_scored = batch_candidates.select(pl.col(['session', 'aid']))
        batch_candidates_scored = batch_candidates_scored.with_columns(pl.lit(scores).alias('score'))

        batch_candidates_scored = batch_candidates_scored.sort(by=['session', 'score'], reverse=[False, True])

        batch_recommendations = batch_candidates_scored.groupby('session').agg(pl.col('aid'))

        batch_recommendations = batch_recommendations.select([(pl.col('session').cast(str) + pl.lit(f'_{type}')).alias('session_type'), \
                                                               pl.col('aid').apply(lambda x: ' '.join([str(i) for i in x[:k]])).alias('labels')])

        recommendations.append(batch_recommendations)
        batch_num += 1

    recommendations = pl.concat(recommendations)
    
    return recommendations