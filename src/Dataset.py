import os
import cudf
import pickle
import random
import numba as nb
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from pathlib import Path
from scipy import sparse
from gensim.models import Word2Vec

from Utils import LOGGER, log_line
from Evaluate import calc_recall
from preprocess.train_val_split import cv_split
from preprocess.parquet_2_npz import parquet_2_npz
from preprocess.json_2_parquet import json_2_df, label_json_2_df
from generate.gene_covisit import gene_covisit_weight, inference
from generate.gene_itemSimMatrix import ItemMatrix_fn, ItemSimilarityMatrix_fn, gene_itemSimMatrix
from generate.gene_hot_items import carts_recall, clicks_recall, orders_recall

from pandarallel import pandarallel
# Initialization
pandarallel.initialize(nb_workers=8, progress_bar=True)

# ========================================
# 1. 数据预处理
# ========================================
def data_processing(cfg):
    LOGGER.info(f'\n\n{"="*25}  Data Preprocess   {"="*25}')
    log_line()

    processer = DataPreprocessor(cfg.seed)
    processer.cross_validation(cfg)
    processer.json_2_parquet(cfg)
    processer.parquet_2_npz(cfg)
    processer.parquet_2_csv(cfg)

    log_line()
    generator = DataGenerator(cfg.seed)

    generator.covisit(cfg, name='valid3', top_k=cfg.top)
    generator.covisit(cfg, name='valid4', top_k=cfg.top)
    generator.covisit(cfg, name='test', top_k=cfg.top)
    
    
    generator.word2vec(cfg, name='valid3', window_size = 10, add_type=True, min_count = 5)
    generator.word2vec(cfg, name='valid4', window_size = 10, add_type=True, min_count = 5)
    generator.word2vec(cfg, name='test', window_size = 10, add_type=True, min_count = 5)

    generator.word2vec(cfg, name='valid3', window_size = 20, add_type=True, min_count = 5)
    generator.word2vec(cfg, name='valid4', window_size = 20, add_type=True, min_count = 5)
    generator.word2vec(cfg, name='test', window_size = 20, add_type=True, min_count = 5)

    generator.hot_item(cfg, name='valid3', top=10)
    generator.hot_item(cfg, name='valid4', top=10)
    generator.hot_item(cfg, name='test', top=10)

    generator.item_cf(cfg, name='valid3')
    generator.item_cf(cfg, name='valid4')
    generator.item_cf(cfg, name='test')

    generator.item_cf_prepare(cfg, name='valid3')
    generator.item_cf_prepare(cfg, name='valid4')
    generator.item_cf_prepare(cfg, name='test')

    generator.item_cf_feature(cfg, name='valid3')
    generator.item_cf_feature(cfg, name='valid4')
    generator.item_cf_feature(cfg, name='test')
# ========================================
# 数据预处理模块
# ========================================
class DataPreprocessor:
    def __init__(self, seed=42):
        self.seed = seed
        random.seed(self.seed)

    # ========================================
    # 划分训练集和验证集，30min~
    # ========================================
    '''
    all_train  : before=2022-08-01 06:00:00 - 2022-08-29 05:59:59
    all_test  : before=2022-08-29 06:00:00 - 2022-09-05 05:59:51
    train12 : before=2022-08-01 06:00:00 - 2022-08-15 05:59:59
    valid3  : before=2022-08-15 06:00:00 - 2022-08-22 05:59:56
    train123 : before=2022-08-01 06:00:00 - 2022-08-22 05:59:59
    valid4  : before=2022-08-22 06:00:00 - 2022-08-29 05:59:56
    '''
    def cross_validation(self, cfg):
        LOGGER.info(f' ======= Splitting the Training set and Validation set ======= ')
        
        train_path = f'{cfg.data_path}/raw/all_train.jsonl'
        output_path = f'{cfg.data_path}/cv'
        val_days = 7  # 最后7天作为验证集
        seed = 42
        train_name = 'train123'
        valid_name = 'valid4'
        train_path = Path(train_path)
        output_path = Path(output_path)
        if not os.path.isfile(f'{output_path}/{train_name}.jsonl'):
            cv_split(train_path, output_path, val_days, seed, train_name, valid_name)
            LOGGER.info(f' ======= 1. The {train_name} and {valid_name} were successfully split and saved ======= ')
        
        train_path = f'{cfg.data_path}/cv/train123.jsonl'
        output_path = f'{cfg.data_path}/cv'
        val_days = 7  # 最后7天作为验证集
        seed = 42
        train_name = 'train12'
        valid_name = 'valid3'
        train_path = Path(train_path)
        output_path = Path(output_path)
        
        if not os.path.isfile(f'{output_path}/{train_name}.jsonl'):
            cv_split(train_path, output_path, val_days, seed, train_name, valid_name)
            LOGGER.info(f' ======= 2. The {train_name} and {valid_name} were successfully split and saved ======= ')

    # 12 min~
    def json_2_parquet(self, cfg):
        LOGGER.info(f' ======= Converting jsonl files to parquet files =======')

        # all_train, all_test
        if not os.path.isfile(f'{cfg.data_path}/cv/all_train.parquet'):
            file_list = ['all_train', 'test']
            path_list = [f'{cfg.data_path}/raw/{file}.jsonl' for file in file_list]
            out_path_list = [f'{cfg.data_path}/cv/{file}.parquet' for file in file_list]
            for input_path, out_path in zip(path_list, out_path_list):
                df = json_2_df(input_path)
                df.to_parquet(out_path)


        if not os.path.isfile(f'{cfg.data_path}/cv/{cfg.valid_files[-1]}_labels.parquet'):
            path_list = [f'{cfg.data_path}/cv/{file}.jsonl' for file in cfg.train_files + cfg.valid_files]
            out_path_list = [f'{cfg.data_path}/cv/{file}.parquet' for file in cfg.train_files + cfg.valid_files]
            for input_path, out_path in zip(path_list, out_path_list):
                df = json_2_df(input_path)
                df.to_parquet(out_path)


            path_list = [f'{cfg.data_path}/cv/{file}_labels.jsonl' for file in cfg.valid_files]
            out_path_list = [f'{cfg.data_path}/cv/{file}_labels.parquet' for file in cfg.valid_files]
            for input_path, out_path in zip(path_list, out_path_list):
                df = label_json_2_df(input_path)
                df.to_parquet(out_path)
            
            # 转换label格式
            for file in out_path_list:
                valid_label = pd.read_parquet(file)
                # 处理clicks
                df_clicks = valid_label[['session', 'clicks']]
                df_clicks = df_clicks[df_clicks['clicks'] != -1].rename(columns={'clicks': 'aid'})
                df_clicks['type'] = 0

                # 处理carts
                df_carts = valid_label[['session', 'carts']]
                df_carts = df_carts.explode('carts').dropna().rename(columns={'carts': 'aid'})
                df_carts['type'] = 1

                # 处理orders
                df_orders = valid_label[['session', 'orders']]
                df_orders = df_orders.explode('orders').dropna().rename(columns={'orders': 'aid'})
                df_orders['type'] = 2

                # 按照session进行拼接
                df = pd.concat([df_clicks, df_carts, df_orders], axis=0, ignore_index=True, sort=False)
                # 按照session排序
                df = df.sort_values('session')
                # 重置索引
                df = df.reset_index(drop=True)
                df.to_parquet(file)        
        
            LOGGER.info(f' ======= Successfully convert jsonl file to parquet file ======= ')

    # 1min~
    def parquet_2_npz(self, cfg):
        LOGGER.info(f' ======= Converting parquet files to npz files ======= ')
        # all_train , test ， train123, valid4, train12, valid3
        if not os.path.isfile(f'{cfg.data_path}/cv/{cfg.valid_files[-1]}.npz'):
            file_name = ['all_train', 'test'] + cfg.train_files + cfg.valid_files
            path_list = [f'{cfg.data_path}/cv/{file}.parquet' for file in file_name]
            out_path_list = [f'{cfg.data_path}/cv/{file}.npz' for file in file_name]
            for input_path, out_path in zip(path_list, out_path_list):
                df = pd.read_parquet(input_path)
                ts_min = df.groupby('session').min()['ts']
                df['ts_min'] = df.session.map(ts_min)
                df['ts'] = df['ts'] - df['ts_min']
                df = df.loc[:, ['aid', 'ts', 'type']]
                df = df.to_numpy()
                np.savez(out_path, aids=df[:, 0].tolist(), ts=df[:, 1].tolist(), type=df[:, 2].tolist())
            
            LOGGER.info(f' ======= Successfully convert parquet file to npz file ======= ')
    

    def parquet_2_csv(self, cfg):
        LOGGER.info(f' ======= Converting parquet files to csv files ======= ')
        # all_train , test
        if not os.path.isfile(f'{cfg.data_path}/raw/all.csv'):
            df1 = pd.read_parquet(f'{cfg.data_path}/raw/all_train.parquet')
            df2 = pd.read_parquet(f'{cfg.data_path}/raw/test.parquet')
            df1 = df1.astype(np.int64)
            df1 = df1.groupby('session').agg(Min=('ts', np.min),Count=('ts', np.count_nonzero))
            df1.columns = ['start_time', 'length']
            df1.reset_index(inplace=True, drop=False)
            df1.to_csv(f'{cfg.data_path}/raw/all_train.csv', index = False)

            df2 = df2.astype(np.int64)
            df2 = df2.groupby('session').agg(Min=('ts', np.min),Count=('ts', np.count_nonzero))
            df2.columns = ['start_time', 'length']
            df2.reset_index(inplace=True, drop=False)
            df2.to_csv(f'{cfg.data_path}/raw/all_test.csv', index = False)
            
            df = pd.concat([df1, df2]).reset_index(drop = True)
            df.to_csv(f'{cfg.data_path}/raw/all.csv', index = False)
        
        if not os.path.isfile(f'{cfg.data_path}/cv/valid4.csv'):
            file_name = ['all_train', 'test'] + cfg.train_files + cfg.valid_files
            path_list = [f'{cfg.data_path}/cv/{file}.parquet' for file in file_name]
            out_path_list = [f'{cfg.data_path}/cv/{file}.csv' for file in file_name]
            for input_path, out_path in zip(path_list, out_path_list):
                df = pd.read_parquet(input_path)
                df = df.astype(np.int64)
                df = df.groupby('session').agg(Min=('ts', np.min),Count=('ts', np.count_nonzero))
                df.columns = ['start_time', 'length']
                df.reset_index(inplace=True, drop=False)
                df.to_csv(out_path, index = False)
            



# ========================================
# 数据生成模块
# ========================================
class DataGenerator:
    def __init__(self, seed=0):
        self.seed = seed
        random.seed(self.seed)
    
    ''' 生成共同访问矩阵 '''
    def covisit(self, cfg, name, top_k=20):
        LOGGER.info(f' ======= Genetate {cfg.data_path}/cv/{name}_covisit_top{top_k}.csv =======')
        if not os.path.isfile(f'{cfg.data_path}/cv/{name}_covisit_top{top_k}.csv'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'

            npz1 = np.load(f"{cfg.data_path}/cv/{train_name}.npz")
            npz2 = np.load(f"{cfg.data_path}/cv/{name}.npz")
            aids = np.concatenate([npz1['aids'], npz2['aids']])
            ts = np.concatenate([npz1['ts'], npz2['ts']])
            type = np.concatenate([npz1['type'], npz2['type']])
            del npz1, npz2

            df1 = pd.read_csv(f'{cfg.data_path}/cv/{train_name}.csv')
            df2 = pd.read_csv(f'{cfg.data_path}/cv/{name}.csv')
            df = pd.concat([df1, df2]).reset_index(drop = True)
            del df1, df2

            df["idx"] = np.cumsum(df.length) - df.length
            df["end_time"] = df.start_time + ts[df.idx + df.length - 1]

            # covisit 权重
            if not os.path.isfile(f'{cfg.data_path}/covisit/{name}_top{top_k}_type_weight.pkl'):          
                topks_type_weight, topks_time_weight = gene_covisit_weight(df, aids, ts, type, top_k, cfg)  # 42min~
                with open(f'{cfg.data_path}/covisit/{name}_top{top_k}_type_weight.pkl', "wb") as f:
                    pickle.dump(topks_type_weight, f)
                with open(f'{cfg.data_path}/covisit/{name}_top{top_k}_time_weight.pkl', "wb") as f:
                    pickle.dump(topks_time_weight, f)
            else:
                with open(f'{cfg.data_path}/covisit/{name}_top{top_k}_type_weight.pkl', "rb") as f:
                    topks_type_weight = pickle.load(f)
                with open(f'{cfg.data_path}/covisit/{name}_top{top_k}_time_weight.pkl', "rb") as f:
                    topks_time_weight = pickle.load(f)
               
            topks  = {}
            for mode in [0, 1]:  
                topk = nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.int64[:])
                if mode == 0:
                    for k,v in topks_type_weight.items():
                        topk[k] = np.array(v)[:]
                elif mode == 1:
                    for k,v in topks_time_weight.items():
                        topk[k] = np.array(v)[:]
                topks[mode] = topk


            

            result_clicks = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64[:])
            result_buy = nb.typed.Dict.empty(key_type=nb.types.int64, value_type=nb.types.int64[:])

            df_valid = pd.read_csv(f'{cfg.data_path}/cv/{name}.csv')
            start = len(df) - len(df_valid)
            end = len(df)
            PARALLEL = 5000
            TYPE_WEIGHT = np.array([1.0, 6.0, 3.0])
            for idx in tqdm(range(start, end, PARALLEL), desc=f" Genetate {name}_covisit_top{top_k}.csv "):
                row = df.iloc[idx : min(idx + PARALLEL, len(df))][['session', 'idx', 'length']].values
                inference(aids, type, row, result_clicks, result_buy, topks[1], topks[0], TYPE_WEIGHT, cfg.top)  # 4min~

            subs = []
            type_names = ["clicks", "carts", "orders"]
            for result, type in tqdm(zip([result_clicks, result_buy, result_buy], type_names), desc=f" Calculate each type", total=len(type_names)):
                sub = pd.DataFrame({"session_type": result.keys(), "labels": result.values()})
                sub.session_type = sub.session_type.astype(str) + f"_{type}"
                sub.labels = sub.labels.apply(lambda x: " ".join(x.astype(str)))
                # sub['type'] = type
                subs.append(sub)

            sub = pd.concat(subs).reset_index(drop=True)
            sub.to_csv(f'{cfg.data_path}/cv/{name}_covisit_top{top_k}.csv', index = False)
            if name != 'test':
                recall_score = calc_recall(f'{cfg.data_path}/cv/{name}_covisit_top{top_k}.csv', f'{cfg.data_path}/cv/{name}_labels.parquet')
                LOGGER.info(f'\n{"="*25} {name} Recall Score   {"="*25}\n')
                LOGGER.info(f'{recall_score}\n')
            LOGGER.info(f' ======= Successfully get {cfg.data_path}/cv/{name}_covisit_top{top_k}.csv file ======= ')


# ------------------------------------------------------------------------

    ''' 训练word2vec模型 '''
    def word2vec(self, cfg, name, window_size, add_type, min_count):
        if not os.path.exists(f'{cfg.data_path}/word2vec'):
            os.makedirs(f'{cfg.data_path}/word2vec')

        LOGGER.info(f' ======= Genetate {cfg.data_path}/word2vec/{name}_w2v_wind_{window_size}.model =======')
        if not os.path.isfile(f'{cfg.data_path}/word2vec/{name}_w2v_wind_{window_size}.model'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'

            df1 = pd.read_parquet(f'{cfg.data_path}/cv/{train_name}.parquet')
            df2 = pd.read_parquet(f'{cfg.data_path}/cv/{name}.parquet')
            df = pd.concat([df1, df2])
            del df1, df2
            


            corpus_name = f'{cfg.data_path}/word2vec/corpus_add_type.txt' if add_type else f'{cfg.data_path}/word2vec/corpus.txt'
            if not os.path.isfile(corpus_name):
                if add_type:
                    df.aid = df.aid.astype(str) + '_' + df.type.astype(str)
                else:
                    df.aid = df.aid.astype(str)

                aid_count = df.aid.value_counts()  # 统计商品出现的次数
                aid_count = aid_count[aid_count < min_count]  # 筛选出现次数小于mincount的商品
                aid_count = pd.Series(aid_count.index.str.slice(-1), index=aid_count.index)
                df.aid = df.aid.map(aid_count).fillna(df.aid).astype(str) # 将sessions中的aid列映射为aid_count中的值，如果映射不成功，则保留原来的值。

                sessions_aid_seq = df.groupby('session')['aid'].apply(list)
                x = [' '.join(sessions_aid_seq[idx]) for idx in tqdm(sessions_aid_seq.index)]   

                with open(corpus_name, 'w') as f:
                    for line in x:
                        f.write(f"{line}\n")

            model = Word2Vec(corpus_file = corpus_name,
                            vector_size = 64,
                            window = window_size, 
                            min_count = min_count, 
                            workers = 8)
            
            # vector_size：是指特征向量的维度，默认为100
            # window：窗口大小，表示当前词与预测词在一个句子中的最大距离是多少
            # min_count:可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。

            # window_size = str(window_size).zfill(2)
            model_path = f'{cfg.data_path}/word2vec/{name}_w2v_wind_{window_size}.model'
            model.save(model_path)
            LOGGER.info(f' ===== Successfully get {model_path} =====')
    



    def hot_item(self, cfg, name, top):
        if not os.path.exists(f'{cfg.data_path}/hot_items'):
            os.makedirs(f'{cfg.data_path}/hot_items')
        LOGGER.info(f' ======= Genetate {cfg.data_path}/hot_items/{name}_all_hot_items.parquet =======')
        if not os.path.isfile(f'{cfg.data_path}/hot_items/{name}_all_hot_items.parquet'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'  
            
            df_train = pd.read_parquet(f'{cfg.data_path}/cv/{train_name}.parquet')
            df_valid = pd.read_parquet(f'{cfg.data_path}/cv/{name}.parquet')   
            # 基于规则，热门商品
            top_clicks = df_valid.loc[df_valid['type'] == 0, 'aid'].value_counts()[:top].to_dict()
            top_carts = df_valid.loc[df_valid['type'] == 1, 'aid'].value_counts()[:top].to_dict()
            top_orders = df_valid.loc[df_valid['type'] == 2, 'aid'].value_counts()[:top].to_dict()
            type_weight = {0: 1, 1: 5, 2: 4}

            if not os.path.isfile(f'{cfg.data_path}/hot_items/{name}_top_clicks_items_last_month.pkl'):
                # 修改权重
                print("test hot")
                df_valid['score'] = df_valid['type'].map(type_weight)
                top_hot_items = df_valid.groupby('aid')['score'].apply(lambda x: x.sum()) \
                                    .sort_values(ascending=False)[:top].to_dict()
                
                print('train hot')
                df_train['score'] = df_train['type'].map(type_weight)
                top_hot_items_last_month = df_train.groupby('aid')['score'].apply(lambda x: x.sum()) \
                                            .sort_values(ascending=False)[:top].to_dict()

                print('train click hot')
                df_train['score'] = 1
                top_clicks_items_last_month = df_train.groupby('aid')['score'].apply(lambda x: x.sum()) \
                                                .sort_values(ascending=False)[:top].to_dict()
                with open(f'{cfg.data_path}/hot_items/{name}_top_hot_items.pkl', 'wb') as f:
                    pickle.dump(top_hot_items, f)
                with open(f'{cfg.data_path}/hot_items/{name}_top_hot_items_last_month.pkl', 'wb') as f:
                    pickle.dump(top_hot_items_last_month, f)
                with open(f'{cfg.data_path}/hot_items/{name}_top_clicks_items_last_month.pkl', 'wb') as f:
                    pickle.dump(top_clicks_items_last_month, f)                                        
            else:
                with open(f'{cfg.data_path}/hot_items/{name}_top_hot_items.pkl', 'rb') as f:
                    top_hot_items = pickle.load(f)
                with open(f'{cfg.data_path}/hot_items/{name}_top_hot_items_last_month.pkl', 'rb') as f:
                    top_hot_items_last_month = pickle.load(f)
                with open(f'{cfg.data_path}/hot_items/{name}_top_clicks_items_last_month.pkl', 'rb') as f:
                    top_clicks_items_last_month = pickle.load(f)                     


            if not os.path.isfile(f'{cfg.data_path}/hot_items/{name}_clicks_hot_items.parquet'):
                print(' clicks hot recall ！！！')
                df_clicks = df_valid.sort_values(["session", "ts"]).groupby(["session"]).apply(\
                    lambda x: clicks_recall(x,top_hot_items,top_hot_items_last_month,top_clicks,top_clicks_items_last_month))
                print(df_clicks.head())
                df_clicks = pd.DataFrame(df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
                df_clicks.to_parquet(f'{cfg.data_path}/hot_items/{name}_clicks_hot_items.parquet')
            else:
                df_clicks = pd.read_parquet(f'{cfg.data_path}/hot_items/{name}_clicks_hot_items.parquet')
            
            if not os.path.isfile(f'{cfg.data_path}/hot_items/{name}_carted_hot_items.parquet'):
                print(' carted hot recall ！！！')
                df_carts = df_valid.sort_values(["session", "ts"]).groupby(["session"]).apply(\
                    lambda x: carts_recall(x, top_hot_items, top_orders, top_carts, top_hot_items_last_month))
                df_carts = pd.DataFrame(df_carts.add_suffix("_carts"), columns=["labels"]).reset_index()            
                df_carts.to_parquet(f'{cfg.data_path}/hot_items/{name}_carted_hot_items.parquet')
            
            if not os.path.isfile(f'{cfg.data_path}/hot_items/{name}_orders_hot_items.parquet'):
                print(' orders hot recall ！！！')
                df_orders = df_valid.sort_values(["session", "ts"]).groupby(["session"]).apply(\
                    lambda x: orders_recall(x, top_hot_items, top_orders, top_carts, top_hot_items_last_month))
                df_orders = pd.DataFrame(df_orders.add_suffix("_orders"), columns=["labels"]).reset_index()            
                df_orders.to_parquet(f'{cfg.data_path}/hot_items/{name}_orders_hot_items.parquet')


            df_candidates = pd.concat([df_clicks, df_carts, df_orders])
            df_candidates.columns = ["session_type", "labels"]
            df_candidates["labels"] = df_candidates.labels.apply(lambda x: " ".join(map(str, x)))            
            print(' save df_candidates ！！！')
    
            df_candidates.to_parquet(f'{cfg.data_path}/hot_items/{name}_all_hot_items.parquet')
            LOGGER.info(f' ===== Successfully get {cfg.data_path}/hot_items/{name}_all_hot_items.parquet =====')





    ''' 得到物品相似度矩阵 '''
    def itemSimMatrix(self, cfg, name):
        if not os.path.exists(f'{cfg.data_path}/itemCF'):
            os.makedirs(f'{cfg.data_path}/itemCF')
        LOGGER.info(f' ======= Genetate {cfg.data_path}/itemCF/{name}_item_sim_matrix.pkl =======')
        if not os.path.isfile(f'{cfg.data_path}/itemCF/{name}_item_sim_matrix.pkl'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'


            df1 = pd.read_parquet(f'{cfg.data_path}/cv/{train_name}.parquet')
            df2 = pd.read_parquet(f'{cfg.data_path}/cv/{name}.parquet')
            df = pd.concat([df1, df2])
            del df1, df2

            # user_item_dict: 用户商品字典   key：11098528  value: [11830 1679529 92401 1055218 ...]
            user_item_dict = df.groupby("session")[['aid']].collect().to_pandas()
            user_item_dict = user_item_dict['aid'].to_dict()
            
            # itemMatrix： 一个用于存储商品之间关联度的二维字典
            itemMatrix = nb.typed.Dict.empty(
                            key_type = nb.types.int64,
                            value_type = nb.typeof(
                                                nb.typed.Dict.empty(
                                                            key_type = nb.types.int64,
                                                            value_type = nb.types.float64)))
            item_counts = nb.typed.Dict.empty(
                            key_type = nb.types.int64,
                            value_type = nb.types.int64)
            
            for user, item_sequence in tqdm(user_item_dict.items()):
                ItemMatrix_fn(itemMatrix, item_counts, item_sequence)
                
            itemSimMatrix = nb.typed.Dict.empty(
                                key_type = nb.types.int64,
                                value_type = nb.typeof(nb.typed.Dict.empty(
                                                    key_type = nb.types.int64, 
                                                    value_type = nb.types.float64)))
            
            for item_id, related_items in tqdm(itemMatrix.items()):
                ItemSimilarityMatrix_fn(itemSimMatrix, item_id, item_counts, related_items)
            
            
            item_sim = {}
            for k,v in tqdm(itemSimMatrix.items()):
                item_sim[k] = dict(v)            
            
            with open(f"{cfg.data_path}/itemCF/{name}_item_sim_matrix.pkl", "wb") as f:
                pickle.dump(item_sim, f)



    def item_cf(self, cfg, name):
        if not os.path.exists(f'{cfg.data_path}/item_cf'):
            os.makedirs(f'{cfg.data_path}/item_cf')
        LOGGER.info(f' ======= Genetate {cfg.data_path}/item_cf/item_sim_{name}.pkl" =======')
        if not os.path.isfile(f'{cfg.data_path}/item_cf/item_sim_{name}.pkl'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'  

            df_train = cudf.read_parquet(f'{cfg.data_path}/cv/{train_name}.parquet')
            df_valid = cudf.read_parquet(f'{cfg.data_path}/cv/{name}.parquet') 

            gene_itemSimMatrix(df_train, df_valid, name, cfg)



            LOGGER.info(f' ===== Successfully get {cfg.data_path}/item_cf/item_sim_{name}.pkl =====')


    def item_cf_prepare(self, cfg, name):
        if not os.path.exists(f'{cfg.data_path}/item_cf'):
            os.makedirs(f'{cfg.data_path}/item_cf')
        LOGGER.info(f' ======= Genetate {cfg.data_path}/item_cf/{name}_user_item.npz" =======')
        if not os.path.isfile(f'{cfg.data_path}/item_cf/{name}_user_item.npz'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'  

            df_train = cudf.read_parquet(f'{cfg.data_path}/cv/{train_name}.parquet')
            df_valid = cudf.read_parquet(f'{cfg.data_path}/cv/{name}.parquet') 
            df_train = df_train[df_train['ts'] >= df_train['ts'].max() - 7 * 24 * 3600]
            df = cudf.concat([df_train, df_valid])
            del df_train, df_valid
            df = df.groupby('session')['aid'].agg(list).reset_index()
            df = df.to_pandas()

            from pandarallel import pandarallel
            pandarallel.initialize(progress_bar=True, nb_workers=12)
            def mat_func(items):
                tmp = np.zeros(1855603, dtype=bool)
                for item in items:
                    tmp[item] += 1
                return sparse.csr_matrix(tmp,dtype=bool)
            user_items_mat = df['aid'].parallel_apply(mat_func)
            user_items_mat = sparse.vstack(user_items_mat.values)
            sparse.save_npz(f'{cfg.data_path}/item_cf/{name}_user_item.npz',user_items_mat)

            LOGGER.info(f' ===== Successfully get {cfg.data_path}/item_cf/{name}_user_item.npz =====')




    def item_cf_feature(self, cfg, name):
        if not os.path.exists(f'{cfg.data_path}/item_cf'):
            os.makedirs(f'{cfg.data_path}/item_cf')
        LOGGER.info(f' ======= Genetate {cfg.data_path}/item_cf/{name}_sim.pkl" =======')
        if not os.path.isfile(f'{cfg.data_path}/item_cf/{name}_sim.pkl'):
            if name == 'valid3':
                train_name = 'train12'
            elif name == 'valid4':
                train_name = 'train123'
            elif name == 'test':
                train_name = 'all_train'  

            df_valid = cudf.read_parquet(f'{cfg.data_path}/cv/{name}.parquet') 
            user_items_train = sparse.load_npz(f'{cfg.data_path}/item_cf/{name}_user_item.npz')
            
            import implicit
            bm25_model = implicit.nearest_neighbours.BM25Recommender(K=50,num_threads=12)
            bm25_model.fit(user_items_train)
            unique_aid = list(df_valid['aid'].unique().to_pandas().values)
            bm25_sim = {}
            for aid in tqdm(unique_aid):
                key, value = bm25_model.similar_items(aid)
                sim = dict(zip(key, value))
                bm25_sim[aid] = sim
            with open(f"{cfg.data_path}/item_cf/{name}_sim.pkl", "wb") as f:
                pickle.dump(bm25_sim, f)    

            LOGGER.info(f' ===== Successfully get {cfg.data_path}/item_cf/{name}_sim.pkl =====')