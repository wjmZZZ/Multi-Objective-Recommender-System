import os
import pickle
import pandas as pd
import polars as pl
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer
from pandarallel import pandarallel
# Initialization
pandarallel.initialize(nb_workers=10, progress_bar=True)


from Config import CFG
from Utils import LOGGER, get_metric, log_line
from generate.gene_hot_items import carts_recall, clicks_recall, orders_recall


def generate_candidates(data_name, cfg):
    if not os.path.exists(cfg.data_path + '/candidates'):
        os.makedirs(cfg.data_path + '/candidates')

    log_line()
    all_candidates = pl.DataFrame()

    LOGGER.info(f'\n\n{"="*25}  Multiple Recall --->>  Covisit_Candidates  {"="*25}')
    covisit_cands = Covisit_Candidates(cfg.data_path, data_name, suffix=f'covisit_top{cfg.top}', save_suffix=f'covist_cands_{cfg.top}', type=None)
    covisit_cands = covisit_cands.load_candidates_file()
    all_candidates = covisit_cands
    del covisit_cands

    covisit_clicks = Covisit_Candidates(cfg.data_path, data_name, suffix=f'covisit_top{cfg.top}', save_suffix=f'clicks_covist_cands_{cfg.top}', type='clicks')
    covisit_clicks = covisit_clicks.load_candidates_file()
    all_candidates = covisit_clicks
    del covisit_clicks

    covisit_carts = Covisit_Candidates(cfg.data_path, data_name, suffix=f'covisit_top{cfg.top}', save_suffix=f'carts_covist_cands_{cfg.top}', type='carts')
    covisit_carts = covisit_carts.load_candidates_file()
    all_candidates = all_candidates.join(covisit_carts, on=['session', 'aid'], how='outer')
    del covisit_carts

    covisit_orders = Covisit_Candidates(cfg.data_path, data_name, suffix=f'covisit_top{cfg.top}', save_suffix=f'orders_covist_cands_{cfg.top}', type='orders')
    covisit_orders = covisit_orders.load_candidates_file()
    all_candidates = all_candidates.join(covisit_orders, on=['session', 'aid'], how='outer')
    del covisit_orders
    get_metric()


    LOGGER.info(f'\n{"="*25}  Multiple Recall --->>  Word2Vec_Candidates  {"="*25}')
    w2v_window10 = Word2Vec_Candidates(cfg.data_path, data_name, 'w2v_window10_cands_20', window_size=10, max_cands=20) 
    w2v_window10 = w2v_window10.load_candidates_file() 
    all_candidates = all_candidates.join(w2v_window10, on=['session', 'aid'], how='outer')
    del w2v_window10

    w2v_window20 = Word2Vec_Candidates(cfg.data_path, data_name, 'w2v_window20_cands_20', window_size=20, max_cands=20) 
    w2v_window20 = w2v_window20.load_candidates_file() 
    all_candidates = all_candidates.join(w2v_window20, on=['session', 'aid'], how='outer')
    del w2v_window20
    get_metric()
    

    LOGGER.info(f'\n{"="*25}  Multiple Recall --->>  Recent_Candidates  {"="*25}')
    clicked_in_session = Recent_Candidates(cfg.data_path, data_name, save_suffix = f'clicks_recent_cands', type=0)
    clicked_in_session = clicked_in_session.load_candidates_file()

    carted_in_session = Recent_Candidates(cfg.data_path, data_name, save_suffix = f'carted_recent_cands', type=1)
    carted_in_session = carted_in_session.load_candidates_file()

    ordered_in_session = Recent_Candidates(cfg.data_path, data_name, save_suffix = f'orders_recent_cands', type=2)
    ordered_in_session = ordered_in_session.load_candidates_file()    

    recent_cands = pl.concat([clicked_in_session, carted_in_session, ordered_in_session])
    recent_cands = recent_cands.pivot(values='rank', index=['session', 'aid'], columns='name')
    all_candidates = all_candidates.join(recent_cands, on=['session', 'aid'], how='outer')
    del recent_cands
    get_metric()
    

    LOGGER.info(f'\n{"="*25}  Multiple Recall --->>  Hot_Items_Candidates  {"="*25}')
    click_hot_items_cands = Hot_Items_Candidates(cfg.data_path, data_name, type=0, save_suffix = f'clicks_hot_items')
    click_hot_items_cands = click_hot_items_cands.load_candidates_file()
    click_hot_items_cands = click_hot_items_cands.drop('type')
    all_candidates = all_candidates.join(click_hot_items_cands, on=['session', 'aid'], how='outer') 
    del click_hot_items_cands
    
    carted_hot_items_cands = Hot_Items_Candidates(cfg.data_path, data_name, type=1, save_suffix = f'carts_hot_items')
    carted_hot_items_cands = carted_hot_items_cands.load_candidates_file()
    carted_hot_items_cands = carted_hot_items_cands.drop('type')
    all_candidates = all_candidates.join(carted_hot_items_cands, on=['session', 'aid'], how='outer') 
    del carted_hot_items_cands

    orders_hot_items_cands = Hot_Items_Candidates(cfg.data_path, data_name, type=2, save_suffix = f'orders_hot_items')
    orders_hot_items_cands = orders_hot_items_cands.load_candidates_file()
    orders_hot_items_cands = orders_hot_items_cands.drop('type')
    all_candidates = all_candidates.join(orders_hot_items_cands, on=['session', 'aid'], how='outer')
    del orders_hot_items_cands
    
    hot_items = pl.concat([click_hot_items_cands, carted_hot_items_cands, orders_hot_items_cands])
    all_candidates = all_candidates.join(hot_items, on=['session', 'aid'], how='outer') # outer
    del click_hot_items_cands, carted_hot_items_cands, orders_hot_items_cands, hot_items

    get_metric()
    

    LOGGER.info(f'\n >>>>> All_candidates total {len(all_candidates)} row ')
    all_candidates.write_parquet(f'{cfg.data_path}/candidates.parquet')
    all_candidates = all_candidates.fill_null(999)
    cands_cols = all_candidates.columns
    cands_cols.remove('aid')
    cands_cols.remove('session')
    

    all_candidates = all_candidates.with_columns(pl.col(cands_cols))#.cast(pl.UInt16))
    get_metric()

    return all_candidates


# ========================================
# 核心候选生成类，都继承这个父类
# ========================================
class CandidatesGen():
    def __init__(self, data_path, data_name, suffix, save_suffix):
        self.data_path = data_path
        self.data_name = data_name
        self.suffix = suffix           # 要读取的文件的后缀
        self.save_suffix = save_suffix # 要保存的candidates文件名

    def prepare_candidates(self):
        raise NotImplementedError

    def load_candidates_file(self):
        file = f'{self.data_path}/candidates/{self.data_name}_{self.save_suffix}.parquet'
        if not os.path.isfile(file):
            LOGGER.info(f'   --> :( {file} does not exit, start generation !!! ')
            self.prepare_candidates()
        else:
            LOGGER.info(f'   --> :) Reading candidates {file} ')
            
        df = pl.read_parquet(file)  
        return df
    

# ========================================
# 
# ========================================
class Covisit_Candidates(CandidatesGen):
    def __init__(self, data_path, data_name, suffix, save_suffix, type):
        super().__init__(data_path, data_name, suffix, save_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.suffix = suffix
        self.save_suffix = save_suffix
        self.type = type

    def prepare_candidates(self):
        df = pl.read_csv(f'{self.data_path}/cv/{self.data_name}_{self.suffix}.csv').lazy() 
        df = df.with_columns(pl.col('labels').apply(lambda x: [int(i) for i in x.split()]).alias('candidates'))  
        df = df.with_columns(pl.col('session_type').str.split(by='_').alias('session_type2')) 
        df = df.with_columns(pl.col('session_type2').apply(lambda x: int(x[0])).alias('session')) 
        df = df.with_columns(pl.col('session_type2').apply(lambda x: x[1]).alias('type_str')) 
        if self.type != None:
            df = df.filter(pl.col('type_str') == self.type)
        df = df.drop(['session_type', 'labels', 'session_type2'])
        df = df.explode('candidates')  
        df = df.with_columns(pl.lit(1).alias('one'))
        cand_col_name = f'covisit_{self.type}'
        df = df.with_columns((pl.col('one').cumsum() - 1).over('session').alias(cand_col_name)) # reverse=self.reverse
        df = df.drop('one')
        df = df.select(
            [pl.col('session').cast(pl.Int32), pl.col('candidates').cast(pl.Int32).alias('aid'), pl.col(cand_col_name).cast(pl.Int32)]).collect()
        df.write_parquet(
            f'{self.data_path}/candidates/{self.data_name}_{self.save_suffix}.parquet')



'''
通过Word2Vec模型计算每个商品与其他商品的相似度，然后选取相似度最高的前200个商品作为候选项
'''
class Word2Vec_Candidates(CandidatesGen):
    def __init__(self, data_path, data_name, save_suffix, window_size, max_cands, suffix=None):
        super().__init__(data_path, data_name, suffix, save_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.save_suffix = save_suffix
        self.window_size = window_size
        self.max_cands = max_cands

    def prepare_candidates(self):
        df = pl.read_parquet(f'{self.data_path}/cv/{self.data_name}.parquet').lazy()
        model = Word2Vec.load(f'{self.data_path}/word2vec/{self.data_name}_w2v_wind_{self.window_size}.model')

        df = df.unique(subset=['session'], keep='last')  
        df = df.with_columns((pl.col('aid').cast(str) + pl.lit('_') + pl.col('type').cast(str)).alias('aid_type'))  
        df = df.with_columns(pl.col('type').cast(str)).collect()
        vocab = list(set(model.wv.index_to_key))
        vocab = pl.DataFrame(vocab, columns=['aid_type'])

        df = df.join(vocab, on='aid_type')
        df = df.select(pl.col(['session', 'aid_type']))

        annoy_index = AnnoyIndexer(model, 64)
        candidates = []
        for aid_type in tqdm(df.select(pl.col('aid_type').unique()).to_dict()['aid_type']):
            candidates.append(self.get_w2v_reco(aid_type, model, annoy_index))  

        candidates = pl.concat(candidates)
        df = df.join(candidates, on='aid_type').drop('aid_type')
        df = df.select(pl.col('*').cast(pl.Int32))

        columns = df.columns
        columns.remove('session')
        columns.remove('aid')
        df = df.filter(df.select(columns).min(1) <= 5)  

        df.write_parquet(
            f'{self.data_path}/candidates/{self.data_name}_{self.save_suffix}.parquet')

    def get_w2v_reco(self, aid_type, model, indexer):
        cands = []
        rank_clicks = 0
        rank_carts = 0
        rank_orders = 0

        # 使用gensim库中的Word2Vec模型，通过训练得到的词向量来寻找与给定aid_str最相似的Top K 个 商品aid
        # 返回一个列表，其中每个元素是一个二元组，第一个元素是与aidstr最相似的aid，第二个元素是相似度得分
        recos = model.wv.most_similar(aid_type, topn=200, indexer=indexer)  
        for reco in recos:
            if len(reco[0]) > 1:  # 过滤掉出现次数太少的商品
                # 根据拼接的类型判断操作类型， 形如aid_0 
                if reco[0][-1] == '0' and rank_clicks < self.max_cands:
                    cands.append([aid_type, int(reco[0][:-2]), f'w2v_{self.window_size}_clicks', rank_clicks])
                    rank_clicks += 1
                elif reco[0][-1] == '1' and rank_carts < self.max_cands:
                    cands.append([aid_type, int(reco[0][:-2]), f'w2v_{self.window_size}_carts', rank_carts])
                    rank_carts += 1
                elif rank_orders < self.max_cands:
                    cands.append([aid_type, int(reco[0][:-2]), f'w2v_{self.window_size}_orders', rank_orders])
                    rank_orders += 1

        cands = pl.DataFrame(cands, orient='row', columns=['aid_type', 'aid', 'col_name', 'rank'])

        cands = cands.pivot(index=['aid_type', 'aid'],columns='col_name', values='rank')

        if f'w2v_{self.window_size}_clicks' not in cands.columns:
            cands = cands.with_columns(pl.lit(None).cast(pl.Int64).alias(f'w2v_{self.window_size}_clicks'))

        if f'w2v_{self.window_size}_carts' not in cands.columns:
            cands = cands.with_columns(pl.lit(None).cast(pl.Int64).alias(f'w2v_{self.window_size}_carts'))

        if f'w2v_{self.window_size}_orders' not in cands.columns:
            cands = cands.with_columns(pl.lit(None).cast(pl.Int64).alias(f'w2v_{self.window_size}_orders'))

        columns = cands.columns
        columns.sort()

        cands = cands.select(pl.col(columns))

        return cands
    

''' 生成最近事件的候选商品 '''
class Recent_Candidates(CandidatesGen):
    def __init__(self, data_path, data_name, save_suffix, type, suffix=None ):
        super().__init__(data_path, data_name, suffix, save_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.suffix = suffix
        self.save_suffix = save_suffix
        self.type = type

    def prepare_candidates(self):
        df = pl.read_parquet(
            f'{self.data_path}/cv/{self.data_name}.parquet').lazy()
        if self.type != None:
            df = df.filter(pl.col('type') == self.type)  # 筛选出来对应事件编号的数据
        df = df.sort(by='ts', reverse=True)  # 按时间那一列排序，降序排列
        df = df.select(pl.col(['session', 'aid']))
        df = df.unique(keep='first')
        # 选出session和aid两列，对aid进行分组求索引，即每个session中的aid是按顺序的索引。并且别名成为rank
        df = df.select([pl.col(['session', 'aid']), pl.col('aid').cumcount().over("session").alias('rank')])
        # 新增加一列name，值都为传入的str名字
        df = df.with_column(pl.lit(self.save_suffix).alias('name')).collect()
        df.write_parquet(f'{self.data_path}/candidates/{self.data_name}_{self.save_suffix}.parquet')




    ''' 热门商品召回 '''
class Hot_Items_Candidates(CandidatesGen):
    def __init__(self, data_path, data_name, save_suffix, type, suffix=None):
        super().__init__(data_path, data_name, suffix, save_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.suffix = suffix
        self.save_suffix = save_suffix
        self.type = type
    def prepare_candidates(self):
        df = pl.read_parquet(f'{self.data_path}/hot_items/{self.data_name}_all_hot_items.parquet')
        df = df.drop('__index_level_0__')

        type_transform = {"clicks": 0, "carts": 1, "orders": 2}
        df = df.with_columns([
            pl.col('session_type').apply(lambda x: x.split('_')[0]).alias('session').cast(pl.Int32),
            pl.col('session_type').apply(lambda x: type_transform[x.split('_')[1]]).alias('aid_type').cast(pl.Int8)
            ])
        df = df.drop('session_type')
        df = df.filter(pl.col('aid_type') == self.type)
        df = df.drop(['aid_type'])

        df = df.with_columns(pl.col('labels').apply(lambda x: x.split(" ")))
        df = df.explode('labels')
        candidate_type_dic = {1: 'history_aid', 2: 'top_hot_aid', 3: 'top_orders_aid', 4: 'top_carts_aid',
                                5: 'top_hot_aid_last_month'}
        candidate_type_scores_dic = {1: 'history_aid_score', 2: 'top_hot_aid_score', 3: 'top_orders_aid_score',
                                        4: 'top_carts_aid_score', 5: 'top_hot_aid_last_month_score'}

        df  = df.with_columns([
                        pl.col('labels').apply(lambda x: x.split('#')[0]).alias('aid').cast(pl.Int32),
                        pl.col('labels').apply(lambda x: x.split('#')[1]).alias('cands_type').cast(pl.Int8),
                        pl.col('labels').apply(lambda x: x.split('#')[2]).alias('cands_score').cast(pl.Float32),
                        ])
        df = df.drop(['labels'])
        
        cands_1= df.filter(pl.col('cands_type') == 1)
        cands_1 = cands_1.rename({'cands_type':'history_aid', 'cands_score':'history_aid_score' })#), pl.col('cands_score').rename('history_aid_sacore')])
        cands_1 = cands_1.with_columns(pl.lit(1).alias('one'))
        # 对one这一列进行分组求索引，即每个session中的one是按顺序的索引。并且别名为self.name，相当于加一列作为辅助列，来求每个session的排序
        cands_1 = cands_1.with_columns((pl.col('one').cumsum() - 1).over('session').alias('history_aid_rank')) 
        cands_1 = cands_1.sort(by =["session", "history_aid_score"], reverse=[False, True])
        cands_1 = cands_1.drop('one')

        for i in range(2, 6):
            tmp = df.filter(pl.col('cands_type') == i)

            tmp = tmp.rename({'cands_type':f'{candidate_type_dic[i]}', 'cands_score':f'{candidate_type_scores_dic[i]}'})
            tmp = tmp.with_columns(pl.lit(1).alias('one'))
            tmp = tmp.with_columns((pl.col('one').cumsum() - 1).over('session').alias(f'{candidate_type_scores_dic[i]}_rank')) # reverse=self.reverse
            tmp = tmp.sort(by =["session", f'{candidate_type_scores_dic[i]}'], reverse=[False, True])
            tmp = tmp.drop('one')
            tmp = tmp.select(pl.col(['session', 'aid', f'{candidate_type_scores_dic[i]}', f'{candidate_type_dic[i]}', f'{candidate_type_scores_dic[i]}_rank']))
            cands_1 = cands_1.join(tmp, on=['session', 'aid'], how='outer')
        # cands_1 = cands_1.drop('type')
        # cands_1 = cands_1.fill_null(-1)#.collect() #.select(pl.col('*').cast(pl.Int32))
        cands_1.write_parquet(f'{self.data_path}/candidates/{self.data_name}_{self.save_suffix}.parquet')



