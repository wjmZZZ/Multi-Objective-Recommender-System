import os
import pandas as pd
import numpy as np
import polars as pl

from Utils import LOGGER, get_metric, log_line


def generate_features(candidates, data_name, cfg):
    if not os.path.exists(cfg.data_path + '/features'):
        os.makedirs(cfg.data_path + '/features')    
    log_line()

    all_features = pl.DataFrame()
    LOGGER.info(f'\n\n{"="*25}  Feature Engineering --->>  Session_Features  {"="*25}')
    session_base_feature_ = Session_Features(cfg.data_path, data_name, sava_suffix=f'session_base_{cfg.top}')
    session_base_feature = session_base_feature_.load_feature_file()
    all_features = candidates.join(session_base_feature, how='left', on='session')
    all_features = session_base_feature_.fill_null(all_features)
    del session_base_feature, session_base_feature_

    session_clicks_feature_ = Session_Features(cfg.data_path, data_name, sava_suffix=f'session_clicks_{cfg.top}', type_id=0)
    session_clicks_feature = session_clicks_feature_.load_feature_file()
    all_features = all_features.join(session_clicks_feature, how='left', on='session')
    all_features = session_clicks_feature_.fill_null(all_features)
    del session_clicks_feature, session_clicks_feature_

    session_carts_feature_ = Session_Features(cfg.data_path, data_name, sava_suffix=f'session_carts_{cfg.top}', type_id=1)
    session_carts_feature = session_carts_feature_.load_feature_file()
    all_features = all_features.join(session_carts_feature, how='left', on='session')
    all_features = session_carts_feature_.fill_null(all_features)
    del session_carts_feature, session_carts_feature_

    session_orders_feature_ = Session_Features(cfg.data_path, data_name, sava_suffix=f'session_orders_{cfg.top}', type_id=2)
    session_orders_feature = session_orders_feature_.load_feature_file()
    all_features = all_features.join(session_orders_feature, how='left', on='session')  
    all_features = session_orders_feature_.fill_null(all_features)
    del session_orders_feature, session_orders_feature_
    get_metric()


    LOGGER.info(f'\n{"="*25}  Feature Engineering --->>  Aid_Features  {"="*25}')
    aid_feature_ = Aid_Features(cfg.data_path, data_name, sava_suffix ='aid_features')
    aid_feature = aid_feature_.load_feature_file()
    all_features = all_features.join(aid_feature, how='left', on='aid')
    all_features = aid_feature_.fill_null(all_features)
    del aid_feature, aid_feature_
    get_metric()


    LOGGER.info(f'\n{"="*25}  Feature Engineering --->>  Interaction_Features  {"="*25}')
    interaction_features_ = Interaction_Features(cfg.data_path, data_name, sava_suffix=f'interaction')
    interaction_features = interaction_features_.load_feature_file()
    all_features = all_features.join(interaction_features, how='left', on=['session', 'aid'])
    all_features = interaction_features_.fill_null(all_features)
    del interaction_features, interaction_features_

    click_interaction_ = Interaction_Features(cfg.data_path, data_name, sava_suffix=f'clicks_interaction', type_id=0)
    click_inter_features = click_interaction_.load_feature_file()
    all_features = all_features.join(click_inter_features, how='left', on=['session', 'aid'])
    all_features = click_interaction_.fill_null(all_features)
    del click_inter_features, click_interaction_

    cart_interaction_ = Interaction_Features(cfg.data_path, data_name, sava_suffix=f'carted_interaction', type_id=1)
    cart_inter_features = cart_interaction_.load_feature_file()
    all_features = all_features.join(cart_inter_features, how='left', on=['session', 'aid'])
    all_features = cart_interaction_.fill_null(all_features)
    del cart_inter_features, cart_interaction_

    order_interaction_ = Interaction_Features(cfg.data_path, data_name, sava_suffix=f'orders_interaction', type_id=2)
    order_inter_features = order_interaction_.load_feature_file()
    all_features = all_features.join(order_inter_features, how='left', on=['session', 'aid'])
    all_features = order_interaction_.fill_null(all_features)
    del order_inter_features, order_interaction_
    get_metric()


    LOGGER.info(f'\n{"="*25} Feature Engineering --->>  Conversion_Features  {"="*25}')
    conversion_features = Conversion_Features(cfg.data_path, data_name, sava_suffix=f'conversion_features')
    conversion_features = conversion_features.load_feature_file()
    all_features = all_features.join(conversion_features, how='left', on='aid')
    del conversion_features
    get_metric()


    LOGGER.info(f'\n{"="*25} Feature Engineering --->>  Type_Features  {"="*25}')
    all_features = all_features.with_column((
            (pl.col('cart_inter_last_time') >= 0) & (pl.col('order_inter_last_time') < 0)).alias('cart_without_order1'))
    all_features = all_features.with_column(((
        pl.col('cart_inter_last_time') >= 0) & (pl.col('order_inter_last_time') >= 0) & ( \
        pl.col('order_inter_last_time') > pl.col('cart_inter_last_time'))).alias('cart_without_order2'))
    all_features = all_features.with_column((
        pl.col('cart_without_order1') | pl.col('cart_without_order2')).alias('cart_without_order')).drop(['cart_without_order1', 'cart_without_order2'])

    # 特征 点击但是没有下单
    all_features = all_features.with_column(((pl.col('click_inter_last_time') >= 0) & (
        pl.col('order_inter_last_time') < 0)).alias('click_without_order1'))
    all_features = all_features.with_column(((pl.col('click_inter_last_time') >= 0) & (pl.col('order_inter_last_time') >= 0) & (
        pl.col('order_inter_last_time') > pl.col('click_inter_last_time'))).alias('click_without_order2'))
    all_features = all_features.with_column((pl.col('click_without_order1') | pl.col('click_without_order2')).alias(
        'click_without_order')).drop(['click_without_order1', 'click_without_order2'])

    # 特征 点击但是没有加购物车
    all_features = all_features.with_column(((pl.col('click_inter_last_time') >= 0) & (
        pl.col('cart_inter_last_time') < 0)).alias('click_without_cart1'))
    all_features = all_features.with_column(((pl.col('click_inter_last_time') >= 0) & (pl.col('cart_inter_last_time') >= 0) & (
        pl.col('cart_inter_last_time') > pl.col('click_inter_last_time'))).alias('click_without_cart2'))
    all_features = all_features.with_column((pl.col('click_without_cart1') | pl.col('click_without_cart2')).alias(
        'click_without_cart')).drop(['click_without_cart1', 'click_without_cart2'])    
    
    get_metric()


    exclude_cols = ['session', 'aid', 'y_clicks', 'y_carts', 'y_orders']
    cfg.features = [feature for feature in all_features.columns if feature not in exclude_cols]
    LOGGER.info(f'\n >>>>> All_features total {len(all_features)} row ')

    LOGGER.info(f'\n >>>>> Total {len(cfg.features)} features :\n         {cfg.features}')
    return all_features

# ========================================
# 核心特征生成类，都继承这个父类
# ========================================
class FeatureGen():
    def __init__(self,data_path, data_name, sava_suffix):
        self.data_path = data_path
        self.data_name = data_name
        self.sava_suffix = sava_suffix

    def prepare_features(self):
        raise NotImplementedError

    def load_feature_file(self):
        file = f'{self.data_path}/features/{self.data_name}_{self.sava_suffix}.parquet'
        if not os.path.isfile(file):
            LOGGER.info(f'   --> :( {file} does not exit, start generation !!! ')
            self.prepare_features()
        else:
            LOGGER.info(f'   --> :) Reading feature {file} ')
        df = pl.read_parquet(file) 
        return df
    

# ========================================
# 用户特征，每个用户点击、加购物车、下单多少商品
# ========================================
class Session_Features(FeatureGen):
    def __init__(self, data_path, data_name, sava_suffix, type_id=None):
        super().__init__(data_path = data_path, data_name = data_name, sava_suffix=sava_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.sava_suffix = sava_suffix
        self.type_id = type_id
        if type_id is not None:
            id2type = {0: 'click', 1: 'cart', 2: 'order'}
            self.type = id2type[type_id]
        else:
            self.type = 'session'

    def prepare_features(self):
        df = pl.read_parquet(f'{self.data_path}/cv/{self.data_name}.parquet')
        if self.type_id:
            df = df.filter(pl.col('type') == self.type_id)

        # 每一个商品有多少用户发生交互，用来衡量每个商品的热度，即被多少不同用户所关注
        # 一个产品被越多不同的用户关注，它的热度就越高，因为越多的用户可能会购买它
        aid_count = df.groupby('aid').agg(pl.col('session').n_unique().alias(f'{self.type}_aid_count'))

        df = df.join(aid_count, on='aid')
        df = df.groupby('session').agg(
                    [pl.col('aid').count().alias(f'{self.type}_count'),  # 用户交互商品的总数
                    pl.col('aid').n_unique().alias(f'{self.type}_unique_count'), # 用户交互的不同商品的总数
                    pl.col(f'{self.type}_aid_count').median().alias(f'{self.type}_session_aid_count_median') # 每个商品被多少不同用户交互的中位数
                                    ])
        # 每个商品平均被多少不同用户交互，衡量的是用户在type交互中，平均关注了多少不同的产品
        # 如果一个用户平均关注的不同产品数比较少，说明他的行为偏好比较集中，可能更容易受到某些产品的吸引
        df = df.with_columns((pl.col(f'{self.type}_count') / pl.col(f'{self.type}_unique_count')).alias(f'{self.type}_session_aid_atten_avg'))

        df.write_parquet(
            f'{self.data_path}/features/{self.data_name}_{self.sava_suffix}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col([f'{self.type}_count', f'{self.type}_unique_count']).fill_null(0))
        df = df.with_columns(
            pl.col([f'{self.type}_session_aid_atten_avg', f'{self.type}_session_aid_count_median']).fill_null(-1))
        return df


# ========================================
# 从产品的视角做特征，每个产品被多少用户点击等
# 按照aid分组，统计每个aid的最大和最小时间戳、session个数、在该产品上发生事件的session个数等信息，
# 计算clicktocart、clicktoorder和carttoorder三个指标
# ========================================
class Aid_Features(FeatureGen):
    def __init__(self, data_path, data_name, sava_suffix, type_id=None):
        super().__init__(data_path = data_path, data_name = data_name, sava_suffix=sava_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.sava_suffix = sava_suffix

    def prepare_features(self):
        if self.data_name == 'valid3':
            train_name = 'train12'
        elif self.data_name == 'valid4':
            train_name = 'train123'
        elif self.data_name == 'test':
            train_name = 'all_train'

        df_test = pl.read_parquet(f'{self.data_path}/cv/{self.data_name}.parquet')
        df_train = pl.read_parquet(f'{self.data_path}/cv/{train_name}.parquet')
        
        # 训练集里的最晚的时间
        max_train_ts = df_train.select(pl.col('ts').max().alias('max'))[0, 0]
        
        # 筛选出训练集，最近7天的数据更能反映当前的市场情况和用户行为
        df_train = df_train.filter(pl.col('ts') > (max_train_ts - (7 * 24 * 3600))) 
        df = pl.concat([df_test, df_train])
        del df_train, df_test, max_train_ts

        ts_max = df.select(pl.col('ts').max())[0, 0]
        ts_min = df.select(pl.col('ts').min())[0, 0]

        # 总体的aid特征
        aid_stats = df.groupby('aid').agg([
            pl.col('ts').max().alias('aid_max_ts'),
            pl.col('ts').min().alias('aid_min_ts'),
            pl.col('session').count().alias('aid_count'), 
            pl.col('session').n_unique().alias('aid_session_cnt')])  
        aid_stats = aid_stats.with_columns((ts_max - pl.col('aid_max_ts')))
        aid_stats = aid_stats.with_columns((pl.col('aid_min_ts') - ts_min))

        # clicks
        aid_click_stats = df.filter(pl.col('type') == 0).groupby('aid').agg([
            pl.col('ts').max().alias('aid_click_max_ts'),
            pl.col('ts').min().alias('aid_click_min_ts'),
            pl.col('session').count().alias('aid_click_cnt'),
            pl.col('session').n_unique().alias('aid_session_click_cnt')])
        
        aid_click_stats = aid_click_stats.with_columns((ts_max - pl.col('aid_click_max_ts')))
        aid_click_stats = aid_click_stats.with_columns((pl.col('aid_click_min_ts') - ts_min))

        # carted
        aid_cart_stats = df.filter(pl.col('type') == 1).groupby('aid').agg([
            pl.col('ts').max().alias('aid_cart_max_ts'),
            pl.col('ts').min().alias('aid_cart_min_ts'),
            pl.col('session').count().alias('aid_cart_cnt'),
            pl.col('session').n_unique().alias('aid_session_cart_cnt')])
        aid_cart_stats = aid_cart_stats.with_columns((ts_max - pl.col('aid_cart_max_ts')))
        aid_cart_stats = aid_cart_stats.with_columns((pl.col('aid_cart_min_ts') - ts_min))

        # orders
        aid_order_stats = df.filter(pl.col('type') == 2).groupby('aid').agg([
            pl.col('ts').max().alias('aid_order_max_ts'),
            pl.col('ts').min().alias('aid_order_min_ts'),
            pl.col('session').count().alias('aid_order_cnt'),
            pl.col('session').n_unique().alias('aid_session_order_cnt')])
        aid_order_stats = aid_order_stats.with_columns((ts_max - pl.col('aid_order_max_ts')))
        aid_order_stats = aid_order_stats.with_columns((pl.col('aid_order_min_ts') - ts_min))

        aid_stats = aid_stats.join(aid_click_stats, on='aid', how='left')
        aid_stats = aid_stats.join(aid_cart_stats, on='aid', how='left')
        aid_stats = aid_stats.join(aid_order_stats, on='aid', how='left')

        aid_stats = aid_stats.with_columns(
            pl.col(['aid_max_ts', 'aid_min_ts', 'aid_click_max_ts', 'aid_click_min_ts', 'aid_cart_max_ts', 'aid_cart_min_ts', 'aid_order_max_ts', 'aid_order_min_ts']).fill_null(999999))
        aid_stats = aid_stats.with_columns(
            pl.col(['aid_count', 'aid_click_cnt', 'aid_cart_cnt', 'aid_order_cnt', 'aid_session_cnt', 'aid_session_click_cnt', 'aid_session_cart_cnt', 'aid_session_order_cnt']).fill_null(0))

        aid_stats = aid_stats.with_columns(
            (pl.col('aid_cart_cnt') / pl.col('aid_click_cnt')).alias('click2cart'))
        aid_stats = aid_stats.with_columns(
            (pl.col('aid_order_cnt') / pl.col('aid_click_cnt')).alias('click2order'))
        aid_stats = aid_stats.with_columns(
            (pl.col('aid_order_cnt') / pl.col('aid_cart_cnt')).alias('cart2order'))

        aid_stats.write_parquet(
            f'{self.data_path}/features/{self.data_name}_{self.sava_suffix}.parquet')

    def fill_null(self, df):
        df = df.with_columns(
            pl.col(['aid_max_ts', 'aid_min_ts', 'aid_click_max_ts', 'aid_click_min_ts', 'aid_cart_max_ts', 'aid_cart_min_ts', \
                    'aid_order_max_ts', 'aid_order_min_ts']).fill_null(999999))
        df = df.with_columns(
            pl.col(['aid_count', 'aid_click_cnt', 'aid_cart_cnt', 'aid_order_cnt', 'aid_session_cnt', 'aid_session_click_cnt', \
                    'aid_session_cart_cnt', 'aid_session_order_cnt']).fill_null(0))
        df = df.with_columns(
            pl.col(['click2cart', 'click2order', 'cart2order']).fill_null(-1))
        return df
    
# ========================================
# 从交互的视角做特征，用户与商品之间的交互，主要考虑时间问题
# ========================================
class Interaction_Features(FeatureGen):
    def __init__(self, data_path, data_name, sava_suffix, type_id=None):
        super().__init__(data_path = data_path, data_name = data_name, sava_suffix=sava_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.sava_suffix = sava_suffix
        self.type_id = type_id
        if type_id is not None:
            id2type = {0: 'click', 1: 'cart', 2: 'order'}
            self.type = id2type[type_id]
        else:
            self.type = 'session'

    def prepare_features(self):
        df = pl.read_parquet(f'{self.data_path}/cv/{self.data_name}.parquet').lazy()
        session_ts_max = df.groupby('session').agg(pl.col('ts').max().alias('session_ts_max')) # 每个session中交互的最晚时间
        if self.type_id:
            df = df.filter(pl.col('type') == self.type_id)
        # 现在有ts列和ts_max列，当前与当前商品交互的时间和当前session交互的最晚时间
        df = df.join(session_ts_max, on='session')  
        df = df.with_column((pl.col('session_ts_max') - pl.col('ts')).alias('session_ts_diff_inter')) # 时间差特征
            
        df = df.groupby(['session', 'aid']).agg([
            pl.col('ts').count().alias(f'{self.type}_inter_cnt'),   # 同一session中，针对aid之间的交互次数
            pl.col('session_ts_diff_inter').min().alias(f'{self.type}_inter_last_time'), # 同一session中，最近一次交互aid持续时间
            pl.col('ts').max() # 同一session中，最近一次交互发生的时间
        ])

        df = df.with_columns(pl.col(['session', 'aid', f'{self.type}_inter_cnt', f'{self.type}_inter_last_time']))
        
        df = df.with_columns([
            # 最后一次交互所处的星期几，取值为0~6，分别代表周一到周日
            (pl.col('ts').cast(pl.Int64) *1000000).cast(pl.Datetime).dt.weekday().alias(f'{self.type}_last_weekday'),
            # 最后一次交互所处的一天中的时间段，将一天分为6个时段，每个时段4小时，取值为0~3，表示从零点开始第几个4小时时段
            ((pl.col('ts').cast(pl.Int64) * 1000000).cast(pl.Datetime).dt.hour()/6).cast(pl.Int16).alias(f'{self.type}_last_time_of_day')
        ]).drop('ts')
        

        df = df.collect()

        df.write_parquet(f'{self.data_path}/features/{self.data_name}_{self.sava_suffix}.parquet')

    def fill_null(self, df):
        df = df.with_column(pl.col(f'{self.type}_inter_cnt').fill_null(0))
        df = df.with_column(pl.col(f'{self.type}_inter_last_time').fill_null(-1))
        df = df.with_column(pl.col(f'{self.type}_last_weekday').fill_null(99))
        df = df.with_column(pl.col(f'{self.type}_last_time_of_day').fill_null(99))

        return df
    

# ========================================
# 转化特征，用户的行为序列，分析用户转化的情况
# 1. 点击后加购物车 2. 点击后下订单 3. 加购物车后下订单
# ========================================
class Conversion_Features(FeatureGen):
    def __init__(self, data_path, data_name, sava_suffix):
        super().__init__(data_path = data_path, data_name = data_name, sava_suffix=sava_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.sava_suffix = sava_suffix

    
    def prepare_features(self):
        if self.data_name == 'valid3':
            train_name = 'train12'
        elif self.data_name == 'valid4':
            train_name = 'train123'
        elif self.data_name == 'test':
            train_name = 'all_train'

        df_test = pl.read_parquet(f'{self.data_path}/cv/{self.data_name}.parquet')
        df_train = pl.read_parquet(f'{self.data_path}/cv/{train_name}.parquet')            
        
        df = pl.concat([df_test, df_train])
        del df_train, df_test

        click_train = df.filter(pl.col('type') == 0).drop('type').rename({'ts':'click_ts'})
        cart_train = df.filter(pl.col('type') == 1).drop('type').rename({'ts':'cart_ts'})
        order_train = df.filter(pl.col('type') == 2).drop('type').rename({'ts':'order_ts'})
        
        # ======================================== 1. 点击后加购物车 ========================================
        click_cart_diff = click_train.join(cart_train, on=['session', 'aid'], how='inner')
        click_cart_diff = click_cart_diff.with_columns((
            pl.col('click_ts') - pl.col('cart_ts')).alias('click_cart_diff'))
        click_cart_diff.filter(pl.col('click_cart_diff') <= 0).drop('click_cart_diff') #筛选出点击后又加购物车的商品
        click_cart_diff = click_cart_diff.groupby(['session', 'aid']).agg([
            pl.min('click_ts').alias('click_ts_min'),
            pl.min('cart_ts').alias('cart_ts_min')
        ])
        click_cart_diff = click_cart_diff.with_columns((pl.col('cart_ts_min') - pl.col('click_ts_min')).alias('cart_click_diff'))
        click_cart_diff = click_cart_diff.groupby('aid').agg([
                                    pl.count('aid').alias('click2cart_count'), # 商品
                                    pl.mean('cart_click_diff').alias('click2cart_diff_avg'), # 从点击到加购物车时间的均值
                                ])
        
        # ======================================== 2. 点击后下订单 ========================================
        click_order_diff = click_train.join(order_train, on=['session', 'aid'], how='inner')
        click_order_diff = click_order_diff.with_columns((
            pl.col('click_ts') - pl.col('order_ts')).alias('click_order_diff'))
        click_order_diff.filter(pl.col('click_order_diff') <= 0).drop('click_order_diff') #筛选出点击后又下订单的商品
        click_order_diff = click_order_diff.groupby(['session', 'aid']).agg([
            pl.min('click_ts').alias('click_ts_min'),
            pl.min('order_ts').alias('order_ts_min')
        ])
        click_order_diff = click_order_diff.with_columns((pl.col('order_ts_min') - pl.col('click_ts_min')).alias('order_click_diff'))
        click_order_diff = click_order_diff.groupby('aid').agg([
                                    pl.count('aid').alias('click2order_count'), # 商品
                                    pl.mean('order_click_diff').alias('click2order_diff_avg'), # 从点击到加购物车时间的均值
                                ])

        # ======================================== 3. 加购物车后下订单 ========================================
        cart_order_diff = cart_train.join(order_train, on=['session', 'aid'], how='inner')
        cart_order_diff = cart_order_diff.with_columns((
            pl.col('cart_ts') - pl.col('order_ts')).alias('cart_order_diff'))
        cart_order_diff.filter(pl.col('cart_order_diff') <= 0).drop('cart_order_diff') #筛选出点击后又下订单的商品
        cart_order_diff = cart_order_diff.groupby(['session', 'aid']).agg([
            pl.min('cart_ts').alias('cart_ts_min'),
            pl.min('order_ts').alias('order_ts_min')
        ])
        cart_order_diff = cart_order_diff.with_columns((pl.col('order_ts_min') - pl.col('cart_ts_min')).alias('order_cart_diff'))
        cart_order_diff = cart_order_diff.groupby('aid').agg([
                                    pl.count('aid').alias('cart2order_count'), # 商品
                                    pl.mean('order_cart_diff').alias('cart2order_diff_avg'), # 从点击到加购物车时间的均值
                                ])
        
        conversion_features = click_cart_diff.join(click_order_diff, on='aid', how='outer')
        conversion_features = conversion_features.join(cart_order_diff, on='aid', how='outer')
        
        conversion_features = conversion_features.fill_null(-1)  

        conversion_features.write_parquet(f'{self.data_path}/features/{self.data_name}_{self.sava_suffix}.parquet')




# ========================================
# 用户特征，每个用户点击、加购物车、下单多少商品
# ========================================
class Session_Features2(FeatureGen):
    def __init__(self, data_path, data_name, sava_suffix):
        super().__init__(data_path = data_path, data_name = data_name, sava_suffix=sava_suffix)
        self.data_path = data_path
        self.data_name = data_name
        self.sava_suffix = sava_suffix

    def prepare_features(self):
        if self.data_name == 'valid3':
            train_name = 'train12'
        elif self.data_name == 'valid4':
            train_name = 'train123'
        elif self.data_name == 'test':
            train_name = 'all_train'

        data = pd.read_parquet(f'{self.data_path}/cv/{self.data_name}.parquet')
        # df_train = pd.read_parquet(f'{self.data_path}/cv/{train_name}.parquet')            
        
        # data = pd.concat([df_test, df_train])
        # del df_train, df_test

        data = data.sort_values(['session', 'ts'], ascending=[True, False])
        data['ui_action_reverse'] = data.groupby('session')['aid'].cumcount()
        action_reverse = data.groupby(['session', 'aid'])['ui_action_reverse'].min()
        data['ui_action_reverse_by_type'] = data.groupby(['session', 'type'])['aid'].cumcount()
        action_reverse_by_type = pd.pivot_table(data, index=['session', 'aid'], columns=['type'],
                                                values=['ui_action_reverse_by_type'], aggfunc='min', fill_value=-1)
        action_reverse_by_type.columns = ['ui_action_reverse_click', 'ui_action_reverse_cart', 'ui_action_reverse_order']

        data = data.sort_values(['session', 'ts'], ascending=[True, True])


        data['session_length'] = data.groupby('session')['aid'].transform('count')
        data['ui_log_score'] = 2 ** (0.1 + ((1 - 0.1) / (data['session_length'] - 1)) * (
                data['session_length'] - data['ui_action_reverse'] - 1)) - 1
        data['ui_log_score'] = data['ui_log_score'].fillna(1.0)
        type_weights = {0: 1, 1: 6, 2: 3}
        data['ui_type_weight_log_score'] = data['type'].map(type_weights) * data['ui_log_score']

        type_weight_log_score = data.groupby(['session', 'aid'])['ui_type_weight_log_score'].sum()
        log_score = data.groupby(['session', 'aid'])['ui_log_score'].sum()
        type_weight_log_score = type_weight_log_score.round(4)
        log_score = log_score.round(4)

        session_aid_count = data.groupby(['session', 'aid'])['ts'].count().rename('ui_session_aid_count')

        history_count = pd.pivot_table(data, index=['session', 'aid'], columns=['type'], values=['ts'], aggfunc='count',
                                        fill_value=0)
        history_count.columns = ['ui_history_click_count', 'ui_history_cart_count', 'ui_history_order_count']

        data['ts_diff'] = data.groupby('session')['ts'].transform('max') - data['ts']
        last_ts_diff = np.log(data.groupby(['session', 'aid'])['ts_diff'].min() + 1)
        last_type = data.groupby(['session', 'aid'])['type'].last().astype('int8')
        user_item_feature = pd.merge(last_ts_diff, last_type, how='left', on=['session', 'aid'])
        user_item_feature.columns = ['ui_last_ts_diff', 'ui_last_type']
        user_item_feature = pd.merge(user_item_feature, history_count, how='left', on=['session', 'aid'])
        user_item_feature = pd.merge(user_item_feature, action_reverse, how='left', on=['session', 'aid'])
        user_item_feature = pd.merge(user_item_feature, action_reverse_by_type, how='left', on=['session', 'aid'])
        user_item_feature = pd.merge(user_item_feature, type_weight_log_score, how='left', on=['session', 'aid'])
        user_item_feature = pd.merge(user_item_feature, log_score, how='left', on=['session', 'aid'])
        user_item_feature = pd.merge(user_item_feature, session_aid_count, how='left', on=['session', 'aid'])
        user_item_feature = reduce_mem_usage(user_item_feature)

        user_item_feature.to_parquet(f'{self.data_path}/features/{self.data_name}_{self.sava_suffix}.parquet')


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df