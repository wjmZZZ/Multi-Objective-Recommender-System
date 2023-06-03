from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import numba as nb
# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=10, progress_bar=True)

# 使用 Numba 优化循环过程
@nb.njit(parallel=True)
def scale_scores(scores):
    min_ = np.min(scores)
    max_ = np.max(scores)
    if min_ == max_:
        return np.full_like(scores, 1)
    else:
        return (scores - min_) / (max_ - min_)

@nb.jit(nopython = True, cache = True)
def calculate_weights(aids_temp ,aids, types, type_weight, start =0.1):
    weights = np.power(2, np.linspace(start, 1, len(aids))) - 1
    n = len(aids)
    for i in nb.prange(n):
        aid = aids[i]
        w = weights[i]
        t = types[i]          
        if aid not in aids_temp:
            aids_temp[aid] = 0.0
        aids_temp[aid] += w * type_weight[t]

    return sorted(aids_temp.items(), key=lambda x: x[1], reverse=True)    



def clicks_recall(df, top_hot_items, top_hot_items_last_month, top_clicks, top_clicks_items_last_month):
    aids = np.array(df.aid)
    types = np.array(df.type)


    aids_temp = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)
    type_weight = np.array([1.0, 5.0, 4.0])
    most_common = calculate_weights(aids_temp ,aids, types, type_weight)
    history_aids = [k for k, v in most_common]
    type_1 = [1] * len(history_aids)
    history_score = [v for k, v in most_common]
    if len(set(history_score)) == 1:
        history_score = [1] * len(history_score)
    else:
        min_ = min(history_score)
        max_ = max(history_score)
        history_score = [(j - min_) / (max_ - min_) for j in history_score]


    # 基于规则召回n个
    # 热门商品召回k个,类别加权
    top_hot_items_k = list(top_hot_items.keys())[:10]
    type_2 = [2] * (len(top_hot_items_k))
    top_hot_items_score = list(top_hot_items.values())[:10]

    # 点击最多的商品召回k个
    top_clicks_k = list(top_clicks.keys())[:10]
    type_3 = [3] * (len(top_clicks_k))
    top_clicks_score = list(top_clicks.values())[:10]

    # 过去一个月点击最多的商品召回k个
    top_clicks_last_month_k = list(top_clicks_items_last_month.keys())[:10]
    type_4 = [4] * (len(top_clicks_last_month_k))
    top_clicks_last_month_score = list(top_clicks_items_last_month.values())[:10]

    # 过去一个月热度最高的k个商品
    top_hot_items_one_month_k = list(top_hot_items_last_month.keys())[:10]
    type_5 = [5] * (len(top_clicks_last_month_k))
    top_hot_items_one_month_score = list(top_hot_items_last_month.values())[:10]

                
    result = history_aids + top_hot_items_k + top_clicks_k + top_clicks_last_month_k + top_hot_items_one_month_k
    type = type_1 + type_2 + type_3 + type_4 + type_5
    score = history_score + top_hot_items_score + top_clicks_score + top_clicks_last_month_score + top_hot_items_one_month_score 
    
    info = [str(result[i]) + "#" + str(type[i]) + "#" + str(score[i]) for i in range(len(result))]
    return info






def carts_recall(df,top_hot_items,top_orders,top_carts,top_hot_items_last_month):
    aids = np.array(df.aid)
    types = np.array(df.type)


    aids_temp = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)
    type_weight = np.array([1.0, 5.0, 4.0])
    most_common = calculate_weights(aids_temp ,aids, types, type_weight, start=0.5)
    history_aids = [k for k, v in most_common]
    type_1 = [1] * len(history_aids)
    history_score = [v for k, v in most_common] 

    if len(set(history_score)) == 1:
        history_score = [1] * len(history_score)
    else:
        min_ = min(history_score)
        max_ = max(history_score)
        history_score = [(j - min_) / (max_ - min_) for j in history_score]

    # 基于规则召回k个
    # 热门商品召回k个,类别加权
    top_hot_items_k = list(top_hot_items.keys())[:10]
    type_2 = [2] * len(top_hot_items_k)
    top_hot_items_score = list(top_hot_items.values())[:10]
    # 购买最多的商品召回k个
    top_orders_k = list(top_orders.keys())[:10]
    type_3 = [3] * len(top_orders_k)
    top_orders_score = list(top_orders.values())[:10]
    # 加购最多的商品召回k个
    top_carts_k = list(top_carts.keys())[:10]
    type_4 = [4] * len(top_carts_k)
    top_carts_score = list(top_carts.values())[:10]
    # 过去一个月热度最高的k个商品
    top_hot_items_one_month_k = list(top_hot_items_last_month.keys())[:10]
    type_5 = [5] * len(top_hot_items_one_month_k)
    top_hot_items_one_month_score = list(top_hot_items_last_month.values())[:10]


    result = history_aids + top_hot_items_k + top_orders_k + top_carts_k + \
             top_hot_items_one_month_k 
    type = type_1 + type_2 + type_3 + type_4 + type_5

    score = history_score + top_hot_items_score + top_orders_score + top_carts_score + top_hot_items_one_month_score 

    info = [str(result[i]) + "#" + str(type[i]) + "#" + str(score[i]) for i in range(len(result))]
    return info




def orders_recall(df, top_hot_items, top_orders, top_carts, top_hot_items_last_month):
    aids = np.array(df.aid)
    types = np.array(df.type)

    aids_temp = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)
    type_weight = np.array([1.0, 5.0, 4.0])
    most_common = calculate_weights(aids_temp ,aids, types, type_weight, start=0.5)
    history_aids = [k for k, v in most_common]
    type_1 = [1] * len(history_aids)
    history_score = [v for k, v in most_common] 

    if len(set(history_score)) == 1:
        history_score = [1] * len(history_score)
    else:
        min_ = min(history_score)
        max_ = max(history_score)
        history_score = [(j - min_) / (max_ - min_) for j in history_score]

    # 基于规则召回n个
    # 热门商品召回k个,类别加权
    top_hot_items_k = list(top_hot_items.keys())[:10]
    type_2 = [2] * len(top_hot_items_k)
    top_hot_items_score = list(top_hot_items.values())[:10]
    # 购买最多的商品召回k个
    top_orders_k = list(top_orders.keys())[:10]
    type_3 = [3] * len(top_orders_k)
    top_orders_score = list(top_orders.values())[:10]
    # 加购最多的商品召回k个
    top_carts_k = list(top_carts.keys())[:10]
    type_4 = [4] * len(top_carts_k)
    top_carts_score = list(top_carts.values())[:10]
    # 过去一个月热度最高的k个商品
    top_hot_items_one_month_k = list(top_hot_items_last_month.keys())[:10]
    type_5 = [5] * len(top_hot_items_one_month_k)
    top_hot_items_one_month_score = list(top_hot_items_last_month.values())[:10]

    

    result = top_hot_items_k + top_orders_k + top_carts_k + top_hot_items_one_month_k 
    type = type_1 + type_2 + type_3 + type_4 + type_5
    score =  top_hot_items_score + top_orders_score + top_carts_score + top_hot_items_one_month_score 
    info = [str(result[i]) + "#" + str(type[i]) + "#" + str(score[i]) for i in range(len(result))]
    return info

