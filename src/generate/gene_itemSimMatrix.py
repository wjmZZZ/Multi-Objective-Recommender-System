import heapq
import math
import pickle
import numpy as np
import numba as nb
from tqdm import tqdm
import cudf 


'''计算item之间的关联度（co-occurrence）
参数
    items：一个用户的历史商品序列
    itemMatrix： 一个用于存储商品之间关联度的二维字典
    item_count：每个商品出现的次数

遍历每个用户的aid数组，判断每一个商品aid，如果不在商品矩阵中，就添加，并将其值也设置为一个字典。
再次检查是否在item_count中，如果不在，则添加，并且将值设置为0。如果在就累加1，
再次遍历aid数组，跳过同一个aid
'''
@nb.jit(nopython = True, cache = True)
def ItemMatrix_fn(itemMatrix, item_count, item_sequence):
    for i_loc, aid_i in enumerate(item_sequence):  # 每个用户的aid数组
        if aid_i not in itemMatrix: 
            itemMatrix[aid_i] = {0: 0.0 for _ in range(0)}

        # 统计商品出现的次数，商品被多少个不同用户购买
        if aid_i not in item_count: 
            item_count[aid_i] = 0

        item_count[aid_i] += 1
        
        for j_loc, aid_j in enumerate(item_sequence):
            if aid_i == aid_j:
                continue
            # 对于每对不同的商品(aid_i, aid_j)，计算一个局部权重值loc_weight，该值随着它们在序列中的位置距离的增加而指数级减小。
            # 这个权重值loc_weight越大，表示商品i和商品j之间的关联度越高；
            # 根据元素在序列中的位置，计算一个权重值，这个权重值随着位置之间的距离增加而指数级减小
            loc_alpha = 1.0 if i_loc > j_loc else 0.6
            loc_weight = loc_alpha * (0.5 **(np.abs(j_loc - i_loc) - 1))  # local_weight是一个局部权重值，值越大，表示两个商品i和j之间的关联越强

            # 在itemMatrix中，对于每个商品i，将其与所有与之不同的商品j的关联度加入到itemMatrix[i][j]中，
            # 同时也需要在itemMatrix[j][i]中将商品i与商品j的关联度加入，保证itemMatrix的对称性
            count = itemMatrix[aid_i]
            if aid_j not in count: 
                count[aid_j] = 0.0
            count[aid_j] += loc_weight


'''堆排序（heap sort）求top-k
输入是一个字典item_count，其中包含了若干个元素及其对应的计数值，k是需要输出的计数值最大的前k个元素的个数
输出是一个列表top_k_items，其中包含了前k个计数值最大的元素及其对应的计数值
算法核心是通过堆数据结构（heap）来维护当前最大的k个元素及其计数值
同时在遍历字典item_count的过程中，将每个元素及其计数值都加入堆中。如果堆的大小超过了k，
就从堆中弹出计数值最小的元素。最后将堆中剩下的元素按照计数值从大到小排序，将它们及其计数值存入列表item_count中并返回
'''
@nb.jit(nopython = True, cache = True)
def heap_topk(item_count,  k):
    heap = [(0.0, 0, 0) for _ in range(0)]
    for i, (item_id, count) in enumerate(item_count.items()):
        heapq.heappush(heap, (count, i, item_id))
        if len(heap) > k:
            heapq.heappop(heap)
    top_k_items = [heapq.heappop(heap) for _ in range(len(heap))][::-1]
    top_k_items = [(r[2], r[0]) for r in top_k_items]
    # top_k_items = [(item_id, count) for (_, _, item_id), count in top_k_items]
    return top_k_items            


'''为每个商品计算其与其他商品的相似度，并将相似度矩阵保存在itemSimMatrix字典中
参数
    itemSimMatrix: 一个字典，保存了每个商品与其他商品的相似度矩阵；
    i: 当前需要计算相似度的商品的ID；
    item_counts: 一个字典，保存了每个商品与多少用户发生交互；
    related_items: 一个字典，保存了与当前商品相似的其他商品以及它们之间的相似度；
首先检查itemSimMatrix中是否已经有当前商品的相似度矩阵，如果没有，则初始化一个空的相似度矩阵
从related_items中选择与当前商品相似度最高的500个商品，并根据余弦相似度公式计算它们与当前商品的相似度，并将结果保存在itemSimMatrix中
'''
@nb.jit(nopython = True, cache = True)
def ItemSimilarityMatrix_fn(item_sim_matrix, item_id, item_counts, related_items):
    if item_id not in item_sim_matrix: 
        item_sim_matrix[item_id] = {0: 0.0 for _ in range(0)}
    top_k_related_items = heap_topk(related_items, 500)
    
    for j, cij in top_k_related_items:
        count = item_sim_matrix[item_id]
        if j not in count: 
            count[j] = 0.0
        count[j] =  cij / math.sqrt(item_counts[item_id] * item_counts[j]) # 余弦相似度



def gene_itemSimMatrix(df_train, df_test, name, cfg):
    # last_week_time = df_train['ts'].max() - 7 * 24 * 3600
    # df_train = df_train[df_train['ts']>last_week_time]
    df = cudf.concat([df_train, df_test])
    del df_train, df_test
    user_item_dict = df.groupby("session")[['aid']].collect().to_pandas()
    uidict = user_item_dict['aid'].to_dict()
    
    # itemMatrix： 一个用于存储商品之间关联度的二维字典
    itemMatrix = nb.typed.Dict.empty(
                key_type = nb.types.int64,
                value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
    
    # item_count：记录每个商品出现的次数的字典
    item_count = nb.typed.Dict.empty(
                    key_type = nb.types.int64,
                    value_type = nb.types.int64)

    for user, items in tqdm(uidict.items()):
        ItemMatrix_fn(itemMatrix, item_count, items)

    # itemSimMatrix: 一个字典，保存了每个商品与其他商品的相似度矩阵；
    itemSimMatrix = nb.typed.Dict.empty(
                key_type = nb.types.int64,
                value_type = nb.typeof(nb.typed.Dict.empty(key_type = nb.types.int64, value_type = nb.types.float64)))
    for i, related_items in tqdm(itemMatrix.items()):
        ItemSimilarityMatrix_fn(itemSimMatrix, i, item_count, related_items)

    item_sim = {}
    for k,v in tqdm(itemSimMatrix.items()):
        item_sim[k] = dict(v)

    del itemSimMatrix,itemMatrix

    with open(f"{cfg.data_path}/item_cf/item_sim_{name}.pkl", "wb") as f:
        pickle.dump(item_sim, f)

    df = df.groupby('session')['aid'].agg(list).reset_index()

