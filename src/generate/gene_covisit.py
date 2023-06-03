import gc
import pickle

import pandas as pd
import numpy as np
import numba as nb
import heapq
from tqdm import tqdm


TAIL = 30
PARALLEL = 2048
TOPN = 20
TYPES_WEIGTHS = np.array([1.0, 6.0, 3.0])
TYPE_WEIGHT = 0
TIME_WEIGHT = 1



'''
这个方法的作用是计算给定数据集中每个商品对之间的相似度。
具体来说，它使用给定的商品权重和时间权重计算每对商品之间的相似度，并将结果存储在一个字典中。
其中，相似度的计算方式取决于传递给方法的模式参数。
如果模式为OPWEIGHT，则相似度是根据用户的商品权重计算的。
如果模式为TIMEWEIGHT，则相似度是根据用户的时间权重计算的。
在计算相似度时，方法会遍历每个用户的商品序列，并计算每对商品之间的相似度。
如果两个商品之间的时间间隔超过24小时，则停止计算。如果两个操作属于同一个商品，则跳过它们。
最后，方法将每对商品之间的相似度存储在一个字典中，并返回该字典。
'''
@nb.jit(nopython=True, cache=True)
def get_single_pairs(pairs, aids, ts, ops, idx, length, start_time, ops_weights, mode):
    # 只考虑操作序列中最近的一段时间内的商品，从而提高计算效率
    max_idx = idx + length
    min_idx = max(max_idx - TAIL, idx)
    for i in range(min_idx, max_idx):
        for j in range(i + 1, max_idx):
            # 判断两个操作之间的时间间隔是否超过了24小时，如果两个操作之间的时间间隔超过了24小时，则停止计算它们之间的相似度。因为，如果两个操作之间的时间间隔太长，它们之间的关联性可能会变得很弱，从而影响相似度的计算结果。
            if ts[j] - ts[i] >= 24 * 60 * 60:  
                break
            # 判断第i个操作和第j个操作是否属于同一个商品。跳过同一个商品的相似度计算，因为同一个用户的操作之间的关联性很强，计算它们之间的相似度没有意义。
            if aids[i] == aids[j]:
                continue

            # 这段代码计算了两个商品之间的权重，用于后续的推荐。
            # 其中，w1和w2分别表示第i个商品和第j个商品的权重。
            if mode == TYPE_WEIGHT:
                # 不同事件类型对应的权重不同
                w1 = ops_weights[ops[j]]
                w2 = ops_weights[ops[i]]
                
            elif mode == TIME_WEIGHT:  # FIXME
                # 这里的权重计算方式是基于时间的，即两个操作之间的时间间隔越短，它们之间的权重就越大。
                # 具体来说，这里使用了一个线性函数，将时间间隔映射到[1,4]的区间上，其中1659304800和1662328791分别是总体数据集train test的起始时间和结束时间。
                w1 = 1 + 3 * (ts[i] + start_time - 1659304800) / (1662328791 - 1659304800)
                w2 = 1 + 3 * (ts[j] + start_time - 1659304800) / (1662328791 - 1659304800)
            
            # 计算所有商品之间的权重，用于后续的推荐。
            pairs[(aids[i], aids[j])] = w1
            pairs[(aids[j], aids[i])] = w2


'''
函数使用Numba库进行JIT编译，以提高性能。并行获取每个会话的字典对 将对合并成嵌套的字典格式 (cnt)
函数的作用是计算aids中的每对不同商品对之间的权重，并将它们存储在cnts中。
具体来说，函数首先将row中的每个元素作为参数传递给get_single_pairs函数，该函数计算单个session中商品对的权重，并将它们存储在一个字典中。
然后，函数将所有字典合并到一个列表中，并将它们传递给一个循环，该循环将所有配对的权重添加到cnts中。最后，函数返回cnts。
'''
@nb.jit(nopython=True, parallel=True, cache=True)
def get_pairs(aids, ts, ops, row, cnts, ops_weights, mode):
    par_n = len(row)
    # 这行代码是在初始化一个长度为parn的列表pairs，其中每个元素都是一个字典，字典中只有一个键值对{(0, 0): 0.0}。这个字典用来存储每个pair之间的权重
    pairs = [{(0, 0): 0.0 for _ in range(0)} for _ in range(par_n)]
    for par_i in nb.prange(par_n):
        _, idx, length, start_time = row[par_i]
        get_single_pairs(pairs[par_i], aids, ts, ops, idx, length, start_time, ops_weights, mode)
        
    # 遍历pairs中的每个元素，将其中的aid1和aid2作为字典cnts的键，将w作为字典cnts中aid1对应的值中aid2对应的值加上去。
    # 如果aid1或aid2不在cnts中，则会将其加入cnts中，并将其值初始化为0.0。
    # 这段代码的作用是统计每个aid与其他aid的权重之和，以便后续进行Top-K筛选。
    for par_i in range(par_n):
        for (aid1, aid2), w in pairs[par_i].items():
            if aid1 not in cnts:
                cnts[aid1] = {0: 0.0 for _ in range(0)}
            cnt = cnts[aid1]
            if aid2 not in cnt:
                cnt[aid2] = 0.0
            cnt[aid2] += w

'''
堆排序的实现，用于从一个字典中获取前k个最大的键。其中，参数cnt是一个字典，表示每个键对应的值，overwrite表示是否覆盖已有的键，cap表示最多返回的键的数量。
函数首先将字典中的键值对转化为元组，然后将元组按照值从大到小排序，最后返回前k个键。如果overwrite为1，则表示覆盖已有的键，否则表示不覆盖已有的键。
使用最小堆从cnt字典中获取最常用的键  overwrite == 1 表示权重相同的后面的项目更重要  否则，意味着前一个权重相同的项目更重要  结果从高权重到低权重排序
'''
@nb.jit(nopython=True, cache=True)
def heap_topk(cnt, overwrite, cap):
    q = [(0.0, 0, 0) for _ in range(0)]
    for i, (k, n) in enumerate(cnt.items()):
        if overwrite == 1:
            heapq.heappush(q, (n, i, k))
        else:
            heapq.heappush(q, (n, -i, k))
        if len(q) > cap:
            heapq.heappop(q)
    return [heapq.heappop(q)[2] for _ in range(len(q))][::-1]


'''
对于每个aid1，从cnts中获取与之相关的计数字典cnt，然后使用heap_topk函数获取cnt中计数最大的前k个aid2，并将结果存储在topk字典中。
其中，topk字典的键为aid1，值为一个长度为k的ndarray，存储了与aid1相关的计数最大的前k个aid2。
'''
@nb.jit(nopython=True, cache=True)
def get_topk(cnts, topk, k):
    for aid1, cnt in cnts.items():
        topk[aid1] = np.array(heap_topk(cnt, 1, k))

    


'''
这个函数的作用是训练一个模型，返回一个字典，其中包含两个键值对，分别对应两种模式（OPWEIGHT和TIMEWEIGHT）下的前TOPN个操作。
具体实现过程是，首先定义一个空字典topks，然后对于每种模式，定义一个空字典cnts，用于存储操作对出现的次数，
然后对于数据集df中的每一行，调用getpairs函数计算操作对的权重，并将结果存储在cnts中。
接着，调用gettopk函数，将cnts中每个操作对出现次数前TOPN的操作存储在一个新的字典topk中。
最后，将topk存储在topks中，并返回topks。
'''
def gene_covisit_weight(df, aids, ts, types, TOPN, cfg):
    topks = {}
    # mode 0：计数器按操作类型加权，TYPE_WEIGHTs = [1.0, 6.0, 3.0]。这将用于购物车和订单预测。 方式一：计数器按运行时间加权。这将用于点击预测。
    for mode in [TYPE_WEIGHT, TIME_WEIGHT]:  # 类型， 时间 两种模式
        # 这行代码定义了一个空的字典cnts，其中键为int64类型，值为另一个字典，键为int64类型，值为float64类型。
        # 这个字典用于存储商品对出现的次数。
        # cnts的每个键代表一个商品的id，每个值是一个字典，其中键代表另一个商品的id，值代表这两个商品组成的商品对出现的次数。
        cnts = nb.typed.Dict.empty(
                        key_type = nb.types.int64,
                        value_type = nb.typeof(
                                            nb.typed.Dict.empty(
                                                key_type = nb.types.int64, 
                                                value_type = nb.types.float64
                                                                )
                                                )
                                                     )
        max_idx = len(df)
        for idx in tqdm(range(0, max_idx, PARALLEL), desc=f" Genetate mode {mode} Covisit csv "):
            row = df.iloc[idx : min(idx + PARALLEL, max_idx)][['session', 'idx', 'length', 'start_time']].values
            get_pairs(aids, ts, types, row, cnts, TYPES_WEIGTHS, mode)
        # 空字典 topk，它的键是整数类型，值是整数类型的数组。这个字典将在 train 函数中被填充。具体来说，get_topk 函数将会使用 cnts 字典中的数据来填充 topk 字典。
        topk = nb.typed.Dict.empty(
                        key_type=nb.types.int64,
                        value_type=nb.types.int64[:])
        get_topk(cnts, topk, TOPN)
        gc.collect()
        topks[mode] = topk

    topks_type_weight = {}
    for k, v in topks[TYPE_WEIGHT].items():
        topks_type_weight[k] = v
    topks_time_weight = {}
    for k, v in topks[TIME_WEIGHT].items():
        topks_time_weight[k] = v
        

    return topks_type_weight, topks_time_weight


'''
与训练函数一样，我们在一次 jit 函数调用中处理 parallel=1024 个会话。对于每个会话，如果unique的aid超过 20，我们将按时间衰减权重对它们进行重新排序。
否则，我们使用当前的aid从他们的 topk 共同访问候选人中召回新的aid，并根据查询的操作类型 (test_ops_weights=[1.0, 6.0, 3.0]) 进行加权。
'''
@nb.jit()
def inference_(aids, ops, row, result, topk, test_ops_weights, seq_weight, TOPK):
    for session, idx, length in row:
        unique_aids = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)
        cnt = nb.typed.Dict.empty(
            key_type=nb.types.int64, value_type=nb.types.float64)

        candidates = aids[idx:idx + length][::-1]
        candidates_ops = ops[idx:idx + length][::-1]
        for a in candidates:
            unique_aids[a] = 0



            result_candidates = list(unique_aids)
            sequence_weight = np.power(2, np.linspace(0.1, 1, len(result_candidates)))[::-1] - 1

            for a,w in zip(result_candidates,sequence_weight):
                if a not in topk:
                    continue
                for b in topk[a]:
                    if b in unique_aids:
                        continue

                    if b not in cnt: 
                        cnt[b] = 0    
                    cnt[b] += w                    
            result_candidates.extend(heap_topk(cnt, 0, TOPK)) # 20 - len(result_candidates)
        result[session] = np.array(result_candidates)


@nb.jit()
def inference(aids, ops, row,
              result_clicks, result_buy,
              topk_clicks, topk_buy,
              test_ops_weights, TOPK):
    inference_(aids, ops, row, result_clicks,
               topk_clicks, test_ops_weights, 0.1, TOPK)
    inference_(aids, ops, row, result_buy, topk_buy, test_ops_weights, 0.5, TOPK)
