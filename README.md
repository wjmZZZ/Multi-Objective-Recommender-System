# Multi-Objective Recommender System

## 任务描述

为每个user (session)推荐 top20 点击(clicks)、加入购物车(carts)、购买(buys)的商品(aids)







## Data Format

数据集：[OTTO Recommender Systems Dataset](https://github.com/otto-de/recsys-dataset)

会话session被存储为`JSON`对象，包含一个唯一的session id和一个event列表：

```python
{
    "session": 42,
    "events": [
        { "aid": 0, "ts": 1661200010000, "type": "clicks" },
        { "aid": 1, "ts": 1661200020000, "type": "clicks" },
        { "aid": 2, "ts": 1661200030000, "type": "clicks" },
        { "aid": 2, "ts": 1661200040000, "type": "carts"  },
        { "aid": 3, "ts": 1661200050000, "type": "clicks" },
        { "aid": 3, "ts": 1661200060000, "type": "carts"  },
        { "aid": 4, "ts": 1661200070000, "type": "clicks" },
        { "aid": 2, "ts": 1661200080000, "type": "orders" },
        { "aid": 3, "ts": 1661200080000, "type": "orders" }
    ]
}
```

- `train.jsonl ` 训练数据，包含完整的会话数据
- `session` - 唯一的会话ID
- `events` -- 会话中的事件(按时间排序)
  - `aid` - 商品ID
  - `ts` - Unix时间戳
  - `type` - 事件类型，点击商品，添加购物车，下订单



对于测试集中的每个session，预测最后一个时间戳之后发生的每种类型的商品id，即测试集包含了按时间戳截断的session，预测截断点之后发生的事情

对于点击来说，每个session只有一个真实值，即在该会话中点击的下一个商品（尽管仍然可以预测多达20个商品）。加购物车和下订单的真实值包含在session期间分别被添加到购物车和订购的所有商品。

![ground_truth.png](https://github.com/otto-de/recsys-dataset/blob/main/.readme/ground_truth.png?raw=true)

## 交叉验证

共5周的数据，训练集是4周，测试集是一周

```python
all_train  : before=2022-08-01 06:00:00 - 2022-08-29 05:59:59
all_test   : before=2022-08-29 06:00:00 - 2022-09-05 05:59:51
```

 ![train_test_split.png](https://github.com/otto-de/recsys-dataset/blob/main/.readme/train_test_split.png?raw=true)

划分方式：

前两周作为训练集`train12`，第三周作为验证集`valid3`

前三周作为训练集`train123`，第四周作为验证集`valid4`



模型第一阶段以`valid3`训练，以`valid4`评估，得到本地验证结果

第二阶段以`valid4`训练，得到`test`的推荐结果

> 不拿训练集去训练模型，反而拿验证集训练？这里训练集，主要是在生成候选以及特征工程中起到了帮助作用，而当我们在训练ranker模型的时候，利用的是训练集（和验证集）得到候选以及特征，来帮助验证集去构建可用于ranker的训练数据，



## 评估指标 Recall@20

三种行为的加权
$$
score = 0.10 \cdot R_{clicks} + 0.30 \cdot R_{carts} + 0.60 \cdot R_{orders}
$$

$$
R_{type} = \frac{ \sum\limits_{i=1}^N | \{ \text{predicted aids} \}_{i, type} \cap \{ \text{ground truth aids} \}_{i, type} | }{ \sum\limits_{i=1}^N \min{( 20, | \{ \text{ground truth aids} \}_{i, type} | )}}
$$


# code

```
├── data
│   ├── candidates 
│   ├── covisit
│   ├── cv
│   ├── features
│   ├── hot_items
│   ├── item_cf
│   ├── raw
│   └── word2vec
├── outputs
└── src
    ├── Candidates.py
    ├── Config.py
    ├── Dataset.py
    ├── Evaluate.py
    ├── Features.py
    ├── Main.py  
    ├── Model.py
    ├── Utils.py
    ├── generate
    │   ├── gene_covisit.py
    │   ├── gene_hot_items.py
    │   ├── gene_itemcf.ipynb
    │   ├── gene_itemSimMatrix.py
    ├── preprocess
    │   ├── json_2_parquet.py
    │   ├── parquet_2_npz.py
    └── └── train_val_split.py
```



## stage1：多路召回

共同访问矩阵召回

最近事件召回

word2vec召回

热门商品召回

## Stage2：特征工程

用户特征

商品特征

用户商品交互特征

转化特征

相似度特征



### 工程优化问题

海量数据的分析和存储，代码计算的性能优化

1、 将数据提前转换为parquet、npz等节约内存的数据格式，提前准备需要的数据，实验过程中对已做好的召回和特征进行保存，方便快速实验

2、 利用cudf加速pandas的处理，利用polars加速Dataframe处理，利用numba加速循环代码，利用多线程并行加速apply处理

3、 在拼接表之后，及时回收内存，避免内存溢出
