import argparse
from pathlib import Path
import pandas as pd
# import cudf as pd
import numpy as np
import gc
from tqdm import tqdm

def json_2_df(data_path):
    
    sessions_df = pd.DataFrame()
    chunks = pd.read_json(data_path, lines=True, chunksize=1000000)

    for chunk in tqdm(chunks, desc=f" '{data_path}' --> JSON to DataFrame "):
        dict = {
            'session': [],
            'aid': [],
            'ts': [],
            'type': [],
        }
    
        # 遍历每一个session
        for session_id, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
            # 每个session下对应的多个event共享同一个session id
            for event in events: 
                dict['session'].append(session_id)
                dict['aid'].append(event['aid'])
                dict['ts'].append(event['ts'])
                dict['type'].append(event['type'])
        chunk_session = pd.DataFrame(dict)
        chunk_session['session'] = chunk_session['session'].astype(np.int32)
        chunk_session['aid'] = chunk_session['aid'].astype(np.int32)
        chunk_session['ts'] = chunk_session['ts']/1000
        chunk_session['ts'] = chunk_session['ts'].astype(np.int32)
        chunk_session['type'] = chunk_session['type'].map({'clicks': 0, 'carts': 1,  'orders': 2}).astype(np.int8)
        sessions_df = pd.concat([sessions_df, chunk_session])
        del chunk_session, dict
        gc.collect()

    sessions_df = sessions_df.reset_index(drop=True)
    
    return sessions_df


def label_json_2_df(data_path):

    chunks = pd.read_json(data_path, lines=True, chunksize=1000000)
    labels_df = pd.DataFrame()
    
    for chunk in tqdm(chunks, desc=f" '{data_path}' --> Label JSON to DataFrame "):
        dict = {
            'session': [],
            'clicks': [],
            'carts': [],
            'orders': [],
            }
        # label_dic = {
        #     'session':[],
        #     'type':[],
        #     'aid':[]
        # }
        for session, labels in zip(chunk['session'].tolist(), chunk['labels'].tolist()):
            dict['session'].append(session)
            if 'clicks' in labels:
                dict['clicks'].append(labels['clicks'])
            else:
                dict['clicks'].append(-1)
            if 'carts' in labels:
                dict['carts'].append(labels['carts'])
            else:
                dict['carts'].append([])
            if 'orders' in labels:
                dict['orders'].append(labels['orders'])
            else:
                dict['orders'].append([])
        chunk_labels = pd.DataFrame(dict)
        chunk_labels['session'] = chunk_labels['session'].astype(np.int32)
        chunk_labels['clicks'] = chunk_labels['clicks'].astype(np.int32)
        labels_df = pd.concat([labels_df, chunk_labels])
    labels_df = labels_df.reset_index(drop=True)
    return labels_df


# def main():
#     file_list = ['train', 'test']
#     path_list = [f'data/raw/{file}.jsonl' for file in file_list]
#     out_path_list = [f'data/raw/{file}.parquet' for file in file_list]
#     for input_path, out_path in zip(path_list, out_path_list):
#         # print(f" {input_path} --> JSON to DataFrame ")
#         df = json_2_df(input_path)
#         df.to_parquet(out_path)
    
#     file_list = ['train_123', 'valid_4']
#     path_list = [f'data/cross_validation/{file}.jsonl' for file in file_list]
#     out_path_list = [f'data/cross_validation/{file}.parquet' for file in file_list]
#     for input_path, out_path in zip(path_list, out_path_list):
#         # print(f"\n {input_path} --> JSON to DataFrame ")
#         df = json_2_df(input_path)
#         df.to_parquet(out_path)

    
#     file_list = ['valid_4_labels']
#     path_list = [f'data/cross_validation/{file}.jsonl' for file in file_list]
#     out_path_list = [f'data/cross_validation/{file}.parquet' for file in file_list]
#     for input_path, out_path in zip(path_list, out_path_list):
#         df = label_json_2_df(input_path)
#         df.to_parquet(out_path)
    


# if __name__ == '__main__':
#     main()
