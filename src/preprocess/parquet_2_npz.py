import pandas as pd
import numpy as np

def parquet_2_npz(path, file_name):
    df = pd.read_parquet(f'{path}{file_name}.parquet')
    ts_min = df.groupby('session').min()['ts']
    df['ts_min'] = df.session.map(ts_min)
    df['ts'] = df['ts'] - df['ts_min']
    df = df.loc[:, ['aid', 'ts', 'type']]
    df = df.to_numpy()
    np.savez(f"{path}{file_name}.npz", aids=df[:, 0].tolist(), ts=df[:, 1].tolist(), type=df[:, 2].tolist())
    
# def main():
#     path = './data/cross_validation/'
#     files = ['train_123', 'valid_4']
#     for file in files:
#         print(f'processing {file}')
#         parquet_2_npz(path, file)

#     path = './data/raw/'
#     files = ['train', 'test']
#     for file in files:
#         print(f'processing {file}')
#         parquet_2_npz(path, file)
    

# if __name__ == '__main__':
#     main()
