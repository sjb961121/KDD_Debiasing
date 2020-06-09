import time
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

train_path = '../data/underexpose_train'
test_path = '../data/underexpose_test'

# 获取当前阶段所有数据
def get_data_files(phase=6):
    train_files=[]
    test_files=[]
    for i in range(0,phase+1):
      train_files.append(train_path+'/underexpose_train_click-{}.csv'.format(i))
      test_files.append(test_path+'/underexpose_test_click-{}.csv'.format(i))
    return train_files,test_files


def get_all_data(train_files=None, test_files=None):
    """
    获取全阶段所有数据集
    """
    if train_files and test_files:
        # 归并全部数据集
        whole_train = pd.DataFrame()
        whole_test = pd.DataFrame()
        for train_file in train_files:
          dt = pd.read_csv(train_file, header=None, names=['user_id', 'item_id', 'time'])
          dt = dt.drop_duplicates()  # 去重
          whole_train = whole_train.append(dt)
          whole_train['click']=1
        for test_file in test_files:
          dt=pd.read_csv(test_file, header=None, names=['user_id', 'item_id', 'time'])
          dt=dt.drop_duplicates()
          whole_test = whole_test.append(dt)
    return whole_train, whole_test

train_files, test_files = get_data_files(6)
whole_train, whole_test = get_all_data(train_files, test_files)

def get_Negative_Sampleling(df):
    """df: 全T训练集
    """
    # 每个user_id只保留一次
    df = df.drop_duplicates(['user_id'],keep='last')
    # 构造user_item列可以查看某一个user_item组合是否已经出现过
    df['user_item'] = df['user_id'].apply(lambda x: str(x)) + '_' + df['item_id'].apply(lambda x:str(x))
    # 为所有用户，每个用户都召回100个负样本
    negative_sample = []
    for user in tqdm(df['user_id'].unique()):
        # 为每个用户召回100个负样本
        user_negative_sample = []
        for item in df['item_id'].unique():
          if len(user_negative_sample)>=100:
              break
          else:
              user_item = str(user) + '_' + str(item)
              if user_item in df['user_item'].unique():
                  continue
              else:
                  user_negative_sample.append([user, item])
        negative_sample.extend(user_negative_sample)
    negative_sample = pd.DataFrame(data=negative_sample, columns=['user_id','item_id'])
    df = pd.concat([df,negative_sample])
    df = df.reset_index(drop=True)
    df['click'] = df['click'].fillna(0.0)
    return df

t = time.time()
Negative_Sampleling_data = get_Negative_Sampleling(whole_train)
Negative_Sampleling_data.to_csv('Negative_Sampleling_data2.csv',index=False)
print("Running time:{}".format(time.time()-t))
