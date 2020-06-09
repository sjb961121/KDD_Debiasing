import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from data_loader import load_data
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math
import csv

def get_sim_item(df_, user_col, item_col, use_iif=False):
    df = df_.copy()
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))

    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()  # 引入时间因素
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    sim_item = {}
    item_cnt = defaultdict(int)  # 商品被点击次数
    for user, items in tqdm(user_item_dict.items()):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                t1 = user_time_dict[user][loc1]  # 点击时间提取
                t2 = user_time_dict[user][loc2]
                sim_item[item].setdefault(relate_item, 0)
                if not use_iif:
                    if loc1 - loc2 > 0:
                        sim_item[item][relate_item] += 1 * 0.7 * (0.8 ** (loc1 - loc2 - 1)) * (
                                    1 - (t1 - t2) * 10000) / math.log(1 + len(items))  # 逆向
                    else:
                        sim_item[item][relate_item] += 1 * 1.0 * (0.8 ** (loc2 - loc1 - 1)) * (
                                    1 - (t2 - t1) * 10000) / math.log(1 + len(items))  # 正向
                else:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))

    sim_item_corr = sim_item.copy()  # 引入AB的各种被点击次数
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2)

    return sim_item_corr, user_item_dict

def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):
    '''
    input:item_sim_list, user_item, uid, 500, 50
    # 用户历史序列中的所有商品均有关联商品,整合这些关联商品,进行相似性排序
    '''
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.7 ** loc)


    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]

# from model import MyModel
# model=MyModel()

item_list=load_data()
item_set = set(item_list.keys())

#读取数据
# train_data=np.loadtxt('../data/ratings_train_0.txt',delimiter='\t')
# np.random.seed(100)
# np.random.shuffle(train_data)
# train_X=train_data[:,3:]
# train_X_1=train_data[:,0]
# train_X_2=train_data[:,1]
# train_X_1=np.expand_dims(train_X_1,-1)
# train_X_2=np.expand_dims(train_X_2,-1)
# train_Y=train_data[:,2]
test_data=np.loadtxt('../data/ratings_test_0.txt',delimiter='\t')
np.random.seed(100)
np.random.shuffle(test_data)
test_X=test_data[:,3:]
test_X_1=test_data[:,0]
test_X_2=test_data[:,1]
test_X_1=np.expand_dims(test_X_1,-1)
test_X_2=np.expand_dims(test_X_2,-1)
test_Y=test_data[:,2]
print('finish rating load...')

#模型结构
input1=tf.keras.Input(shape=(1,),dtype=tf.float32)
input2=tf.keras.Input(shape=(1,),dtype=tf.float32)
input=tf.keras.Input(shape=(256,),dtype=tf.float32)
user_embedding_matrix=tf.keras.layers.Embedding(40000,256,embeddings_regularizer=tf.keras.regularizers.l2(1e-6),embeddings_initializer=tf.initializers.TruncatedNormal(mean=0,stddev=0.01))
# item_embedding_matrix = tf.keras.layers.Embedding(120000, 256, embeddings_regularizer=tf.keras.regularizers.l2(0.01))
user_embedding=user_embedding_matrix(input1)
user_emd=Dense(units=512,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-6),kernel_initializer=tf.initializers.TruncatedNormal(mean=0,stddev=0.01))(user_embedding)
user_emd=tf.keras.layers.Dropout(0.2)(user_emd)
# user_emd=Dense(units=512,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-6),kernel_initializer=tf.initializers.TruncatedNormal(mean=0,stddev=0.01))(user_emd)
user_emd=Dense(units=256,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(1e-6),kernel_initializer=tf.initializers.TruncatedNormal(mean=0,stddev=0.01))(user_emd)
# user_emd=tf.keras.layers.Dropout(0.1)(user_emd)
user_emd=tf.keras.layers.BatchNormalization()(user_emd)
# item_embedding=item_embedding_matrix(input2)
# item_emd=Dense(units=256,activation='relu')(input)
# item_emd=tf.nn.l2_normalize(input)
item_emd=tf.keras.layers.BatchNormalization()(input)
user_emd=tf.reshape(user_emd,[-1,256])
# output=tf.keras.layers.Lambda(muti(user_emd,item_emd))
# scores = tf.reduce_sum(user_emd * item_emd)
output=tf.nn.sigmoid(tf.reduce_sum(user_emd * item_emd,axis=1))
model=tf.keras.Model(inputs=[input1,input2,input],outputs=output)

#selfModel
# start=0
# for epoch in range(epochs):
#     start=0
#     step_per_each=train_X.shape[0]//batch_size
#     optimizers = tf.keras.optimizers.Adam(learning_rate=0.01)
#     # with tqdm(total=step_per_each) as t:
#     for step in tqdm(range(step_per_each)):
#         with tf.GradientTape() as tape:
#             x_batch,y_batch=train_X[start:start+batch_size],train_Y[start:start+batch_size]
#             y_pred=model.train(x_batch)
#             loss=tf.keras.losses.binary_crossentropy(y_batch,y_pred)
#             g = tape.gradient(loss, model.trainable_variables)
#         optimizers.apply_gradients(grads_and_vars=zip(g, model.trainable_variables))
#         start+=batch_size
#         # t.set_postfix(loss=loss.numpy())
#     model.save_weights('../weights/model_weights')
#     # train_acc=model.eval(train_X,train_Y)
#     # print('epoch %d train_acc %f'%(epoch,train_acc))

#编译模型
model.compile(optimizer=tf.keras.optimizers.SGD(momentum=0.9,learning_rate=0.01),
                loss=tf.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy],
              )

#测试
model.load_weights('../weights/weights_model')
# model.evaluate([train_X_1,train_X_2,train_X],train_Y,batch_size=4096)
# model.evaluate([test_X_1,test_X_2,test_X],test_Y,batch_size=4096)
# start=time.time()
# print(tf.nn.top_k(model.predict([train_X_1[0:500],train_X_2[0:500],train_X[0:500]]),50))
print(model.predict([test_X_1[0:500],test_X_2[500:1000],test_X[500:1000]]))
x1=tf.convert_to_tensor(26654)
x1=tf.expand_dims(x1,0)
x2=tf.convert_to_tensor(67966)
x2=tf.expand_dims(x2,0)
x3=tf.convert_to_tensor(item_list[67966])
x3=tf.expand_dims(x3,0)
print(model.predict([x1,x2,x3]))
# print(time.time()-start)

#训练模型
# model.fit([train_X_1,train_X_2,train_X],
#           train_Y,
#           batch_size=4096,
#           epochs=100,
#
#           shuffle=True)
#
# model.summary()
# model.save_weights('../weights/weights_model')

# topK
# now_phase = 6
# train_path = '../data/underexpose_train'
# test_path = '../data/underexpose_test'
# recom_item = []
# top_k=dict()
# whole_click = pd.DataFrame()
# for c in range(now_phase + 1):
#     print('phase:', c)
#     click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c), header=None,
#                               names=['user_id', 'item_id', 'time'])
#     click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c, c), header=None,
#                              names=['user_id', 'item_id', 'time'])
#
#     all_click = click_train.append(click_test)
#     whole_click = whole_click.append(all_click)
#     whole_click = whole_click.drop_duplicates(subset=['user_id', 'item_id', 'time'], keep='last')
#     whole_click = whole_click.sort_values('time')
#
#     item_sim_list, user_item = get_sim_item(whole_click, 'user_id', 'item_id', use_iif=False)
#
#     for i in tqdm(click_test['user_id'].unique()):
#         # start=time.time()
#         recom_item=[]
#         rank_item = recommend(item_sim_list, user_item, i, 500, 500)
#         for j in rank_item:
#             if j[0] in item_list:
#                 recom_item.append([i, j[0], item_list[j[0]]])
#         if len(recom_item)<50:
#                 for ran in np.random.choice(list(item_set),size=50-len(recom_item)):
#                     recom_item.append([i,ran,item_list[ran]])
#         recom_item=np.array(recom_item)
#         p=[]
#         for item in recom_item[:,2]:
#             y = np.reshape(item, [-1, 256])
#             p.append(y)
#         p = np.array(p)
#         p = np.squeeze(p)
#         p1=np.expand_dims(recom_item[:,0],-1)
#         p2=np.expand_dims(recom_item[:,1],-1)
#         p3=recom_item[:,1]
#         p3=p3.astype(np.int64)
#         p1=p1.astype(np.float64)
#         p2=p2.astype(np.float64)
#         top_50=tf.nn.top_k(model.predict([p1,p2,p]),k=50)[1].numpy()
#         if i not in top_k:
#             top_k[i]=[]
#             top_k[i].append(i)
#             for top in top_50:
#                 top_k[i].append(p3[top])
#         # print(time.time()-start)
#
# #top_k写入csv
# f = open('../answer.csv', 'w', encoding='utf-8', newline="")
# d=sorted(top_k.items(), key = lambda k: k[0])
# csv_writer = csv.writer(f)
# for item in tqdm(d):
#     csv_writer.writerow(item[1])
# f.close()

