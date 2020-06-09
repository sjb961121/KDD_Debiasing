import tensorflow as tf
# import tensorflow_probability as tp
import math
import tensorflow.keras.backend as K
from keras import backend as k
# tf.enable_eager_execution()
# tf.compat.v1.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )
# import pandas as pd
import numpy as np
# x=[1,2,3,4,5]
#
# x=tf.keras.layers.BatchNormalization(x)

# x = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
# for i,j in x.iterrows():
#     print(i)
#     print(j)

x=[[1,2,[3,4]],[2,3,[4,5]],[3,4,[5,6]]]
print(len(x))
# x=[1,2,3,4,5,6]
x.append(7)
print(x)
# print(np.shape(x[1]))
# print(np.shape([x[0],x[1]]))
# input1=tf.constant(value=1)
# print(input1)
# input2=tf.keras.Input(shape=(1,),dtype='int32')
# print(input2)
# output=input2
# model=tf.keras.Model(input2,output)
#
# print(K.is_keras_tensor(input1))
# print(K.is_keras_tensor(input2))
# print(k.eval(input2))

# t=tf.convert_to_tensor([30,20,50],dtype=tf.float32)
# t=tf.nn.l2_normalize(t)
# t=tf.keras.layers.BatchNormalization()(t)
# # t=tf.keras.layers.LayerNormalization(t)
# t=tf.nn.sigmoid(t)

# print(t.numpy())

# np.random.seed(100)
# np.random.shuffle(x)
# print(tf.nn.top_k(x,2)[1].numpy())
# print(x)
# x=np.array(x)
#
# p1=np.expand_dims(x[:,0],-1)
# p1=p1.astype(np.float64)
# p2=np.expand_dims(x[:,1],-1)
# p2=p2.astype(np.float64)

from tqdm import tqdm
import csv
y=dict()
y[1]=[2,3,4]
y[3]=[4,5,6]
y[2]=[3,4,5]
d=sorted(y.items(), key = lambda k: k[0])
print(d)
f = open('../test.csv','w',encoding='utf-8',newline="")
csv_writer = csv.writer(f)
for item in tqdm(d):
    csv_writer.writerow(item[1])
f.close()
# p=[]
# for item in x[:,2]:
#     y=np.reshape(item,[-1,2])
#     p.append(y)
# p=np.array(p)
# p=np.squeeze(p)
# print(p)