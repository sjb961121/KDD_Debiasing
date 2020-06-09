import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from data_loader import load_data
import numpy as np

# global item_list
# item_list=load_data()

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # item_list=load_data()
        self.build_layer()

    def build_layer(self):
        self.mlp_1=Dense(units=512,activation=None)
        self.mlp_2=Dense(units=256,activation='sigmoid')
        self.mlp_3=Dense(units=512,activation=None)
        self.mlp_4 = Dense(units=256, activation='sigmoid')
        self.user_embedding_matrix = tf.keras.layers.Embedding(40000, 256, embeddings_regularizer=tf.keras.regularizers.l2)
        self.item_embedding_matrix = tf.keras.layers.Embedding(120000, 256, embeddings_regularizer=tf.keras.regularizers.l2)
        self.bn=tf.keras.layers.BatchNormalization()

    def train(self,inputs):
        user_index_list=inputs[:,0]
        item_index_list=inputs[:,1]
        user_embedding=self.user_embedding_matrix(user_index_list)
        # item_embedding=[]
        # for item in item_index_list:
        #     item_embedding.append(item_list[item])
        # item_embedding=tf.convert_to_tensor(item_embedding)
        item_embedding=self.item_embedding_matrix(item_index_list)
        item_embedding=self.mlp_3(item_embedding)
        item_embedding=self.mlp_4(item_embedding)
        user_embedding=self.mlp_1(user_embedding)
        user_embedding=self.mlp_2(user_embedding)
        # item_embedding=self.bn(item_embedding)
        scores = tf.reduce_sum(user_embedding * item_embedding)
        scores_normalized = tf.nn.sigmoid(scores)
        return scores_normalized

    def eval(self,train_x,train_y):
        scores=self.train(train_x)
        labels=train_y
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return acc

# print(max(item_list.keys()))

