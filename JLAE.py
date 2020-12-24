# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
import time
import math
import evaluate
from keras.layers import Lambda, Input, Dense
from keras import backend as K
from keras.models import Model
from keras.losses import binary_crossentropy
from preprocessor import *
from test import parse_args
import threading
import os
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt

args = parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
cpus = [x.name for x in device_lib.list_local_devices() if x.device_type == 'CPU']
data_generator = ml1m(args.batch_size)
intermediate_dim = 512
latent_dim = 160


class JoVA():
    def __init__(self, args, data):
        self.args = args
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.max_item = data.max_item
        self.max_user = data.max_user
        self.item_train = data.item_train
        self.user_train = data.user_train
        self.train_R = data.train_R
        self.test_R = data.test_R
        self.coffi_u= data.coffi
        self.coffi_i =data.coffi.T

        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size

        self.num_batch_u = int(math.ceil(self.num_users / float(self.batch_size)))
        self.num_batch_i = int(math.ceil(self.num_items / float(self.batch_size)))

        self.lr = args.lr

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.drop = tf.placeholder(tf.float32)
        self.input_R_U = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name="input_R_U")
        self.input_R_I = tf.placeholder(dtype=tf.float32, shape=[self.num_users, None], name="input_R_I")
        self.input_P_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_P_cor")
        self.input_N_cor = tf.placeholder(dtype=tf.int32, shape=[None, 2], name="input_N_cor")
        self.train_coffi_u = tf.nn.embedding_lookup(self.coffi_u, self.users)
        self.train_coffi_i = tf.nn.embedding_lookup(self.coffi_i, self.items)
        self.u_embeddings_c = self.user_e()
        self.i_embeddings = self.user_d()

        self.i_embeddings_c = self.item_e()
        self.u_embeddings = self.item_d()


        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def user_e(self):
        # encoder
        self.Wue = tf.get_variable('Wue', [self.num_items, latent_dim], tf.float32, xavier_initializer())
        self.nor_input_u = tf.square(tf.nn.l2_normalize(self.input_R_U, 1))
        self.nor_input_u  = tf.nn.dropout(self.nor_input_u, self.drop)
        u_embedding_c = tf.matmul(self.nor_input_u, self.Wue)

        return u_embedding_c

    def user_d(self):
        # encoder
        self.Wud = tf.get_variable('Wud', [self.num_items, latent_dim], tf.float32, xavier_initializer())
        i_embedding = self.Wud

        return i_embedding

    def item_e(self):
        # encoder

        self.nor_input_i = tf.square(tf.nn.l2_normalize(tf.transpose(self.input_R_I), 1))
        self.nor_input_i= tf.nn.dropout(self.nor_input_i, self.drop)
        self.Wie = tf.get_variable('Wie', [self.num_users, latent_dim], tf.float32, xavier_initializer())
        # self.h = tf.get_variable('h', [1, latent_dim], tf.float32, xavier_initializer())
        i_embedding_c = tf.matmul(self.nor_input_i, self.Wie)

        return i_embedding_c

    def item_d(self):
        # encoder
        self.Wid = tf.get_variable('Wid', [self.num_users , latent_dim], tf.float32, xavier_initializer())
        u_embedding = self.Wid

        return u_embedding

    def get_loss(self):
        self.u_decoder = tf.matmul(self.u_embeddings_c, self.i_embeddings, transpose_a=False, transpose_b=True)
        self.i_decoder = tf.matmul(self.i_embeddings_c, self.u_embeddings, transpose_a=False, transpose_b=True)
        self.decoder = ((tf.transpose(tf.gather_nd(tf.transpose(self.u_decoder), self.items)))
                        + tf.gather_nd(tf.transpose(self.i_decoder), self.users)) / 2.0
        pos_data = tf.gather_nd(self.Decoder, self.input_P_cor)
        neg_data = tf.gather_nd(self.Decoder, self.input_N_cor)
        pre_cost1 = tf.maximum(neg_data - pos_data + 0.15,
                               tf.zeros(tf.shape(neg_data)[0]))
        self.hinge = tf.reduce_sum(pre_cost1)
        self.loss_u = tf.reduce_sum(tf.square(self.u_decoder-self.input_R_U)*self.train_coffi_u)
        self.loss_i = tf.reduce_sum(tf.square(self.i_decoder-tf.transpose(self.input_R_I))*self.train_coffi_i)

        self.loss = (self.loss_u  + self.loss_i)/2.0 + self.hinge



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    model = JoVA(args, data_generator)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # tf.set_random_seed(777)

    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())

    for epoch in range(1, args.train_epoch + 1):
        t1 = time.time()
        loss = 0.
        random_row_idx = np.random.permutation(model.num_users)
        random_col_idx = np.random.permutation(model.num_items)
        for i in range(model.num_batch_u):
            if i == model.num_batch_u - 1:
                row_idx = random_row_idx[i * model.batch_size:]
            else:
                row_idx = random_row_idx[(i * model.batch_size):((i + 1) * model.batch_size)]

            for j in range(model.num_batch_i):
                if j == model.num_batch_i - 1:
                    col_idx = random_col_idx[j * model.batch_size:]
                else:
                    col_idx = random_col_idx[(j * model.batch_size):((j + 1) * model.batch_size)]
                input_R_U = model.train_R[row_idx, :]
                input_R_I = model.train_R[:, col_idx]
                p_input, n_input = evaluate.pairwise_neg_sampling(model.train_R, row_idx, col_idx, 1)
                _, batch_loss = sess.run(
                    [model.optimizer, model.loss],
                    feed_dict={model.users: row_idx, model.input_R_U: input_R_U, model.items: col_idx,
                               model.input_R_I: input_R_I, model.input_P_cor: p_input,
                               model.input_N_cor: n_input, model.drop: 0.3})
                loss += batch_loss / model.num_batch_i * model.num_batch_u

        print("Epoch %d //" % (epoch), " cost = {:.8f}".format(loss), "Elapsed time : %d sec" % (time.time() - t1))
        evaluate.test_all(sess, model)

        print("=" * 100)
