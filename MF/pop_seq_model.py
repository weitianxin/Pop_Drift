import tensorflow as tf
import numpy as np
import os
import sys
import random
import collections
import heapq
import math
import logging
from time import time
import multiprocessing
import argparse
from scipy.special import softmax, expit
from matplotlib import pyplot as plt


class Seq():
    def __init__(self, maxSeqLength, n_batch, alpha):
        drop_prob = 0.5
        lstmUnits = 2
        lr = 1e-3
        self.drop_prob = tf.placeholder_with_default(drop_prob, shape=())
        self.data = tf.placeholder(tf.float32, [None, maxSeqLength, 1])
        self.label = tf.placeholder(tf.float32, [None, maxSeqLength-2, 1])
        lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
        lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=1-self.drop_prob)
        value, _ = tf.nn.dynamic_rnn(lstmCell, self.data, dtype=tf.float32)
        self.pred = tf.layers.dense(value, 1, name='pre_fc')
        self.loss_ori = tf.reduce_mean((self.pred[:,:-2,:]-self.label)**2)
        self.tra_vars = tf.trainable_variables('pre_fc')
        self.l2_norm = 0
        for var in self.tra_vars:
            self.l2_norm += tf.nn.l2_loss(var)
        self.loss = self.loss_ori + alpha*self.l2_norm
        global_step = tf.Variable(0, trainable=False)
        lr_decayed = tf.train.cosine_decay_restarts(lr, global_step,
                                        n_batch)
        self.opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, global_step=global_step)

def load_pop_data(path="data/ml_10m/"):
    with open(path+"item_pop_seq_ori.txt") as f:
        item_list = []
        pop_item_all = []
        for line in f:
            line = line.strip().split()
            item, pop_list = int(line[0]), [float(x) for x in line[1:]]
            item_list.append(item)
            pop_item_all.append(pop_list)
    return item_list, np.array(pop_item_all)

def load_batch_data(data, label, i , b, S):
    # N x T x 1
    T_cur = data.shape[1]
    b_cur = len(data[i*b:(i+1)*b])
    out_data = np.zeros([b_cur, S, 1])
    out_data[:, :T_cur, :] = data[i*b:(i+1)*b]
    out_label = label[i*b:(i+1)*b]
    return out_data, out_label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run pop_bias.")
    parser.add_argument('--cuda', type=str, default='1',
                        help='Avaiable GPU ID')
    parser.add_argument('--load', type=int, default=0,
                        help='Avaiable GPU ID')
    parser.add_argument('--a', type=float, default=0,
                        help='Avaiable GPU ID')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    batch_size = 128
    epoch = 200
    path = "data/ml_10m/"
    item_list, pop_item_all = load_pop_data(path=path)
    N, S = pop_item_all.shape
    # N x S x 1
    pop_item_all = pop_item_all[:, :, np.newaxis]
    train_input = pop_item_all[:, :-2, :]
    train_output = pop_item_all[:, 1:-1, :]
    
    test_input = pop_item_all[:, :-1, :]
    test_gt = pop_item_all[:, -1:, :]
    print(pop_item_all.shape, train_input.shape, train_output.shape)
    if N//batch_size==N/batch_size:
        n_batch = N//batch_size
    else:
        n_batch = N//batch_size + 1
    model = Seq(S, n_batch, args.a)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config = gpu_config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    if args.load==1:
        model_file = 'checkpoint/pop/best_alpha_ori'
        saver.restore(sess, model_file)
        test_loss_list = []
        pop_pred = []
        for i in range(n_batch):
            test_data, test_label = load_batch_data(test_input, test_gt, i , batch_size, S)
            # N x T x 1
            pred = sess.run(model.pred, feed_dict = {model.data: test_data, model.drop_prob: 0 })
            pred = pred[:, -1: ,:]
            pop_pred.extend(pred[:,0,0])
            loss = np.mean((pred - test_label)**2)
            test_loss_list.append(loss)
        mean_test_loss = np.mean(test_loss_list)
        print("test loss: ", mean_test_loss)
        with open(path+"pop_predict.txt", "w") as f:
            for i, item  in enumerate(item_list):
                f.write(str(item)+" "+str(pop_pred[i])+"\n")
    else:
        best_test_mse = 1e9
        # initial
        test_metric_dict = {"RMSE":[], "MAE":[]}
        for i in range(n_batch):
            test_data, test_label = load_batch_data(test_input, test_gt, i , batch_size, S)
            # N x T x 1
            pred = sess.run(model.pred, feed_dict = {model.data: test_data, model.drop_prob: 0})
            pred = pred[:, -1: ,:]
            # pred = 1e-5*np.ones([pred.shape[0],1,1])
            test_metric_dict["RMSE"].extend((pred - test_label)**2)
            test_metric_dict["MAE"].extend(np.abs(pred - test_label))
        mean_test_rmse = np.sqrt(np.mean(test_metric_dict["RMSE"]))
        mean_test_mse = np.mean(test_metric_dict["RMSE"])
        print("initial test rmse: ", mean_test_rmse, " test mse: ", mean_test_mse)
        for e in range(epoch):
            # train
            loss_list = []
            for i in range(n_batch):
                train_data, train_label = load_batch_data(train_input, train_output, i , batch_size, S)
                loss, _ = sess.run([model.loss, model.opt], feed_dict = {model.data: train_data,
                                                    model.label: train_label})
                loss_list.append(loss)
            print("{} epoch:".format(e),np.mean(loss_list))
            # test
            test_metric_dict = {"RMSE":[], "MAE":[]}
            for i in range(n_batch):
                test_data, test_label = load_batch_data(test_input, test_gt, i , batch_size, S)
                # N x T x 1
                pred = sess.run(model.pred, feed_dict = {model.data: test_data, model.drop_prob: 0})
                pred = pred[:, -1: ,:]
                test_metric_dict["RMSE"].extend((pred - test_label)**2)
                test_metric_dict["MAE"].extend(np.abs(pred - test_label))
            mean_test_rmse = np.sqrt(np.mean(test_metric_dict["RMSE"]))
            mean_test_mse = np.mean(test_metric_dict["RMSE"])
            if mean_test_mse<best_test_mse:
                best_test_mse = mean_test_mse
                saver.save(sess, 'checkpoint/pop/best_alpha_ori')
            print("{} epoch: test rmse: ".format(e), mean_test_rmse, " test mse: ", mean_test_mse)
    