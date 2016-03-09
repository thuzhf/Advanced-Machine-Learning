#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-09 02:41:34
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-09 22:49:10

from __future__ import print_function,division,unicode_literals,absolute_import
import sys,os,re,json,gzip,math,time,datetime,functools,contextlib,itertools
import multiprocessing as mp
import subprocess as sp
if sys.version_info < (3,): # version 2.x
    range2 = range
    range = xrange
    import ConfigParser as configparser
    import cPickle as pickle
else:
    import configparser
    import pickle
import tensorflow as tf

from extract_data import DataSet

class HiddenLayer(object):
    """docstring for HiddenLayer"""
    def __init__(self, x, n_out, activate_func):
        self.x = x
        self.n_in = self.x.get_shape().as_list()[-1]
        self.n_out = n_out
        self.activate_func = activate_func
        self.W = tf.Variable(tf.zeros([self.n_in, self.n_out]))
        self.b = tf.Variable(tf.zeros([self.n_out]))
        self.y = self.activate_func(tf.matmul(self.x, self.W) + self.b)


class LogisticRegression(object):
    """docstring for LogisticRegression"""
    def __init__(self, x, n_out):
        self.x = x
        self.n_in = self.x.get_shape().as_list()[-1]
        self.n_out = n_out
        self.W = tf.Variable(tf.zeros([self.n_in, self.n_out]))
        self.b = tf.Variable(tf.zeros([self.n_out]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

    def cross_entropy(self, y_):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(self.y))
        return cross_entropy

    def accuracy(self, y_):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

class MLP(object):
    """docstring for MLP"""
    def __init__(self, dataset, n_out, learning_rate, n_iter, batch_size, \
            validate_frequency, hiddenlayer_params, activate_func):
        # self.dataset_file = dataset_file
        # self.train_validate_ratio = train_validate_ratio
        self.dataset = dataset
        self.n_in = self.dataset.n_in
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.validate_frequency = validate_frequency
        self.hiddenlayer_params = hiddenlayer_params
        self.activate_func = activate_func

        # self.get_dataset()
        self.build_model()

    # def get_dataset(self):
    #     self.dataset = DataSet(self.dataset_file, self.train_validate_ratio)
    #     self.dataset.extract_data(self.n_out)
    #     self.n_in = self.dataset.n_in

    def build_model(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_in])
        self.y_ = tf.placeholder(tf.float32, [None, self.n_out])
        self.hidden_layers = []
        for i, v in enumerate(self.hiddenlayer_params):
            if i == 0:
                self.hidden_layers.append(HiddenLayer(self.x, v, self.activate_func))
            else:
                self.hidden_layers.append(HiddenLayer(self.hidden_layers[-1].y, v, self.activate_func))
        if len(self.hidden_layers) == 0:
            self.logistic_regression_layer = LogisticRegression(self.x, self.n_out)
        else:
            self.logistic_regression_layer = LogisticRegression(self.hidden_layers[-1].y, self.n_out)

        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.logistic_regression_layer.cross_entropy(self.y_))

    def train(self):
        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)
        for i in range(self.n_iter):
            train_batch = self.dataset.next_train_batch(self.batch_size)
            batch_xs = self.dataset.get_x(train_batch)
            batch_ys = self.dataset.get_y(train_batch)
            if i % self.validate_frequency == 0: # TODO
                train_accuracy = self.logistic_regression_layer.accuracy(self.y_).eval(feed_dict={self.x: batch_xs, self.y_: batch_ys})
                print("step {:d}, training accuracy {:g}".format(i, train_accuracy))
            sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def train_mnist(self):
        mnist = self.dataset
        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)
        for i in range(self.n_iter):
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            if i % self.validate_frequency == 0: # TODO
                train_accuracy = self.logistic_regression_layer.accuracy(self.y_).eval(feed_dict={self.x: batch_xs, self.y_: batch_ys})
                print("step {:d}, training accuracy {:g}".format(i, train_accuracy))
            sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def test(self):
        test_x = self.dataset.get_x(self.dataset.test_data)
        test_y = self.dataset.get_y(self.dataset.test_data)
        test_accuracy = self.logistic_regression_layer.accuracy(self.y_).eval(feed_dict={self.x: test_x, self.y_: test_y})
        print('test accuracy {:g}'.format(test_accuracy))

def main():
    data_set_file_KDD_ICDM = '../data/data_set_KDD_ICDM.pkl'
    data_set_file_SIGMOD_ICDE = '../data/data_set_SIGMOD_ICDE.pkl'
    data_set_file_NIPS_ICML = '../data/data_set_NIPS_ICML.pkl'
    train_validate_ratio = 3

    n_out = 2
    learning_rate=0.00001
    n_iter=1000
    batch_size=200
    validate_frequency = 100
    hiddenlayer_params = []
    activate_func = tf.tanh
    if 1:
        dataset = DataSet(data_set_file_KDD_ICDM, train_validate_ratio)
        dataset.extract_data(n_out)
        model = MLP(dataset, n_out, learning_rate, n_iter, batch_size, \
            validate_frequency, hiddenlayer_params, activate_func)
        model.train()

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
