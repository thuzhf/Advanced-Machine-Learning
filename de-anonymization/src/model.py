#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-09 02:41:34
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-10 18:10:11

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
        self.initialize_weights()
        self.y = self.activate_func(tf.matmul(self.x, self.W) + self.b)

    def initialize_weights(self):
        self.initial_bound = 4 * math.sqrt(6 / (self.n_in + self.n_out))
        # self.W = tf.Variable(tf.random_uniform([self.n_in, self.n_out], -self.initial_bound, self.initial_bound))
        # self.b = tf.Variable(tf.random_uniform([self.n_out], -self.initial_bound, self.initial_bound))
        self.W = tf.Variable(tf.truncated_normal([self.n_in, self.n_out]))
        self.b = tf.Variable(tf.truncated_normal([self.n_out]))


class LogisticRegression(object):
    """docstring for LogisticRegression"""
    def __init__(self, x, n_out):
        self.x = x
        self.n_in = self.x.get_shape().as_list()[-1]
        self.n_out = n_out
        self.initialize_weights()
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)

    def initialize_weights(self):
        self.initial_bound = 4 * math.sqrt(6 / (self.n_in + self.n_out))
        # self.W = tf.Variable(tf.random_uniform([self.n_in, self.n_out], -self.initial_bound, self.initial_bound))
        # self.b = tf.Variable(tf.random_uniform([self.n_out], -self.initial_bound, self.initial_bound))
        self.W = tf.Variable(tf.truncated_normal([self.n_in, self.n_out]))
        self.b = tf.Variable(tf.truncated_normal([self.n_out]))

    def cross_entropy(self, y_):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(self.y))
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.y, y_)
        return cross_entropy

    def accuracy(self, y_):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

class MLP(object):
    """docstring for MLP"""
    def __init__(self, dataset, n_out, learning_rate, n_iter, batch_size, \
            validate_frequency, hiddenlayer_params, activate_func):
        self.dataset = dataset
        self.n_in = self.dataset.n_in
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.validate_frequency = validate_frequency
        self.hiddenlayer_params = hiddenlayer_params
        self.activate_func = activate_func
        self.build_model()

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

    def check_logistic_regression_layer_weight(self):
        return [self.logistic_regression_layer.W.eval(session=self.sess), \
            self.logistic_regression_layer.b.eval(session=self.sess)]

    def train(self):
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        for i in range(self.n_iter):
            train_batch = self.dataset.next_train_batch(self.batch_size)
            batch_xs = self.dataset.get_x(train_batch)
            batch_ys = self.dataset.get_y(train_batch)
            if i % self.validate_frequency == 0: # TODO
                train_accuracy = self.logistic_regression_layer.accuracy(self.y_).eval(\
                    feed_dict={self.x: batch_xs, self.y_: batch_ys})
                print("step {:d}, training accuracy {:g}".format(i, train_accuracy))
                validate_accuracy = self.logistic_regression_layer.accuracy(self.y_).eval(feed_dict={\
                    self.x: self.dataset.get_x(self.dataset.validata_data), \
                    self.y_: self.dataset.get_y(self.dataset.validata_data)})
                print("step {:d}, validate accuracy {:g}\n".format(i, validate_accuracy))
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

    def train_mnist(self):
        mnist = self.dataset
        init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        for i in range(self.n_iter):
            batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
            if i % self.validate_frequency == 0: # TODO
                train_accuracy = self.logistic_regression_layer.accuracy(self.y_).eval(feed_dict={self.x: batch_xs, self.y_: batch_ys})
                print("step {:d}, training accuracy {:g}".format(i, train_accuracy))
            self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})

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
    negative_positive_ratio = 1

    n_out = 2
    learning_rate=0.001
    n_iter=1000
    batch_size=100
    validate_frequency = 100
    hiddenlayer_params = [32]
    activate_func = tf.sigmoid
    if 1:
        dataset = DataSet(data_set_file_NIPS_ICML, train_validate_ratio, negative_positive_ratio)
        dataset.extract_data(n_out)
        print('num train data: {:d}'.format(dataset.train_data_boundary))
        model = MLP(dataset, n_out, learning_rate, n_iter, batch_size, \
            validate_frequency, hiddenlayer_params, activate_func)
        model.train()
        # W, b = model.check_logistic_regression_layer_weight()
        # tmp = [i for i in W if i[0] or i[1]]
        # print(tmp)
        model.test()

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
