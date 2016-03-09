#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-09 22:21:31
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-09 22:48:38

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

import mnist_input_data
from model import MLP

def main():
    n_out = 10
    learning_rate=0.001
    n_iter=1000
    batch_size=100
    validate_frequency = 100
    hiddenlayer_params = []
    activate_func = tf.tanh
    if 1:
        mnist = mnist_input_data.read_data_sets("../data/MNIST_data/", one_hot=True)
        model = MLP(mnist, n_out, learning_rate, n_iter, batch_size, \
            validate_frequency, hiddenlayer_params, activate_func)
        model.train_mnist()

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
