#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-09 00:18:42
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-10 20:15:27

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
import numpy as np
import random

from preprocess_data import get_author_info_from_mongodb, harmonic_mean_of_two_arrays

class DataSet(object):
    """dataset for training and testing"""
    def __init__(self, data_set_file, train_test, negative_positive_ratio):
        self.data_set_file = data_set_file
        self.train_test = train_test
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.negative_positive_ratio = negative_positive_ratio
        self.filename = '{:s}_{:d}'.format(self.data_set_file, self.negative_positive_ratio)
        
    def construct_data(self, config_file, venues):
        if os.path.isfile(self.filename):
            return
        venues_list = list(venues)
        author_info = get_author_info_from_mongodb(config_file, venues)
        tmp = list(author_info.aggregate([{'$group': {'_id': None, 'max_author_index': {'$max': '$index'}}}]))[0]
        max_author_index = tmp['max_author_index']
        positive_x = []
        indexes = []
        for doc in author_info.find({'num_as_coauthor.{:s}'.format(venues_list[0]): {'$gte': 1}, 'num_as_coauthor.{:s}'.format(venues_list[1]): {'$gte': 1}}):
            indexes.append(doc['index'])
            tmp = doc['coauthor_feature']['harmonic_mean']
            x = np.zeros(max_author_index + 3)
            for i in tmp:
                x[i['index']] = i['value']
            x[-2] = 1 # label
            x[-1] = 0 # label
            positive_x.append(x)
        num_positive_x = len(positive_x)

        negative_x = []
        for i in range(self.negative_positive_ratio):
            for doc in author_info.find({'num_as_coauthor.{:s}'.format(venues_list[0]): {'$gte': 1}, 'num_as_coauthor.{:s}'.format(venues_list[1]): {'$gte': 1}}):
                index = doc['index']
                coauthors_tf_idf = doc['coauthors_tf_idf']
                while True:
                    rand_index = random.choice(indexes)
                    if rand_index != index:
                        break
                arr1 = coauthors_tf_idf[venues_list[0]]
                arr2 = author_info.find_one({'index': rand_index})['coauthors_tf_idf'][venues_list[1]]
                tmp = harmonic_mean_of_two_arrays(arr1, arr2)
                x = np.zeros(max_author_index + 3)
                for i in tmp:
                    x[i['index']] = i['value']
                x[-2] = 0 # label
                x[-1] = 1 # label
                negative_x.append(x)
        data_set = np.asarray(positive_x + negative_x)
        np.random.shuffle(data_set)
        print('Begin pickling data_set to file {:s}'.format(self.filename))
        with open(self.filename, 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def load_data(self, n_out):
        with open(self.filename, 'rb') as f:
            self.data_set = pickle.load(f)
        self.num_data = len(self.data_set)
        self.n_out = n_out
        self.n_in = self.data_set.shape[1] - self.n_out

        tmp = sum(self.train_test)
        self.num_one_fold = int(self.num_data / tmp)
        self.train_data_boundary = int(self.num_one_fold * self.train_test[0])
        # self.test_data_boundary = self.num_data

        self.train_data = self.data_set[0:self.train_data_boundary]
        self.test_data = self.data_set[self.train_data_boundary:self.num_data]

    def split_data_k_fold(self, k_fold, test_index):
        self.num_one_fold = self.num_data // k_fold
        self.test_data_start = self.num_one_fold * test_index
        self.test_data_end = self.test_data_start + self.num_one_fold
        self.test_data = self.data_set[self.test_data_start:self.test_data_end]
        self.train_data = np.concatenate((self.data_set[0:self.test_data_start], self.data_set[self.test_data_end:]))
        # print('test_data_start: {:d}, test_data_end: {:d}'.format(self.test_data_start, self.test_data_end))

    def get_x(self, data):
        return data[:, 0:-self.n_out]

    def get_y(self, data):
        return data[:, -self.n_out:]

    def next_train_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.train_data_boundary:
            self._epochs_completed += 1
            # print('{:d} epochs completed'.format(self._epochs_completed))
            np.random.shuffle(self.train_data)
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.train_data_boundary
        end = self._index_in_epoch
        return self.train_data[start:end]

def main():
    config_file = 'mongo.cfg'
    coauthor_file = '../data/publications.txt'
    KDD_ICDM = {'KDD': 1, 'ICDM': 1}
    SIGMOD_ICDE = {'SIGMOD': 1, 'ICDE': 1}
    NIPS_ICML = {'NIPS': 1, 'ICML': 1}
    data_set_file_KDD_ICDM = '../data/data_set_KDD_ICDM.pkl'
    data_set_file_SIGMOD_ICDE = '../data/data_set_SIGMOD_ICDE.pkl'
    data_set_file_NIPS_ICML = '../data/data_set_NIPS_ICML.pkl'
    train_test = [4, 1]
    n_out = 2
    negative_positive_ratio = 1
    if 1:
        dataset = DataSet(data_set_file_NIPS_ICML, train_test, negative_positive_ratio)
        dataset.construct_data(config_file, NIPS_ICML)
        dataset.load_data(n_out)


if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
