#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-09 00:18:42
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-09 02:40:51

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
    def __init__(self, data_set_file, train_validate_ratio):
        self.data_set_file = data_set_file
        self.train_validate_ratio = train_validate_ratio
        
    def construct_data(self, config_file, venues):
        if os.path.isfile(self.data_set_file):
            return
        venues_list = list(venues)
        author_info = get_author_info_from_mongodb(config_file, venues)
        tmp = list(author_info.aggregate([{'$group': {'_id': None, 'max_author_index': {'$max': '$index'}}}]))[0]
        max_author_index = tmp['max_author_index']
        # num_positive_instances = author_info.find({'num_as_coauthor.ICDM': {'$gte': 1}, \
        #     'num_as_coauthor.KDD': {'$gte': 1}}).count()
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
        # positive_y = np.ones((num_positive_x, 1))

        negative_x = []
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
        print('Begin pickling data_set to file {:s}'.format(self.data_set_file))
        with open(self.data_set_file, 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def extract_data(self):
        with open(self.data_set_file, 'rb') as f:
            self.data_set = pickle.load(f)
        self.num_data = len(self.data_set)
        tmp = self.train_validate_ratio + 1 + 1
        self.num_one_fold = int(self.num_data / tmp)
        self.train_data_boundary = int(self.num_one_fold * self.train_validate_ratio)
        self.validata_data_boundary = self.train_data_boundary + self.num_one_fold
        self.test_data_boundary = self.num_data

        self.train_data = self.data_set[0:self.train_data_boundary]
        self.train_x = self.train_data[:, 0:-2]
        self.train_y = self.train_data[:, -2:]
        self.validata_data = self.data_set[self.train_data_boundary:self.validata_data_boundary]
        self.validata_x = self.validata_data[:, 0:-2]
        self.validata_y = self.validata_data[:, -2:]
        self.test_data = self.data_set[self.validata_data_boundary:self.test_data_boundary]
        self.test_x = self.test_data[:, 0:-2]
        self.test_y = self.test_data[:, -2:]


def main():
    config_file = 'mongo.cfg'
    coauthor_file = '../data/publications.txt'
    KDD_ICDM = {'KDD': 1, 'ICDM': 1}
    SIGMOD_ICDE = {'SIGMOD': 1, 'ICDE': 1}
    NIPS_ICML = {'NIPS': 1, 'ICML': 1}
    data_set_file_KDD_ICDM = '../data/data_set_KDD_ICDM.pkl'
    data_set_file_SIGMOD_ICDE = '../data/data_set_SIGMOD_ICDE.pkl'
    data_set_file_NIPS_ICML = '../data/data_set_NIPS_ICML.pkl'
    train_validate_ratio = 3
    if 1:
        dataset = DataSet(data_set_file_NIPS_ICML, train_validate_ratio)
        dataset.construct_data(config_file, NIPS_ICML)
        dataset.extract_data()
        

if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
