#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-07 18:03:23
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-08 01:44:13

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

from mongoservice import MongoService

def preprocess_data(coauthor_file, venues):
    config_file = 'mongo.cfg'
    config_parser = configparser.SafeConfigParser()
    with open(config_file) as f:
        config_parser.readfp(f)
    host = config_parser.get('mongo', 'host')
    port = config_parser.getint('mongo', 'port')
    db = config_parser.get('mongo', 'db')
    username = config_parser.get('mongo', 'username')
    passwd = config_parser.get('mongo', 'passwd')
    mongo = MongoService(host, port, db, username, passwd)
    KDD_ICDM = mongo.db['thuzhf_ML_author_info_in_KDD_ICDM']
    SIGMOD_ICDE = mongo.db['thuzhf_ML_author_info_in_SIGMOD_ICDE']
    NIPS_ICML = mongo.db['thuzhf_ML_author_info_in_NIPS_ICML']
    if 'KDD' in venues:
        author_info = KDD_ICDM
    elif 'SIGMOD' in venues:
        author_info = SIGMOD_ICDE
    elif 'NIPS' in venues:
        author_info = NIPS_ICML
    else:
        print('Invalid parameters.')
        sys.exit()

    authors = {} # author: {'name': author, 'index': i, 'coauthors': {coauthor_id: 1, ...}}
    index = 0
    with open(coauthor_file) as f:
        lines = f.readlines()
    print('{0:s} has been loaded into memory.'.format(coauthor_file))
    current_paper = {'authors': None, 'venue': None, 'index': None}
    for line in lines:
        if re.match(r"#@.*", line):
            authors_list = line[2:].strip().split(',')
            if authors_list == ['']:
                continue
            current_paper['authors'] = authors_list
        elif re.match(r"#c.*", line):
            venue = line[2:].strip()
            if venue == '':
                continue
            current_paper['venue'] = venue
        elif re.match(r"#index.*", line):
            current_paper['index'] = int(line[6:].strip())
        elif not line.strip(): # paper separatrix
            if not current_paper['authors'] or \
                not current_paper['venue'] in venues:\
                continue
            else:
                for a in current_paper['authors']:
                    if not a: # debug
                        print(current_paper['index'])
                        sys.exit()
                    if a not in authors:
                        authors[a] = {'name': a, 'index': index, 'coauthors': {}}
                        index += 1
                for a in current_paper['authors']:
                    for b in current_paper['authors']:
                        b_index = str(authors[b]['index'])
                        if b_index not in authors[a]['coauthors']:
                            authors[a]['coauthors'][b_index] = 0
                        authors[a]['coauthors'][b_index] += 1
    print('Begin writing to mongodb.')
    for a in authors:
        if not author_info.find_one({'index': authors[a]['index']}):
            author_info.insert_one(authors[a])


def main():
    coauthor_file = '../data/publications.txt'
    KDD_ICDM = {'KDD': 1, 'SIGKDD': 1, 'ICDM': 1}
    SIGMOD_ICDE = {'SIGMOD': 1, 'ICDE': 1}
    NIPS_ICML = {'NIPS': 1, 'ICML': 1}
    preprocess_data(coauthor_file, NIPS_ICML)


if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
