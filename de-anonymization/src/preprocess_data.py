#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-07 18:03:23
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-08 23:08:13

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

def get_author_info_from_mongodb(config_file, venues):
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
    return author_info

def preprocess_data(coauthor_file, venues, config_file):
    author_info = get_author_info_from_mongodb(config_file, venues)

    authors = {} # author: {'name': author, 'index': i, 'num_papers': n, 'coauthors': {coauthor_id: 1, ...}}
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
            elif venue == 'SIGKDD':
                venue = 'KDD'
            elif venue == 'SIGMOD Conference':
                venue = 'SIGMOD'
            current_paper['venue'] = venue
        elif re.match(r"#index.*", line):
            current_paper['index'] = int(line[6:].strip())
        elif not line.strip(): # paper separatrix
            if not current_paper['authors'] or \
                not current_paper['venue'] in venues:
                continue
            else:
                for a in current_paper['authors']:
                    if not a: # debug
                        print(current_paper['index'])
                        sys.exit()
                    if a not in authors:
                        authors[a] = {'name': a, 'index': index, 'num_papers': 0, 'coauthors': {}}
                        for v in venues:
                            authors[a]['coauthors'][v] = {}
                        index += 1
                    authors[a]['num_papers'] += 1
                for a in current_paper['authors']:
                    for b in current_paper['authors']:
                        b_index = authors[b]['index']
                        if b_index not in authors[a]['coauthors'][current_paper['venue']]:
                            authors[a]['coauthors'][current_paper['venue']][b_index] = 0
                        authors[a]['coauthors'][current_paper['venue']][b_index] += 1
    print('Begin writing to mongodb.')
    for a in authors:
        if not author_info.find_one({'index': authors[a]['index']}):
            coauthors = []
            for v in authors[a]['coauthors']:
                for i in authors[a]['coauthors'][v]:
                    coauthors.append({'venue': v, 'index': i, 'num': authors[a]['coauthors'][v][i]})
            authors[a]['coauthors'] = coauthors
            author_info.insert_one(authors[a])

def calc_num_as_coauthor(config_file, venues):
    author_info = get_author_info_from_mongodb(config_file, venues)
    for doc in author_info.find({}):
        num_as_coauthor = {}
        for v in venues:
            num_as_coauthor[v] = 0
        for c in doc['coauthors']:
            num_as_coauthor[c['venue']] += 1
        # num_as_coauthor = [{'venue': k, 'num_as_coauthor': num_as_coauthor[k]} for k in num_as_coauthor]
        author_info.update_one({'_id': doc['_id']}, {'$set': {'num_as_coauthor': num_as_coauthor}})

def get_num_all_authors(config_file, venues):
    author_info = get_author_info_from_mongodb(config_file, venues)
    ret = {}
    for v in venues:
        n = author_info.find({'num_as_coauthor.{:s}'.format(v): {'$gte': 1}}).count()
        ret[v] = n
    return ret

def calc_tf_idf_for_coauthors(config_file, venues):
    author_info = get_author_info_from_mongodb(config_file, venues)
    num_all_authors = get_num_all_authors(config_file, venues)
    for doc in author_info.find({}):
        tf_idf_num_coauthors = {}
        for v in venues:
            tf_idf_num_coauthors[v] = []
        for c in doc['coauthors']:
            tmp = author_info.find_one({'index': c['index']})
            df = tmp['num_as_coauthor'][c['venue']]
            N = num_all_authors[c['venue']] + 1
            tf_idf_value = c['num'] * math.log(N / df)
            tf_idf_num_coauthors[c['venue']].append({'index': c['index'], 'value': tf_idf_value})
        author_info.update({'_id': doc['_id']}, {'$set': {'coauthors_tf_idf': tf_idf_num_coauthors}})

def calc_coauthor_feature(config_file, venues):
    author_info = get_author_info_from_mongodb(config_file, venues)
    for doc in author_info.find({}):
        tmp = []
        for v in doc['coauthors_tf_idf']:
            tmp.append(doc['coauthors_tf_idf'][v])
        harmonic_mean = harmonic_mean_of_two_arrays(*tmp)
        binary = AND_of_two_arrays(*tmp)
        cosine = cosine_of_two_arrays(*tmp)
        coauthor_feature = {'harmonic_mean': harmonic_mean, 'cosine': cosine, 'binary': binary}
        author_info.update({'_id': doc['_id']}, {'$set': {'coauthor_feature': coauthor_feature}})

# def calc_coauthor_feature(config_file, venues):
#     author_info = get_author_info_from_mongodb(config_file, venues)
#     for doc in author_info.find({}):
#         coauthor_feature = {'harmonic_mean': {}, 'cosine': {}, 'binary': {}}
#         for v in doc['coauthors_tf_idf']:
#             for c in doc['coauthors_tf_idf'][v]:
#                 if c['index'] not in coauthor_feature['harmonic_mean']:
#                     coauthor_feature['harmonic_mean'][c['index']] = {'value': c['value'], 'num': 1}
#                     coauthor_feature['cosine'][c['index']] = {'value': c['value'], 'num': 1}
#                     coauthor_feature['binary'][c['index']] = {'value': 1, 'num': 1}
#                 else:
#                     tmp1 = coauthor_feature['harmonic_mean'][c['index']]['value']
#                     tmp2 = c['value']
#                     tmp = 2 * tmp1 * tmp2 / (tmp1 + tmp2)
#                     coauthor_feature['harmonic_mean'][c['index']]['value'] = tmp
#                     coauthor_feature['harmonic_mean'][c['index']]['num'] += 1
#                     coauthor_feature['cosine'][c['index']]['value'] *= tmp2
#                     coauthor_feature['cosine'][c['index']]['num'] += 1
#                     coauthor_feature['binary'][c['index']]['num'] += 1
#         tmp = []
#         for i in coauthor_feature['harmonic_mean']:
#             tmp.append(i)
#         for i in tmp:
#             if coauthor_feature['harmonic_mean'][i]['num'] == 1:
#                 coauthor_feature['harmonic_mean'].pop(i)
#                 coauthor_feature['cosine'].pop(i)
#                 coauthor_feature['binary'].pop(i)
#         cosine = {}
#         for v in doc['coauthors_tf_idf']:
#             cosine[v] = 0
#             for c in doc['coauthors_tf_idf'][v]:
#                 cosine[v] += math.pow(c['value'], 2)
#         numerator = 0
#         denominator = 1
#         for v in cosine:
#             denominator *= math.pow(cosine[v], 0.5)
#         for i in coauthor_feature['cosine']:
#             numerator += coauthor_feature['cosine'][i]['value']
#         if denominator != 0:
#             coauthor_feature['cosine'] = numerator / denominator
#         else:
#             coauthor_feature['cosine'] = 0
#         coauthor_feature['harmonic_mean'] = [{'index': i, 'value': coauthor_feature['harmonic_mean'][i]['value']} \
#             for i in coauthor_feature['harmonic_mean']]
#         coauthor_feature['binary'] = [{'index': i, 'value': coauthor_feature['binary'][i]['value']} \
#             for i in coauthor_feature['binary']]
#         author_info.update({'_id': doc['_id']}, {'$set': {'coauthor_feature': coauthor_feature}})

def harmonic_mean_of_two_arrays(arr1, arr2):
    ans = {}
    tmp = [arr1, arr2]
    for i in range(len(tmp)):
        for j in tmp[i]:
            if j['index'] not in ans:
                ans[j['index']] = {'value': j['value'], 'num': 1}
            else:
                t1 = ans[j['index']]['value']
                t2 = j['value']
                if t1 + t2 == 0:
                    t = 0
                else:
                    t = 2 * t1 * t2 / (t1 + t2)
                ans[j['index']]['value'] = t
                ans[j['index']]['num'] += 1
    tmp = []
    for i in ans:
        if ans[i]['num'] == 2:
            tmp.append({'index': i, 'value': ans[i]['value']})
    return tmp

def AND_of_two_arrays(arr1, arr2):
    ans = {}
    tmp = [arr1, arr2]
    for i in range(len(tmp)):
        for j in tmp[i]:
            if j['index'] not in ans:
                ans[j['index']] = {'value': 1, 'num': 1}
            else:
                ans[j['index']]['num'] += 1
    tmp = []
    for i in ans:
        if ans[i]['num'] == 2:
            tmp.append({'index': i, 'value': ans[i]['value']})
    return tmp

def cosine_of_two_arrays(arr1, arr2):
    ans = {}
    tmp = [arr1, arr2]
    for i in range(len(tmp)):
        for j in tmp[i]:
            if j['index'] not in ans:
                ans[j['index']] = {'value': j['value'], 'num': 1}
            else:
                ans[j['index']]['value'] *= j['value']
                ans[j['index']]['num'] += 1
    numerator = 0
    denominator = {}
    for i in ans:
        if ans[i]['num'] == 2:
            numerator += ans[i]['value']
    for i in range(len(tmp)):
        denominator[i] = 0
        for j in tmp[i]:
            denominator[i] += math.pow(j['value'], 2)
    denominator2 = 1
    for i in denominator:
        denominator2 *= math.pow(denominator[i], 0.5)
    if denominator2 == 0:
        return 0
    else:
        return numerator / denominator2

def main():
    config_file = 'mongo.cfg'
    coauthor_file = '../data/publications.txt'
    KDD_ICDM = {'KDD': 1, 'ICDM': 1}
    SIGMOD_ICDE = {'SIGMOD': 1, 'ICDE': 1}
    NIPS_ICML = {'NIPS': 1, 'ICML': 1}
    if 0:
        # preprocess_data(coauthor_file, KDD_ICDM, config_file)
        preprocess_data(coauthor_file, SIGMOD_ICDE, config_file)
        # preprocess_data(coauthor_file, NIPS_ICML, config_file)
    if 0:
        # calc_num_as_coauthor(config_file, KDD_ICDM)
        calc_num_as_coauthor(config_file, SIGMOD_ICDE)
        # calc_num_as_coauthor(config_file, NIPS_ICML)
    if 0:
        tmp = get_num_all_authors(config_file, SIGMOD_ICDE)
        print(tmp)
    if 0:
        calc_tf_idf_for_coauthors(config_file, KDD_ICDM)
        calc_tf_idf_for_coauthors(config_file, SIGMOD_ICDE)
        calc_tf_idf_for_coauthors(config_file, NIPS_ICML)
    if 1:
        # calc_coauthor_feature(config_file, KDD_ICDM)
        # calc_coauthor_feature(config_file, SIGMOD_ICDE)
        calc_coauthor_feature(config_file, NIPS_ICML)


if __name__ == '__main__':
    start_t = time.time()
    main()
    end_t = time.time()
    t = end_t - start_t
    print('Time elapsed: {:.4f} minutes'.format(t / 60.))
