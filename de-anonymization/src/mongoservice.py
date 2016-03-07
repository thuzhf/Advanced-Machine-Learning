#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author: zhangfang
# @Email:  thuzhf@gmail.com
# @Date:   2016-03-07 18:12:12
# @Last Modified by:   zhangfang
# @Last Modified time: 2016-03-07 18:32:25

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
from pymongo import MongoClient

class MongoService(object):
	def __init__(self, host, port, db, username, passwd):
		self.client = MongoClient(host, port)
		self.db = self.client[db]
		self.db.authenticate(username, passwd)
