#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyspark import SparkConf, SparkContext
import re

conf = SparkConf()
sc = SparkContext(conf=conf)


E = np.e
DELTA = E**-5
EPSILON = E*1e-4
N_BUCKETS = int(np.ceil(E / EPSILON))
P = 123457


def hash_fun(a, b, x):
    '''Returns hash(x) for hash function given by parameters a,b '''
    y = x % P
    hash_val = (a*y + b) % P
    return hash_val % N_BUCKETS

def count(path_stream, path_param):
    '''Returns the counter c_{j,b} and the parameters a's and b's '''
    rdd = sc.textFile(path_param)
    hash_params = np.array(rdd.map(lambda line: [int(x) for x in re.split(r'\t+', line)]).collect())
    a1 = hash_params[0,0]
    b1 = hash_params[0,1]
    a2 = hash_params[1,0]
    b2 = hash_params[1,1]
    a3 = hash_params[2,0]
    b3 = hash_params[2,1]
    a4 = hash_params[3,0]
    b4 = hash_params[3,1]
    a5 = hash_params[4,0]
    b5 = hash_params[4,1]
    
    counter = np.zeros((5, N_BUCKETS))
    
    # Stream elements arriving
    with open(path_stream) as f:
        for line in f:
            i = int(line)
            counter[0, hash_fun(a1, b1, i)] += 1
            counter[1, hash_fun(a2, b2, i)] += 1
            counter[2, hash_fun(a3, b3, i)] += 1
            counter[3, hash_fun(a4, b4, i)] += 1
            counter[4, hash_fun(a5, b5, i)] += 1
    return counter, hash_params

def estimate_fun(hash_params, i, counter):
    '''Returns the estimate of F[i]'''
    counter_values = []
    
    for j in range(5):
        counter_values.append(counter[j, hash_fun(hash_params[j,0], hash_params[j,1], i)])

    return np.min(counter_values) # pick the minimum

def estimate(counter, hash_params, path_count):
    '''Returns counts[:,0]: i; counts[:,1]: F[i]; counts[:,2]: Ftilde[i]'''
    rdd = sc.textFile(path_count)
    counts = rdd.map(lambda line: [int(x) for x in re.split(r'\t+', line)])\
                .map(lambda line: [line[0], line[1], estimate_fun(hash_params, line[0], counter)])
    
    return np.array(counts.collect())

def plot(counts, counter):
    
    t = np.sum(counter[0])
    x_axis = counts[:,1]/t
    y_axis = (counts[:,2] - counts[:,1]) / counts[:,1]
    
    plt.figure()
    plt.plot(x_axis, y_axis, '.')
    plt.plot([np.min(x_axis), np.max(x_axis)], [1,1], 'r-')
    plt.yscale('log')
    plt.xscale('log')
    plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    plt.xlabel('word frequency F[i]/t')
    plt.ylabel('relative error Er[i]')
    plt.title('Relative error vs. word frequency')
    plt.show()


if __name__ == '__main__':
    counter, hash_params = count('./data/words_stream.txt', './data/hash_params.txt')
    counts = estimate(counter, hash_params, './data/counts.txt')
    plot(counts, counter)

