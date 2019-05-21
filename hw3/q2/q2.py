#!/usr/bin/env python
# coding: utf-8


from pyspark import SparkConf, SparkContext
import re
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf()
sc = SparkContext(conf=conf)


BETA = 0.8
MAX_ITER = 40
N = 1000


# Load edges
def load_edges(path):
    data = sc.textFile(path)
    return data.map(lambda line:\
        (int(re.split(r'\t+', line)[0]),\
            int(re.split(r'\t+', line)[1:][0])))

# Combine same edges
def combine(rdd):
    return rdd.groupByKey()\
              .mapValues(lambda x: sorted(list(set([v for v in x]))))\
              .sortByKey()

# Obtain outgoing degrees
def inv_degrees(rdd):
    return rdd.map(lambda x: 1/len(x[1])).collect()

# Graph
def M(rdd, inv_d):
    return rdd.flatMapValues(lambda x: x)\
              .map(lambda x: (x[1], x[0]))\
              .groupByKey()\
              .mapValues(lambda x: [(v, inv_d[v-1]) for v in x])\
              .sortByKey()

# Initialize r
def initialize_r():
    return [1/N]*N

# PageRank
def pagerank(r, m):
    return m.mapValues(lambda x: sum([r[v[0]-1]*v[1]*BETA for v in x]))\
            .mapValues(lambda x: x+(1-BETA)/N)\
            .map(lambda lines: lines[1]).collect()

# Iterate
def iterate(r, m):
    for j in range(MAX_ITER):
        r = pagerank(r, m)
    return r

# Find the top and bottom nodes
def top_and_bottom(r):
    r_sorted = sorted(r)
    r = np.array(r)
    bottom = []
    top = []
    for j in range(5):
        bottom.append((np.where(r==r_sorted[j])[0][0]+1, r_sorted[j]))
        top.append((np.where(r==r_sorted[-j-1])[0][0]+1, r_sorted[-j-1]))
    return top, bottom

# Print results
def print_results(top, bottom):
    print('------Solution for 2a------')
    print('Top:')
    for j in range(5):
        print('id: '+ str(top[j][0]) + ', score: '+str(top[j][1]))
    print('\nBottom:')
    for j in range(5):
        print('id: '+ str(bottom[j][0]) + ', score: '+str(bottom[j][1]))

# Solution
def solve_2a(path):
    edges = load_edges(path)
    edges = combine(edges)
    inv_d = inv_degrees(edges)
    graph = M(edges, inv_d)
    r = initialize_r()
    r = iterate(r, graph)
    top, bottom = top_and_bottom(r)
    print_results(top, bottom)


# Initialize h
def initialize_h():
    return [1]*N

# Link matrix
def LT(rdd):
    return rdd.flatMapValues(lambda x: x)\
              .map(lambda x: (x[1], x[0]))\
              .groupByKey()\
              .mapValues(lambda x: [v for v in x])\
              .sortByKey()

# Compute a
def A(h, lt):
    return lt.mapValues(lambda x: sum([h[v-1] for v in x]))\
             .map(lambda lines: lines[1]).collect()

# Compute h
def H(a, l):
    return l.mapValues(lambda x: sum([a[v-1] for v in x]))\
            .map(lambda lines: lines[1]).collect()

# Iterate
def iterate_b(h,l,lt):
    for j in range(MAX_ITER):
        a = A(h, lt)
        a_max = max(a)
        for j in range(len(a)): a[j] /= a_max
        h = H(a, l)
        h_max = max(h)
        for j in range(len(h)): h[j] /= h_max
    return a, h

# Print results
def print_results_b(top_a, bottom_a, top_h, bottom_h):
    print('\n------Solution for 2b------')
    print('Top Hubbiness:')
    for j in range(5):
        print('id: '+ str(top_h[j][0]) + ', score: '+str(top_h[j][1]))
    print('\nBottom Hubbiness:')
    for j in range(5):
        print('id: '+ str(bottom_h[j][0]) + ', score: '+str(bottom_h[j][1]))
    print('\nTop Authority:')
    for j in range(5):
        print('id: '+ str(top_a[j][0]) + ', score: '+str(top_a[j][1]))
    print('\nBottom Authority:')
    for j in range(5):
        print('id: '+ str(bottom_a[j][0]) + ', score: '+str(bottom_a[j][1]))

# Solution for b
def solve_2b(path):
    edges = load_edges(path)
    l = combine(edges)
    lt = LT(l)
    h = initialize_h()
    a, h = iterate_b(h,l,lt)
    top_a, bottom_a = top_and_bottom(a)
    top_h, bottom_h = top_and_bottom(h)
    print_results_b(top_a, bottom_a, top_h, bottom_h)


if __name__ == '__main__':
    solve_2a("data/graph-full.txt")
    solve_2b("data/graph-full.txt")




