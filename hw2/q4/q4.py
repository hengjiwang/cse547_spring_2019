
# coding: utf-8

# Question 4


import numpy as np
import pandas as pd
import re


M = 9985
N = 563


# Load data
def load(path):
    return pd.read_csv(path, header=None, sep=' ')

# Compute P
def compute_p(r):
    l = np.zeros(M)
    r = r.values
    for j in range(M):
        l[j] = np.sum(r[j])
    return np.diag(l)

# Compute Q
def compute_q(r):
    l = np.zeros(N)
    r = r.values
    for j in range(N):
        l[j] = np.sum(r[:,j])
    return np.diag(l)

# Compute the -1/2 power of a diagonal matrix
def invsqrt(m):
    l = []
    for j in range(len(m)):
        l.append(1/np.sqrt(m[j,j]))
    return np.diag(l)

# Compute Gamma in user-user case
def compute_gamma_user(r):
    p = compute_p(r)
    return invsqrt(p)@r@r.T@invsqrt(p)@r

# Compute Gamma in item-item case
def compute_gamma_item(r):
    q = compute_q(r)
    return r@invsqrt(q)@r.T@r@invsqrt(q)

# Find the names based on indices
def names(col, path):
    l = []
    with open(path) as f:
        for line in f: 
            l.append(line.strip('\n').strip('"'))
            
    name = [l[i] for i in col]
    return name

# Recommend five shows for Alex
def recommend(path1, path2, ind, case):
    
    r = load(path1)
    
    if case == 'user-user':
        gamma = compute_gamma_user(r)
    elif case == 'item-item':
        gamma = compute_gamma_item(r)
    else:
        raise ValueError('Case can only be user-user or item-item.')
    
    scores = gamma.iloc[ind-1, :100]
    col5 = np.argsort(-scores)[:5]
    name5 = names(col5, path2)
    return name5  

# Solution
def solve_4d(path1, path2):
    shows1 = recommend(path1, path2, 500, 'user-user')
    shows2 = recommend(path1, path2, 500, 'item-item')
    print(shows1)
    print(shows2)


if __name__ == '__main__':
    solve_4d('data/user-shows.txt', 'data/shows.txt')

