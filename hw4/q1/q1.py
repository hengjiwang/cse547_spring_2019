#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


class GD:
    
    def _cost_func(self, x, y, w, b):
        
        ksi = 1 - y * (x@w + b)
        ksi[ksi < 0] = 0
        
        return 0.5 * w@w + 100 * np.sum(ksi)
    
    def _cost_change(self, cost_list):
        n = len(cost_list)
        if n < 2:
            return cost_list[-1]
        else:
            return np.abs(cost_list[-1] - cost_list[-2]) / cost_list[-2] * 100
        
    def _update_w_b(self, x, y_, w, b, eta):
        pass
    
    def train(self, features, target, eta, epsl):
        pass


class BGD(GD):
        
    def _update_w_b(self, x, y_, w, b, eta):
        y = np.copy(y_)
        y[y * (x@w + b) >= 1] = 0
        deriv_w = w - 100 * x.T@y
        deriv_b = - 100 * np.sum(y)
        return w - eta * deriv_w, b - eta * deriv_b
    
    def train(self, features, target, eta, epsl):
        
        w, b = np.zeros(len(features[0])), 0
        cost_change = np.Infinity
        costs = []
        
        while cost_change >= epsl:
            cost = self._cost_func(features, target, w, b)
            costs.append(cost)
            cost_change = self._cost_change(costs)
            w_new, b_new = self._update_w_b(features, target, w, b, eta)
            w = np.copy(w_new)
            b = np.copy(b_new)
            
        return costs

class SGD(GD):
    def _update_w_b(self, x, y, w, b, eta, i):
        if y[i] * (w@x[i] + b) >= 1:
            deriv_w = w
            deriv_b = 0
        else:
            deriv_w = w - 100 * (x[i] * y[i])
            deriv_b = - 100 * y[i]
        
        return w - eta * deriv_w, b - eta * deriv_b
        
    def train(self, features_, target_, eta, epsl):
        features = np.copy(features_)
        target = np.copy(target_)
        np.random.seed(70)
        np.random.shuffle(features)
        np.random.seed(70)
        np.random.shuffle(target)
        
        w, b = np.zeros(len(features[0])), 0
        cost_change = np.Infinity
        cost_changek = 0
        costs = []
        i = 1
        n = len(target)
        
        while cost_changek == 0 or cost_changek >= epsl:
            
            cost = self._cost_func(features, target, w, b)
            costs.append(cost)
            cost_change = self._cost_change(costs)
            cost_changek = 0.5 * cost_changek + 0.5 * cost_change
            
            w_new, b_new = self._update_w_b(features, target, w, b, eta, i)
            w = np.copy(w_new)
            b = np.copy(b_new)
            i = i % n + 1
        return costs


class MBGD(GD):
    def _update_w_b(self, x_, y_, w, b, eta, i, size, n):
        if i + size - 1 >= n:
            y = np.copy(y_[i:n])
            x = np.copy(x_[i:n])
        else:
            y = np.copy(y_[i:i+size-1])
            x = np.copy(x_[i:i+size-1])
            
        y[y * (x@w + b) >= 1] = 0
        deriv_w = w - 100 * x.T@y
        deriv_b = - 100 * np.sum(y)
        return w - eta * deriv_w, b - eta * deriv_b 
        

    def train(self, features_, target_, eta, epsl, size):
        features = np.copy(features_)
        target = np.copy(target_)
        np.random.seed(70)
        np.random.shuffle(features)
        np.random.seed(70)
        np.random.shuffle(target)
        
        w, b = np.zeros(len(features[0])), 0
        cost_change = np.Infinity
        cost_changek = 0
        costs = []
        l = 0
        n = len(target)
        
        while cost_changek == 0 or cost_changek >= epsl:
            cost = self._cost_func(features, target, w, b)
            costs.append(cost)
            cost_change = self._cost_change(costs)
            cost_changek = 0.5 * cost_changek + 0.5 * cost_change
            
            i = l * size + 1
            w_new, b_new = self._update_w_b(features, target, w, b, eta, i, size, n)
 
            w = np.copy(w_new)
            b = np.copy(b_new)
            l = (l+1) % int(np.ceil(n/size))
        return costs


class SVM:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.n = len(features)
        self.m = len(features[:])
    
    def _gradient_descent(self, method):
        # return the cost function and the total time taken for convergence
        if method == 'BGD':
            start = time.time()
            f = BGD().train(self.features, self.target, 3e-7, 0.25)
            t = (time.time() - start)
        if method == 'SGD':
            start = time.time()
            f = SGD().train(self.features, self.target, 0.0001, 0.001)
            t = (time.time() - start)
        if method == 'MBGD':
            start = time.time()
            f = MBGD().train(self.features, self.target, 1e-5, 0.01, 20)
            t = (time.time() - start)
        
        return f, t
    
    def output(self):
        fb, tb = self._gradient_descent('BGD')
        fs, ts = self._gradient_descent('SGD')
        fm, tm = self._gradient_descent('MBGD')
        plt.figure()
        plt.plot(fb, 'b-', label = 'BGD')
        plt.plot(fs, 'r-', label = 'SGD')
        plt.plot(fm, 'g-', label = 'MBGD')
        plt.xlabel('iteration number')
        plt.ylabel('cost function')
        plt.legend()
        plt.show()
        
        print('The total time of BGD is %.4f s' % tb)
        print('The total time of SGD is %.4f s' % ts)
        print('The total time of MBGD is %.4f s' % tm)


if __name__ == '__main__':
    features = pd.read_csv('data/features.txt', header = None).values
    target = pd.read_csv('data/target.txt', header = None).values
    target = np.reshape(target, len(target))
    SVM(features, target).output()

