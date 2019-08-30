# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import random
import numpy as np
import traceback

__all__ = ['LinUCB']

class LinUCB:
    """ This class is implementation of LinUCB with disjoint linear models algorithm.
    Reference from 'A Contextual-Bandit Approach to Personalized News Article Recommendation'.
    """
    def __init__(self, arms, context, dim, alpha=1):
        self.arms    = arms
        self.context = context
        self.dim     = dim
        self.alpha   = alpha
        
        self.param = {}
        for arm in range(self.arms):
            self.param[arm] = {'A': np.eye(self.dim),
                               'b':np.zeros(self.dim)}
        
    def __solve(self, A, b):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b)[0]
        
    def __get_parameter(self, arm, ctx):
        params = {}
        
        A_inv = self.__solve(self.param[arm]['A'], np.eye(self.dim))
        theta = np.matmul(A_inv, self.param[arm]['b'])
        var = np.matmul(A_inv, ctx)
        #theta = self.__solve(self.param[arm]['A'], self.param[arm]['b'])
        #var = self.__solve(self.param[arm]['A'], ctx)
        var = np.sqrt(np.dot(ctx, var))
        
        params['theta'] = theta
        params['var'] = var
        
        return params
            
    def select_arm(self):
        selected_arm, best = 0, 0
        for arm in range(self.arms):
            ctx = self.context[arm]
            params = self.__get_parameter(arm, ctx)
            
            tmp = np.dot(params['theta'], ctx)
            tmp += (self.alpha * params['var'])
                
            if tmp > best:
                selected_arm = arm
                best = tmp
            elif tmp == best:
                selected_arm = random.choice(selected_arm, arm)
    
        return selected_arm
    
    def update_parameter(self, selected_arm, reward):
        x = self.context[selected_arm]
        self.param[selected_arm]['A'] += np.outer(x, x)
        self.param[selected_arm]['b'] += reward * x