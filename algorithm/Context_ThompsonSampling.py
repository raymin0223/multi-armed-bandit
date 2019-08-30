# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import random
import numpy as np
import traceback

__all__ = ['Context_ThompsonSampling']

class Context_ThompsonSampling:
    """ This class is implementation of Context_ThompsonSampling algorithm.
    Reference from 'Thompson Sampling for Contextual Bandits with Linear Payoffs'.
    """
    def __init__(self, arms, context, dim, v=0.15):
        self.arms    = arms
        self.context = context
        self.dim     = dim
        
        self.param = {'B': np.eye(self.dim),
                      'u_hat': np.zeros(self.dim),
                      'f': np.zeros(self.dim),
                      'v': v}
        
    def __solve(self, A, b):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b)[0]
        
    def __get_parameter(self):
        params = {}
        
        B_inv = self.__solve(self.param['B'], np.eye(self.dim))
        cov = pow(self.param['v'], 2) * B_inv
        params['u_hat'] = self.param['u_hat']
        params['cov'] = cov
        
        return params
            
    def select_arm(self):
        selected_arm, best = 0, 0
        params = self.__get_parameter()
        
        try:
            u_sample = np.random.multivariate_normal(params['u_hat'], params['cov'], tol=1e-6)
        except Exception as e:
            traceback.print_exc(e)
            u_sample = np.zeros(self.dim)
            
        for arm in range(self.arms):
            ctx = self.context[arm]
            tmp = np.dot(u_sample, ctx)
                
            if tmp > best:
                selected_arm = arm
                best = tmp
            elif tmp == best:
                selected_arm = random.choice(selected_arm, arm)
    
        return selected_arm
    
    def update_parameter(self, selected_arm, reward):
        b = self.context[selected_arm]
        self.param['B'] += np.outer(b, b)
        self.param['f'] += reward * b
        
        self.param['u_hat'] = self.__solve(self.param['B'], self.param['f'])