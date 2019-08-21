# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import random
import numpy as np
import datetime
__all__ = ['KL_UCB']

class KL_UCB:
    """ This class is implementation of KL_UCB algorithm.
    Reference from 'The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond'.
    """
    def __init__(self, arms, c=0):
        self.arms = arms
        self.c = c
        
        self.n = {'rounds': self.arms}
        self.reward_mean = {}
        # for the initialization of KL_UCB algorithm param
        for arm in range(self.arms):
            self.n[arm] = 1
            self.reward_mean[arm] = np.random.binomial(1, 0.5)

    # This Bernoulli Kullback-Leibler divergence formulation is from the paper
    def __kl(self, P, Q):
        div = 0 if P == Q else np.sum(P * np.log(P / Q) + (1 - P) * np.log((1 - P) / (1 - Q)))

        return div
    
    # For each arm the upper-confidence bound can be efficiently computed using Newton iterations
    def __newton_method(self, arm, tol=1e-2, maxiter=3):
        epsilon = 1e-5
        # initial value of q can be different, but will be good to set higher than mean reward value
        q = self.reward_mean[arm] + 0.05
        q = 1 if q > 1 else q
        
        for i in range(maxiter):
            # For avoiding RuntimeWarning: division by zero in log
            q = (q + epsilon) if q == 0 else (q - epsilon) if q == 1 else q
            self.reward_mean[arm] = (self.reward_mean[arm] + epsilon) if self.reward_mean[arm] == 0 else (self.reward_mean[arm] - epsilon) if self.reward_mean[arm] == 1 else self.reward_mean[arm]
            
            if q == self.reward_mean[arm]:
                return q
            
            y = self.n[arm] * self.__kl(self.reward_mean[arm], q) - (np.log(self.n['rounds']) + self.c * np.log(np.log(self.n['rounds'])))
            dy = (1 - self.reward_mean[arm]) * (self.n[arm] / (1 - q)) - self.n[arm] * (self.reward_mean[arm] / q)
            
            q_next = min(max(q - y / dy, 0), 1)
            if q_next in (0, 1):
                return q_next
            # give some tolerance to convergence
            if abs((q_next - q) / q_next) <= tol:
                return q_next
            
            q = q_next
            
        return q
        
    def select_arm(self):
        selected_arm, best = 0, 0
        for arm in range(self.arms):
            tmp = self.__newton_method(arm)
            if tmp > best:
                selected_arm = arm
                best = tmp
            elif tmp == best:
                selected_arm = random.choice([selected_arm, arm])

        return selected_arm

    def update_parameter(self, selected_arm, reward):
        self.reward_mean[selected_arm] *= self.n[selected_arm]
        self.reward_mean[selected_arm] += reward
        self.n[selected_arm] += 1
        self.reward_mean[selected_arm] /= self.n[selected_arm]
        
        self.n['rounds'] += 1
