# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import random
import numpy as np

__all__ = ['Egreedy']

class Egreedy:
    """ This class is implementation of Egreedy algorithm.
    Reference from 'Finite-time Analysis of the Multiarmed Bandit Problem'.
    """
    # d should be in interval 0 and (best reward - second reward) & c just positive scalar
    def __init__(self, arms, c=6, d=0.2):
        self.arms = arms
        self.e, self.c, self.d = 1, c, d
        
        self.n = {'rounds': 1}
        self.reward_mean = {}
        # for the initialization of Egreedy algorithm param
        for arm in range(self.arms):
            self.n[arm] = 0
            self.reward_mean[arm] = 0

    def select_arm(self):
        def _get_epsilon(arms, c, d, n):
            e = min(1, (c * arms) / (pow(d, 2) * n))
            return e

        # if True Exploit, False Explore
        if not np.random.binomial(1, _get_epsilon(self.arms, self.c, self.d, self.n['rounds'])):
            selected_arm, best = 0, 0
            for arm in range(self.arms):
                tmp = self.reward_mean[arm]
                if tmp > best:
                    selected_arm = arm
                    best = tmp
                elif tmp == best:
                    selected_arm = random.choice([selected_arm, arm])

        else:
            selected_arm = random.choice(list(range(self.arms)))

        return selected_arm

    def update_parameter(self, selected_arm, reward):
        self.reward_mean[selected_arm] *= self.n[selected_arm]
        self.reward_mean[selected_arm] += reward
        self.n[selected_arm] += 1
        self.reward_mean[selected_arm] /= self.n[selected_arm]
        
        self.n['rounds'] += 1
