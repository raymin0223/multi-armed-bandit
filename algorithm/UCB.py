# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import random
import numpy as np

__all__ = ['UCB']

class UCB:
    def __init__(self, arms):
        self.arms = arms
        self.n = {'rounds': arms}
        self.reward_mean = {}
        # for the initialization of UCB algorithm param
        for arm in range(self.arms):
            self.n[arm] = 1
            self.reward_mean[arm] = np.random.binomial(1, 0.5)

    def select_arm(self):
        selected_arm, best = 0, 0
        for arm in range(self.arms):
            tmp = self.reward_mean[arm] + np.sqrt((2 * np.log(self.n['rounds']) / self.n[arm]))
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
