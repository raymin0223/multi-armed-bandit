# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

__all__ = ['ThompsonSampling']

class ThompsonSampling:
    """ This class is implementation of ThompsonSampling algorithm.
    Reference from 'Analysis of Thompson Sampling for the Multi-armed Bandit Problem'.
    """
    def __init__(self, arms, prior_alpha=1, prior_beta=1):
        self.arms = arms
        self.alpha, self.beta = {}, {}
        
        for arm in range(self.arms):
            self.alpha[arm] = prior_alpha
            self.beta[arm] = prior_beta

    def select_arm(self):
        selected_arm, best = 0, 0
        for arm in range(self.arms):
            tmp = np.random.beta(self.alpha[arm], self.beta[arm])
            if tmp > best:
                selected_arm = arm
                best = tmp
            elif tmp == best:
                selected_arm = random.choice(selected_arm, arm)

        return selected_arm

    def update_parameter(self, selected_arm, reward):
        if reward == 1:
            self.alpha[selected_arm] += 1
        else:
            self.beta[selected_arm] += 1
