# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from .ThompsonSampling import *

__all__ = ['Decay_ThompsonSampling']

class Decay_ThompsonSampling(ThompsonSampling):
    """ This class is implementation of ThompsonSampling algorithm.
    Reference from 'Analysis of Thompson Sampling for the Multi-armed Bandit Problem'.
    """
    def __init__(self, arms, prior_alpha=1, prior_beta=1, decay_ratio=0.95):
        super(Decay_ThompsonSampling, self).__init__(arms, prior_alpha, prior_beta)
        self.decay_ratio = decay_ratio
        
    def __decay_parameter(self):
        """ If multiply (0, 1) value on both alpha and beta, expectation will be same but variance will be larger.
        """
        for arm in range(self.arms):
            self.alpha[arm] *= self.decay_ratio
            self.beta[arm] *= self.decay_ratio

    def update_parameter(self, selected_arm, reward):
        if reward == 1:
            self.alpha[selected_arm] += 1
        else:
            self.beta[selected_arm] += 1
            
        self.__decay_parameter()