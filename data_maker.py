# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import random
import pickle
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.conf_loader import *


class DataMaker:
    def __init__(self, conf_fname):
        self.opt = ConfLoader(conf_fname).opt
        
        self.fpath = 'r%d-a%d-br%0.2f-off%0.2f' % (self.opt.param.rounds, self.opt.param.arms, self.opt.param.best_reward, self.opt.param.offset)
        self._make_dirs()
        
        fpath = os.path.join(self.opt.dir, self.fpath, 'data_maker.log')
        self._logging(fpath)

        self.data = {}
        
    def _make_dirs(self):
        fpath = os.path.join(self.opt.dir, self.fpath)
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
            
    def _logging(self, fpath):
        if os.path.isfile(fpath):
            os.remove(fpath)
            
        self.logger = logging.getLogger('DataMaker')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.FileHandler(fpath)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def _get_arms_reward(self, param):
        """ One of the arms will get best reward probability of bernoulli distribution, and others will get some value as low as offset amount.
        Return class attribute reward_list (list type) that each index arm have its own reward value
        """
        reward = param.best_reward - param.offset
        self.reward_list = [reward] * param.arms

        self.best_arm_idx = np.random.randint(param.arms)
        self.logger.info('(%s) is the best arm' % self.best_arm_idx)
        self.reward_list[self.best_arm_idx] += param.offset

    def __change_arms_reward(self, round, param):
        changed = False
        if param.change_type == 'abruptly':
            change_round = int(param.rounds / param.get('change_num', 10))
            if round > 0 and round % change_round == 0:
                changed = True
                random.shuffle(self.reward_list)
                self.best_arm_idx = np.argmax(self.reward_list)
                self.logger.info('In %s non-stationary setting, now (%s) is the best arm in round %s' % (param.change_type, self.best_arm_idx, round))
        
        elif param.change_type == 'slowly':
            change_round = int(param.rounds / param.get('change_num', 10))
            change_by = param.offset / change_round
            if round % change_round == 0:
                self.prev_best = self.best_arm_idx
                self.next_best = np.random.randint(param.arms)
            
            for a, r in enumerate(self.reward_list):
                if a == self.prev_best:
                    self.reward_list[a] = min(max(r - change_by, param.best_reward - param.offset), param.best_reward)
                if a == self.next_best:
                    self.reward_list[a] = min(max(r + change_by, param.best_reward - param.offset), param.best_reward)
            
            if round % change_round != 0 and round % (change_round / 2) == 0:
                changed = True
                self.best_arm_idx = self.next_best
                self.logger.info('In %s non-stationary setting, now (%s) is the best arm in round %s' % (param.change_type, self.best_arm_idx, round))
                
        return changed
    
    def _make_data(self, param):
        rounds = param.rounds
        arms = param.arms

        fig = plt.figure(figsize=(7, 5))
        color = (random.random(), random.random(), random.random())
        changed = False
        for r in tqdm(range(rounds), ascii=True, desc='rounds'):
            self.data[r] = []
            if param.stationary:
                changed = self.__change_arms_reward(r, param)
                
            if changed:
                color = (random.random(), random.random(), random.random())
                
            plt.plot(r, self.reward_list[self.best_arm_idx], 'o', c=color)
            
            for arm in range(arms):
                reward = np.random.binomial(1, self.reward_list[arm])
                if reward == 1:
                    self.data[r].append(arm)

        fpath = os.path.join(self.opt.dir, self.fpath, 'data.pickle')
        with open(fpath, 'wb') as f:
            pickle.dump(self.data, f)
            
        plt.plot(range(rounds), [param.best_reward-param.offset]*rounds, 'o', c=(0.5, 0.5, 0.5))
        plt.ylim(0, 1)
        fig.savefig(os.path.join(self.opt.dir, self.fpath, 'arms_reward.png'))
        
        self.logger.info('Making data is accomplished')

    def __get_arms_context(self, info, param):
        arms_context = {}
        for arm in range(param.arms):
            arms_context[arm] = np.random.rand(1, param.get('context_dim', 40))
        
        info['arms_context'] = arms_context
    
    def _store_data_info(self, param):
        info = {}
        
        info['rounds'] = param.rounds
        info['arms'] = param.arms
        info['best_reward'] = param.best_reward
        info['offset'] = param.offset
        info['stationary'] = param.stationary
        info['contextual'] = param.contextual
        
        if param.contextual:
            self.__get_arms_context(info, param)
        
        fpath = os.path.join(self.opt.dir, self.fpath, 'info.pickle')
        with open(fpath, 'wb') as f:
            pickle.dump(info, f)
            
        self.logger.info('Storing data information is accomplished')

    def run(self):
        _begin = datetime.datetime.now()
        
        param = self.opt.param
        self._get_arms_reward(param)
        self._make_data(param)
        self._store_data_info(param)
        
        _end = datetime.datetime.now()
        self.logger.info('(%s) elapsed for data_maker.py' % (str(_end - _begin)))

if __name__ == '__main__':
    data_maker = DataMaker(sys.argv[1])
    data_maker.run()