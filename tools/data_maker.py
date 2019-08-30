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
from .conf_loader import *

__all__ = ['DataMaker']

class DataMaker:
    """ This class is for making synthetic reward dataset and its information as pickle type file.
    If data is already made, making data process will be skipped.
    """
    def __init__(self, opt):
        self.opt = opt
        self.data = {}
        
        self._get_dirs()
        
    def _get_dirs(self):
        param = self.opt.data.param
        stat = '' if param.stationary else '%s_non' % param.change_type
        cont = '' if param.contextual else 'non_'
        
        self.dir = './data/{}stationary_{}contextual'.format(stat, cont)
        self.sub_dir = 'r%d-a%d-br%0.2f-off%0.2f' % (param.rounds, param.arms, param.best_reward, param.offset)
            
    def _logging(self, fpath):
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
        self.reward_list[self.best_arm_idx] += param.offset
        
        self.best_arm_idx_list = [self.best_arm_idx]

    def __change_arms_reward(self, round, param):
        """ This function is for non_stationary data setting.
        Returns changed flag and change the reward_list according to `change_type` and `change_num`.
        
        In abruptly change_type, randomly choose one of the arms to be next best arm and it will get `best_reward` in sudden point.
        In slowly change_type, randomly choose one of the arms to be next best arm and its reward will slowly increase to `best_reward`. Oppositely, previous best arm's reward will slowly decrease.
        """
        changed = False
        if param.change_type == 'abruptly':
            change_round = int(param.rounds / param.get('change_num', 10))
            if round > 0 and round % change_round == 0:
                changed = True
                random.shuffle(self.reward_list)
                self.best_arm_idx = np.argmax(self.reward_list)
                self.best_arm_idx_list.append(self.best_arm_idx)
        
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
                self.best_arm_idx_list.append(self.best_arm_idx)
                
        return changed
    
    def _make_data(self, param):
        rounds = param.rounds
        arms = param.arms
        self.round_rewards = {}

        changed = False
        checkpoint = 0
        best_rewards = []
        fig = plt.figure(figsize=(7, 5))
        color = (random.random(), random.random(), random.random())
        
        for r in tqdm(range(rounds), ascii=True, desc='rounds'):
            self.data[r] = []
            if not param.stationary:
                changed = self.__change_arms_reward(r, param)
                
            if changed:
                plt.plot(range(checkpoint, checkpoint + len(best_rewards)), best_rewards, 'o', c=color)
                color = (random.random(), random.random(), random.random())
                best_rewards = []
                checkpoint = r
            
            best_rewards.append(self.reward_list[self.best_arm_idx])
            # to calculate regret easily, just store best_arm_idx and its reward value
            self.round_rewards[r] = (self.best_arm_idx, self.reward_list[self.best_arm_idx])
            
            for arm in range(arms):
                reward = np.random.binomial(1, self.reward_list[arm])
                if reward == 1:
                    self.data[r].append(arm)

        fpath = os.path.join(self.dir, self.sub_dir, 'data.pickle')
        with open(fpath, 'wb') as f:
            pickle.dump(self.data, f)
            
        if best_rewards:
            plt.plot(range(checkpoint, checkpoint + len(best_rewards)), best_rewards, 'o', c=color)
        plt.plot(range(rounds), [param.best_reward-param.offset]*rounds, 'o', c=(0.5, 0.5, 0.5))
        plt.ylim(0, 1)
        fig.savefig(os.path.join(self.dir, self.sub_dir, 'arms_reward.png'))
        
        self.logger.info('Making data is accomplished')
        self.logger.info('=' * 60)

    # If `contextual=True` setting, contexts of each arm will be also stored in info.pickle
    def __get_arms_context(self, info, param):
        arms_context = {}
        dim = param.get('context_dim', 40)
        for arm in range(param.arms):
            arms_context[arm] = np.random.rand(dim)
        
        info['arms_context'] = arms_context
        info['arms_context_dim'] = dim
    
    def _store_data_info(self, param):
        info = {}
        
        info['rounds'] = param.rounds
        info['arms'] = param.arms
        info['best_reward'] = param.best_reward
        info['offset'] = param.offset
        info['best_arm_idx'] = self.best_arm_idx_list
        info['round_rewards'] = self.round_rewards
        self.logger.info('rounds, arms_number, best_reward, offset, best_arm_index, each rounds reward, and stationary or contextual information is stored')
        
        info['stationary'] = param.stationary
        if not param.stationary:
            info['change_type'] = param.change_type
            info['change_num'] = param.change_num
            self.logger.info('non-stationary type and change_number infomation is stored')
        
        info['contextual'] = param.contextual
        if param.contextual:
            self.__get_arms_context(info, param)
            self.logger.info('context vector of arms information is stored')
            
        fpath = os.path.join(self.dir, self.sub_dir, 'info.pickle')
        with open(fpath, 'wb') as f:
            pickle.dump(info, f)
            
        self.logger.info('Storing data information is accomplished')
        self.logger.info('=' * 60)

    def run(self):
        _begin = datetime.datetime.now()
        
        self._logging(os.path.join('./results', self.sub_dir, self.opt.name, 'mab_experiment.log'))
        fpath = os.path.join(self.dir, self.sub_dir)
        if os.path.isfile(os.path.join(fpath, 'data.pickle')):
            self.logger.debug('Data is already made')
            self.logger.info('=' * 60)
            
        else:
            if not os.path.isdir(fpath):
                os.makedirs(fpath)

            param = self.opt.data.param
            self._get_arms_reward(param)
            self._make_data(param)
            self._store_data_info(param)
        
        _end = datetime.datetime.now()
        
        self.logger.info('(%s) elapsed for data_maker.py' % (str(_end - _begin)))
        self.logger.info('=' * 60)