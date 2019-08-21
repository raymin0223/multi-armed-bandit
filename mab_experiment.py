# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import pickle
import logging
import datetime

from tqdm import tqdm
from tools import *
from algorithm import *

class MABexp:
    """ This class is for doing experiment of MAB algorithm on some synthetic reward dataset.
    Can create reward data using `DataMaker` class, get regret info of various MAB algorithm, and plot them.
    Any experiment setting can be controlled by `./config/mab_experiment.json` file.
    """
    # If you implement some MAB algorithms, put them here
    ALGO_MAP = {'egreedy': Egreedy,
                'ucb': UCB,
                'kl_ucb' : KL_UCB,
                'thompson' : ThompsonSampling}
    
    def __init__(self, conf_fname):
        self.opt = ConfLoader(conf_fname).opt
        self.data_maker = DataMaker(self.opt.data)
        
        self.fpath = os.path.join('./results', self.data_maker.sub_dir)
        if not os.path.isdir(self.fpath):
            os.makedirs(self.fpath)
        self._logging(os.path.join(self.fpath, 'mab_experiment.log'))
        
    def _logging(self, fpath):
        if os.path.isfile(fpath):
            os.remove(fpath)

        self.logger = logging.getLogger('MAB_experiment')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.FileHandler(fpath)
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
    def _load_data(self):
        """ Try to load reward dataset and its information pickle file.
        If except statement works, program will be shutdown and you should set `enabled=True` in json file.
        """
        try:
            with open(os.path.join(self.data_maker.dir, self.data_maker.sub_dir, 'data.pickle'), 'rb') as f:
                self.data = pickle.load(f)

            with open(os.path.join(self.data_maker.dir, self.data_maker.sub_dir, 'info.pickle'), 'rb') as f:
                self.data_info = pickle.load(f)
                
            self.logger.info('data and information is loaded')
                
        except FileNotFoundError:
            self.logger.info('Data is not found. It should be created first')
            raise
    
    def _explore_exploit(self, algo_name, param, flag=True):
        """ This function is for selecting arm, getting real reward from dataset, and updating parameters using that rewards. And plotting the results of regret values.
        Returns flag to plot regret lowerbound just only once.
        """
        _begin = datetime.datetime.now()
        
        algo = self.ALGO_MAP[algo_name](self.data_info['arms'], **param)
        
        for r in tqdm(range(self.data_info['rounds']), ascii=True, desc='rounds-%s' % algo_name):
            selected_arm = algo.select_arm()
            reward = 1 if selected_arm in self.data[r] else 0
            algo.update_parameter(selected_arm, reward)
            
            self.plotter._get_algo_regret(r, algo_name, selected_arm)
            if flag:
                self.plotter._get_lowerbound(r)
                
        flag = False
        
        _end = datetime.datetime.now()
        
        self.logger.info('(%s) elapsed for %s algorithm' % (str(_end - _begin), algo_name))
        self.logger.info('=' * 60)
        
        return flag
        
    def run(self):
        _begin = datetime.datetime.now()
        
        if self.opt.data.enabled:
            self.data_maker.run()
        self._load_data()
        
        self.plotter = Plotter(self.data, self.data_info)
    
        flag=True
        for algo_name, param in self.opt.algo.items():
            flag = self._explore_exploit(algo_name, param, flag)
        
        self.plotter._plot_regret(self.opt.name, os.path.join(self.fpath, '%s.png' % self.opt.name))
        
        _end = datetime.datetime.now()
        
        self.logger.info('(%s) elapsed for mab_experiment.py' % (str(_end - _begin)))
        self.logger.info('=' * 60)
        
if __name__ == '__main__':
    mab_exp = MABexp(sys.argv[1])
    mab_exp.run()