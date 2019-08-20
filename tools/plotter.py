import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Plotter']

class Plotter:
    def __init__(self, data, data_info):
        self.data = data
        self.data_info = data_info
    
        self.lowerbound = []
        self.algo_regret = {}
        
    def __kl(self, p, q):
        epsilon = 0.00001
        
        p += epsilon
        q += epsilon
        
        div = np.sum(p * np.log(p/q))

        return div
    
    def _get_lowerbound(self, round):
        ''' This is for asymptotic lower bound for regret.
        Reference from 'An Empirical Evaluation of Thompson Sampling'
        '''
        best_arm_idx, best_reward = self.data_info['round_rewards'][round]
        not_best_reward = self.data_info['best_reward'] - self.data_info['offset']
        
        tmp = ((best_reward - not_best_reward) / self.__kl(best_reward, not_best_reward))
        tmp *= self.data_info['arms']
        
        self.lowerbound.append((np.log(round + 0.00001) * (tmp + 0.1)))

    def _get_algo_regret(self, round, algo_name, selected_arm):
        best_arm_idx, best_reward = self.data_info['round_rewards'][round]
        not_best_reward = self.data_info['best_reward'] - self.data_info['offset']
        
        regret = 0 if selected_arm == best_arm_idx else best_reward - not_best_reward
        
        if algo_name not in self.algo_regret:
            self.algo_regret[algo_name] = [regret]
        else:
            tmp = self.algo_regret[algo_name][-1]
            self.algo_regret[algo_name].append(tmp + regret)

    def __plot(self, x):
        ylim = 0
        for algo_name, regret in self.algo_regret.items():
            plt.plot(x, regret, label='%s' % algo_name)
            ylim = max(ylim, regret[-1])
            
        plt.plot(x, self.lowerbound, label='asymptotic_lower_bound')
        
        return ylim

    def _plot_regret(self, title, fpath):
        x = range(self.data_info['rounds'])
        
        ylim = self.__plot(x)

        plt.xlim(0, self.data_info['rounds'])
        plt.ylim(0, ylim + 500)
        plt.xscale('symlog')
        plt.xlabel('rounds')
        plt.ylabel('regret')

        plt.title('Regret graph of (%s)' % title)
        plt.legend()
        plt.savefig(fpath)