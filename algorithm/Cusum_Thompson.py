# -*- coding: utf-8 -*-
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from .ThompsonSampling import *

__all__ = ['CUSUM_ThompsonSampling']

class CUSUM_ThompsonSampling(ThompsonSampling):
    """ This class is implementation of ThompsonSampling with CUSUM change detection algorithm.
    Reference from 'A Change-Detection based Framework for Piecewise-stationary Multi-Armed Bandit Problem'.
    """
    def __init__(self, arms, prior_alpha=1, prior_beta=1):
        super(CUSUM_ThompsonSampling, self).__init__(arms, prior_alpha, prior_beta)
        self.trial, self.avg, self.mean_shift_minus, self.mean_shift_plus = {}, {}, {}, {}
        
        for arm in range(self.arms):
            self.trial[arm] = 1
            self.avg[arm] = 0
            self.mean_shift_minus[arm] = 0
            self.mean_shift_plus[arm] = 0

    def __detect_change(self, param, imp, reward):
        changed = False
        s_k_minus = param['average'] * imp - self.opt.cusum.epsilon * imp - reward
        s_k_plus = reward - param['average'] * imp - self.opt.cusum.epsilon * imp
        # self.logger.info("(cusum) s_k_minus: %f, s_k_plus: %f" % (s_k_minus, s_k_plus))

        param['mean_shift_minus'] = max(0, param['mean_shift_minus'] + s_k_minus)
        param['mean_shift_plus'] = max(0, param['mean_shift_plus'] + s_k_plus)
        # self.logger.info("(cusum) threshold: %d" % self.opt.cusum.threshold)
        # self.logger.info("(cusum) mean_shift_minus: %d, mean_shift_plus: %d" % (param['mean_shift_minus'], param['mean_shift_plus']))
        if param['mean_shift_minus'] >= self.opt.cusum.threshold or param['mean_shift_plus'] >= self.opt.cusum.threshold:
            self.logger.info("(cusum) User preference is changed")
            self.logger.info("(cusum) trial and average: (%s, %s)" % (param['trial'], param['average']))
            changed = True
        return changed

    def gamma_restart(self, arm, data_k):
        for arms, info in self.data[data_k].iteritems():
            info['click'] *= self.opt.cusum.gamma
            info['unclick'] *= self.opt.cusum.gamma
            if arms == arm:
                info['trial'] = 1.0
                info['average'] = 0.0
                info['mean_shift_minus'] = 0.0
                info['mean_shift_plus'] = 0.0

    def cusum(self, arm, imp, samples, data_k="arms"):
        reward = 0.0
        if self.data[data_k][arm]['trial'] <= self.opt.cusum.sample_size:
            sample_sum_tmp = self.data[data_k][arm]['trial'] * self.data[data_k][arm]['average']
            self.data[data_k][arm]['trial'] += imp
            if samples.get(arm, None):
                hit = samples[arm]
                hit = hit['hit'] if isinstance(hit, dict) else hit
                reward += hit
            self.data[data_k][arm]['average'] = (reward + sample_sum_tmp) / max(1, self.data[data_k][arm]['trial'])
        else:
            if samples.get(arm, None):
                hit = samples[arm]
                hit = hit['hit'] if isinstance(hit, dict) else hit
                reward += hit
            param = self.data[data_k][arm]
            if self.__detect_change_cusum(param, imp, reward):
                self.gamma_restart(arm, data_k)

    def update_arms(self, samples, impression):
        total_click = 0
        if self.dgp_enabled and time.time() > self.data['dgp_t'] + self.dgp_ttl:
            for arm, v in self.data["dgp_arms"].iteritems():
                v['click'] *= self.dgp_gamma
                v['unclick'] *= self.dgp_gamma
                v['trial'] *= self.dgp_gamma
            self.data['dgp_t'] = time.time()

        for arm, hit in samples.iteritems():
            str_arm = str(arm)
            if str_arm == self.key:
                continue
            hit = hit['hit'] if isinstance(hit, dict) else hit
            self._update_arm(str_arm, hit, 0)
            if self.dgp_enabled:
                self._update_arm(str_arm, hit, 0, data_k="dgp_arms")
            total_click += hit
        for arm in self.data['selected_arms']:
            imp = impression.get(arm, 0) if isinstance(impression, dict) else impression
            self._update_arm(arm, 0, imp)
            if self.dgp_enabled:
                self._update_arm(arm, 0, imp, data_k="dgp_arms")

            if self.cusum_enabled:
                self.cusum(arm, imp, samples)
                if self.dgp_enabled:
                    self.cusum(arm, samples, data_k="dgp_arms")

        if not self.moment_method.get("enabled", False):
            return
        self.logger.info("(cusum) moment_method enabled...")
        use_global = self.moment_method.get("use_global", True)
        global_key = self.moment_method.get("global_key", "global")
        if use_global and self.key != global_key:
            self.logger.info("(cusum) global moment_method enabled...")
            gb_ts = CUSUM_ThompsonSampling(self.conf_fname, self.cache, global_key)
            gb_ts.load_from_cache()
            self.prior_alpha = gb_ts.prior_alpha
            self.prior_beta = gb_ts.prior_beta
            self._update_prior(self.prior_alpha, self.prior_beta)
            return
        min_clks = self.moment_method.get("min_clicks", 100)
        min_arms = self.moment_method.get("min_arms", 100)
        mu_pool = []
        for arm in self.data['arms']:
            clks = self.data['arms'][arm]['click']
            if clks >= min_clks:
                unclks = self.data['arms'][arm]['unclick']
                imp = clks + max(unclks, 0)
                mu = clks / float(imp)
                mu_pool.append(mu)
        if len(mu_pool) >= min_arms:
            mean_0 = np.mean(mu_pool)
            n = len(mu_pool)
            var_0 = (np.var(mu_pool)) * n / (n - 1.0)
            est = mean_0 * (1.0 - mean_0) / var_0 - 1.0
            if est > 0 and mean_0 > 0 and mean_0 < 1.0:
                self.prior_alpha = mean_0 * est
                self.prior_beta = (1.0 - mean_0) * est
                self._update_prior(self.prior_alpha, self.prior_beta)
                self.logger.info("(cusum) mu_pool: prior_alpha(%.3f) prior_beta(%.3f)" % (self.prior_alpha, self.prior_beta))
