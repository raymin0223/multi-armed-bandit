# -*- coding: utf-8 -*-
import time
import numpy as np

from thompson import ThompsonSampling


class CI_ThompsonSampling(ThompsonSampling):

    CONFIDENCE_INTERVAL_VALUE = {"50%": 0.68, "55%": 0.76, "60%": 0.84, "65%": 0.93, "70%": 1.04,
                                 "75%": 1.15, "80%": 1.28, "85%": 1.44, "90%": 1.65, "95%": 1.96}

    def __init__(self, conf_fname, cache, key):
        super(CI_ThompsonSampling, self).__init__(conf_fname, cache, key)
        self.ci_enabled = self.opt.ci.get("enabled", False)
        self.confidence_level = self.__set_confidence_level()

    def initialize(self, arms, data_k="arms"):
        if not arms:
            return False
        arms = map(str, arms)  # ensure key type
        t = time.time()
        self.data[data_k] = {_arm: {'selected_at': t,
                                    'click': 0.0,
                                    'unclick': 0.0,
                                    'trial': 1.0,
                                    'average': 0.0,
                                    'threshold_high': 0.0,
                                    'threshold_low': 0.0,
                                    'sampled': False
                                    }
                             for _arm in arms
                             if _arm != self.key}
        return True

    def _add_new_arm(self, arm, data_k="arms"):
        if self.is_valid_arm(arm, data_k=data_k):
            return False
        self.data[data_k][arm] = {'selected_at': time.time(),
                                  'click': 0.0,
                                  'unclick': 0.0,
                                  'trial': 1.0,
                                  'average': 0.0,
                                  'threshold_high': 0.0,
                                  'threshold_low': 0.0,
                                  'sampled': False}
        self.logger.info('new arm inserted: %s' % arm)
        return True

    def __set_confidence_level(self):
        value = CI_ThompsonSampling.CONFIDENCE_INTERVAL_VALUE[self.opt.ci.confidence_interval]
        return value

    def __detect_change_ci(self, param):
        changed = False
        if (param['average'] < param['threshold_low'] or param['average'] > param['threshold_high']) and param['average'] > self.opt.ci.get('average_threshold', 0.06):
            self.logger.info("(ci) User preference is changed")
            self.logger.info("(ci) threshold_high: %s, threshold_low: %s" % (param['threshold_high'], param['threshold_low']))
            self.logger.info("(ci) arm average reward: %s" % param['average'])
            self.logger.info("(ci) arm selected time: %s" % param['trial'])
            changed = True
        return changed

    def gamma_restart(self, arm, data_k):
        for arms, info in self.data[data_k].iteritems():
            # do not gamma restart to unpopular arms
            if info['average'] > self.opt.ci.get('average_threshold', 0.06):
                info['click'] *= self.opt.ci.gamma
                info['unclick'] *= self.opt.ci.gamma

            if arms == arm:
                info['trial'] = 1.0
                info['average'] = 0.0
                info['threshold_high'] = 0.0
                info['threshold_low'] = 0.0
                info['sampled'] = False

    def __calculate_average_reward(self, arm, param, imp, samples, reward):
        sample_sum_tmp = param['trial'] * param['average']
        param['trial'] += imp
        if samples.get(arm, None):
            hit = samples[arm]
            hit = hit['hit'] if isinstance(hit, dict) else hit
            reward += hit
        param['average'] = (reward + sample_sum_tmp) / max(1, param['trial'])

    def ci(self, arm, imp, samples, data_k="arms"):
        reward = 0.0
        param = self.data[data_k][arm]
        if not param['sampled']:
            self.__calculate_average_reward(arm, param, imp, samples, reward)

            if param['trial'] >= self.opt.ci.sample_size:
                variance = param['average'] - pow(param['average'], 2)
                self.logger.info("(ci) average and variance of %s: %f, %f" % (arm, param['average'], variance))
                param['threshold_high'] = param['average'] + self.confidence_level * np.sqrt(variance / max(1, param['trial']))
                param['threshold_low'] = param['average'] - self.confidence_level * np.sqrt(variance / max(1, param['trial']))

                param['average'] = 0.0
                param['trial'] = 1.0
                param['sampled'] = True

        else:
            self.__calculate_average_reward(arm, param, imp, samples, reward)
            if param['trial'] >= self.opt.ci.sample_size and self.__detect_change_ci(param):
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

            if self.ci_enabled:
                self.ci(arm, imp, samples)
                if self.dgp_enabled:
                    self.ci(arm, imp, samples, data_k="dgp_arms")

        if not self.moment_method.get("enabled", False):
            return
        self.logger.info("(ci) moment_method enabled...")
        use_global = self.moment_method.get("use_global", True)
        global_key = self.moment_method.get("global_key", "global")
        if use_global and self.key != global_key:
            self.logger.info("(ci) global moment_method enabled...")
            gb_ts = CI_ThompsonSampling(self.conf_fname, self.cache, global_key)
            gb_ts.load_from_cache()
            self.prior_alpha = gb_ts.prior_alpha
            self.prior_beta = gb_ts.prior_beta
            self._update_prior(self.prior_alpha, self.prior_beta)
            return
        min_clks = self.moment_method.get("min_clicks", 100)
        min_arms = self.moment_method.get("min_arms", 100)
        prior_beta = self.moment_method.get("prior_beta", 15)
        mu_pool = []
        for arm in self.data['arms']:
            clks = self.data['arms'][arm]['click']
            # Not good for CI (if average_of_arm > average_threshold: click get gamma-decay)
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
                tmp = max((1.0 - mean_0) * est, 0.01)
                self.prior_alpha = (mean_0 * est) * (prior_beta / tmp)
                self.prior_beta = prior_beta
                self._update_prior(self.prior_alpha, self.prior_beta)
                self.logger.info("(ci) mu_pool: prior_alpha(%.3f) prior_beta(%.3f)" % (self.prior_alpha, self.prior_beta))
