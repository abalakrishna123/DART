"""
    Experiment script intended to test DAgger
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import gym
import numpy as np
from tools import statistics, noise, utils
from tools.supervisor import GaussianSupervisor
import argparse
import scipy.stats
import time as timer
import framework

def main():
    title = 'test_bias_variance_switch'
    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)                         # OpenAI gym environment
    ap.add_argument('--t', required=True, type=int)                     # time horizon
    ap.add_argument('--iters', required=True, type=int, nargs='+')      # iterations to evaluate the learner on
    
    args = vars(ap.parse_args())
    # args['arch'] = [64, 64]
    args['arch'] = [64]
    args['lr'] = .01
    args['epochs'] = 100
    args['mode'] = 'bias_variance_switch'

    TRIALS = framework.TRIALS


    test = Test(args)
    start_time = timer.time()
    test.run_trials(title, TRIALS)
    end_time = timer.time()

    print "\n\n\nTotal time: " + str(end_time - start_time) + '\n\n'





class Test(framework.Test):


    def run_iters(self):
        T = self.params['t']

        results = {
            'rewards': [],
            'sup_rewards': [],
            'surr_losses': [],
            'sup_losses': [],
            'sim_errs': [],
            'data_used': [],
            'biases': [],
            'variances': [],
            'biases_learner': [],
            'variances_learner': [],
            'covariate_shifts': []
        }
        
        trajs = []
        snapshots = []
        dist_gen_agents = []
        learner_bias, learner_variance = None, None

        for i in range(self.params['iters'][-1]):
            print "\tIteration: " + str(i)

            if i == 0:
                states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
                trajs.append((states, i_actions))
                states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
                self.lnr.add_data(states, i_actions)
                self.lnr.train()
                learner_last = False
                dist_gen_agent = self.sup
            else:
                # if was learner last time and variance > some quantity switch to supervisor
                if learner_last and float(learner_variance)/(float(learner_bias) + float(learner_variance)) > 0.5: # TODO: can modify this threshold in various ways as see fit...
                    states, i_actions, _, _ = statistics.collect_traj(self.env, self.sup, T, False)
                    trajs.append((states, i_actions))
                    states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
                    self.lnr.add_data(states, i_actions)
                    self.lnr.train()
                    learner_last = False
                    dist_gen_agent = self.sup
                else:
                    states, _, _, _ = statistics.collect_traj(self.env, self.lnr, T, False)
                    i_actions = [self.sup.intended_action(s) for s in states]
                    states, i_actions, _ = utils.filter_data(self.params, states, i_actions)
                    self.lnr.add_data(states, i_actions)
                    self.lnr.train(verbose=True)
                    learner_last = True
                    learner_bias, learner_variance = statistics.evaluate_bias_variance_learner_cont(self.env, self.lnr, self.sup, T, num_samples=20)
                    dist_gen_agent = self.lnr

            if ((i + 1) in self.params['iters']):
                snapshots.append((self.lnr.X[:], self.lnr.y[:]))
                dist_gen_agents.append(dist_gen_agent)


        for j in range(len(snapshots)):
            X, y = snapshots[j]
            self.lnr.X, self.lnr.y = X, y
            self.lnr.train(verbose=True)
            print "\nData from snapshot: " + str(self.params['iters'][j])
            it_results = self.iteration_evaluation(dist_gen_agent=dist_gen_agents[j])
            
            results['sup_rewards'].append(it_results['sup_reward_mean'])
            results['rewards'].append(it_results['reward_mean'])
            results['surr_losses'].append(it_results['surr_loss_mean'])
            results['sup_losses'].append(it_results['sup_loss_mean'])
            results['sim_errs'].append(it_results['sim_err_mean'])
            results['biases'].append(it_results['biases_mean'])
            results['variances'].append(it_results['variances_mean'])
            results['biases_learner'].append(it_results['biases_learner_mean'])
            results['variances_learner'].append(it_results['variances_learner_mean'])
            results['covariate_shifts'].append(it_results['covariate_shifts_mean'])
            results['data_used'].append(len(y))


        for key in results.keys():
            results[key] = np.array(results[key])
        return results




if __name__ == '__main__':
    main()

