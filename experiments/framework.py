import time as timer
import numpy as np
from tools.expert import load_policy
from tools import statistics, noise, utils
from tools import learner
from tools.supervisor import GaussianSupervisor, Supervisor
import tensorflow as tf
from net import knet
import gym
import os
import pandas as pd
import scipy.stats


TRIALS = 20
ENSEMBLE_SIZE = 5
BOOTSTRAP_RATIO = 0.5

class Test(object):

    def __init__(self, params):
        self.params = params
        return

    def reset_learner(self, params):
        """
            Initializes new neural network and learner wrapper
        """
        est = [knet.Network(params['arch'], learning_rate=params['lr'], epochs=params['epochs']) for _ in range(ENSEMBLE_SIZE)]
        lnr = learner.Learner(est, bootstrap_ratio=BOOTSTRAP_RATIO)
        return est, lnr


    def prologue(self):
        """
            Preprocess hyperparameters and initialize learner and supervisor
        """
        self.params['filename'] = './experts/' + self.params['envname'] + '.pkl'
        self.env = gym.envs.make(self.params['envname'])

        self.params['d'] = self.env.action_space.shape[0]

        sess = tf.Session()
        policy = load_policy.load_policy(self.params['filename'])
        net_sup = Supervisor(policy, sess)
        init_cov = np.zeros((self.params['d'], self.params['d']))
        sup = GaussianSupervisor(net_sup, init_cov)
        est, lnr = self.reset_learner(self.params)

        self.lnr, self.sup, self.net_sup = lnr, sup, net_sup
        return self.params


    def run_iters(self):
        """
            To be implemented by learning methods (e.g. behavior cloning, dart, dagger...)
        """
        raise NotImplementedError


    def run_trial(self):
        """
            Run a trial by first preprocessing the parameters and initializing
            the supervisor and learner. Then run each iterations (not implemented here)
        """
        start_time = timer.time()

        self.prologue()
        results = self.run_iters()

        end_time = timer.time()
        results['start_time'] = start_time
        results['end_time'] = end_time
        results['total_time'] = end_time - start_time

        return results



    def iteration_evaluation(self, dart_sup=None, dagger_beta=None, mixed_switch_idx=None, dist_gen_agent=None):
        """
            Evaluate learner and supervisor given the current amount of data
            Supervisor is averaged over p trajectories
            Learner is averaged over q trajectories
        """
        # Asserting limited data per iteration. 
        # See experiments from Ho and Ermon, 2016 for sampling method

        print "Data: " + str(len(self.lnr.X))
        assert len(self.lnr.X) <= (self.params['iters'][-1] * 50)
        
        it_results = {}

        p = 1
        q = 3
        sup_rewards = np.zeros(p)
        sup_losses = np.zeros(q)
        rewards = np.zeros(q)
        surr_losses = np.zeros(q)
        sim_errs = np.zeros(q)
        biases = np.zeros(q)
        variances = np.zeros(q)
        biases_learner = np.zeros(q)
        variances_learner = np.zeros(q)
        cov_shifts = np.zeros(q)

        for j in range(p):
            sup_rewards[j] = statistics.eval_rewards(self.env, self.sup, self.params['t'], 1)
        
        for j in range(q):
            eval_results = self.evals(dart_sup, dagger_beta, mixed_switch_idx, dist_gen_agent)
            rewards[j] = eval_results['rewards']
            surr_losses[j] = eval_results['surr_losses']
            sup_losses[j] = eval_results['sup_losses']
            sim_errs[j] = eval_results['sim_errs'] 
            biases[j] = eval_results['biases']
            variances[j] = eval_results['variances']
            biases_learner[j] = eval_results['biases_learner']
            variances_learner[j] = eval_results['variances_learner']
            cov_shifts[j] = eval_results['covariate_shifts']

        it_results['sup_reward_mean'], it_results['sup_reward_std'] = np.mean(sup_rewards), np.std(sup_rewards)
        it_results['reward_mean'], it_results['reward_std'] = np.mean(rewards), np.std(rewards)
        it_results['surr_loss_mean'], it_results['surr_loss_std'] = np.mean(surr_losses), np.std(surr_losses)
        it_results['sup_loss_mean'], it_results['sup_loss_std'] = np.mean(sup_losses), np.std(sup_losses)
        it_results['sim_err_mean'], it_results['sim_err_std'] = np.mean(sim_errs), np.std(sim_errs)
        it_results['biases_mean'], it_results['biases_std'] = np.mean(biases), np.std(biases)
        it_results['variances_mean'], it_results['variances_std'] = np.mean(variances), np.std(variances)
        it_results['biases_learner_mean'], it_results['biases_learner_std'] = np.mean(biases_learner), np.std(biases_learner)
        it_results['variances_learner_mean'], it_results['variances_learner_std'] = np.mean(variances_learner), np.std(variances_learner)
        it_results['covariate_shifts_mean'], it_results['covariate_shifts_std'] = np.mean(cov_shifts), np.std(cov_shifts)

        print "\t\tSup reward: " + str(it_results['sup_reward_mean']) + " +/- " + str(it_results['sup_reward_std'])
        print "\t\tLnr_reward: " + str(it_results['reward_mean']) + " +/- " + str(it_results['reward_std'])
        print "\t\tSurr loss: " + str(it_results['surr_loss_mean']) + " +/- " + str(it_results['surr_loss_std'])
        print "\t\tSup loss: " + str(it_results['sup_loss_mean']) + "+/-" + str(it_results['sup_loss_std'])
        print "\t\tSim err: " + str(it_results['sim_err_mean']) + " +/- " + str(it_results['sim_err_std'])
        print "\t\tBiases: " + str(it_results['biases_mean']) + " +/- " + str(it_results['biases_std'])
        print "\t\tVariances: " + str(it_results['variances_mean']) + " +/- " + str(it_results['variances_std'])
        print "\t\tBiases Learner: " + str(it_results['biases_learner_mean']) + " +/- " + str(it_results['biases_learner_std'])
        print "\t\tVariances Learner: " + str(it_results['variances_learner_mean']) + " +/- " + str(it_results['variances_learner_std'])
        print "\t\tCovariate Shifts: " + str(it_results['covariate_shifts_mean']) + " +/- " + str(it_results['covariate_shifts_std'])
        print "\t\tTrace: " + str(np.trace(self.sup.cov))

        return it_results


    # TODO: for bias, variance, covariate shift need current distribution generating parameter to understand during TRAINING performance
    def evals(self, dart_sup, dagger_beta, mixed_switch_idx, dist_gen_agent):
        """
            Evaluate on all metrics including 
            reward, loss, and simulated error of the supervisor
        """
        results = {}
        results['rewards'] = statistics.eval_rewards(self.env, self.lnr, self.params['t'], 1)
        results['surr_losses'] = statistics.evaluate_lnr_cont(self.env, self.lnr, self.sup, self.params['t'], 1)
        results['sup_losses'] = statistics.evaluate_sup_cont(self.env, self.lnr, self.sup, self.params['t'], 1)
        results['sim_errs'] = statistics.evaluate_sim_err_cont(self.env, self.sup, self.params['t'], 1)
        biases, variances = statistics.evaluate_bias_variance_cont(self.env, self.lnr, self.sup, self.params['mode'], self.params['t'], dart_sup, dagger_beta, mixed_switch_idx, dist_gen_agent, 20)
        biases_learner, variances_learner = statistics.evaluate_bias_variance_learner_cont(self.env, self.lnr, self.sup, self.params['t'], 20)
        results['biases'] = biases 
        results['variances'] = variances
        results['biases_learner'] = biases_learner 
        results['variances_learner'] = variances_learner
        results['covariate_shifts'] = statistics.evaluate_covariate_shift_cont(self.env, self.lnr, self.sup, self.params['mode'], self.params['t'], dart_sup, dagger_beta, mixed_switch_idx, dist_gen_agent, 20)
        return results



    def run_trials(self, title, TRIALS):
        """
            Runs and saves all trials. Generates directories under 'results/experts/'
            where sub-directory names are based on the parameters. Data is saved after
            every trial, so it is safe to interrupt program.
        """
        iters = self.params['iters']
        sub_dir = 'experts'
        paths = {}
        for it in iters:
            self.params['it'] = it
            parent_data_dir = utils.generate_data_dir(title, sub_dir, self.params)
            save_path = parent_data_dir + 'data.csv'
            paths[it] = save_path
            if not os.path.exists(parent_data_dir):
                os.makedirs(parent_data_dir)
            print "Creating directory at " + str(save_path)


        m = len(iters)
        self.rewards_all, self.sup_rewards_all = np.zeros((TRIALS, m)), np.zeros((TRIALS, m))       # reward obtained from learner and (noisy) supervisor
        self.surr_losses_all, self.sup_losses_all = np.zeros((TRIALS, m)), np.zeros((TRIALS, m))    # loss obtained on supervisor's distribution and on learner's distribution
        self.sim_errs_all = np.zeros((TRIALS, m))                                                   # Empirical simulated error of supervisor (trace of covariance matrix)
        self.data_used_all = np.zeros((TRIALS, m))
        self.biases = np.zeros((TRIALS, m)) 
        self.variances = np.zeros((TRIALS, m)) 
        self.biases_learner = np.zeros((TRIALS, m)) 
        self.variances_learner = np.zeros((TRIALS, m)) 
        self.covariate_shifts = np.zeros((TRIALS, m)) 
        self.total_times_all = np.zeros((TRIALS))

        for t in range(TRIALS):
            print "\n\nTrial: " + str(t)
            results = self.run_trial()
            total_time = results['end_time'] - results['start_time']

            self.rewards_all[t, :], self.sup_rewards_all[t, :] = results['rewards'], results['sup_rewards']
            self.surr_losses_all[t, :], self.sup_losses_all[t, :] = results['surr_losses'], results['sup_losses']
            self.sim_errs_all[t, :] = results['sim_errs']
            self.data_used_all[t, :] = results['data_used']
            self.biases[t, :] = results['biases']
            self.variances[t, :] = results['variances']
            self.biases_learner[t, :] = results['biases_learner']
            self.variances_learner[t, :] = results['variances_learner']
            self.covariate_shifts[t, :] = results['covariate_shifts']
            self.total_times_all[t] = results['total_time']


            print "trial time: " + str(total_time)
            self.save_all(t + 1, paths)



    def save_all(self, t, paths):
        rewards_all = self.rewards_all[:t, :]
        surr_losses_all = self.surr_losses_all[:t, :]
        sup_rewards_all = self.sup_rewards_all[:t, :]
        sup_losses_all = self.sup_losses_all[:t, :]
        sim_errs_all = self.sim_errs_all[:t, :]
        data_used_all = self.data_used_all[:t, :]
        biases_all = self.biases[:t, :]
        variances_all = self.variances[:t, :]
        biases_learner_all = self.biases_learner[:t, :]
        variances_learner_all = self.variances_learner[:t, :]
        covariate_shifts_all = self.covariate_shifts[:t, :]
        total_times_all = self.total_times_all[:t]


        iters = self.params['iters']

        for i in range(len(iters)):
            it = iters[i]
            rewards = rewards_all[:, i]
            surr_losses = surr_losses_all[:, i]
            sup_rewards = sup_rewards_all[:, i]
            sup_losses = sup_losses_all[:, i]
            sim_errs = sim_errs_all[:, i]
            data_used = data_used_all[:, i]
            biases = biases_all[:, i]
            variances = variances_all[:, i]
            biases_learner = biases_learner_all[:, i]
            variances_learner = variances_learner_all[:, i]
            covariate_shifts = covariate_shifts_all[:, i]
            total_time = total_times_all[:]
            save_path = paths[iters[i]]

            print "Saving to: " + str(save_path)


            d = {'reward': rewards, 'surr_loss': surr_losses, 
                'sup_reward': sup_rewards, 'sup_loss': sup_losses,
                'sim_err': sim_errs, 'biases': biases,
                'variances': variances, 'biases_learner': biases_learner,
                'variances_learner': variances_learner, 
                'covariate_shifts': covariate_shifts,
                'data_used': data_used, 'total_time': total_time}
            df = pd.DataFrame(d)
            df.to_csv(save_path)

            reward_mean, reward_sem = np.mean(rewards), scipy.stats.sem(rewards)
            surr_loss_mean, surr_loss_sem = np.mean(surr_losses), scipy.stats.sem(surr_losses)
            sup_reward_mean, sup_reward_sem = np.mean(sup_rewards), scipy.stats.sem(sup_rewards)
            sup_loss_mean, sup_loss_sem = np.mean(sup_losses), scipy.stats.sem(sup_losses)
            sim_err_mean, sim_err_sem = np.mean(sim_errs), scipy.stats.sem(sim_errs)
            data_used_mean, data_used_sem = np.mean(data_used), scipy.stats.sem(data_used)
            biases_mean, biases_sem = np.mean(biases), scipy.stats.sem(biases)
            variances_mean, variances_sem = np.mean(variances), scipy.stats.sem(variances)
            biases_learner_mean, biases_learner_sem = np.mean(biases_learner), scipy.stats.sem(biases_learner)
            variances_learner_mean, variances_learner_sem = np.mean(variances_learner), scipy.stats.sem(variances_learner)
            covariate_shifts_mean, covariate_shifts_sem = np.mean(covariate_shifts), scipy.stats.sem(covariate_shifts)
            total_time_mean, total_time_sem = np.mean(total_time), scipy.stats.sem(total_time)

            print "Iteration " + str(it) + " results:"
            print "For iteration: " + str(it)
            print "Lnr reward: " + str(reward_mean) + ' +/- ' + str(reward_sem)
            print "Surr loss: " + str(surr_loss_mean) + " +/- " + str(surr_loss_sem)
            print "Sup reward: " + str(sup_reward_mean) + " +/- " + str(sup_reward_sem)
            print "Sup loss: " + str(sup_loss_mean) + " +/- " + str(sup_loss_sem)
            print "Sim err: " + str(sim_err_mean) + " +/- " + str(sim_err_sem)
            print "Data used: " + str(data_used_mean) + " +/- " + str(data_used_sem)
            print "Bias: " + str(biases_mean) + " +/- " + str(biases_sem)
            print "Variance: " + str(variances_mean) + " +/- " + str(variances_sem)
            print "Bias Learner: " + str(biases_learner_mean) + " +/- " + str(biases_learner_sem)
            print "Variance Learner: " + str(variances_learner_mean) + " +/- " + str(variances_learner_sem)
            print "Covariate Shift: " + str(covariate_shifts_mean) + " +/- " + str(covariate_shifts_sem)
            print "Total time: " + str(total_time_mean) + " +/- " + str(total_time_sem)
            print "\n\n\n"



        return





