import numpy as np
import scipy.stats
import random

def eval_agent_statistics_cont(env, agent, sup, T, num_samples=1):
    """
        evaluate loss in the given environment along the agent's distribution
        for T timesteps on num_samples
    """
    losses = []
    for i in range(num_samples):
        # collect trajectory with states visited and actions taken by agent
        tmp_states, _, tmp_actions, _ = collect_traj(env, agent, T)
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (sup_actions - tmp_actions) ** 2.0
        # compute the mean error on that trajectory (may not be T samples since game ends early on failures)
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))

    # compute the mean and sem on averaged losses.
    return stats(losses)


def eval_bias_variance_cont(env, agent, sup, mode, T, dart_sup, dagger_beta, mixed_switch_idx, num_samples=1):
    s = env.reset()
    biases = []
    variances = []

    if mode in ['bc', 'iso', 'rand']:
        dist_gen = sup 
    elif mode in ['dagger', 'dart', 'mixed']:
        dist_gen = None

    for i in range(num_samples):
        bias, variance, t = 0, 0, 0
        while t < T:
            a = agent.sample_action(s) # \E_D(\pi^D_\theta(s))
            a_sup = sup.intended_action(s) # \pi^*(s)
            # For variance, at each state sample actions from a random model and compare
            # to that from expected model
            a_ensemble_list = agent.intended_actions(s)
            ensemble_idx = np.random.randint(len(a_ensemble_list))
            a_ensemble = a_ensemble_list[ensemble_idx] # \pi^D_\theta(s)

            # Need to evaluate bias/variance on distribution generating parameter
            if dist_gen:
                a_dist_gen = dist_gen.sample_action(s)
            else:
                if mode == 'dagger':
                    if random.random() > dagger_beta:
                        a_dist_gen = agent.sample_action(s)
                    else:
                        a_dist_gen = sup.sample_action(s)
                elif mode == 'dart':
                        a_dist_gen = dart_sup.sample_action(s)
                elif mode == 'mixed':
                    if t < mixed_switch_idx:
                        a_dist_gen = agent.sample_action(s)
                    else:
                        a_dist_gen = sup.sample_action(s)
                else:
                    raise NotImplementedError("Unsupported Mode")


                # a_dist_gen = sup.sample_action(s)

            next_s, r, done, _ = env.step(a_dist_gen) 
            s = next_s
            bias += np.sum((a - a_sup)**2)
            variance += np.sum((a - a_ensemble)**2)
            t += 1

            if done == True:
                break

        bias /= float(t)
        variance /= float(t)
        biases.append(bias)
        variances.append(variance)
            
    return stats(biases), stats(variances)

def eval_bias_variance_learner_cont(env, agent, sup, T, num_samples=1):
    s = env.reset()
    biases = []
    variances = []

    for i in range(num_samples):
        bias, variance, t = 0, 0, 0
        while t < T:
            a = agent.sample_action(s) # \E_D(\pi^D_\theta(s))
            a_sup = sup.intended_action(s) # \pi^*(s)
            # For variance, at each state sample actions from a random model and compare
            # to that from expected model
            a_ensemble_list = agent.intended_actions(s)
            ensemble_idx = np.random.randint(len(a_ensemble_list))
            a_ensemble = a_ensemble_list[ensemble_idx] # \pi^D_\theta(s)

            # Need to evaluate bias/variance on learner's dist
            next_s, r, done, _ = env.step(a) 
            s = next_s
            bias += np.sum((a - a_sup)**2)
            variance += np.sum((a - a_ensemble)**2)
            t += 1

            if done == True:
                break

        bias /= float(t)
        variance /= float(t)
        biases.append(bias)
        variances.append(variance)
            
    return stats(biases), stats(variances)


def eval_covariate_shift_cont(env, agent, sup, mode, T, dart_sup, dagger_beta, mixed_switch_idx, num_samples=1):
    s = env.reset()
    learned_dist_losses = []
    dist_gen_param_losses = []

    if mode in ['bc', 'iso', 'rand']:
        dist_gen = sup 
    elif mode in ['dagger', 'dart', 'mixed']:
        dist_gen = None

    for i in range(num_samples):
        learner_loss, dist_gen_loss = 0, 0

        t = 0
        while t < T:
            a = agent.sample_action(s)
            a_sup = sup.intended_action(s)
            # Evaluate losses on learner's distribution
            next_s, r, done, _ = env.step(a) 
            s = next_s
            learner_loss += np.sum((a - a_sup)**2)
            t += 1

            if done == True:
                break

        learner_loss /= float(t)
        learned_dist_losses.append(learner_loss)

        t = 0
        while t < T:
            a = agent.sample_action(s)
            a_sup = sup.intended_action(s)
            # Need to evaluate bias/variance on distribution generating parameter
            if dist_gen:
                a_dist_gen = dist_gen.sample_action(s)
            else:
                if mode == 'dagger':
                    if random.random() > dagger_beta:
                        a_dist_gen = agent.sample_action(s)
                    else:
                        a_dist_gen = sup.sample_action(s)
                elif mode == 'dart':
                        a_dist_gen = dart_sup.sample_action(s)
                elif mode == 'mixed':
                    if t < mixed_switch_idx:
                        a_dist_gen = agent.sample_action(s)
                    else:
                        a_dist_gen = sup.sample_action(s)
                else:
                    raise NotImplementedError("Unsupported Mode")
            # Evaluate losses on dist gen's distribution
            next_s, r, done, _ = env.step(a_dist_gen) 
            s = next_s
            dist_gen_loss += np.sum((a - a_sup)**2)
            t += 1

            if done == True:
                break

        dist_gen_loss /= float(t)
        dist_gen_param_losses.append(dist_gen_loss)

            
    return stats(np.array(learned_dist_losses) - np.array(dist_gen_param_losses))


def eval_sup_statistics_cont(env, agent, sup, T, num_samples=1):
    """
        Evaluate loss on the supervisor's trajectory in the given env
        for T timesteps
    """
    losses = []
    for i in range(num_samples):
        # collect states made by the supervisor (actions are sampled so not collected)
        tmp_states, _, _, _ = collect_traj(env, sup, T)

        # get inteded actions from the agent and supervisor
        tmp_actions = np.array([ agent.intended_action(s) for s in tmp_states ])
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = (sup_actions - tmp_actions) ** 2.0

        # compute the mean error on that traj
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))

    # generate statistics, same as above
    return stats(losses)


def eval_sim_err_statistics_cont(env, sup, T, num_samples = 1):
    losses = []
    for i in range(num_samples):
        tmp_states, int_actions, taken_actions, _ = collect_traj(env, sup, T)
        int_actions = np.array(int_actions)
        taken_actions = np.array(taken_actions)
        errors = (int_actions - taken_actions) ** 2.0
        errors = np.sum(errors, axis=1)
        losses.append(np.mean(errors))
    return stats(losses)


def eval_rewards(env, agent, T, num_samples=1):
    reward_samples = np.zeros(num_samples)
    for j in range(num_samples):
        _, _, _, reward = collect_traj(env, agent, T)
        reward_samples[j] = reward
    return np.mean(reward_samples)


def stats(losses):
    if len(losses) == 1: sem = 0.0
    else: sem = scipy.stats.sem(losses)

    d = {
        'mean': np.mean(losses),
        'sem': sem
    }
    return d

def ste(trial_rewards):
    if trial_rewards.shape[0] == 1:
        return np.zeros(trial_rewards.shape[1])
    return scipy.stats.sem(trial_rewards, axis=0)

def mean(trial_rewards):
    return np.mean(trial_rewards, axis=0)


def mean_sem(trial_data):
    s = ste(trial_data)
    m = mean(trial_data)
    return m, s


def evaluate_lnr_cont(env, agent, sup, T, num_samples = 1):
    stats = eval_agent_statistics_cont(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sup_cont(env, agent, sup, T, num_samples = 1):
    stats = eval_sup_statistics_cont(env, agent, sup, T, num_samples)
    return stats['mean']

def evaluate_sim_err_cont(env, sup, T, num_samples = 1):
    stats = eval_sim_err_statistics_cont(env, sup, T, num_samples)
    return stats['mean']

def evaluate_bias_variance_cont(env, agent, sup, mode, T, dart_sup, dagger_beta, mixed_switch_idx, num_samples=1):
    stats_bias, stats_variance = eval_bias_variance_cont(env, agent, sup, mode, T, dart_sup, dagger_beta, mixed_switch_idx, num_samples)
    return stats_bias['mean'], stats_variance['mean']

def evaluate_covariate_shift_cont(env, agent, sup, mode, T, dart_sup, dagger_beta, mixed_switch_idx, num_samples=1):
    stats = eval_covariate_shift_cont(env, agent, sup, mode, T, dart_sup, dagger_beta, mixed_switch_idx, num_samples)
    return stats['mean']

def evaluate_bias_variance_learner_cont(env, agent, sup, T, num_samples=1):
    stats_bias, stats_variance = eval_bias_variance_learner_cont(env, agent, sup, T, num_samples)
    return stats_bias['mean'], stats_variance['mean']


def collect_traj(env, agent, T, visualize=False):
    """
        agent must have methods: sample_action and intended_action
        Run trajectory on sampled actions
        record states, sampled actions, intended actions and reward
    """
    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()

    reward = 0.0

    for t in range(T):

        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if done == True:
            break
            
    return states, intended_actions, taken_actions, reward

def collect_traj_beta(env, sup, lnr, T, beta, visualize=False):

    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()

    reward = 0.0
    count = 0
    for t in range(T):

        a_intended = lnr.intended_action(s)

        if random.random() > beta:
            a = lnr.sample_action(s)
        else:
            a = sup.sample_action(s)
            count += 1
        
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if done == True:
            break
     
    print "Beta: " + str(beta), "empirical beta: " + str(float(count) / (t + 1))       
    return states, intended_actions, taken_actions, reward

def collect_traj_mixed(env, sup, lnr, T, i, max_iter, visualize=False):

    states = []
    taken_actions = []

    s = env.reset()

    switch_idx = int(T * (i + 1)/max_iter) # TODO: make this some function of i in the future, not necessarily this one...
    print "Switch Idx: " + str(switch_idx)

    reward = 0.0
    for t in range(T):
        if t < switch_idx:
            a = lnr.sample_action(s)
        else:
            a = sup.sample_action(s)
        
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()


        if done == True:
            break

    post_switch_states = states[switch_idx:]
    post_switch_sup_actions = taken_actions[switch_idx:]
          
    return post_switch_states, post_switch_sup_actions, switch_idx, reward

