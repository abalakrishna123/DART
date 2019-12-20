import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import argparse
import pandas as pd
import scipy.stats
from tools import statistics, utils
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*', 's')) 
color = itertools.cycle(( "#FCB716", "#2D3956", "#A0B2D8", "#988ED5", "#F68B20", "#15d134", "#008F00"))



def main():

    # In the event that you change the sub_directory within results, change this to match it.
    sub_dir = 'experts'

    ap = argparse.ArgumentParser()
    ap.add_argument('--envname', required=True)
    ap.add_argument('--t', required=True, type=int)
    ap.add_argument('--iters', required=True, type=int, nargs='+')
    ap.add_argument('--update', required=True, nargs='+', type=int)
    ap.add_argument('--save', action='store_true', default=False)
    
    params = vars(ap.parse_args())
    params['arch'] = [64]
    params['lr'] = .01
    params['epochs'] = 100

    should_save = params['save']
    del params['save']

    plt.style.use('ggplot')

    iters = params['iters']

    # Behavior Cloning loss on sup distr
    title = 'test_bc'
    params['mode'] = 'bc'
    ptype = 'sup_loss'
    params_bc = params.copy()
    del params_bc['update']     # Updates are used in behavior cloning

    c = next(color)

    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    # plt.plot(iters, means, color=c, linestyle='--')

    ptype = 'covariate_shifts'
    means, sems = utils.extract_data(params_bc, iters, title, sub_dir, ptype)
    means = abs(means)
    plt.plot(iters, means, label='Behavior Cloning', color=c)
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)


    # DAgger
    beta = .5
    title = 'test_dagger'
    params['mode'] = 'dagger'
    ptype = 'sup_loss'
    params_dagger = params.copy()
    params_dagger['beta'] = .5      # You may adjust the prior to whatever you chose.
    del params_dagger['update']
    c = next(color)

    means, sems = utils.extract_data(params_dagger, iters, title, sub_dir, ptype)
    # plt.plot(iters, means, color=c, linestyle='--')

    ptype = 'covariate_shifts'
    means, sems = utils.extract_data(params_dagger, iters, title, sub_dir, ptype)
    means = abs(means)
    plt.plot(iters, means, label='DAgger', color=c)
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)


    # Bias Variance Thresh
    title = 'test_bias_variance_switch'
    params['mode'] = 'bias_variance_switch'
    ptype = 'sup_loss'
    params_bias_variance_switch = params.copy()
    del params_bias_variance_switch['update']
    c = next(color)

    means, sems = utils.extract_data(params_bias_variance_switch, iters, title, sub_dir, ptype)
    # plt.plot(iters, means, color=c, linestyle='--')

    ptype = 'covariate_shifts'
    means, sems = utils.extract_data(params_bias_variance_switch, iters, title, sub_dir, ptype)
    means = abs(means)
    plt.plot(iters, means, label='Switch', color=c)
    plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)


    # # DAgger B
    # beta = .5
    # title = 'test_dagger_b'
    # ptype = 'sup_loss'
    # params_dagger_b = params.copy()
    # params_dagger_b['beta'] = beta      # You may adjust the prior to whatever you chose.
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
    #     plt.plot(iters, means, color=c, linestyle='--')

    #     ptype = 'surr_loss'
    #     means, sems = utils.extract_data(params_dagger_b, iters, title, sub_dir, ptype)
    #     plt.plot(iters, means, label='DAgger-B', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass


    # # Isotropic noise
    # title = 'test_iso'
    # params['mode'] = 'iso'
    # ptype = 'sup_loss'
    # params_iso = params.copy()
    # params_iso['scale'] = 1.0
    # del params_iso['update']
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
    #     ptype = 'covariate_shifts'                
    #     means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='Isotropic Noise 1.0', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass

    # # Isotropic noise
    # title = 'test_iso'
    # params['mode'] = 'iso'
    # ptype = 'sup_loss'
    # params_iso = params.copy()
    # params_iso['scale'] = 0.5
    # del params_iso['update']
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
    #     ptype = 'covariate_shifts'        
    #     means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='Isotropic Noise 0.5', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass

    # # Isotropic noise
    # title = 'test_iso'
    # params['mode'] = 'iso'
    # ptype = 'sup_loss'
    # params_iso = params.copy()
    # params_iso['scale'] = 2.0
    # del params_iso['update']
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
    #     ptype = 'covariate_shifts'
    #     means, sems = utils.extract_data(params_iso, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='Isotropic Noise 2.0', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass



    # DART
    partition = 450
    title = 'test_dart'
    params['mode'] = 'dart'
    ptype = 'sup_loss'
    params_dart = params.copy()
    params_dart['partition'] = partition
    c = next(color)
    try:
        means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
        # plt.plot(iters, means, color=c, linestyle='--')
        ptype = 'covariate_shifts'        
        means, sems = utils.extract_data(params_dart, iters, title, sub_dir, ptype)
        means = abs(means)
        plt.plot(iters, means, label='DART ' + str(partition), color=c)
        plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    except IOError:
        pass



    # # DART
    # partition = 450
    # title = 'test_dart_min_var'
    # params['mode'] = 'dart'
    # ptype = 'sup_loss'
    # params_dart_min_var = params.copy()
    # params_dart_min_var['partition'] = partition
    # params_dart_min_var['reg_penalty'] = 0.3
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_dart_min_var, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
        
    #     ptype = 'covariate_shifts'
    #     means, sems = utils.extract_data(params_dart_min_var, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='DART Reg 0.3 ' + str(partition), color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass



    # # DART
    # partition = 450
    # title = 'test_dart_min_var'
    # params['mode'] = 'dart'
    # ptype = 'sup_loss'
    # params_dart_min_var = params.copy()
    # params_dart_min_var['partition'] = partition
    # params_dart_min_var['reg_penalty'] = 0.7
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_dart_min_var, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
        
    #     ptype = 'covariate_shifts'
    #     means, sems = utils.extract_data(params_dart_min_var, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='DART Reg 0.7 ' + str(partition), color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass



    # # MIXED with Dagger mixed
    # title = 'test_mixed'
    # params['mode'] = 'mixed'
    # ptype = 'sup_loss'
    # params_mixed = params.copy()
    # params_mixed['dagger_mixed'] = 1
    # del params_mixed['update']
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_mixed, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
        
    #     ptype = 'covariate_shifts'
    #     means, sems = utils.extract_data(params_mixed, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='MIXED DAgger', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass



    # # MIXED without Dagger mixed
    # title = 'test_mixed'
    # params['mode'] = 'mixed'
    # ptype = 'sup_loss'
    # params_mixed = params.copy()
    # params_mixed['dagger_mixed'] = 0
    # del params_mixed['update']
    # c = next(color)
    # try:
    #     means, sems = utils.extract_data(params_mixed, iters, title, sub_dir, ptype)
    #     # plt.plot(iters, means, color=c, linestyle='--')
        
    #     ptype = 'covariate_shifts'
    #     means, sems = utils.extract_data(params_mixed, iters, title, sub_dir, ptype)
    #     means = abs(means)
    #     plt.plot(iters, means, label='MIXED', color=c)
    #     plt.fill_between(iters, (means - sems), (means + sems), alpha=.3, color=c)
    # except IOError:
    #     pass



    plt.title("Covariate Shift on " + str(params['envname']))
    plt.legend()
    plt.xticks(iters)
    plt.legend(loc='upper right')

    save_path = 'images/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if should_save == True:
        plt.savefig(save_path + str(params['envname']) + "_covariate_shift.pdf")
    else:
        plt.show()



if __name__ == '__main__':
    main()


