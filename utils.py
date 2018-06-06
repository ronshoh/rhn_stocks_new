#!/usr/bin/env python

import sys
import re
import os, shutil
import numpy as np
import tensorflow as tf
import argparse
import scipy.io

def get_command_line_args(Config):
    ap = argparse.ArgumentParser()

    ap.add_argument("--weight_decay", type=float, nargs=1, default=None, help='L2 weight normalization')
    ap.add_argument("--max_grad_norm", type=float, nargs=1, default=None, help='truncation of gradient magnitude')
    ap.add_argument("--drop_i", type=float, nargs=1, default=None, help='drop rate for input')
    ap.add_argument("--drop_e", type=float, nargs=1, default=None, help='drop rate for input')
    ap.add_argument("--drop_h", type=float, nargs=1, default=None, help='drop rate for state')
    ap.add_argument("--drop_o", type=float, nargs=1, default=None, help='drop rate for RHN output')
    ap.add_argument("--drop_g", type=float, nargs=1, default=None, help='drop rate for global')
    ap.add_argument("--mc_est", type=int, nargs=1, default=None, help='flag to decide if us monte carlo estimation')
    ap.add_argument("--mc_drop_i", type=float, nargs=1, default=None, help='MC drop rate for input')
    ap.add_argument("--mc_drop_e", type=float, nargs=1, default=None, help='MC drop rate for input')
    ap.add_argument("--mc_drop_h", type=float, nargs=1, default=None, help='MC drop rate for state')
    ap.add_argument("--mc_drop_o", type=float, nargs=1, default=None, help='MC drop rate for RHN output')
    ap.add_argument("--mc_drop_g", type=float, nargs=1, default=None, help='MC drop rate for global')
    ap.add_argument("--mc_steps", type=int, nargs=1, default=None, help='number of MC iterations')
    ap.add_argument("--hidden_size", type=int, nargs=1, default=None, help='state size')
    ap.add_argument("--mask", type=float, nargs='*', default=None, help='target msking')
    ap.add_argument("--num_steps", type=int, nargs=1, default=None, help='length of BPTT')
    ap.add_argument("--init_scale", type=float, nargs=1, default=None, help='scaling for weight initialization')
    ap.add_argument("--state_gate", type=int, nargs=1, default=None, help='flag to use state gating')
    ap.add_argument("--init_bias", type=float, nargs=1, default=None, help='bias for gating')
    ap.add_argument("--num_layers", type=int, nargs=1, default=None, help='number of rhn layers')
    ap.add_argument("--depth", type=int, nargs=1, default=None, help='depth of each layer')
    ap.add_argument("--depth_out", type=int, nargs=1, default=0, help='layers after recurrent layers')
    ap.add_argument("--emb_size", type=int, nargs=1, default=None, help='number of embedding neurons')
    ap.add_argument("--emb_groups", type=str, nargs=1, default=None, help='what type of embedding groups')
    ap.add_argument("--glob_feat_in_size", type=int, nargs=1, default=None, help='number of global neurons')
    ap.add_argument("--glob_feat_groups", type=str, nargs=1, default=None, help='what type of global groups')
    ap.add_argument("--glob_feat_conf", type=str, nargs=1, default=None, help='what type of global configuration')
    ap.add_argument("--out_size", type=int, nargs=1, default=None, help='size of output')
    ap.add_argument("--adaptive_optimizer", type=str, nargs=1, default=None, help='which adaptive optimizer to use')
    ap.add_argument("--loss_func", type=str, nargs=1, default=None, help='what loss function to use')
    ap.add_argument("--reset_weights_flag", type=int, nargs=1, default=None,
                    help='flag to reste weights between each time window')
    ap.add_argument("--start_time", type=int, nargs=1, default=None, help='time to start testing')
    ap.add_argument("--wind_step_size", type=int, nargs=1, default=None, help='time between test windows')
    ap.add_argument("--switch_to_asgd", type=int, nargs=1, default=None, help='what epoch to switch to ASGD')
    ap.add_argument("--decay_epochs", type=int, nargs='*', default=None,
                    help='list of what epochs to decay learning rate')
    ap.add_argument("--learning_rate", type=float, nargs=1, default=None, help='initial learning rate')
    ap.add_argument("--lr_decay", type=float, nargs='*', default=None, help='list of what decays to use')
    ap.add_argument("--max_max_epoch", type=int, nargs=1, default=None, help='number of total epochs')
    ap.add_argument("--DB_name", type=str, nargs=1, default=None, help='database name')
    # ap.add_argument("--random", type=int, nargs=1, default=1)
    ap.add_argument("--server", type=int, nargs=1, default=0, help='flag to say if use linux server')
    ap.add_argument("--gpu", type=int, nargs='*', default=-1, help='which GPU to use for simulation')
    ap.add_argument("--num_of_proc", type=int, nargs=1, default=1,
                    help='how many simulations to run simultaneously')
    ap.add_argument("--tf_seed", type=int, nargs=1, default=None, help='tensorflow seed')
    ap.add_argument("--numpy_seed", type=int, nargs=1, default=None, help='numpy seed')
    ap.add_argument("--n_experts", type=int, nargs=1, default=None,
                    help='number of regression layers (mixture of regression)')
    ap.add_argument("--h_last", type=int, nargs=1, default=None, help='In case of using mixture, what size of layer')
    ap.add_argument("--drop_l", type=float, nargs=1, default=None, help='drop rate for latent')
    ap.add_argument("--mc_drop_l", type=float, nargs=1, default=None, help='MC drop rate for latent')

    args = ap.parse_args()
    for arg in vars(args):
        attr = getattr(args, arg)
        if attr is not None:
            if type(attr) == list:
                if hasattr(Config, arg):
                    if type(getattr(Config, arg)) == list:
                        setattr(Config, arg, attr)
                    elif len(attr) == 1:
                        setattr(Config, arg, attr[0])
                    else:
                        print('you enter list where its not suppose to be list.. exiting')
                        print('you enter ' + str(attr))
                        print('original ' + str(getattr(Config, arg)))
                        sys.exit()
                elif len(attr) == 1:
                    setattr(Config, arg, attr[0])
                else:
                    setattr(Config, arg, attr)
            else:
                setattr(Config, arg, attr)

# def isRandom():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--random", type=bool, nargs=1, default=True)
#     args = ap.parse_args()
#     return args.random

def get_relevant_features_idx(arg):
    if arg == "all":
        return np.array([i for i in range(385)])
    elif arg == "opt_a":
        l1 = np.array([i for i in range(132)])
        l2 = np.array([i for i in range(289,293)])
        return np.concatenate([l1,l2])
    elif arg == "opt_b":
        return np.array([i for i in range(132,385)])
    elif arg == "old":
        return np.array([i for i in range(179)])
    else:
        print("wrong argument for \'get_relevant_features_idx\'!! exiting")
        exit()


def get_scores_mask(y, config):
    # mask can be just rectangle if it's float
    # if its a list of float then the first one is the rectangle mask value and the second one is the probability
    #       of the others to be unmasked
    # nan targets (with value 0) will be masked anyway
    if len(config.mask) == 1: # treshold
        return np.array(abs(y) > config.mask[0], dtype=np.float32)
    elif len(config.mask) == 2: # threshold + probability for low targets
        random_mask = np.array(abs(y) > config.mask[0], dtype=np.float32) + np.random.random_sample(y.shape)
        return np.array(random_mask > 1-config.mask[1], dtype=np.float32) * (y != 0)
    elif len(config.mask) == 3: # threshold + probability for high targets + probability for low targets
        thresholds = config.mask[1] * np.array(abs(y) > config.mask[0], dtype=np.float32) + \
                     config.mask[2] * np.array(abs(y) <= config.mask[0], dtype=np.float32)
        return np.array(thresholds + np.random.random_sample(thresholds.shape), dtype=np.float32) * (y != 0)
    else:
        print("wrong argument for get_scores_mask!! mask=" + str(config.mask))
        exit()


def get_noise(m, drop_i, drop_h, drop_o, drop_l, drop_e, drop_g):
    keep_i, keep_h, keep_o, keep_l, keep_e, keep_g = 1.0 - drop_i, 1.0 - drop_h, 1.0 - drop_o, 1 - drop_l, 1 - drop_e, 1 - drop_g
    if keep_i < 1.0:
        noise_i = (np.random.random_sample((m.batch_size, m.in_size, m.num_layers)) < keep_i).astype(np.float32) / keep_i
    else:
        noise_i = np.ones((m.batch_size, m.in_size, m.num_layers), dtype=np.float32)
    if m.emb_size != 0:
        if keep_e < 1.0:
            noise_e = (np.random.random_sample((m.batch_size, m.emb_size)) < keep_e).astype(np.float32) / keep_e
        else:
            noise_e = np.ones((m.batch_size, m.emb_size), dtype=np.float32)
    else:
        noise_e = None
    if keep_h < 1.0:
        noise_h = (np.random.random_sample((m.batch_size, m.size, m.num_layers)) < keep_h).astype(np.float32) / keep_h
    else:
        noise_h = np.ones((m.batch_size, m.size, m.num_layers), dtype=np.float32)
    if keep_o < 1.0:
        noise_o = (np.random.random_sample((m.batch_size, 1, m.size)) < keep_o).astype(np.float32) / keep_o
    else:
        noise_o = np.ones((m.batch_size, 1, m.size), dtype=np.float32)
    if keep_g < 1.0:
        noise_g = (np.random.random_sample((m.batch_size, 1, m.glob_feat_in_size)) < keep_g).astype(np.float32) / keep_g
    else:
        noise_g = np.ones((m.batch_size, 1, m.glob_feat_in_size), dtype=np.float32)
    if m.n_experts > 1 and m.h_last > 1:
        if keep_l < 1.0:
            noise_l = (np.random.random_sample((m.batch_size, 1, m.n_experts*m.h_last)) < keep_l).astype(np.float32) / keep_l
        else:
            noise_l = np.ones((m.batch_size, 1, m.n_experts*m.h_last), dtype=np.float32)
    else:
        noise_l = None
    return noise_i, noise_h, noise_o, noise_l, noise_e, noise_g


def reset_optimizer(session, name):
    print("reseting optimizer " + name)
    optimizer_scope = [v for v in tf.global_variables() if name in v.name]
    session.run(tf.variables_initializer(optimizer_scope))

def reset_weights():
    print("reseting weights")
    tf.global_variables_initializer().run()


def get_gpu_device_list(args_gpu):
    if type(args_gpu) == int:
        return str(args_gpu)
    if type(args_gpu) == list:
        return [str(g) for g in args_gpu]

# def get_num_of_proc():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--num_of_proc", type=int, nargs=1, default=1)
#     args = ap.parse_args()
#     return args.num_of_proc

# def isServer():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--server", type=bool, nargs=1, default=False)
#     args = ap.parse_args()
#     print('')
#     print("#############################")
#     if args.server:
#         print("# running on server!!!")
#     else:
#         print("# NOT running on server!!!")
#     print("#############################")
#     print('')
#     return args.server

class Logger(object):
    def __init__(self, file_path_and_name):
        self.terminal = sys.stdout
        self.log_file_name = file_path_and_name
        log = open(self.log_file_name, "w")
        log.close()

    def write(self, message):
        self.terminal.write(message)
        log = open(self.log_file_name, "a")
        log.write(message)
        log.close()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

    def encoding(self):
        pass


def get_rand_seed():
    proc_id = os.getpid()
    num = np.random.randint(np.iinfo(np.int32).max)
    for i in range(proc_id):
        num = np.random.randint(np.iinfo(np.int32).max)
    return num


def targets_nan_to_num(targets):
    print("in targets_nan_to_num")
    idxes = np.array(np.where(np.isnan(targets[:, :, 0])))
    for i in range(idxes.shape[1]):
        targets[idxes[0, i], idxes[1, i], 0] = 2
        targets[idxes[0, i], idxes[1, i], 1] = 3
        targets[idxes[0, i], idxes[1, i], 2] = 0.0
    return targets

def features_nan_to_num(features):
    print("in features_nan_to_num")
    for i in range(features.shape[2]):
        features[:, :, i] = np.nan_to_num(features[:, :, i])
        if i%50 == 0:
            print("finished converting %d features" % (i+1))
    return features

def get_documentation(config, Config):
    simulation_name = 'rhn_dep=' + str(config.depth) + '__hid_size=' + str(config.hidden_size) + \
                      '__DB=' + config.DB_name + \
                      '__mask=' + str(config.mask).replace('.','_').replace(' ','').replace('[','').replace(']','') + \
                      '__wind=' + str(config.wind_step_size)

    if config.state_gate:
        simulation_name = simulation_name + '_stg'

    if config.reset_weights_flag:
        simulation_name = simulation_name + '_Rst'

    # if Config.random:
    #     simulation_name = simulation_name + '_Rnd'

    if Config.server:
        simulation_name = simulation_name + '_Srv'

    if Config.mc_est:
        simulation_name = simulation_name + '_MC'

    count = 0
    documentation_dir = os.path.join('./documentation/' + simulation_name)
    while os.path.isdir(documentation_dir + '__' + str(count)):
        count += 1
    simulation_name = simulation_name + '__' + str(count)

    documentation_dir = os.path.join('./documentation/' + simulation_name)
    os.makedirs(documentation_dir)
    sys.stdout = Logger(documentation_dir + "/logfile.log")
    os.makedirs(documentation_dir + '/saver')
    print('simulation is saved to %s' % documentation_dir)
    print("process id = " + str(os.getpid()))

    scripts = ['main.py', 'rhn_stocks.py', 'utils.py']
    for sc in scripts:
        dst_file = os.path.join(documentation_dir, os.path.basename(sc))
        shutil.copyfile(sc, dst_file)

    # documentation of configurations
    print('')
    print('')
    print("######################################  CONFIGURATIONS  ######################################")
    text_file = open(documentation_dir + "/configurations.txt", "w")
    for attr, value in sorted(vars(Config).items()):
        if str(attr).startswith("__"): continue
        line = str(attr) + '=' + str(value)
        print("# " + line)
        text_file.write(line + '\n')
    text_file.close()
    print("##############################################################################################")
    print('')
    print('')
    return simulation_name, documentation_dir

def get_sess_config(config):
    if config.server:
        sess_config = tf.ConfigProto(device_count={"CPU": 2},
                                     inter_op_parallelism_threads=2,
                                     intra_op_parallelism_threads=8)
        sess_config.gpu_options.visible_device_list = get_gpu_device_list(config.gpu)
    else:
        sess_config = tf.ConfigProto()

    sess_config.gpu_options.allow_growth = True
    return sess_config

def train_windows_producer(config, max_time):
    print("generationg training windows:")
    end_list = [config.start_time]
    start_list = [0]
    while (end_list[-1] + config.wind_step_size) < max_time:
        print(start_list[-1], end_list[-1])
        end_list.append(end_list[-1] + config.wind_step_size)
        start_list.append(0)
    print(start_list[-1], end_list[-1])
    return zip(start_list, end_list)


def get_embedding_groups(emb_groups):
    if emb_groups == "none":
        return None
    groups = scipy.io.loadmat('groups.mat')
    return groups[emb_groups][0]


def get_idx_group(groups, idx):
    for i in range(len(groups)):
        if idx+1 in groups[i]:
            return i


def get_glob_feat_idx_list(conf):
    l_1 = [0,1,2,3,7,8,9,10]
    l_2 = [i for i in range(25)]
    l_3 = [i for i in range(198,211)]
    if conf == "conf_1":
        return l_1
    elif conf == "conf_2":
        return l_2
    elif conf == "conf_3":
        return l_1 + l_3
    elif conf == "conf_4":
        return l_2 + l_3
    elif conf == "conf_5":
        return l_3
    elif conf == "all":
        return [i for i in range(211)]
    else:
        print("non valid configuration for get_glob_feat_idx_list (conf=" + str(conf) + ").. exiting!")
        exit()
