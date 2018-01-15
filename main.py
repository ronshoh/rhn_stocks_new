from __future__ import absolute_import, division, print_function

from multiprocessing import Process
from copy import deepcopy
import time

import mat4py as m4p
from utils import *
from sacred import Experiment

from rhn_stocks import Model
ex = Experiment('rhn_prediction_stocks')
logging = tf.logging


import h5py
import stocks_black_box as bb

class Config():

    # architecture
    weight_decay = 1e-07
    max_grad_norm = 0.8
    drop_i = 0.05
    drop_h = 0.3
    drop_o = 0.75
    hidden_size = 200
    mask = [0.6, 0.05] # should be a list
    num_steps = 25
    init_scale = 0.04
    state_gate = True
    init_bias = -2.5
    num_layers = 1
    depth = 5
    out_size = 1
    loss_func = "mse"

    estimation_flag = True
    esimation_epoch = 5

    # which optimmizer to use - "RMSProp" "Adam"
    adaptive_optimizer = "RMSProp"

    # monte carlo estimation
    mc_est = True
    mc_steps = 200
    mc_drop_i = 0.0
    mc_drop_h = 0.2
    mc_drop_o = 0.5

    # windows
    reset_weights_flag = True
    start_time = 4000
    wind_step_size = 100
    switch_to_asgd = 30
    decay_epochs = [15,23,30]
    learning_rate = 0.001
    lr_decay = [2.0,2.0,0.025]
    max_max_epoch = 35

    # database
    DB_name = 'CCver5_db'

    numpy_seed = None
    tf_seed = None

get_command_line_args(Config)


def run_full_epoch(session, m, feats, tars, eval_op, config, verbose=False, test_wind="all", asgd_flag=False):
    prediction_tot = np.zeros_like(tars)
    num_steps = m.num_steps
    epoch_size = feats.shape[1] // num_steps
    if test_wind=="all":
        test_wind = [0, epoch_size]
    start_time = time.time()

    grad_sum = 0.0
    max_grad = 0.0
    costs = 0.0
    state = [x.eval() for x in m.initial_state]
    noise_i, noise_h, noise_o = get_noise(m, config.drop_i, config.drop_h, config.drop_o)
    for i in range(epoch_size):
        if i==0 and verbose:
            lr = session.run(m.lr)
            m.assign_lr(session, 0.0)
        elif i==1 and verbose:
            m.assign_lr(session, lr)
        x = feats[:, i * num_steps:(i + 1) * num_steps,:]
        y = tars[:, i * num_steps:(i + 1) * num_steps]

        scores_mask = get_scores_mask(y, config)

        feed_dict = {m.input_data: x, m.targets: y, m.mask: scores_mask,
                    m.noise_i: noise_i, m.noise_h: noise_h, m.noise_o: noise_o}
        feed_dict.update({m.initial_state[i]: state[i] for i in range(m.num_layers)})

        if not asgd_flag:
            cost, state, predictions, grad_norm, _ = session.run([m.cost, m.final_state, m.predictions,
                                                                  m.global_norm, eval_op], feed_dict)
        else:
            cost, state, predictions, grad_norm, _, _, _ = \
                session.run([m.cost, m.final_state, m.predictions, m.global_norm, m.asgd_acc_op,
                             m.add_counter_op, eval_op], feed_dict)

        if grad_norm != 0.0:
            max_grad = max(max_grad, grad_norm)
            grad_sum += grad_norm

        if (i >= test_wind[0]) and (i < test_wind[1]):
            costs += cost

        prediction_tot[:,i * m.num_steps:(i + 1) * m.num_steps] = predictions

    if verbose:
        l2_loss = session.run(m.l2_loss) * config.weight_decay
        print("epoch took %.0f sec. cost: %.4f. average grad norm: %.4f. maximal grad %.4f. l2_loss: %.5f" %
              (time.time() - start_time, costs/(test_wind[1] - test_wind[0]), grad_sum/epoch_size, max_grad, l2_loss))

    return (costs / (test_wind[1] - test_wind[0])), prediction_tot


def run_mc_epoch(session, m, feats, tars, eval_op, config, test_wind):
    print("start monte carlo evaluation")
    if (m.batch_size != tars.shape[0]) or (m.num_steps != 1) or (
                    config.drop_i + config.drop_h + config.drop_o != 0.0): print("not good properties my friend!")
    mc_final = np.zeros([tars.shape[0],test_wind[1]-test_wind[0]])
    mc_std = np.zeros([tars.shape[0],test_wind[1]-test_wind[0]])
    num_steps = m.num_steps
    epoch_size = test_wind[1]

    start_time = time.time()

    state = [x.eval() for x in m.initial_state]
    noise_i, noise_h, noise_o = get_noise(m, config.drop_i, config.drop_h, config.drop_o)
    for i in range(epoch_size):

        x = feats[:, i * num_steps:(i + 1) * num_steps,:]
        y = tars[:, i * num_steps:(i + 1) * num_steps]

        scores_mask = get_scores_mask(y, config)

        if i >= test_wind[0]:
            mc_preds = np.zeros([tars.shape[0], config.mc_steps])
            for j in range(config.mc_steps):
                mc_noise_i, mc_noise_h, mc_noise_o = get_noise(m, config.mc_drop_i, config.mc_drop_h, config.mc_drop_o)


                feed_dict = {m.input_data: x, m.targets: y, m.mask: scores_mask,
                             m.noise_i: mc_noise_i, m.noise_h: mc_noise_h, m.noise_o: mc_noise_o}
                feed_dict.update({m.initial_state[i]: state[i] for i in range(m.num_layers)})

                mc_preds[:, j:j+1], _ = session.run([m.predictions,eval_op], feed_dict)

            mc_final[:, i - test_wind[0]] = mc_preds.mean(axis=1)
            mc_std[:, i - test_wind[0]] = np.std(mc_preds, axis=1)

            if (i - test_wind[0]) % ((test_wind[1] - test_wind[0])//10) == 0:
                print("finished %.3f. time passed %.0f" % (((i - test_wind[0]) / (test_wind[1] - test_wind[0]))
                                                           , time.time() - start_time))

        feed_dict = {m.input_data: x, m.targets: y, m.mask: scores_mask,
                    m.noise_i: noise_i, m.noise_h: noise_h, m.noise_o: noise_o}
        feed_dict.update({m.initial_state[i]: state[i] for i in range(m.num_layers)})

        state, _ = session.run([m.final_state, eval_op], feed_dict)

    return mc_final, mc_std


def run_algo():

    Config.numpy_seed = Config.numpy_seed if Config.numpy_seed is not None else get_rand_seed()
    Config.tf_seed = Config.tf_seed if Config.tf_seed is not None else get_rand_seed()
    Config.batch_size = targets.shape[0]
    Config.num_of_features = features.shape[2]

    config = Config()

    test_config = deepcopy(config)

    test_config.drop_x = 0.0
    test_config.drop_i = 0.0
    test_config.drop_h = 0.0
    test_config.drop_o = 0.0
    test_config.num_steps = 1
    # test_config.batch_size = targets.shape[0]

    simulation_name, documentation_dir = get_documentation(config, Config)

    matrix_epilog = '_org-db' if (targets.shape[1] == 4638) else ''

    print("setting seeds for tf and numpy")
    np.random.seed(config.numpy_seed)
    tf.set_random_seed(config.tf_seed)

    test_feat =  features          #[:,config.train_time:,:]
    test_tar =  targets[:,:,2]    #[:,config.train_time:,2]
    prediction_tot = np.zeros_like(targets[:, :, 2])
    prediction_tot_mc = np.zeros_like(targets[:, :, 2])
    std_tot_mc = np.zeros_like(targets[:, :, 2])
    sess_config = get_sess_config(config)

    with  tf.Graph().as_default(), tf.Session(config=sess_config) as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale, seed=config.tf_seed)

        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = Model(is_training=True, config=config)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mtest = Model(is_training=False, config=test_config)

        train_writer = tf.summary.FileWriter(documentation_dir, graph=tf.get_default_graph())
        tf.global_variables_initializer().run()

        train_windows = train_windows_producer(config, targets.shape[1])
        for train_time_st, train_time_end in train_windows:
            test_window = [train_time_end, min(targets.shape[1],train_time_end + config.wind_step_size)]
            print('Currently testing on times %d:%d. Train times are %d:%d'%
                  (test_window[0] ,test_window[1], train_time_st, train_time_end))

            train_feat = features[:, train_time_st:train_time_end, :]
            train_tar = targets[:, train_time_st:train_time_end, 2]

            reset_optimizer(session, config.adaptive_optimizer)

            if config.reset_weights_flag:
                reset_weights()

            if config.max_max_epoch > config.switch_to_asgd:
                mtrain.reset_asgd(session)
            lr_decay = 1.0
            best_acc = 0.0
            best_corr = 0.0
            best_cost = 100.0
            asgd_flag = False
            for i in range(config.max_max_epoch):
                st_time = time.time()

                if (i % config.esimation_epoch) == 0 and config.estimation_flag:
                    st_time = time.time()
                    if asgd_flag:
                        mtrain.store_set_asgd_weights(session)
                    costs, predictions = run_full_epoch(session, mtest, test_feat, test_tar, tf.no_op(),
                                                        config=test_config, test_wind=test_window)

                    accuracy_window_1, _, _, corr_window_1, _, _, _, _, _, _ = \
                        bb.black_box(predictions,targets, train_time_end,window1=test_window)

                    print('')
                    print("accuracy_window_%d:%d = %.10f"%(test_window[0] ,test_window[1], accuracy_window_1))
                    print("corr_window_%d:%d = %.10f"%(test_window[0] ,test_window[1], corr_window_1))
                    print("test_cost_window_%d:%d = %.10f"%(test_window[0] ,test_window[1], costs))

                    best_acc = accuracy_window_1 if (accuracy_window_1>best_acc) else best_acc
                    best_corr = corr_window_1 if (corr_window_1 > best_corr) else best_corr
                    best_cost = costs if (costs < best_cost) else best_cost

                    print("window %d:%d - best accuracy = %.10f, best corr = %.10f, best cost = %.10f"%
                          (test_window[0] ,test_window[1], best_acc, best_corr, best_cost))
                    print('')

                    tag_name = "accuracy_window_%d:%d"%(test_window[0] ,test_window[1])
                    sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=accuracy_window_1)])
                    train_writer.add_summary(sum, i + 1)

                    tag_name = "corr_window_%d:%d"%(test_window[0] ,test_window[1])
                    sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=corr_window_1)])
                    train_writer.add_summary(sum, i + 1)

                    tag_name = "cost_window_%d:%d"%(test_window[0] ,test_window[1])
                    sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=costs)])
                    train_writer.add_summary(sum, i + 1)

                    print("estimation time was %.0f" % (time.time() - st_time))
                    if asgd_flag:
                        session.run(mtrain.return_regular_weights)

                idx = next((idx for idx, x in enumerate(config.decay_epochs) if x == i), None)
                if idx is not None:
                    lr_decay = lr_decay * config.lr_decay[idx]

                st_time = time.time()
                lr = config.learning_rate / lr_decay
                mtrain.assign_lr(session, lr)

                print("Epoch: %d Learning rate: %.8f" % (i + 1, session.run(mtrain.lr)))

                if i < config.switch_to_asgd:
                    print("optimizer is " + config.adaptive_optimizer)
                    optimizer = mtrain.train_op_ad
                else:
                    print("optimizer is ASGD")
                    optimizer = mtrain.train_op_sgd
                    asgd_flag = True

                costs, _ = run_full_epoch(session, mtrain, train_feat, train_tar, optimizer, config=config,
                                          verbose=True, asgd_flag=asgd_flag)

                print("finished epoch. Time passed: %.0f " % (time.time() - st_time))

            if asgd_flag:
                mtrain.store_set_asgd_weights(session)

            st_time = time.time()
            print('')
            print("finished training on window %d:%d.. final estimation" % (train_time_st, train_time_end))
            costs, predictions = run_full_epoch(session, mtest, test_feat, test_tar, tf.no_op(), config=test_config,
                                                test_wind=test_window)

            accuracy_window_1, _, _, corr_window_1, _, _, _, _, _, _ = bb.black_box(
                predictions, targets, train_time_end, window1=test_window)

            print("final accuracy_window_%d:%d = %.10f" % (test_window[0], test_window[1], accuracy_window_1))
            print("final corr_window_%d:%d = %.10f" % (test_window[0], test_window[1], corr_window_1))
            print('')

            tag_name = "accuracy_window_%d:%d" % (test_window[0], test_window[1])
            sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=accuracy_window_1)])
            train_writer.add_summary(sum, i + 1)

            tag_name = "corr_window_%d:%d" % (test_window[0], test_window[1])
            sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=corr_window_1)])
            train_writer.add_summary(sum, i + 1)

            tag_name = "accuracy_over_time"
            sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=accuracy_window_1)])
            train_writer.add_summary(sum, test_window[1])

            tag_name = "corr_window_over_time"
            sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=corr_window_1)])
            train_writer.add_summary(sum, test_window[1])

            print("estimation time was %.0f" % (time.time() - st_time))

            prediction_tot[:, test_window[0]:test_window[1]] = deepcopy(
                predictions[:, test_window[0]:test_window[1]])

            if config.mc_est:
                st_time = time.time()
                print("starting MC estimation")
                mc_final, mc_std = run_mc_epoch(session, mtest, test_feat, test_tar, tf.no_op(),
                                                config=test_config, test_wind=test_window)
                prediction_tot_mc[:, test_window[0]:test_window[1]] = deepcopy(mc_final)
                std_tot_mc[:, test_window[0]:test_window[1]] = deepcopy(mc_std)


                accuracy_window_1, _, _, corr_window_1, _, _, _, _, _, _ = bb.black_box(
                    prediction_tot_mc, targets, train_time_end, window1=test_window)

                print("MC accuracy_window_%d:%d = %.10f" % (test_window[0], test_window[1], accuracy_window_1))
                print("MC corr_window_%d:%d = %.10f" % (test_window[0], test_window[1], corr_window_1))
                print('')

                tag_name = "MC_accuracy_over_time"
                sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=accuracy_window_1)])
                train_writer.add_summary(sum, test_window[1])

                tag_name = "MC_corr_window_over_time"
                sum = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=corr_window_1)])
                train_writer.add_summary(sum, test_window[1])

                print("MC estimation time was %.0f" % (time.time() - st_time))




    print('')
    print("finished training")
    prediction_tot[:, :config.start_time] = deepcopy(predictions[:, :config.start_time])
    prediction_tot_mc[:, :config.start_time] = deepcopy(predictions[:, :config.start_time])

    accuracy_window_1, accuracy_window_2, total_accuracy, corr_window_1, corr_window_2, corr_total, \
    train_rms_loss, test_rms_loss, wind1_rms_loss, wind2_rms_loss = bb.black_box(prediction_tot, targets,
                                                                                 config.start_time)

    print("train_rms_loss = %.4f" % (train_rms_loss))
    print("test_rms_loss = %.4f" % (test_rms_loss))
    print("wind1_rms_loss = %.4f" % (wind1_rms_loss))
    print("wind2_rms_loss = %.4f" % (wind2_rms_loss))
    print("accuracy_window_1 = %.4f" % (accuracy_window_1))
    print("accuracy_window_2 = %.4f" % (accuracy_window_2))
    print("total_accuracy = %.4f" % (total_accuracy))
    print("corr_window_1 = %.4f" % (corr_window_1))
    print("corr_window_2 = %.4f" % (corr_window_2))
    print("corr_total = %.4f" % (corr_total))

    # documentation of final scores
    text_file = open(documentation_dir + "/final_scores.txt", "w")
    line = "accuracy_window_1 = %.4f\n" % (accuracy_window_1)
    text_file.write(line)
    line = "accuracy_window_2 = %.4f\n" % (accuracy_window_2)
    text_file.write(line)
    line = "total_accuracy = %.4f\n" % (total_accuracy)
    text_file.write(line)
    line = "corr_window_1 = %.4f\n" % (corr_window_1)
    text_file.write(line)
    line = "corr_window_2 = %.4f\n" % (corr_window_2)
    text_file.write(line)
    line = "corr_total = %.4f\n" % (corr_total)
    text_file.write(line)
    text_file.close()



    accuracy_window_1, accuracy_window_2, total_accuracy, corr_window_1, corr_window_2, corr_total, \
    train_rms_loss, test_rms_loss, wind1_rms_loss, wind2_rms_loss = bb.black_box(prediction_tot_mc, targets,
                                                                                 config.start_time)

    print("MC train_rms_loss = %.4f" % (train_rms_loss))
    print("MC test_rms_loss = %.4f" % (test_rms_loss))
    print("MC wind1_rms_loss = %.4f" % (wind1_rms_loss))
    print("MC wind2_rms_loss = %.4f" % (wind2_rms_loss))
    print("MC accuracy_window_1 = %.4f" % (accuracy_window_1))
    print("MC accuracy_window_2 = %.4f" % (accuracy_window_2))
    print("MC total_accuracy = %.4f" % (total_accuracy))
    print("MC corr_window_1 = %.4f" % (corr_window_1))
    print("MC corr_window_2 = %.4f" % (corr_window_2))
    print("MC corr_total = %.4f" % (corr_total))

    # documentation of final scores
    text_file = open(documentation_dir + "/MC_final_scores.txt", "w")
    line = "accuracy_window_1 = %.4f\n" % (accuracy_window_1)
    text_file.write(line)
    line = "accuracy_window_2 = %.4f\n" % (accuracy_window_2)
    text_file.write(line)
    line = "total_accuracy = %.4f\n" % (total_accuracy)
    text_file.write(line)
    line = "corr_window_1 = %.4f\n" % (corr_window_1)
    text_file.write(line)
    line = "corr_window_2 = %.4f\n" % (corr_window_2)
    text_file.write(line)
    line = "corr_total = %.4f\n" % (corr_total)
    text_file.write(line)
    text_file.close()


    data = {'allScores': prediction_tot.tolist()}
    pred_name = '/final_predictions' + matrix_epilog

    save_path = documentation_dir + pred_name + '.mat'
    print("saving total predictions to ",save_path)
    m4p.savemat(save_path, data)

    data = {'allScores': prediction_tot_mc.tolist()}
    pred_name = '/mc_final_predictions' + matrix_epilog

    save_path = documentation_dir + pred_name + '.mat'
    print("saving MC total predictions to ",save_path)
    m4p.savemat(save_path, data)

    data = {'STDs': std_tot_mc.tolist()}
    pred_name = '/mc_final_STDs' + matrix_epilog

    save_path = documentation_dir + pred_name + '.mat'
    print("saving MC total STDs to ",save_path)
    m4p.savemat(save_path, data)


##### pre-main #####
config = Config()

print("loading DB")
f = h5py.File(config.DB_name + '.mat')

data = {}

for k, v in f.items():
    data[k] = np.array(v)

targets = np.transpose(data["targets"], [2, 0, 1])

features = np.transpose(data["features"], [2, 0, 1])

del data
del v
del k
del f

print("converting nan to num")
features = features_nan_to_num(features)
targets = targets_nan_to_num(targets)

del config



def main():
    num_of_process = Config.num_of_proc
    if num_of_process == 1:
        run_algo()
    else:
        processes = [Process(target=run_algo) for _ in range(num_of_process)]
        print('start running %d processes. Not working on linux' % (num_of_process))
        for p in processes:
            p.start()
        print('all processes are running')
        for p in processes:
            p.join()
        print('all processes finished')

if __name__ == "__main__":
    main()