import numpy as np
import os
import datetime

class Config():

    # architecture
    weight_decay = 3e-06
    max_grad_norm = 0.8
    drop_i = 0.0
    drop_h = 0.3
    drop_o = 0.75
    hidden_size = 200
    mask = 0.59
    num_steps = 25
    init_scale = 0.04
    state_gate = True
    init_bias = -2.5
    num_layers = 1
    depth = 4

    # which optimmizer to use - "RMSProp" "Adam"
    adaptive_optimizer = "RMSProp"

    # windows
    reset_weights_flag = True
    start_time = 4000
    wind_step_size = 100
    switch_to_asgd = 40
    decay_epochs = np.array([10,15,20,25,30,35,40])
    learning_rate = 0.001
    lr_decay = [1.3,1.3,1.3,1.3,1.3,1.3,(0.1/(1.3**6))]
    max_max_epoch = 40

    # database
    DB_name = 'CCver5_db'

scores_list = ["accuracy_window_1", "accuracy_window_2", "total_accuracy", "corr_window_1",
               "corr_window_2", "corr_total"]

doc_path_list = ["C:\\Users\\User\\Desktop\\ron_try\\rhn_ptb_new\\documentation",
             "\\\\132.72.53.182\\common_space\\tensorflow-all\\ron\\rhn_stocks_new\\documentation"]

def get_attr_list():
    attr_list = []
    for a in sorted(Config.__dict__):
        if a.startswith("__"): continue
        attr_list.append(a)
    return attr_list

def getKey(item):
    return item[2]

def get_sorted_folder_and_date_list():
    folder_and_date_list = []
    for doc_fold in doc_path_list:
        for (folder, subs, files) in os.walk(doc_fold):
            if os.path.exists(os.path.join(folder, "final_predictions.mat")):
                sim_name = folder.split("\\")[-1]
                sim_time = os.path.getctime(folder)
                folder_and_date_list.append([folder, sim_name, sim_time])
    folder_and_date_list = sorted(folder_and_date_list, key=getKey)
    return folder_and_date_list

def print_first_line(attr_list, csv_file):
    csv_file.write('simulation name, date')
    for attr in attr_list:
        csv_file.write(',' + attr)
    for attr in scores_list:
        csv_file.write(',' + attr)
    csv_file.write(',' + 'bbox_1' + ',' + 'bbox_2' + '\n')

def find_attr_val(attr, lines):
    for line in lines:
        if attr in line:
            val = line.split('=')[-1].replace(' ', '').replace('\n', '').replace(',', ' ; ')
            return val
    print('attribute ' + attr + ' was not found.. returning NULL')
    return 'NULL'

def find_score_val(score, lines):
    for line in lines:
        if score in line:
            val = line.split('=')[-1].replace(' ', '').replace('\n', '').replace(',', ' ; ')
            return val
    print('attribute ' + score + ' was not found.. returning NULL')
    return 'NULL'

def find_bbox_vals(fold):
    if os.path.isfile(fold + "/final_predictions.txt"):
        f = open(fold + "/final_predictions.txt", 'r')
        lines = f.readlines()
        f.close()
        bbox1 = lines[0].replace(' ', '').replace('\n', '')
        bbox2 = lines[1].replace(' ', '').replace('\n', '')
    else:
        return ['not ready', 'not ready']
    return [bbox1, bbox2]



def get_and_print_sim_data(sim, attr_list, csv_file):
    csv_file.write(sim[1] + ',' + str(datetime.datetime.fromtimestamp(sim[2]).date()))
    f = open(sim[0] + "\\configurations.txt", 'r')
    lines = f.readlines()
    f.close()
    for attr in attr_list:
        val = find_attr_val(attr, lines)
        csv_file.write(',' + val)
    if os.path.isfile(sim[0] + "\\final_scores.txt"):
        f = open(sim[0] + "\\final_scores.txt", 'r')
        lines = f.readlines()
        f.close()
        for score in scores_list:
            val = find_score_val(score, lines)
            csv_file.write(',' + val)
    else:
        for score in scores_list:
            csv_file.write(',not ready')
    vals = find_bbox_vals(sim[0])
    csv_file.write(',' + vals[0] + ',' + vals[1] + '\n')

if __name__ == "__main__":
    attr_list = get_attr_list()
    sorted_folder_and_date_list = get_sorted_folder_and_date_list()
    csv_file = open('./documentation/table.csv','w')
    print_first_line(attr_list, csv_file)
    for sim in sorted_folder_and_date_list:
        get_and_print_sim_data(sim, attr_list, csv_file)

    csv_file.close()
