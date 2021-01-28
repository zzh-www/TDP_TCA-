import pandas as pd
import numpy as np
import os
import datetime
import TCA
import TCA_plus
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression


# -------Auxiliary method--------start
def init_record_data():
    record_data = {'Training Set': [], 'Test Set': [], 'Model': [], 'Loop Size': [], 'accuracy_mean': [],
                   'accuracy_std': [], 'AUC_mean': [],
                   'AUC_std': [], 'F-measure_mean': [], 'F-measure_std': [], 'MCC_mean': [], 'MCC_std': []}
    return record_data


def insert_param(training, test, model_name, loop_size, repository_name, data):
    data['Training Set'].append(training)
    data['Test Set'].append(test)
    data['Model'].append(model_name)
    data['Loop Size'].append(str(loop_size))
    data['repository_name'] = repository_name


def insert_result(acc_m, acc_s, auc_m, auc_s, f1_m, f1_s, mcc_m, mcc_s, data):
    data['accuracy_mean'].append(round(acc_m, 3))
    data['accuracy_std'].append(round(acc_s, 3))
    data['AUC_mean'].append(round(auc_m, 3))
    data['AUC_std'].append(round(auc_s, 3))
    data['F-measure_mean'].append(round(f1_m, 3))
    data['F-measure_std'].append(round(f1_s, 3))
    data['MCC_mean'].append(round(mcc_m, 3))
    data['MCC_std'].append(round(mcc_s, 3))


def save_data(file_name, data):
    print("save_________________________")
    df = pd.DataFrame(data=data,
                      columns=['repository_name','Training Set', 'Test Set', 'Model', 'Loop Size', 'accuracy_mean', 'accuracy_std',
                               'AUC_mean', 'AUC_std', 'F-measure_mean', 'F-measure_std',
                               'MCC_mean', 'MCC_std'])

    save_path = 'result/' + file_name + '.csv'

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)


def extract_feature_labels(df, features, label):
    # row = df[df.file_name == file_name][features]
    # row = np.array(row).tolist()
    # row = np.squeeze(row)
    # row = list(row)
    # return row
    # def extract_label(df, file_name):
    #     row = df[df.file_name == file_name]['bug']
    #     row = np.array(row).tolist()
    #     if row[0] > 1:
    #         row[0] = 1
    #     return row
    hand_x_data = []
    label_data = []
    for i in df.index:
        # feature
        row_feature = df[i:i + 1][features]
        row_feature = np.array(row_feature).tolist()
        row_feature = np.squeeze(row_feature)
        row_feature = list(row_feature)
        # label
        row_label = df[i:i + 1][label]
        row_label = np.array(row_label).tolist()[0]
        if row_label[0] > 1:
            row_label[0] = 1
        # append
        hand_x_data.append(row_feature)
        label_data.append(row_label)
    return np.array(hand_x_data), np.array(label_data)

def train_and_save(
    c_train_x,
    c_train_y,
    c_test_x,
    test_label,
    record_data,
    model_name,
    repository_name,
    path,
    c_weights = None,):
    cls = LogisticRegression(class_weight='balanced');
    # 为了平衡样本，有些样本过于不均衡会出现预测结果全为0的情况
    cls.fit(c_train_x, c_train_y, sample_weight=c_weights)
    y_pred = cls.predict(c_test_x)

    acc = accuracy_score(y_true=test_label, y_pred=y_pred)
    auc = roc_auc_score(y_true=test_label, y_score=y_pred)
    f1 = f1_score(y_true=test_label, y_pred=y_pred)
    mcc = matthews_corrcoef(y_true=test_label, y_pred=y_pred)

    # Save the results in a file
    # record_data = init_record_data()
    insert_result(acc, 0, auc, 0, f1, 0, mcc, 0, data=record_data)
    insert_param(training=path[0], test=path[1], model_name=model_name, loop_size=1, repository_name=repository_name,data=record_data)

def Z_Score(train_hand_craft, test_hand_craft):
    nor_train_hand_craft = (train_hand_craft - np.mean(train_hand_craft, axis=0)) / np.std(train_hand_craft, axis=0)
    nor_test_hand_craft = (test_hand_craft - np.mean(test_hand_craft, axis=0)) / np.std(test_hand_craft, axis=0)
    return nor_train_hand_craft, nor_test_hand_craft

def fix_feature(arr,arr_f):
    arrs = []
    arr_fs = []
    for i in range(len(arr)):
        if arr[i].tolist() != np.zeros(len(arr[0])).tolist():
            arrs.append(arr[i])
            arr_fs.append(arr_f[i])
    arr = np.array(arrs)
    arr_f = np.array(arr_fs)
    return arr,arr_f
# some setting
# PROMISE
repository_name = 'PROMISE'
features = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
            'ic', 'cbm', 'amc', 'max_cc', 'avg_cc']
label = ['bug']
# 提供source和target组合的txt文件地址
s_t_text_path = '../../TCNN-master/TCNN-master/data/pairs-one.txt'
# csv文件存放文件夹地址
root_path_csv = '../../TCNN-master/TCNN-master/data/csvs/'

# AEEEM
repository_name = 'AEEEM'
features = ['ck_oo_cbo','ck_oo_numberOfLinesOfCode','numberOfBugsFoundUntil:','numberOfCriticalBugsFoundUntil:','CvsEntropy','CvsLogEntropy','LDHH_cbo','LDHH_numberOfLinesOfCode','WCHU_cbo','WCHU_numberOfLinesOfCode']
label = ['class']
# 提供source和target组合的txt文件地址
s_t_text_path = '../../TCNN-master/TCNN-master/data_csvs_AEEEM.txt'
# csv文件存放文件夹地址
root_path_csv = '../../TCNN-master/TCNN-master/data/csvs/AEEEM/'

# RelinK
repository_name = 'Relink'
features = ['AvgCyclomatic','AvgLine','CountLine','CountStmt','MaxCyclomatic','RatioCommentToCode','SumCyclomatic']
label = ['isDefective']
# 提供source和target组合的txt文件地址
s_t_text_path = '../../TCNN-master/TCNN-master/data_csvs_Relink.txt'
# csv文件存放文件夹地址
root_path_csv = '../../TCNN-master/TCNN-master/data/csvs/Relink/'

tr_methods = ['LR', 'TCA', 'TCA+']
# tr_methods = ['TCA']
result_file_name = 'TCA_PLUS_CDPD_balanced'

# start time
start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

# 获取源项目和目标项目数据
path_train_and_test = []
with open(s_t_text_path,'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        line = line.split(',')
        T_line = [line[1], line[0]]
        path_train_and_test.append(line)
        path_train_and_test.append(T_line)
print(path_train_and_test)
# 存放结果数据
record_data = init_record_data()
for path in path_train_and_test:
    path_train_csv = root_path_csv + path[0] + '.csv'
    path_test_csv = root_path_csv + path[1] + '.csv'
    train_project_name = path[0]
    test_project_name = path[1]
    dp = pd.DataFrame(pd.read_csv(path_train_csv))
    train_feature, train_label = extract_feature_labels(dp,features,label)
    train_feature,train_label = fix_feature(train_feature,train_label)
    dt = pd.DataFrame(pd.read_csv(path_test_csv))
    test_feature, test_label = extract_feature_labels(dt,features,label)
    test_feature,test_label = fix_feature(test_feature,test_label)
    print(dp)
    for model_name in tr_methods:
        train_hand_craft = train_feature
        train_hand_craft[train_hand_craft==0] = 0.000000000001
        test_hand_craft = test_feature
        test_hand_craft[test_hand_craft==0] = 0.000000000001
        nor_train_hand_craft, nor_test_hand_craft = Z_Score(train_hand_craft,test_hand_craft)
        # without transformation 没有迁移
        if model_name in ['LR']:
            train_and_save(
                c_train_x=train_feature,
                c_train_y=train_label,
                c_test_x=test_feature,
                test_label=test_label,
                record_data=record_data,
                model_name=model_name,
                repository_name=repository_name,
                path=path,)
            print('LR Done')
        elif model_name in ['TCA']:
            tca_plus = TCA_plus.TCA_plus(kernel_type='linear', dim=10, lamb=1, gamma=1)
            tca = TCA.TCA(kernel_type='linear', dim=10, lamb=1, gamma=1)
            # 无正则
            normalization_option = 'NoN'
            # new_Xs, new_Xt = tca_plus.get_normalization_result(train_hand_craft, test_hand_craft,
            #                                                    method_type=normalization_option)
            try:
                c_train_x, c_test_x = tca.fit(train_hand_craft, test_hand_craft)
                train_and_save(
                    c_train_x=c_train_x,
                    c_train_y=train_label,
                    c_test_x=c_test_x,
                    test_label=test_label,
                    record_data=record_data,
                    model_name=model_name + '_' + normalization_option,
                    repository_name=repository_name,
                    path=path, )
            except:
                # print(e)
                print("can not use "+normalization_option)
            # N1
            normalization_option = 'N1'
            new_Xs, new_Xt = tca_plus.get_normalization_result(train_hand_craft, test_hand_craft,
                                                               method_type=normalization_option)
            # train_hand_craft 经过N1后 生成 的 new_Xs 有些行 例如第29行里七个元素全为零 而后范数计算也为零从而导致了nan无法计算
            # 解决方法零元素用一个足够小的数去代替例如 0.000000001 或者是 所有元素除去零外绝对值中最小的十分之一?
            new_Xs[new_Xs == 0] = 0.000000000001
            new_Xt[new_Xt == 0] = 0.000000000001
            try:
                c_train_x, c_test_x = tca.fit(new_Xs, new_Xt)
                train_and_save(
                    c_train_x=c_train_x,
                    c_train_y=train_label,
                    c_test_x=c_test_x,
                    test_label=test_label,
                    record_data=record_data,
                    model_name=model_name + '_' + normalization_option,
                    repository_name=repository_name,
                    path=path, )
            except:
                print("can not use "+normalization_option)
            # N2
            normalization_option = 'N2'
            new_Xs, new_Xt = tca_plus.get_normalization_result(train_hand_craft, test_hand_craft,
                                                               method_type=normalization_option)
            # 若果某个实例特征值全为0，经过N2会出现nan，因为mean=0，直接去除或者全部取0.0000000001
            new_Xs[new_Xs == 0] = 0.000000000001
            new_Xt[new_Xt == 0] = 0.000000000001
            try:
                c_train_x, c_test_x = tca.fit(new_Xs, new_Xt)
                train_and_save(
                    c_train_x=c_train_x,
                    c_train_y=train_label,
                    c_test_x=c_test_x,
                    test_label=test_label,
                    record_data=record_data,
                    model_name=model_name + '_' + normalization_option,
                    repository_name=repository_name,
                    path=path, )
            except:
                print("can not use "+normalization_option)
            # N3
            normalization_option = 'N3'
            new_Xs, new_Xt = tca_plus.get_normalization_result(train_hand_craft, test_hand_craft,
                                                               method_type=normalization_option)
            new_Xs[new_Xs == 0] = 0.000000000001
            new_Xt[new_Xt == 0] = 0.000000000001
            try:
                c_train_x, c_test_x = tca.fit(new_Xs, new_Xt)
                train_and_save(
                    c_train_x=c_train_x,
                    c_train_y=train_label,
                    c_test_x=c_test_x,
                    test_label=test_label,
                    record_data=record_data,
                    model_name=model_name + '_' + normalization_option,
                    repository_name=repository_name,
                    path=path, )
            except:
                print("can not use "+normalization_option)
            # N4
            normalization_option = 'N4'
            new_Xs, new_Xt = tca_plus.get_normalization_result(train_hand_craft, test_hand_craft,
                                                               method_type=normalization_option)
            new_Xs[new_Xs == 0] = 0.000000000001
            new_Xt[new_Xt == 0] = 0.000000000001
            try:
                c_train_x, c_test_x = tca.fit(new_Xs, new_Xt)
                train_and_save(
                    c_train_x=c_train_x,
                    c_train_y=train_label,
                    c_test_x=c_test_x,
                    test_label=test_label,
                    record_data=record_data,
                    model_name=model_name + '_' + normalization_option,
                    repository_name=repository_name,
                    path=path, )
            except:
                print("can not use "+normalization_option)
        elif model_name in ['TCA+']:
            tca_plus = TCA_plus.TCA_plus(kernel_type='linear', dim=10, lamb=1, gamma=1)
            DCV_s = tca_plus.get_characteristic_vector(train_hand_craft)
            DCV_t = tca_plus.get_characteristic_vector(test_hand_craft)
            normalization_option = tca_plus.select_normalization_method(DCV_s, DCV_t)
            new_Xs, new_Xt = tca_plus.get_normalization_result(train_hand_craft, test_hand_craft,
                                                               method_type=normalization_option)
            new_Xs[new_Xs == 0] = 0.000000000001
            new_Xt[new_Xt == 0] = 0.000000000001
            c_train_x, c_test_x = tca_plus.fit(new_Xs, new_Xt)
            train_and_save(
                c_train_x=c_train_x,
                c_train_y=train_label,
                c_test_x=c_test_x,
                test_label=test_label,
                record_data=record_data,
                model_name=model_name+'_'+normalization_option,
                repository_name=repository_name,
                path=path, )
            print('TCA+ Done')



save_data(file_name=result_file_name, data=record_data)

