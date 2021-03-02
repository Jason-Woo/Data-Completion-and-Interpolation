from knn import *
from regression import *
from decision_tree import *
from random_forest import *
from basic_completion import *
from math import sqrt

import argparse
import numpy as np
import csv

config = {}


def is_number(s):
    """
        Judge whether the given string can be converted into a number
        ----------
        Parameters
        s: string
        ----------
        Return
        whether s can be be converted into a number
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def cal_acc(label_real, label_predict, task):
    """
        Calculate accuracy / RMSE
        ----------
        Parameters
        label_real: Real label
        label_predict: Predicted label
        task: regression/classification
        ----------
        Return
        accuracy for classification tasks
        RMSE for regression tasks
    """
    if task == 'regression':
        return sqrt(np.square(np.subtract(label_real.astype(np.float64), label_predict.astype(np.float64))).mean())
    elif task == 'classification':
        cnt = 0
        for i in range(len(label_real)):
            if str(label_real[i]) == str(label_predict[i]):
                cnt += 1
        return cnt / len(label_real)


def get_feature_type(data, has_empty=False):
    """
        Get the type of all features
        ----------
        Parameters
        data: Dataset
        ----------
        Return
        List of feature types of each columns
    """
    global config
    feature_type = []
    for i in range(len(data[0])):
        curr_col = data[:, i].tolist()
        if has_empty:
            while '' in curr_col:
                curr_col.remove('')
        if is_number(curr_col[0]):
            tmp_col = list(map(float, curr_col))
            sample = random.sample(tmp_col, int(config['model-n']))
            if len(set(sample)) <= int(config['model-k']):
                feature_type.append('discrete_num')
            else:
                feature_type.append('continuous_num')
        else:
            feature_type.append('string')
    return np.array(feature_type)


def get_task_priority(tasks):
    """
        Get the executing sequence of the tasks
        ----------
        Parameters
        tasks: List of tasks
        ----------
        Return
        The executing sequence of the tasks
    """
    priority = []
    for i, task in enumerate(tasks):
        if task == 'discrete_num' or 'string':  # classification goes first
            priority = [i] + priority
        elif task == 'continuous_num':  # regression goes last
            priority = priority + [i]
    return priority


def normalization(data):
    """
        normalize every column of the data
        ----------
        Parameters
        Data: Dataset column
        ----------
        Return
        The normalized dataset
        Meta data for recovery
    """
    data_min = min(data)
    data_max = max(data)
    data_mean = np.mean(data)
    if data_max == data_min:
        return np.ones(len(data)), [data_max, 0]
    else:
        data = (data - data_mean) / (data_max - data_min)
        return data, [data_max - data_min, data_mean]


def normalization_reverse(data, info):
    """
        Recover normalized data to real data
        ----------
        Parameters
        Data: Dataset column
        info: Meta data for recovery
        ----------
        Return
        The real dataset
    """
    data_max_min = info[0]
    data_mean = info[1]
    data = data * data_max_min + data_mean
    return data


def one_hot_encoding(data):
    """
        One-hot encoding the feature
        ----------
        Parameters
        Data: Dataset column
        ----------
        Return
        The encoded columns
    """
    encoded_data = []
    value = list(set(data))
    value_cnt = len(value)
    for i in range(len(data)):
        data_tmp = np.zeros(value_cnt)
        for j in range(value_cnt):
            if data[i] == value[j]:
                data_tmp[j] = 1
                encoded_data.append(data_tmp)
                continue
    return encoded_data


def continuous_to_discrete(data, k):
    """
        Convert continuous feature to discrete feature
        ----------
        Parameters
        data: dataset
        ----------
        Return
        dataset after convert
    """
    feature_type = get_feature_type(data)
    for i in range(len(feature_type)):
        if feature_type[i] == 'continuous_num':
            min_val = min(data[:, i])
            max_val = max(data[:, i])
            interval = (max_val - min_val) / k
            for j in range(len(data)):
                if data[j][i] == max_val:
                    data[j][i] = k - 1
                else:
                    data[j][i] = (data[j][i] - min_val) // interval
    return data


def impute(data, cols_target, cols_using, classification_model, regression_model):
    """
        Impute all Missing Columns
        ----------
        Parameters
        data: dataset
        cols_target: list of target columns
        cols_using: list of columns we are using
        classification_model: model for classification tasks
        regression_model: model for regression tasks
        ----------
        Return
        imputed data set
    """
    cols_missing = []
    cnt = 0
    while cnt < len(cols_using):
        col = cols_using[cnt]
        if '' in data[:, col]:
            cols_missing.append(col)
            cols_using.remove(col)
        else:
            cnt += 1

    data_using = data[:, cols_using]
    print('--target columns: ', cols_target)
    print('--missing columns: ', cols_missing)
    print('--using columns: ', cols_using)

    var_type_using = get_feature_type(data_using)  # Get all features types

    # Preprocessing
    for i, f in enumerate(var_type_using):
        if f == 'string':  # One hot encoding if string
            encoded_data = one_hot_encoding(data_using[:, i])
            data_using = np.delete(data_using, i, axis=1)
            data_using = np.hstack((data_using, encoded_data))
            var_type_using = np.delete(var_type_using, i)
            var_type_using = np.hstack((var_type_using, np.array(['discrete_num'] * len(encoded_data[0]))))
    data_using = data_using.astype(np.float64)
    for i in range(len(data_using[0])):  # Normalizing all columns
        data_using[:, i], _ = normalization(data_using[:, i])

    # Handel the missing columns in columns we are using
    if cols_missing:
        var_type_missing = get_feature_type(data[:, cols_missing], has_empty=True)
        priority = get_task_priority(var_type_missing)
        for task in priority:
            curr_col = cols_missing[task]
            task_type = var_type_missing[task]
            print("---Processing column ", curr_col)
            label = data[:, curr_col]
            empty_rows = []

            for i, tmp_label in enumerate(label):
                if tmp_label == '':
                    empty_rows.append(i)

            data_testing = data_using[empty_rows, :]

            data_training = np.delete(data_using, empty_rows, axis=0)
            label_training = np.delete(label, empty_rows, axis=0)

            label_predict = impute_col(data_training, label_training, data_testing, task_type, classification_model, regression_model)
            for i, rows in enumerate(empty_rows):
                data[rows][curr_col] = label_predict[i]

            # Encoding/normalizing the new column
            if task_type == 'string':
                encoded_col = np.array(one_hot_encoding(data[:, curr_col]))
                data_using = np.hstack((data_using, encoded_col))

            elif task_type == 'discrete_num' or 'continuous_num':
                normalized_label, _ = normalization(data[:, curr_col].astype(np.float64))
                normalized_label = normalized_label[:, np.newaxis]
                data_using = np.hstack((data_using, normalized_label))
    var_type_target = get_feature_type(data[:, cols_target], has_empty=True)
    priority = get_task_priority(var_type_target)
    for task in priority:
        curr_col = cols_target[task]
        task_type = var_type_target[task]
        print("--Processing column ", curr_col)
        label = data[:, curr_col]
        empty_rows = []

        for i, tmp_label in enumerate(label):
            if tmp_label == '':
                empty_rows.append(i)

        data_testing = data_using[empty_rows, :]

        data_training = np.delete(data_using, empty_rows, axis=0)
        label_training = np.delete(label, empty_rows, axis=0)
        label_predict = impute_col(data_training, label_training, data_testing, task_type, classification_model, regression_model)
        for i, rows in enumerate(empty_rows):
            data[rows][curr_col] = label_predict[i]
    return data


def model_executor(training_data, training_label, testing_data, model_name, meta_data=None):
    """
        Executor ml models
        ----------
        Parameters
        training_data, training_label, testing_data: data for model training
        model_name: model we are using
        meta_data: Meta data for normalization recovery
        ----------
        Return
        Prediction of the ml model
    """
    print(' -Executing ', model_name)
    global config
    predict_label = None
    if model_name == 'knn':
        model = KNN(int(config['knn-k']))
        predict_label = model.train_and_predict(training_data, training_label, testing_data)
    elif model_name == 'decision_tree':
        model = DecisionTree(max_depth=int(config['dt-max_depth']), min_num=int(config['dt-min_num']))
        data_dis = continuous_to_discrete(np.vstack((training_data, testing_data)), int(config['dt-k']))
        data_train_dis = data_dis[:len(training_data)]
        data_test_dis = data_dis[len(training_data):]
        predict_label = model.train_and_predict(data_train_dis, training_label, data_test_dis)
    elif model_name == 'random_forest':
        data_dis = continuous_to_discrete(np.vstack((training_data, testing_data)), int(config['dt-k']))
        data_train_dis = data_dis[:len(training_data)]
        data_test_dis = data_dis[len(training_data):]
        model = RandomForest(int(config['rf-n']), forest_size=int(config['rf-forest_size']))
        predict_label = model.train_and_predict(data_train_dis, training_label, data_test_dis)
    elif model_name == 'basic_completion':
        model = BasicCompletion('discrete')
        predict_label = model.predict(training_label, testing_data)
    elif model_name == 'naive_regression':
        model = Regression(method='naive')
        model.train(training_data, training_label)
        predict_label_normalized = model.predict(testing_data)
        predict_label = normalization_reverse(predict_label_normalized, meta_data)
    elif model_name == 'ridge_regression':
        model = Regression(method='ridge', lmb=float(config['ridge-lambda']))
        model.train(training_data, training_label)
        predict_label_normalized = model.predict(testing_data)
        predict_label = normalization_reverse(predict_label_normalized, meta_data)
    elif model_name == 'lasso_regression':
        model = Regression(method='lasso', lmb=float(config['lasso-lambda']))
        model.train(training_data, training_label)
        predict_label_normalized = model.predict(testing_data)
        predict_label = normalization_reverse(predict_label_normalized, meta_data)
    elif model_name == 'basic_completion':
        model = BasicCompletion('continuous')
        predict_label = model.predict(training_label, testing_data)
    return predict_label


def find_best_model(training_data, training_label, col_type):
    """
        Find best model for current column
        ----------
        Parameters
        training_data, training_label: data for model training
        col_type: type of the feature
        ----------
        Return
        Best model for current column
    """
    training_label = training_label[:, np.newaxis]
    full_data = np.hstack((training_data, training_label))

    np.random.shuffle(full_data)
    training_size = int(0.8 * len(full_data))
    data_training = full_data[:training_size, :-1].astype(np.float64)  # split training set
    label_training = full_data[:training_size, -1].transpose()
    data_testing = full_data[training_size:, :-1].astype(np.float64)  # split validation set
    label_testing = full_data[training_size:, -1].transpose()
    best_model = None

    if col_type == 'discrete_num' or col_type == 'string':
        classification_model_list = ['knn', 'decision_tree', 'random_forest', 'basic_completion']
        best_acc = 0.0
        for model in classification_model_list:
            predict_label = model_executor(data_training, label_training, data_testing, model)
            curr_acc = cal_acc(label_testing, predict_label, 'classification')
            if curr_acc > best_acc:
                best_acc = curr_acc
                best_model = model
            print('    curr_acc: ', curr_acc, 'best_acc: ', best_acc, 'best_model: ', best_model)

    elif col_type == 'continuous_num':
        regression_model_list = ['naive_regression', 'ridge_regression', 'lasso_regression', 'basic_completion']
        best_mse = 99999999
        for model in regression_model_list:
            if model == 'basic_completion':
                predict_label = model_executor(data_training, label_training, data_testing, model)
            else:
                training_label_normalized, meta_data = normalization(label_training.astype(np.float64))
                predict_label = model_executor(data_training, training_label_normalized, data_testing, model,
                                               meta_data)
            curr_mse = cal_acc(label_testing, predict_label, 'regression')
            if curr_mse < best_mse:
                best_mse = curr_mse
                best_model = model
            print('    curr_mse: ', curr_mse, 'best_mse: ', best_mse, 'best_model: ', best_model)
    return best_model


def impute_col(training_data, training_label, testing_data, col_type, classification_model, regression_model):
    """
        Impute target column with specified model
        ----------
        Parameters
        training_data, training_label, testing_data: data for model training
        col_type: type of the feature
        classification_model: model for classification tasks
        regression_model: model for regression tasks
        ----------
        Return
        Prediction for current column
    """
    # Make training and testing dataset
    training_label = training_label.transpose()

    predict_label = None
    if col_type == 'discrete_num' or col_type == 'string':
        if classification_model == 'best_model':
            classification_model = find_best_model(training_data, training_label, col_type)
        predict_label = model_executor(training_data, training_label, testing_data, classification_model)
    elif col_type == 'continuous_num':
        if regression_model == 'best_model':
            regression_model = find_best_model(training_data, training_label, col_type)
        if regression_model == 'basic_completion':
            predict_label = model_executor(training_data, training_label, testing_data, regression_model)
        else:
            training_label_normalized, meta_data = normalization(training_label.astype(np.float64))
            predict_label = model_executor(training_data, training_label_normalized, testing_data, regression_model, meta_data)
    return predict_label


def input_handling(file_path, header):
    """
        Reading dataset and configuration file
        ----------
        Parameters
        file_path: path of dataset
        header: whether dataset has header
        ----------
        Return
        Dataset stored in array and header line
    """
    data_header = None
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = [row for row in reader]
    if header:
        data_header = np.array(rows[0])
        data_full = np.array(rows[1:])
    else:
        data_full = np.array(rows)

    with open('./configure.txt', 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split()
            config[tmp[0]] = tmp[1]
    return data_full, data_header


def output(data, header, data_header):
    """
        Write dataset to a file
        ----------
        Parameters
        data: path of dataset
        header: whether dataset has header
        data_header: header of dataset
    """
    if header:
        data = np.vstack((data_header, data))
    output_file = "generate.csv"
    np.savetxt(output_file, data, delimiter=",", fmt='%s')
    print("--Result is stored in ", output_file)


def meta_main(args):
    file_path = args.file_path
    header = args.header
    cols_tar = args.target
    cols_using = args.using
    classification_model = args.classification_model
    regression_model = args.regression_model

    data_full, data_header = input_handling(file_path, header)

    data_imputed = impute(data_full, cols_tar, cols_using, classification_model, regression_model)
    output(data_imputed, header, data_header)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='file path')
    parser.add_argument('--header', type=str, help='whether table has header', default=True)
    parser.add_argument('--target', nargs='+', type=int, help='list of target columns')
    parser.add_argument('--using', nargs='+', type=int, help='list of columns using')
    parser.add_argument('--classification_model', type=str, help='classification model', default='basic_completion')
    parser.add_argument('--regression_model', type=str, help='regression model', default='basic_completion')
    args = parser.parse_args()
    meta_main(args)

