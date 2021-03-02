from decision_tree import *
import random


class RandomForest:
    def __init__(self, num_feature, forest_size=40):
        """
            num_feature: number of features for each tree
            forest_size: forest size
        """
        self.num_feature = num_feature
        self.forest_size = forest_size

    def sample_data(self, data_train, label, data_test):
        """
            Sample dataset for training and testing
            ----------
            Parameters
            data_train: Original training data
            label:  Original label
            data_test: Original testing data
            ----------
            Return
            Sampled training data
            Sampled label
            Sampled testing data
        """
        m, n = data_train.shape
        label_list = [i for i in range(n)]

        # pick random features
        sample_feature = sorted(random.sample(label_list, self.num_feature))

        data_train_sample = np.zeros((m, self.num_feature))
        label_sample = []
        data_test_sample = np.zeros((len(data_test), self.num_feature))
        for i in range(m):
            # pick m random rows
            j = np.random.randint(0, m)
            for k, feature in enumerate(sample_feature):
                # pick feature
                data_train_sample[i][k] = data_train[j][feature]
            label_sample.append(label[j])
        for i in range(len(data_test)):
            for k, feature in enumerate(sample_feature):
                data_test_sample[i][k] = data_test[i][feature]
        return data_train_sample, np.array(label_sample), data_test_sample

    def train_and_predict(self, data_train, label, data_test):
        predictions_list = []
        for _ in range(len(data_test)):
            predictions_list.append({})
        for t in range(self.forest_size):
            print("\r    Tree {a}/{b}".format(a=t+1, b=self.forest_size), end="")
            training_data, training_label, testing_data = self.sample_data(data_train, label, data_test)
            model = DecisionTree()
            label_predict = model.train_and_predict(training_data, training_label, testing_data)
            for i in range(len(data_test)):
                predictions_list[i][label_predict[i]] = predictions_list[i].get(label_predict[i], 0) + 1
        label_final = []

        for i in range(len(data_test)):
            label_final.append(max(predictions_list[i].items(), key=lambda x: x[1])[0])
        print('\n', end='')
        return np.array(label_final)
