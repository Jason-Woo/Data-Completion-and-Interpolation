import numpy as np


class BasicCompletion:
    def __init__(self, type):
        self.type = type

    def predict(self, training_label, testing_data):
        if self.type == 'discrete':
            # mode of the fetaure
            label_cnt = {}
            for label in training_label:
                label_cnt[label] = label_cnt.get(label, 0) + 1
            mode = max(label_cnt.items(), key=lambda x: x[1])[0]
            predict_label = [mode] * len(testing_data)
            return np.array(predict_label)
        elif self.type == 'continuous':
            # mean of the feature
            mean_val = training_label.astype(np.float64).mean()
            predict_label = [mean_val] * len(testing_data)
            return np.array(predict_label)
