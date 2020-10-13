import helper
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def base_decision_tree(training, test, output):
    training_data = np.loadtxt(training, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    dt = DecisionTreeClassifier()
    dt.fit(training_data[:, :-1], training_data[:, -1])
    test_prediction = dt.predict(test_data[:, :-1])
    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = "./Assig1-Dataset/train_2.csv"#input('Enter path of training data')
    test_path = "./Assig1-Dataset/test_with_label_2.csv"#input('Enter path of test data')
    output_path = "./Assig1-Dataset/Base-DT-DS2.csv"#input('Enter path of output csv')
    base_decision_tree(training_path, test_path, output_path)
