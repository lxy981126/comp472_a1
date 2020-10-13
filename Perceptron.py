import helper
import numpy as np
from sklearn.linear_model import Perceptron


def perceptron(training, test, output):
    training_data = np.loadtxt(training, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    per = Perceptron()
    per.fit(training_data[:, :-1], training_data[:, -1])
    test_prediction = per.predict(test_data[:, :-1])
    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = "./Assig1-Dataset/train_1.csv"#input('Enter path of training data')
    test_path = "./Assig1-Dataset/test_with_label_1.csv"#input('Enter path of test data')
    output_path = "./Output/PER-DS1.csv"#input('Enter path of output csv')
    perceptron(training_path, test_path, output_path)
