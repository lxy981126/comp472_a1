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
    training_path = input('Enter path of training data')  # "./Assig1-Dataset/train_2.csv"
    test_path = input('Enter path of test data')  # "./Assig1-Dataset/test_with_label_2.csv"
    output_path = input('Enter path of output csv')  # "./Assig1-Dataset/Perceptron-DS2.csv"
    perceptron(training_path, test_path, output_path)
