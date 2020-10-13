import numpy as np
import helper
from sklearn.naive_bayes import GaussianNB


def gaussian_naive_bayes(training, test, output):
    training_data = np.loadtxt(training, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    gnb = GaussianNB()
    gnb.fit(training_data[:, :-1], training_data[:, -1])

    test_prediction = gnb.predict(test_data[:, :-1]).astype(int)
    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = input('Enter path of training data')#"./Assig1-Dataset/train_2.csv"
    test_path = input('Enter path of test data')#"./Assig1-Dataset/test_with_label_2.csv"
    output_path = input('Enter path of output csv')#"./Assig1-Dataset/GNB-DS2.csv"
    gaussian_naive_bayes(training_path, test_path, output_path)

