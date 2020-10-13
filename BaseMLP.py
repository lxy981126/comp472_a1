import helper
import numpy as np
from sklearn.neural_network import MLPClassifier


def base_mlp(training, test, output):
    training_data = np.loadtxt(training, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    mlp = MLPClassifier(hidden_layer_sizes=(1, 100), activation='logistic', solver='sgd')
    mlp.fit(training_data[:, :-1], training_data[:, -1])
    test_prediction = mlp.predict(test_data[:, :-1])

    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = input('Enter path of training data')  # "./Assig1-Dataset/train_2.csv"
    test_path = input('Enter path of test data')  # "./Assig1-Dataset/test_with_label_2.csv"
    output_path = input('Enter path of output csv')  # "./Output/Base-MLP-DS2.csv"
    base_mlp(training_path, test_path, output_path)
