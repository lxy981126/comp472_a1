import helper
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def best_mlp(training, validation, test, output):
    training_data = np.loadtxt(training, delimiter=',')
    validation_data = np.loadtxt(validation, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    mlp = MLPClassifier()
    mlp.fit(training_data[:, :-1], training_data[:, -1])

    params = {'activation': ['logistic', 'tanh', 'relu', 'identity'],
              'hidden_layer_sizes': [(20, 30, 50), (30, 10, 10)],
              'solver': ['adam', 'sgd']}
    tuned_model = GridSearchCV(estimator=mlp, param_grid=params)
    tuned_model.fit(validation_data[:, :-1], validation_data[:, -1])

    test_prediction = tuned_model.best_estimator_.predict(test_data[:, :-1])
    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = "./Assig1-Dataset/train_1.csv"#input('Enter path of training data')
    validation_path = "./Assig1-Dataset/val_1.csv"#input('Enter path of validation data')
    test_path = "./Assig1-Dataset/test_with_label_1.csv"#input('Enter path of test data')
    output_path = "./Output/Best-MLP-DS1.csv"#input('Enter path of output csv')
    best_mlp(training_path, validation_path, test_path, output_path)
