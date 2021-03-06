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
              'hidden_layer_sizes': [(30, 50), (10, 10, 10)],
              'solver': ['adam', 'sgd']}
    tuned_model = GridSearchCV(estimator=mlp, param_grid=params, n_jobs=-1, cv=3)
    tuned_model.fit(validation_data[:, :-1], validation_data[:, -1])
    print(tuned_model.best_params_)

    test_prediction = tuned_model.best_estimator_.predict(test_data[:, :-1])
    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = input('Enter path of training data')  # "./Assig1-Dataset/train_1.csv"
    validation_path = input('Enter path of validation data')  # "./Assig1-Dataset/val_1.csv"
    test_path = input('Enter path of test data')  # "./Assig1-Dataset/test_with_label_1.csv"
    output_path = input('Enter path of output csv')  # "./Output/Best-MLS-DS1.csv"
    best_mlp(training_path, validation_path, test_path, output_path)
