import helper
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def best_decision_tree(training, validation, test, output):
    training_data = np.loadtxt(training, delimiter=',')
    validation_data = np.loadtxt(validation, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    dt = DecisionTreeClassifier()
    dt.fit(training_data[:, :-1], training_data[:, -1])

    params = {'criterion': ['gini', 'entropy'],
              'max_depth': [10, None],
              'min_samples_split': [0.01, 0.1, 0.5, 0.9, 1.0],
              'min_impurity_decrease': [0.0, 0.01, 0.1, 0.5, 0.9, 1.0],
              'class_weight': [None, 'balanced']}
    tuned_model = GridSearchCV(estimator=dt, param_grid=params)
    tuned_model.fit(validation_data[:, :-1], validation_data[:, -1])

    test_prediction = tuned_model.best_estimator_.predict(test_data[:, :-1])
    helper.output_computed_metrics(test_data, test_prediction, output)


if __name__ == '__main__':
    training_path = input('Enter path of training data') #"./Assig1-Dataset/train_1.csv"
    validation_path = input('Enter path of validation data') #"./Assig1-Dataset/val_1.csv"
    test_path = input('Enter path of test data') #"./Assig1-Dataset/test_with_label_1.csv"
    output_path = input('Enter path of output csv') #"./Output/Best-DT-DS1.csv"
    best_decision_tree(training_path, validation_path, test_path, output_path)
