import numpy as np
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB


def gaussian_naive_bayes(training, validation, test):
    training_data = np.loadtxt(training, delimiter=',')
    validation_data = np.loadtxt(validation, delimiter=',')
    test_data = np.loadtxt(test, delimiter=',')

    gnb = GaussianNB()
    gnb.fit(training_data[:, :-1], training_data[:, -1])
    gnb.fit(validation_data[:, :-1], validation_data[:, -1])

    test_prediction = gnb.predict(test_data[:, :-1]).astype(int)

    index_column = np.arange(len(test_prediction))
    prediction_with_index = np.vstack((index_column, test_prediction)).transpose()
    np.savetxt("./Assig1-Dataset/GNB-DS1.csv", prediction_with_index, delimiter=',', fmt="%i")

    confusion_matrix = metrics.confusion_matrix(test_data[:, -1], test_prediction)
    print(confusion_matrix)

    precision = metrics.precision_score(test_data[:, -1], test_prediction, average=None, zero_division=1)
    print(precision)
    recall = metrics.recall_score(test_data[:, -1], test_prediction, average=None)
    print(recall)
    f1 = metrics.f1_score(test_data[:, -1], test_prediction, average=None)
    print(f1)

    accuracy = metrics.accuracy_score(test_data[:, -1], test_prediction)
    print(accuracy)
    macro_f1 = metrics.f1_score(test_data[:, -1], test_prediction, average='macro')
    print(macro_f1)
    weighted_f1 = metrics.f1_score(test_data[:, -1], test_prediction, average='weighted')
    print(weighted_f1)


if __name__ == '__main__':
    training_path = "./Assig1-Dataset/train_1.csv"
    validation_path = "./Assig1-Dataset/val_1.csv"
    test_path = "./Assig1-Dataset/test_with_label_1.csv"
    gaussian_naive_bayes(training_path, validation_path, test_path)

