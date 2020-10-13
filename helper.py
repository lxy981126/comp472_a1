import numpy as np
import sklearn.metrics as metrics


def output_computed_metrics(test_data, test_prediction, output_file):
    confusion_matrix = metrics.confusion_matrix(test_data[:, -1], test_prediction)
    precision = metrics.precision_score(test_data[:, -1], test_prediction, average=None, zero_division=1)
    recall = metrics.recall_score(test_data[:, -1], test_prediction, average=None)
    f1 = metrics.f1_score(test_data[:, -1], test_prediction, average=None)
    accuracy = metrics.accuracy_score(test_data[:, -1], test_prediction)
    macro_f1 = metrics.f1_score(test_data[:, -1], test_prediction, average='macro')
    weighted_f1 = metrics.f1_score(test_data[:, -1], test_prediction, average='weighted')

    index_column = np.arange(len(test_prediction))
    prediction_with_index = np.vstack((index_column, test_prediction)).transpose()

    file = open(output_file, 'w')
    np.savetxt(file, prediction_with_index, delimiter=',', fmt="%i")

    file.write('Confusion matrix\n')
    np.savetxt(file, confusion_matrix, delimiter=',', fmt='%i')
    file.write('Precision\n')
    np.savetxt(file, precision, delimiter=',', fmt='%i')
    file.write('Recall\n')
    np.savetxt(file, recall, delimiter=',', fmt='%i')
    file.write('F1-measure\n')
    np.savetxt(file, f1, delimiter=',', fmt='%i')
    file.write('Accuracy: ' + str(accuracy) + '\n')
    file.write('Macro F1: ' + str(macro_f1) + '\n')
    file.write('Weighted F1: ' + str(weighted_f1) + '\n')

    file.close()
