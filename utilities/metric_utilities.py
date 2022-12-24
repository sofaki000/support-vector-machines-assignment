from sklearn import metrics

from utilities.plot_utilities import plot_confusion_matrix


def get_metrics(confusion_file_name, actual, predicted):
    plot_confusion_matrix(confusion_file_name, actual, predicted)

    Accuracy = metrics.accuracy_score(actual, predicted)
    Precision = metrics.precision_score(actual, predicted)
    Sensitivity_recall = metrics.recall_score(actual, predicted, pos_label=1)
    Specificity = metrics.recall_score(actual, predicted, pos_label=-1)
    F1_score = metrics.f1_score(actual, predicted)

    metric_content = f"Accuracy={Accuracy:.2f}, Precision={Precision:.2f}, Sensitivity recall={Sensitivity_recall:.2f}, Specificity={Specificity:.2f},F1_score={F1_score:.2f}"
    print(metric_content)
    return metric_content