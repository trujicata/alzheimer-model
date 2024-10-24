# %%
from research import start  # noqa
import h5py
import matplotlib.pyplot as plt
import numpy as np

# %%
classes = ["AD", "MCI", "CN"]
data_path = "data/P01"
val_file = h5py.File(f"{data_path}/test_csv.hdf5", "r")
test_file = h5py.File(f"{data_path}/pre_pet_diag.hdf5", "r")
train_file = h5py.File(f"{data_path}/train_csv.hdf5", "r")
# %%
# Check the labels of the training set
train_labels = train_file["y"]
label_count_train = {0: 0, 1: 0, 2: 0}
for label in train_labels:
    label_count_train[label] += 1
print(label_count_train)

# %%
# Check the labels of the validation set
val_labels = val_file["y"]
label_count_val = {0: 0, 1: 0, 2: 0}
for label in val_labels:
    label_count_val[label] += 1
print(label_count_val)
# %%
# Check the labels of the test set
test_labels = test_file["y"]
label_count_test = {0: 0, 1: 0, 2: 0}
for label in test_labels:
    label_count_test[label] += 1
label_count_test


# %%
def metrics_from_confusion_matrix(confusion_matrix):
    precision = []
    recall = []
    f1_score = []
    for i in range(3):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix.sum(axis=0)[i] - tp
        fn = confusion_matrix.sum(axis=1)[i] - tp
        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        f1_score.append(
            2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            if (precision[i] + recall[i]) > 0
            else 0
        )
    accuracy_mci_cn = confusion_matrix[1, 1] / (
        confusion_matrix[1, 1] + confusion_matrix[1, 2]
    )
    accuracy_ad_cn = confusion_matrix[0, 0] / (
        confusion_matrix[0, 0] + confusion_matrix[0, 2]
    )

    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1_score),
        accuracy_mci_cn,
        accuracy_ad_cn,
    )


# %%
# If all the predictions were 0 (AD), what would the precision, recall and f1 score be?
confusion_matrix_val = np.array(
    [
        [label_count_val[0], label_count_val[1], label_count_val[2]],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
precision_val, recall_val, f1_score_val, acc_mci_cn_val, acc_ad_cn_val = (
    metrics_from_confusion_matrix(confusion_matrix_val)
)
print(
    f"Validation\nPrecision: {precision_val}\nRecall: {recall_val}\nF1 Score: {f1_score_val}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_val}\nAccuracy AD/CN: {acc_ad_cn_val}\n\n")


confusion_matrix_test = np.array(
    [
        [label_count_test[0], label_count_test[1], label_count_test[2]],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
precision_test, recall_test, f1_score_test, acc_mci_cn_test, acc_ad_cn_test = (
    metrics_from_confusion_matrix(confusion_matrix_test)
)
print(
    f"Test\nPrecision: {precision_test}\nRecall: {recall_test}\nF1 Score: {f1_score_test}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_test}\nAccuracy AD/CN: {acc_ad_cn_test}\n\n")
# %%
# If all the predictions were 1 (MCI), what would the precision, recall and f1 score be?

confusion_matrix_val = np.array(
    [
        [0, 0, 0],
        [label_count_val[0], label_count_val[1], label_count_val[2]],
        [0, 0, 0],
    ]
)

precision_val, recall_val, f1_score_val, acc_mci_cn_val, acc_ad_cn_val = (
    metrics_from_confusion_matrix(confusion_matrix_val)
)
print(
    f"Validation\nPrecision: {precision_val}\nRecall: {recall_val}\nF1 Score: {f1_score_val}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_val}\nAccuracy AD/CN: {acc_ad_cn_val}\n\n")


confusion_matrix_test = np.array(
    [
        [0, 0, 0],
        [label_count_test[0], label_count_test[1], label_count_test[2]],
        [0, 0, 0],
    ]
)

precision_test, recall_test, f1_score_test, acc_mci_cn_test, acc_ad_cn_test = (
    metrics_from_confusion_matrix(confusion_matrix_test)
)
print(
    f"Test\nPrecision: {precision_test}\nRecall: {recall_test}\nF1 Score: {f1_score_test}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_test}\nAccuracy AD/CN: {acc_ad_cn_test}\n\n")


# %%
# If all the predictions were 2 (CN), what would the precision, recall and f1 score be?

confusion_matrix_val = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [label_count_val[0], label_count_val[1], label_count_val[2]],
    ]
)

precision_val, recall_val, f1_score_val, acc_mci_cn_val, acc_ad_cn_val = (
    metrics_from_confusion_matrix(confusion_matrix_val)
)
print(
    f"Validation\nPrecision: {precision_val}\nRecall: {recall_val}\nF1 Score: {f1_score_val}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_val}\nAccuracy AD/CN: {acc_ad_cn_val}\n\n")


confusion_matrix_test = np.array(
    [
        [0, 0, 0],
        [0, 0, 0],
        [label_count_test[0], label_count_test[1], label_count_test[2]],
    ]
)

precision_test, recall_test, f1_score_test, acc_mci_cn_test, acc_ad_cn_test = (
    metrics_from_confusion_matrix(confusion_matrix_test)
)
print(
    f"Test\nPrecision: {precision_test}\nRecall: {recall_test}\nF1 Score: {f1_score_test}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_test}\nAccuracy AD/CN: {acc_ad_cn_test}\n\n")


# %%
# Si las predicciones fueran aleatorias, ¿cuáles serían las métricas?

total_val = sum(label_count_val.values())
total_test = sum(label_count_test.values())

confusion_matrix_val = np.array(
    [
        [
            label_count_val[0] / total_val,
            label_count_val[1] / total_val,
            label_count_val[2] / total_val,
        ],
        [
            label_count_val[0] / total_val,
            label_count_val[1] / total_val,
            label_count_val[2] / total_val,
        ],
        [
            label_count_val[0] / total_val,
            label_count_val[1] / total_val,
            label_count_val[2] / total_val,
        ],
    ]
)

precision_val, recall_val, f1_score_val, acc_mci_cn_val, acc_ad_cn_val = (
    metrics_from_confusion_matrix(confusion_matrix_val)
)
print(
    f"Validation\nPrecision: {precision_val}\nRecall: {recall_val}\nF1 Score: {f1_score_val}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_val}\nAccuracy AD/CN: {acc_ad_cn_val}\n\n")


confusion_matrix_test = np.array(
    [
        [
            label_count_test[0] / total_test,
            label_count_test[1] / total_test,
            label_count_test[2] / total_test,
        ],
        [
            label_count_test[0] / total_test,
            label_count_test[1] / total_test,
            label_count_test[2] / total_test,
        ],
        [
            label_count_test[0] / total_test,
            label_count_test[1] / total_test,
            label_count_test[2] / total_test,
        ],
    ]
)

precision_test, recall_test, f1_score_test, acc_mci_cn_test, acc_ad_cn_test = (
    metrics_from_confusion_matrix(confusion_matrix_test)
)
print(
    f"Test\nPrecision: {precision_test}\nRecall: {recall_test}\nF1 Score: {f1_score_test}\n\n"
)
print(f"Accuracy MCI/CN: {acc_mci_cn_test}\nAccuracy AD/CN: {acc_ad_cn_test}\n\n")

# %%
