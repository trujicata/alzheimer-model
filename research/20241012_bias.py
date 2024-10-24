# %%
import research.start
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

# %%
csv_path = "data/results/best/densenet_P03_0.5435_0.5676_test_results.csv"
df = pd.read_csv(csv_path)
df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()
# %%
classes = ["AD", "MCI", "CN"]
# %%
# First, let's calculate the f1 score and recall for each class


def get_metrics_by_class(df):
    f1_scores = []
    recall_scores = []
    precision_scores = []
    accuracy_scores = []

    for class_ in classes:
        y_true = df["Label"].apply(lambda x: classes[int(x)])
        y_pred = df["Prediction"].apply(lambda x: classes[int(x)])

        y_true_class = y_true[y_true == class_]
        y_pred_class = y_pred[y_true == class_]

        f1 = f1_score(y_true_class, y_pred_class, average="weighted")
        recall = recall_score(y_true_class, y_pred_class, average="weighted")
        precision = precision_score(y_true_class, y_pred_class, average="weighted")
        accuracy = accuracy_score(y_true_class, y_pred_class)

        f1_scores.append(f1)
        recall_scores.append(recall)
        precision_scores.append(precision)
        accuracy_scores.append(accuracy)

    print("AD")
    print(f"F1: {f1_scores[0]}")
    print(f"Recall: {recall_scores[0]}")
    print(f"Precision: {precision_scores[0]}")
    print(f"Accuracy: {accuracy_scores[0]}")

    print("\nMCI")
    print(f"F1: {f1_scores[1]}")
    print(f"Recall: {recall_scores[1]}")
    print(f"Precision: {precision_scores[1]}")
    print(f"Accuracy: {accuracy_scores[1]}")

    print("\nCN")
    print(f"F1: {f1_scores[2]}")
    print(f"Recall: {recall_scores[2]}")
    print(f"Precision: {precision_scores[2]}")
    print(f"Accuracy: {accuracy_scores[2]}")

    return f1_scores, recall_scores, precision_scores, accuracy_scores


get_metrics_by_class(df)

# %%
# Global metrics


def get_global_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)

    print(f"F1: {f1}")
    print(f"Recall: {recall}")
    # print(f"Precision: {precision}")
    # print(f"Accuracy: {accuracy}")

    return f1, recall, precision, accuracy


def get_alzheimer_det_metrics(df):
    predictions_for_ad = df[df["Label"] == 0]["Prediction"]

    true_positives = predictions_for_ad[predictions_for_ad == 0].count()
    predicted_as_cn = predictions_for_ad[predictions_for_ad == 2].count()

    ad_cn_accuracy = true_positives / (true_positives + predicted_as_cn)

    # Check the accuracy for MCI/CN
    predictions_for_mci = df[df["Label"] == 1]["Prediction"]

    true_positives = predictions_for_mci[predictions_for_mci == 1].count()
    predicted_as_cn = predictions_for_mci[predictions_for_mci == 2].count()

    mci_cn_accuracy = true_positives / (true_positives + predicted_as_cn)

    print("AD/CN accuracy:", ad_cn_accuracy)
    print("MCI/CN accuracy:", mci_cn_accuracy)

    return ad_cn_accuracy, mci_cn_accuracy


def get_metrics(df):
    y_true = df["Label"]
    y_pred = df["Prediction"]

    f1, recall, precision, accuracy = get_global_metrics(y_true, y_pred)
    ad_cn_accuracy, mci_cn_accuracy = get_alzheimer_det_metrics(df)

    support = df["Label"].value_counts()
    # print("Support:")
    # print(support)

    return f1, recall, precision, accuracy, ad_cn_accuracy, mci_cn_accuracy


# %%
get_metrics(df)

# %%
# WOMEN
woman_df = df[df["Sex"] == 0]

get_metrics(woman_df)
# %%
# MEN
men_df = df[df["Sex"] == 1]

get_metrics(men_df)
# %%
# Get metrics by age range
ranges = [(50, 60), (60, 70), (70, 80), (80, 90)]
for range_ in ranges:
    age_df = df[(df["Age"] >= range_[0]) & (df["Age"] < range_[1])]
    print(f"Age range: {range_}")
    get_metrics(age_df)
    print("-----------------\n\n\n")
# %%
