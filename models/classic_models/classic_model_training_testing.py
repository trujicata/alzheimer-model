#%%
import os
import pandas as pd
import numpy as np
import nibabel as nib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import h5py
from tqdm import tqdm

# %%
def mask_image(image, atlas, area_of_interest):
    """ Get masked image of area of interest, based on given atlas.

    Args:
        - image (numpy array): numpy array of the image. Must be registered and
        be the same dimensions of atlas.
        - atlas (numpy array): atlas where each pixel has a numeric code
        corresponding to the area assigned to that pixel.
        - area_of_interest (int): numeric code for the area of interest to be
        masked.

    Returns:
        - masked image as numpy array where values outside of area of interest
        are 0 and values inside the area remain untouched.

    """
    return np.where(atlas == area_of_interest, image, 0)


def measure_image(image, atlas, area_of_interest):
    """ Measure the mean intensity and standard deviation for the brain area of
    interest.

    Args:
        - image (numpy array): numpy array of the image. Must be registered and
        be the same dimensions of atlas.
        - atlas (numpy array): atlas where each pixel has a numeric code
        corresponding to the area assigned to that pixel.
        - area_of_interest (int): numeric code for the area of interest to be
        masked.

    Returns:
        - mean of intensity for all the pixels in the area (int)
        - standard deviation of intensity for all the pixels in the area (int)

    """
    masked_image = mask_image(image, atlas, area_of_interest)
    mean_intensity = np.mean(masked_image)
    std_deviation = np.std(masked_image)
    return mean_intensity, std_deviation


def process_h5(h5, atlas, areas_to_measure):
    """ Get measurements (intensity and std) for all brain areas specified for
    all images in the h5 file.

    Args:
        - h5 (h5 file): h5 file containing all registered images as numpy
        arrays.
        - atlas (numpy array): atlas where each pixel has a numeric code.
        - areas_to_measure (list): list of areas to measure according to
        numeric coding of atlas.

    Returns:
        - dataframe containing all the images as rows, first column is the
        label, other columns represent the measurements (intensity and std)
        for different brain areas specified.

    """
    result = pd.DataFrame({})
    for i in tqdm(range(0, h5["X_nii"].shape[0]), desc="Processing"):
        row = {}
        row["label"] = h5["y"][i]
        row["sex"] = h5["X_Sex"][i]
        row["age"] = h5["X_Age"][i]
        for area in areas_to_measure:
            intensity, std_dev = measure_image(h5["X_nii"][i], atlas, area)
            row[f"{area}_intensity"] = intensity
            row[f"{area}_std_dev"] = std_dev
        new_df = pd.DataFrame([row])
        result = pd.concat([result, new_df], ignore_index=True)
    return result


def process_folder(folder_path, metadata, atlas, areas_to_measure):
    result = pd.DataFrame({})
    if os.path.isdir(folder_path):
        # List all files in the directory
        file_list = os.listdir(folder_path)
        for image in file_list:
            if not image.startswith("._") and image.endswith(".nii"):
                file_path = os.path.join(folder_path, image)
                if os.path.isfile(file_path):  # Check if it's a regular file
                    try:
                        nib_image = nib.load(file_path)
                        image_np = np.array(nib_image.dataobj)
                        image_id, _ = os.path.splitext(image)
                        row = {}
                        data_image = metadata.loc[
                            metadata["Image Data ID"] == image_id,
                        ]
                        row["label"] = data_image["Group"].values[0]
                        row["age"] = data_image["Age"].values[0]
                        row["sex"] = data_image["Sex"].values[0]
                        for area in areas_to_measure:
                            intensity, std_dev = measure_image(image_np, atlas, area)
                            # Store intensity and std_dev in the row dictionary
                            row[f"{area}_intensity"] = intensity
                            row[f"{area}_std_dev"] = std_dev
                        # Append the row to the result DataFrame
                        new_df = pd.DataFrame([row])
                        result = pd.concat([result, new_df], ignore_index=True)
                    except nib.ImageFileError as e:
                        print(f"Error loading image file '{file_path}': {e}")
                else:
                    print(f"Ignoring non-regular file: {file_path}")
    return result


def svm_model(
    X_train, X_test, y_train, y_test, output_folder, cudim_pre_x, cudim_pre_y
):
    """ Train and test SVM model for the mean intensities and standard
    deviations dataframe of all images.

    Args:
        - X_train (dataframe): dataframe containing the training data.
        - X_test (dataframe): dataframe containing the test data.
        - y_train (dataframe): dataframe containing the training labels.
        - y_test (dataframe): dataframe containing the test labels.
        - output_folder (str): path to the folder where the confusion matrix
        will be saved.
        - cudim_pre_x (dataframe): dataframe containing the CUDIM data.
        - cudim_pre_y (dataframe): dataframe containing the CUDIM labels.

    Returns:
        - svm model parameters.
        - performance scores of the best model on the test set.
        Accuracy, precision, recall and f1 score.
        - cudim_pre_scores: performance scores of the best model on the CUDIM
        data. Accuracy, precision, recall and f1 score.

    """

    # Define the parameter grid for the SVM
    param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": [0.1, 1, 10]}
    # Create an SVM classifier
    svm = SVC()
    grid_search = GridSearchCV(
        estimator=svm, param_grid=param_grid, scoring="f1_macro", cv=5, refit="f1_macro"
    )

    # Fit the GridSearchCV to your training data
    grid_search.fit(X_train, y_train)
    # Get the best parameter settings
    best_params = grid_search.best_params_
    # Get the best SVM model
    best_svm = grid_search.best_estimator_
    # Use the best model for prediction
    predictions = best_svm.predict(X_test)
    # Evaluate the performance of the best model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")
    print(
        f"SVM = F1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}"
    )
    scores = {
        "f1": f1,
        "precision": precision,
        "accuracy": accuracy,
        "recall": accuracy,
    }
    cm = confusion_matrix(y_test, predictions, labels=best_svm.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_svm.classes_)
    disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_folder, f"svm_confusion_matrix_{timestamp}.png"))
    plt.show()
    # PRE-CUDIM
    cudim_pre_predictions = best_svm.predict(cudim_pre_x)

    cudim_pre_accuracy = accuracy_score(cudim_pre_y, cudim_pre_predictions)
    cudim_pre_precision = precision_score(
        cudim_pre_y, cudim_pre_predictions, average="macro"
    )
    cudim_pre_recall = recall_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    cudim_pre_f1 = f1_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    print(
        f"CUDIM_PRE: svm = F1: {cudim_pre_f1}, accuracy: {cudim_pre_accuracy},"
        f" precision: {cudim_pre_precision}, recall: {cudim_pre_recall}"
    )
    cudim_pre_scores = {
        "f1": cudim_pre_f1,
        "precision": cudim_pre_precision,
        "accuracy": cudim_pre_accuracy,
        "recall": cudim_pre_recall,
    }
    cudim_pre_cm = confusion_matrix(
        cudim_pre_y, cudim_pre_predictions, labels=best_svm.classes_
    )
    cudim_pre_disp = ConfusionMatrixDisplay(
        confusion_matrix=cudim_pre_cm, display_labels=best_svm.classes_
    )
    cudim_pre_disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(output_folder, f"svm_confusion_matrix_cudim_pre{timestamp}.png")
    )
    plt.show()
    return best_params, scores, cudim_pre_scores


def random_forest_model(
    X_train, X_test, y_train, y_test, output_folder, cudim_pre_x, cudim_pre_y
):
    """ Train and test Random Forest model for the mean intensities and
    standard deviations dataframe of all images.

    Args:
        - X_train (dataframe): dataframe containing the training data.
        - X_test (dataframe): dataframe containing the test data.
        - y_train (dataframe): dataframe containing the training labels.
        - y_test (dataframe): dataframe containing the test labels.
        - output_folder (str): path to the folder where the confusion matrix
        will be saved.
        - cudim_pre_x (dataframe): dataframe containing the CUDIM data.
        - cudim_pre_y (dataframe): dataframe containing the CUDIM labels.

    Returns:
        - random forest model parameters
        - performance scores of the best model on the test set.
        Accuracy, precision, recall and f1 score.
        - cudim_pre_scores: performance scores of the best model on the CUDIM
        data. Accuracy, precision, recall and f1 score.

    """

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, scoring="f1_macro", cv=5
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_rf = grid_search.best_estimator_
    predictions = best_rf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")
    print(
        f"Random Forest = F1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}"
    )
    scores = {"f1": f1, "precision": precision, "accuracy": accuracy, "recall": recall}
    cm = confusion_matrix(y_test, predictions, labels=best_rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
    disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(output_folder, f"random_forest_confusion_matrix_{timestamp}.png")
    )
    plt.show()
    # PRE-CUDIM
    cudim_pre_predictions = best_rf.predict(cudim_pre_x)

    cudim_pre_accuracy = accuracy_score(cudim_pre_y, cudim_pre_predictions)
    cudim_pre_precision = precision_score(
        cudim_pre_y, cudim_pre_predictions, average="macro"
    )
    cudim_pre_recall = recall_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    cudim_pre_f1 = f1_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    print(
        f"CUDIM_PRE: RF = F1: {cudim_pre_f1}, accuracy: {cudim_pre_accuracy}, precision: {cudim_pre_precision}, recall: {cudim_pre_recall}"
    )
    cudim_pre_scores = {
        "f1": cudim_pre_f1,
        "precision": cudim_pre_precision,
        "accuracy": cudim_pre_accuracy,
        "recall": cudim_pre_recall,
    }
    cudim_pre_cm = confusion_matrix(
        cudim_pre_y, cudim_pre_predictions, labels=best_rf.classes_
    )
    cudim_pre_disp = ConfusionMatrixDisplay(
        confusion_matrix=cudim_pre_cm, display_labels=best_rf.classes_
    )
    cudim_pre_disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(output_folder, f"RF_confusion_matrix_cudim_pre{timestamp}.png")
    )
    plt.show()

    return best_params, scores, cudim_pre_scores


def gradient_boosting_model(
    X_train, X_test, y_train, y_test, output_folder, cudim_pre_x, cudim_pre_y
):
    """ Train and test Gradient Boosting model for the mean intensities and
    standard deviations dataframe of all images.

    Args:
        - X_train (dataframe): dataframe containing the training data.
        - X_test (dataframe): dataframe containing the test data.
        - y_train (dataframe): dataframe containing the training labels.
        - y_test (dataframe): dataframe containing the test labels.
        - output_folder (str): path to the folder where the confusion matrix
        will be saved.
        - cudim_pre_x (dataframe): dataframe containing the CUDIM data.
        - cudim_pre_y (dataframe): dataframe containing the CUDIM labels.

    Returns:
        - gradient boosting model parameters
        - performance scores of the best model on the test set.
        Accuracy, precision, recall and f1 score.
        - cudim_pre_scores: performance scores of the best model on the CUDIM
        data. Accuracy, precision, recall and f1 score.

    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": [3, 5, 7],
    }

    gb = GradientBoostingClassifier()
    grid_search = GridSearchCV(
        estimator=gb, param_grid=param_grid, scoring="f1_macro", cv=5
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_gb = grid_search.best_estimator_
    predictions = best_gb.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")
    print(
        f"Gradient Boosting = F1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}"
    )
    scores = {"f1": f1, "precision": precision, "accuracy": accuracy, "recall": recall}
    cm = confusion_matrix(y_test, predictions, labels=best_gb.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_gb.classes_)
    disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(
            output_folder, f"gradient_boosting_confusion_matrix_{timestamp}.png"
        )
    )
    plt.show()
    # PRE-CUDIM
    cudim_pre_predictions = best_gb.predict(cudim_pre_x)

    cudim_pre_accuracy = accuracy_score(cudim_pre_y, cudim_pre_predictions)
    cudim_pre_precision = precision_score(
        cudim_pre_y, cudim_pre_predictions, average="macro"
    )
    cudim_pre_recall = recall_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    cudim_pre_f1 = f1_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    print(
        f"CUDIM_PRE: gradient_boosting = F1: {cudim_pre_f1}, accuracy: {cudim_pre_accuracy}, precision: {cudim_pre_precision}, recall: {cudim_pre_recall}"
    )
    cudim_pre_scores = {
        "f1": cudim_pre_f1,
        "precision": cudim_pre_precision,
        "accuracy": cudim_pre_accuracy,
        "recall": cudim_pre_recall,
    }
    cudim_pre_cm = confusion_matrix(
        cudim_pre_y, cudim_pre_predictions, labels=best_gb.classes_
    )
    cudim_pre_disp = ConfusionMatrixDisplay(
        confusion_matrix=cudim_pre_cm, display_labels=best_gb.classes_
    )
    cudim_pre_disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(
            output_folder,
            f"gradient_boosting_confusion_matrix_cudim_pre{timestamp}.png",
        )
    )
    plt.show()

    return best_params, scores, cudim_pre_scores


def xgboost_model(
    X_train, X_test, y_train, y_test, output_folder, cudim_pre_x, cudim_pre_y
):
    """ Train and test XGBoost model for the mean intensities and
    standard deviations dataframe of all images.

    Args:
        - X_train (dataframe): dataframe containing the training data.
        - X_test (dataframe): dataframe containing the test data.
        - y_train (dataframe): dataframe containing the training labels.
        - y_test (dataframe): dataframe containing the test labels.
        - output_folder (str): path to the folder where the confusion matrix
        will be saved.
        - cudim_pre_x (dataframe): dataframe containing the CUDIM data.
        - cudim_pre_y (dataframe): dataframe containing the CUDIM labels.
    Returns:
        - XGBoost model parameters
        - Performance scores of the best model on the test set:
        accuracy, precision, recall, and F1 score.
        - cudim_pre_scores: performance scores of the best model on the CUDIM
        data. Accuracy, precision, recall and f1 score.

    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.1, 0.01, 0.001],
        "max_depth": [3, 5, 7],
    }

    xgb = XGBClassifier()
    grid_search = GridSearchCV(
        estimator=xgb, param_grid=param_grid, scoring="f1_macro", cv=5
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_xgb = grid_search.best_estimator_
    # TRAINING
    predictions = best_xgb.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    recall = recall_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test, predictions, average="macro")
    print(
        f"XGBoost = F1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}"
    )
    scores = {"f1": f1, "precision": precision, "accuracy": accuracy, "recall": recall}
    cm = confusion_matrix(y_test, predictions, labels=best_xgb.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_xgb.classes_)
    disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(output_folder, f"xgboost_confusion_matrix_{timestamp}.png")
    )
    plt.show()
    # PRE-CUDIM
    cudim_pre_predictions = best_xgb.predict(cudim_pre_x)

    cudim_pre_accuracy = accuracy_score(cudim_pre_y, cudim_pre_predictions)
    cudim_pre_precision = precision_score(
        cudim_pre_y, cudim_pre_predictions, average="macro"
    )
    cudim_pre_recall = recall_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    cudim_pre_f1 = f1_score(cudim_pre_y, cudim_pre_predictions, average="macro")
    print(
        f"CUDIM_PRE: XGBoost = F1: {cudim_pre_f1}, accuracy: {cudim_pre_accuracy}, precision: {cudim_pre_precision}, recall: {cudim_pre_recall}"
    )
    cudim_pre_scores = {
        "f1": cudim_pre_f1,
        "precision": cudim_pre_precision,
        "accuracy": cudim_pre_accuracy,
        "recall": cudim_pre_recall,
    }
    cudim_pre_cm = confusion_matrix(
        cudim_pre_y, cudim_pre_predictions, labels=best_xgb.classes_
    )
    cudim_pre_disp = ConfusionMatrixDisplay(
        confusion_matrix=cudim_pre_cm, display_labels=best_xgb.classes_
    )
    cudim_pre_disp.plot()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(
        os.path.join(
            output_folder, f"xgboost_confusion_matrix_cudim_pre{timestamp}.png"
        )
    )
    plt.show()

    return best_params, scores, cudim_pre_scores


#%%
def get_features_folder(folder_path, metadata_path):
    """ Extract features from a folder containing NIFTI images.

    Args:
        - folder_path (str): path to the folder containing the NIFTI images.
        - metadata_path (str): path to the metadata file.

    Returns:
        - features_data (dataframe): dataframe containing the extracted features.

    """
    atlas = nib.load("CerebrA.nii")
    atlas_np = np.array(atlas.dataobj)
    all_areas = np.unique(atlas_np)
    metadata = pd.read_csv(metadata_path)
    features_data = process_folder(folder_path, metadata, atlas_np, all_areas)
    features_data.to_csv("features_FDG.csv")
    return features_data


def get_features_h5(train_path, test_path, atlas_path, output_folder):
    """ Extract features from h5 files containing NIFTI images.

    Args:
        - train_path (str): path to the train h5 file.
        - test_path (str): path to the test h5 file.
        - atlas_path (str): path to the atlas file.
        - output_folder (str): path to the folder where the extracted features
        will be saved.

    Returns:
        - train_export_filename (str): path to the exported train features file.
        - test_export_filename (str): path to the exported test features file.

    """
    atlas = nib.load(atlas_path)
    atlas_np = np.array(atlas.dataobj)
    transposed_atlas = np.transpose(atlas_np, (0, 2, 1))
    all_areas = np.unique(atlas_np)
    train_file = h5py.File(train_path)
    test_file = h5py.File(test_path)
    print("Start feature extraction for Train dataset")
    train_features_data = process_h5(train_file, transposed_atlas, all_areas)
    print("Start feature extraction for Test dataset")
    test_features_data = process_h5(test_file, transposed_atlas, all_areas)
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    train_export_filename = os.path.join(
        output_folder, f"train_features_FDG_{timestamp}.csv"
    )
    test_export_filename = os.path.join(
        output_folder, f"test_features_FDG_{timestamp}.csv"
    )
    # Save the data with the timestamp in the filenames
    train_features_data.to_csv(train_export_filename)
    test_features_data.to_csv(test_export_filename)
    print("Ended feature extraction from train and test h5 files")
    return train_export_filename, test_export_filename


def get_features_cudim_h5(file_path, atlas_path, output_folder):
    """ Extract features from h5 files containing NIFTI images.

    Args:
        - file_path (str): path to the h5 file.
        - atlas_path (str): path to the atlas file.
        - output_folder (str): path to the folder where the extracted features
        will be saved.

    Returns:
        - train_export_filename (str): path to the exported features file.

    """
    atlas = nib.load(atlas_path)
    atlas_np = np.array(atlas.dataobj)
    transposed_atlas = np.transpose(atlas_np, (0, 2, 1))
    all_areas = np.unique(atlas_np)
    train_file = h5py.File(file_path)
    print("Start feature extraction for CUDIM dataset")
    train_features_data = process_h5(train_file, transposed_atlas, all_areas)
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    train_export_filename = os.path.join(output_folder, f"CUDIM_FDG_{timestamp}.csv")
    # Save the data with the timestamp in the filenames
    train_features_data.to_csv(train_export_filename)
    print("Ended feature extraction from CUDIM h5 files")
    return train_export_filename


#%%
# Extract features from h5 files
train_h5_path = "PREPROCESSED DATA/P01/train_csv.hdf5"
test_h5_path = "PREPROCESSED DATA/P01/test_csv.hdf5"
atlas_path = "cerebra_100_120_100.nii"
output_folder = "PREPROCESSED DATA/P01/output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

train_features_file, test_features_file = get_features_h5(
    train_h5_path, test_h5_path, atlas_path, output_folder
)

# UNCOMMENT NEXT LINES TO RUN FEATURE EXTRACTION FROM FOLDER
# folder_path = ""
# metadata_path = ""
# get_features_folder(folder_path, metadata_path)
#%%
# Extract CUDIM data
cudim_h2_path = "PREPROCESSED DATA/P01//pre_pet_diag.hdf5"
cudim_features_file = get_features_cudim_h5(cudim_h2_path, atlas_path, output_folder)
#%%
# Features preprocessing
train_features_data = pd.read_csv(train_features_file)
test_features_data = pd.read_csv(test_features_file)
X = pd.concat([train_features_data, test_features_data], ignore_index=True)
# Remove std_dev columns
columns_to_keep = ["label", "sex", "age"]  # Columns to keep
# Loop to add the intensity columns to the list of columns to keep
for i in range(103):  # Loop from 0.0 to 102.0
    columns_to_keep.append(f"{i}.0_intensity")
# Create a new DataFrame with only the columns you want to keep
X = X[columns_to_keep]
# Remove rows with NaN values
X = X.dropna()
labels = X["label"]
X = X.drop("label", axis=1)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, labels, test_size=0.2, random_state=42
)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
label_counts = np.bincount(y_train_resampled)
print("label counts:", label_counts)

#%%
# PRE-CUDIM data pre-processing
# pre_cudim_data = pd.read_csv(os.path.join(output_folder,'pre_cudim_features_FDG_20240818_114706.csv'))
pre_cudim_data = pd.read_csv(cudim_features_file)
columns_to_keep = ["label", "sex", "age"]  # Columns to keep
# Loop to add the intensity columns to the list of columns to keep
for i in range(103):  # Loop from 0.0 to 102.0
    columns_to_keep.append(f"{i}.0_intensity")
# Create a new DataFrame with only the columns you want to keep
pre_cudim_df = pre_cudim_data[columns_to_keep]
# Remove rows with NaN values
pre_cudim_df = pre_cudim_df.dropna()
pre_cudim_labels = pre_cudim_df["label"]
# pre_cudim_labels = np.where(pre_cudim_labels == 0.0, 2.0, pre_cudim_labels)
pre_cudim_df = pre_cudim_df.drop("label", axis=1)
scaler = StandardScaler()
pre_cudim_df_normalized = scaler.fit_transform(pre_cudim_df)

#%%
# MODEL TRAINING AND TESTING
svm, svm_scores, cudim_pre_svm_scores = svm_model(
    X_train_resampled,
    X_test,
    y_train_resampled,
    y_test,
    output_folder,
    pre_cudim_df_normalized,
    pre_cudim_labels,
)
rf, rf_scores, cudim_pre_rf_scores = random_forest_model(
    X_train_resampled,
    X_test,
    y_train_resampled,
    y_test,
    output_folder,
    pre_cudim_df_normalized,
    pre_cudim_labels,
)
gradient, gradient_scores, cudim_pre_gradient_scores = gradient_boosting_model(
    X_train_resampled,
    X_test,
    y_train_resampled,
    y_test,
    output_folder,
    pre_cudim_df_normalized,
    pre_cudim_labels,
)
xgb, xgb_scores, cudim_pre_xgb_scores = xgboost_model(
    X_train_resampled,
    X_test,
    y_train_resampled,
    y_test,
    output_folder,
    pre_cudim_df_normalized,
    pre_cudim_labels,
)
# %%
# Store and export the scores in csv files
scores = {
    "Metric": list(svm_scores.keys()),
    "SVM": list(svm_scores.values()),
    "Random Forest": list(rf_scores.values()),
    "Gradient Boosting": list(gradient_scores.values()),
    "XGBoost": list(xgb_scores.values()),
}
best_params_all_models = {
    "SVM": svm,
    "Random Forest": rf,
    "Gradient Boosting": gradient,
    "XGBoost": xgb,
}
data_scores = pd.DataFrame(scores)
data_params = pd.DataFrame(best_params_all_models)
now = datetime.datetime.now()

# Format the date and time
timestamp = now.strftime("%Y%m%d_%H%M%S")
# Specify the CSV file name
scores_csv_file = os.path.join(output_folder, f"classic_model_scores_{timestamp}.csv")
params_csv_file = os.path.join(output_folder, f"best_params_models{timestamp}.csv")
# Export the DataFrame to a CSV file
data_scores.to_csv(scores_csv_file, index=False)
data_params.to_csv(params_csv_file, index=False)
print(f"Scores exported to {scores_csv_file}")
print(f"Models params exported to {params_csv_file}")

# PRE-CUDIM
cudim_pre_scores = {
    "Metric": list(cudim_pre_svm_scores.keys()),
    "SVM": list(cudim_pre_svm_scores.values()),
    "Random Forest": list(cudim_pre_rf_scores.values()),
    "Gradient Boosting": list(cudim_pre_gradient_scores.values()),
    "XGBoost": list(cudim_pre_xgb_scores.values()),
}
cudim_pre_data_scores = pd.DataFrame(cudim_pre_scores)

# Specify the CSV file name
cudim_pre_scores_csv_file = os.path.join(
    output_folder, f"cudim_pre_classic_model_scores_{timestamp}.csv"
)
# Export the DataFrame to a CSV file
cudim_pre_data_scores.to_csv(cudim_pre_scores_csv_file, index=False)
print(f"Scores exported to {cudim_pre_scores_csv_file}")
