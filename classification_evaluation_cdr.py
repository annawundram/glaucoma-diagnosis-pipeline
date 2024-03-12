import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


def generate_extended_labels(labels, n_samples):
    """
    Generate extended labels by repeating each label multiple times.

    Parameters:
    - labels (numpy.ndarray): The original array of labels.
    - n_samples (int): The number of times to repeat each label.

    Returns:
    - numpy.ndarray: An array of extended labels where each original label is repeated n_samples times.
    """

    new_labels = np.zeros((len(labels), n_samples))

    for i in range(len(labels)):
        label = labels[i]
        rep_label = np.repeat(label, n_samples)
        new_labels[i] = rep_label

    return new_labels


def generating_datasets(
    list_of_ds_paths_train, list_of_ds_paths_test, list_of_nsamples
):
    """
    Generate datasets for training, validation, and testing from given paths. 
    Note that the index position for the paths for the ground truth dataset and the U-Net dataset have
    to be at index 0 and 1 respectively since their shape differs from the other datasets and therefore
    needs to be processed differently.

    Parameters:
    - list_of_ds_paths_train (list): List of file paths for training datasets.
    - list_of_ds_paths_test (list): List of file paths for testing datasets.
    - list_of_nsamples (list): List of integers specifying the number of samples for each dataset.

    Returns:
    - tuple: A tuple containing lists of training, validation, and testing datasets.
    """
    # define the list of train and test sets
    list_of_trains = []
    list_of_tests = []
    list_of_vals = []

    # do it for the original data and the unet separately
    file_orig = h5py.File(list_of_ds_paths_train[0], mode="r")
    file_orig_test = h5py.File(list_of_ds_paths_test[0], mode="r")
    labels_glaucoma_train = file_orig["train"]["diagnosis"][()]
    rim_thickness_train = file_orig["train"]["vertical_CDR"][()]
    labels_glaucoma_val = file_orig["val"]["diagnosis"][()]
    rim_thickness_val = file_orig["val"]["vertical_cdr"][()]
    labels_glaucoma_test = file_orig_test["diagnosis"][()]
    rim_thickness_test = file_orig_test["vertical_CDR"][()]
    rim_thickness_train_reshaped = rim_thickness_train.reshape((807, 5, 1))
    x_train = rim_thickness_train_reshaped.reshape((807 * 5, 1))
    y_train = labels_glaucoma_train.reshape((807 * 5))
    rim_thickness_val_reshaped = rim_thickness_val.reshape((202, 5, 1))
    x_val = rim_thickness_val_reshaped.mean(axis=-1).reshape((202 * 5, 1))
    y_val = labels_glaucoma_val.reshape((202 * 5))
    rim_thickness_test_reshaped = rim_thickness_test.reshape((336, 5, 1))
    x_test = rim_thickness_test_reshaped.reshape((336 * 5, 1))
    y_test = labels_glaucoma_test.reshape((336 * 5))
    list_of_trains.append((x_train, y_train))
    list_of_tests.append((x_test, y_test))
    list_of_vals.append((x_val, y_val))

    file_unet = h5py.File(list_of_ds_paths_train[1], mode="r")
    file_unet_test = h5py.File(list_of_ds_paths_test[1], mode="r")
    labels_glaucoma_train_unet = file_unet["train"]["majority_vote"][()]
    rim_thickness_train_unet = file_unet["train"]["vertical_CDR"][()]
    labels_glaucoma_val_unet = file_unet["val"]["majority_vote"][()]
    rim_thickness_val_unet = file_unet["val"]["vertical_CDR"][()]
    labels_glaucoma_test_unet = file_unet_test["majority_vote"][()]
    rim_thickness_test_unet = file_unet_test["vertical_CDR"][()]
    rim_thickness_unet_train_reshaped = rim_thickness_train_unet.reshape((807, 1))
    x_train_unet = rim_thickness_unet_train_reshaped.reshape((807, 1))
    y_train_unet = labels_glaucoma_train_unet.reshape((807))
    rim_thickness_val_unet_reshaped = rim_thickness_val_unet.reshape((202, 1))
    x_val_unet = rim_thickness_val_unet_reshaped.mean(axis=-1).reshape((202, 1))
    y_val_unet = labels_glaucoma_val_unet.reshape((202))
    rim_thickness_unet_test_reshaped = rim_thickness_test_unet.reshape((336, 1))
    x_test_unet = rim_thickness_unet_test_reshaped.reshape((336, 1))
    y_test_unet = labels_glaucoma_test_unet.reshape((336))
    list_of_trains.append((x_train_unet, y_train_unet))
    list_of_tests.append((x_test_unet, y_test_unet))
    list_of_vals.append((x_val_unet, y_val_unet))

    # make loop for others
    for i in range(2, len(list_of_ds_paths_train)):
        # for i in range(len(list_of_ds_paths_train)):
        # load the files
        file_tr = h5py.File(list_of_ds_paths_train[i], mode="r")
        file_ts = h5py.File(list_of_ds_paths_test[i], mode="r")

        # extract the train and test input + labels
        labels_glaucoma_train = generate_extended_labels(
            file_tr["train"]["majority_vote"][()], list_of_nsamples[i]
        )
        rim_thickness_train = file_tr["train"]["vertical_CDR"][()]
        rim_thickness_train_reshaped = rim_thickness_train.reshape(
            (807, list_of_nsamples[i], 1)
        )
        x_train = rim_thickness_train_reshaped.reshape((807 * list_of_nsamples[i], 1))
        y_train = labels_glaucoma_train.reshape((807 * list_of_nsamples[i]))

        labels_glaucoma_val = generate_extended_labels(
            file_tr["val"]["majority_vote"][()], list_of_nsamples[i]
        )
        rim_thickness_val = file_tr["val"]["vertical_CDR"][()]
        rim_thickness_val_reshaped = rim_thickness_val.reshape(
            (202, list_of_nsamples[i], 1)
        )
        x_val = rim_thickness_val_reshaped.mean(axis=-1).reshape(
            (202 * list_of_nsamples[i], 1)
        )
        y_val = labels_glaucoma_val.reshape((202 * list_of_nsamples[i]))

        labels_glaucoma_test = file_ts["majority_vote"][()]
        rim_thickness_test = file_ts["vertical_CDR"][()]
        rim_thickness_test_reshaped = rim_thickness_test.reshape(
            (336, list_of_nsamples[i], 1)
        )
        x_test = rim_thickness_test_reshaped.reshape((336, list_of_nsamples[i], 1))
        y_test = labels_glaucoma_test.reshape((336))

        # append to the lists
        list_of_trains.append((x_train, y_train))
        list_of_vals.append((x_val, y_val))
        list_of_tests.append((x_test, y_test))

    return list_of_trains, list_of_vals, list_of_tests


def fit_classifier(list_of_model_names, list_of_train_sets, list_of_val_sets):
    """
    Fit classifiers and determine optimal thresholds.

    Parameters:
    - list_of_model_names (list): List of model names.
    - list_of_train_sets (list): List of tuples containing training sets (X_train, y_train).
    - list_of_val_sets (list): List of tuples containing validation sets (X_val, y_val).

    Returns:
    - tuple: A tuple containing lists of fitted classifiers and corresponding thresholds.
    """
    # define all classifiers
    list_of_classifiers = [
        LogisticRegression(max_iter=4000) for model in list_of_model_names
    ]

    # fit all classifiers
    for i in range(len(list_of_model_names)):
        x, y = list_of_train_sets[i]
        x, y = shuffle(x, y, random_state=0)
        list_of_classifiers[i].fit(x, y)

    # Initialize the best threshold and its corresponding sum of Youden's index

    thresholds = []
    for i, classifier in enumerate(list_of_classifiers):
        x_val, y_val = list_of_val_sets[i]

        # Predict probabilities
        y_proba = classifier.predict_proba(x_val)[:, 1]
        best_threshold = 0
        best_youden_index = 0

        # Youden's J statistic to get the best threshold
        for threshold in [0.05, 0.1, 0.15, 0.2]:
            y_pred_classes = [1 if prob > threshold else 0 for prob in y_proba]
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_classes).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            youdens_ind = tpr - fpr
            if youdens_ind > best_youden_index:
                best_youden_index = youdens_ind
                best_threshold = threshold
        print("best threshold", best_threshold)
        thresholds.append(best_threshold)

    return list_of_classifiers, thresholds


def make_predictions(list_of_classifiers, test_sets, threshold_list):
    """
    Make predictions using trained classifiers. 
    Note again that the index position for the paths for the ground truth dataset and the U-Net dataset have
    to be at index 0 and 1 respectively.

    Parameters:
    - list_of_classifiers (list): List of trained classifiers.
    - test_sets (list): List of tuples containing test sets (X_test, y_test).
    - threshold_list (list): List of threshold values.

    Returns:
    - tuple: A tuple containing lists of binary predictions, probabilities, and mean predictions.
    """
    # Define the lists of predictions, probability predictions, and mean predictions
    list_of_preds = []
    list_of_proba_preds = []
    list_of_mean_of_preds = []

    for i in range(len(list_of_classifiers)):
        classifier = list_of_classifiers[i]
        x_test, _ = test_sets[i]

        if i < 2:
            preds_proba = classifier.predict_proba(x_test)[:, 1]
            preds = (preds_proba >= threshold_list[i]).astype(int)

            list_of_proba_preds.append(preds_proba)
            list_of_preds.append(preds)
        else:
            # Predict probabilities
            preds_proba = np.array(
                [
                    classifier.predict_proba(x_test[:, i, :])[:, 1]
                    for i in range(x_test.shape[1])
                ]
            )

            predictions_all = (preds_proba >= threshold_list[i]).astype(int)
            predictions_mean = predictions_all.mean(axis=0)

            # Apply the threshold to the probabilities to get binary predictions
            preds = (preds_proba.mean(axis=0) >= threshold_list[i]).astype(int)

            # Store the average probabilities and the binary predictions
            list_of_proba_preds.append(preds_proba.mean(axis=0))
            list_of_preds.append(preds)
            list_of_mean_of_preds.append(predictions_mean)

    return list_of_preds, list_of_proba_preds, list_of_mean_of_preds


def compute_summary_statistics(
    list_y_ts, prediction_list, proba_prediction_list, list_of_model_names
):
    """
    Compute summary statistics such as sensitivity, specificity, and AUC score for each model.

    Parameters:
    - list_y_ts (list): List of true labels for each test set.
    - prediction_list (list): List of binary predictions for each model.
    - proba_prediction_list (list): List of predicted probabilities for each model.
    - list_of_model_names (list): List of model names.

    Returns:
    - tuple: A tuple containing lists of sensitivities, specificities, and AUC scores for each model.
    """
    target_names = ["non glaucom", "glaucom"]
    sensitivities = []
    specificities = []
    aucs = []
    for i in range(len(prediction_list)):
        print("")
        print(list_of_model_names[i])
        y_gt = list_y_ts[i][1]

        pred = prediction_list[i]

        pred_proba = proba_prediction_list[i]

        auc_score = roc_auc_score(y_gt, pred_proba)
        report = classification_report(
            y_gt, pred, target_names=target_names, output_dict=True
        )

        sensitivity = report["glaucom"]["recall"]
        specificity = report["non glaucom"]["recall"]

        print("specificity:", report["non glaucom"]["recall"])
        print("sensitivity:", report["glaucom"]["recall"])
        print("auc:", auc_score)

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        aucs.append(auc_score)

    return sensitivities, specificities, aucs


train_paths = ["your_paths_here"]

test_paths = ["your_paths_here"]

list_of_nsamples = [5, 0, 100, 100, 100, 10, 11, 5]

print("generating data")
list_tr, list_val, list_ts = generating_datasets(
    train_paths, test_paths, list_of_nsamples
)

print("fitting classifier")
classifiers, best_thresholds = fit_classifier(train_paths, list_tr, list_val)
print("best threshold", best_thresholds)

print("make predictions")
preds, preds_proba, preds_mean = make_predictions(classifiers, list_ts, best_thresholds)


print("computing statistics")
sens, spec, auc = compute_summary_statistics(list_ts, preds, preds_proba, train_paths)
