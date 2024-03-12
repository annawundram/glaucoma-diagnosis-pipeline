import numpy as np
import h5py


def dice_coefficient(y_true, y_pred):
    """
    Compute the Dice coefficient for a given pair of ground truth and predicted segmentation masks.

    Parameters:
    - y_true (numpy.ndarray): Ground truth segmentation mask.
    - y_pred (numpy.ndarray): Predicted segmentation mask.

    Returns:
    - float: The Dice coefficient.
    """
    dice = 0
    for value in [0, 1, 2]:
        true_binary = y_true == value
        pred_binary = y_pred == value
        intersection = np.logical_and(true_binary, pred_binary)
        dice += (2 * intersection.sum()) / (true_binary.sum() + pred_binary.sum())
    return dice / 3


def dice_coef_ds(y_true_ds, y_pred_ds):
    """
    Compute the mean Dice coefficient for a dataset of ground truth and predicted segmentation masks.

    Parameters:
    - y_true_ds (list of numpy.ndarray): List of ground truth segmentation masks.
    - y_pred_ds (list of numpy.ndarray): List of predicted segmentation masks.

    Returns:
    - float: The mean Dice coefficient for the dataset.
    """
    dice_list = []
    for i in range(len(y_true_ds)):
        y_true = y_true_ds[i]
        y_pred = y_pred_ds[i]
        dice = dice_coefficient(y_true, y_pred)
        dice_list.append(dice)
    return np.mean(dice)


def compute_mean_dice_for_models(models, ground_truth):
    """
    Compute the mean Dice coefficient for each model's segmentation predictions.

    Parameters:
    - models (list of numpy.ndarray): List of segmentation predictions from different models.
    - ground_truth (list of numpy.ndarray): List of ground truth segmentation masks.

    Returns:
    - list of float: List of mean Dice coefficients for each model.
    """
    dice_scores = []
    for model_segm in models:
        if len(model_segm.shape) == 4:  # Check if model provides samples
            model_segm = np.mean(
                model_segm, axis=1
            ).round()  # Compute mean segmentation
            dice_score = dice_coef_ds(ground_truth, model_segm)
        else:  # If model does not provide samples
            dice_score = dice_coef_ds(ground_truth, model_segm)
        dice_scores.append(dice_score)  # Compute dice score and add to list
    return dice_scores


path_gt = "path_to_gt_dataset"
file_gt = h5py.File(path_gt, mode="r")
gt_segms = file_gt["consensus"][()]
file_gt.close()

model_paths = ["your_paths_here"]

model_segm_list = []

for model_path in model_paths:
    file = h5py.File(model_path, mode="r")
    print(model_path)
    model_segm_list.append(file["segmentations"][()])
    file.close()

dices = compute_mean_dice_for_models(model_segm_list, gt_segms)
print(dices)
