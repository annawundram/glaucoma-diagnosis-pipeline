import numpy as np
import os.path as osp
import glob
import math
from skimage import feature, transform
import cv2
from typing import Tuple
from PIL import Image
from src.models import UNet
from utils import harden_softmax_outputs
import torch

# support functions for preprcoessing (author: Sarah Müller)

def expectation(p: np.ndarray, w: np.ndarray) -> Tuple[float]:
    """Expectation step for finding circle in 2D image.

    Basically the 'least-squares circle fit' method.
    See https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf for details.

    Args:
        p: Points on 2D plane.
        w: Point weights.

    Returns:
        Tuple of center points and radius of circle.
    """
    x, y = p
    x_mean = w.T @ x
    y_mean = w.T @ y
    u = x - x_mean
    v = y - y_mean
    S_uu = w.T @ (u**2)
    S_uv = w.T @ (u * v)
    S_vv = w.T @ (v**2)
    S_uuu = w.T @ (u**3)
    S_vvv = w.T @ (v**3)
    S_uvv = w.T @ (u * v**2)
    S_vuu = w.T @ (v * u**2)
    A_inv = np.array([[S_vv, -S_uv], [-S_uv, S_uu]]) / (S_uu * S_vv - S_uv**2)
    B = np.array([S_uuu + S_uvv, S_vvv + S_vuu]) / 2
    u_c, v_c = A_inv @ B
    x_c = u_c + x_mean
    y_c = v_c + y_mean
    alpha = u_c**2 + v_c**2 + S_uu + S_vv
    r = np.sqrt(alpha)
    return (x_c, y_c, r)


def maximization(
    p: np.ndarray,
    circle: Tuple[float],
    λ: float = 1.0,
) -> np.ndarray:
    """Maximization step for finding circle in 2D image.

    Adjust point weighting.

    Args:
        p: Points on 2D plane.
        circle: Tuple of center points and radius.
        λ: Decay constant of exponential function.

    Returns:
        Point weights.
    """
    x, y = p
    x_c, y_c, r = circle
    delta = λ * (np.sqrt((x - x_c) ** 2 + (y - y_c) ** 2) - r) ** 2
    delta = delta - delta.min()
    w = np.exp(-delta)
    w /= w.sum()
    return w


def circle_em(
    p: np.ndarray,
    circle_init: Tuple[float],
    num_steps: int = 100,
    λ: float = 0.1,
) -> Tuple[float]:
    """Expectation-maximization (EM) algorithm to find circle in 2D plane.

    Use the domain knowledge that the circle we search for is the largest possible circle.
    Therefore the algorithm can be started with an initial guess.

    Args:
        p: Points on 2D plane.
        circle_init: Initial circle guess.
        num_steps: Number of EM steps.
        λ: Decay constant of exponential function.

    Returns:
        Tuple of center points and radius of circle.
    """
    x_c, y_c, r = circle_init
    for step in range(num_steps):
        w = maximization(p, (x_c, y_c, r), λ / (num_steps - step))
        x_c, y_c, r = expectation(p, w)
    return (x_c, y_c, r)


def square_padding(im: np.ndarray, add_pad: int = 100, gray: bool = False) -> np.ndarray:
    """Set image into the center and pad around.

    To better find edges and the corresponding circle in the fundus images.

    Args:
        im: Fundus image.
        add_pad: Constant border padding.

    Return:
        Padded image.
    """
    dim_y, dim_x = im.shape[:2]
    dim_larger = max(dim_x, dim_y)
    x_pad = (dim_larger - dim_x) // 2 + add_pad
    y_pad = (dim_larger - dim_y) // 2 + add_pad
    if gray:
        return np.pad(im, ((y_pad, y_pad), (x_pad, x_pad)))
    return np.pad(im, ((y_pad, y_pad), (x_pad, x_pad), (0, 0)))


def get_mask(ratios, target_resolution=1024):
    """Get mask from radius ratios.

    Args:
        ratios: Ratios for radius in mask horizontally and vertically.
        target_resolution: Target image resolution after resize.

    Return:
        Mask.
    """
    Y, X = np.ogrid[:target_resolution, :target_resolution]
    r = target_resolution / 2
    dist_from_center = np.sqrt((X - r) ** 2 + (Y - r) ** 2)
    mask = dist_from_center <= r

    if ratios[0] < 1:
        mask &= Y >= (r - r * ratios[0])
    if ratios[1] < 1:
        mask &= Y <= (r + r * ratios[1])

    if ratios[2] < 1:
        mask &= X >= (r - r * ratios[2])
    if ratios[3] < 1:
        mask &= X <= (r + r * ratios[3])
    return mask

# end support functions

# preprocessing function - adapted from Sarah Müller; returns un-resized, cropped wrt. fundus, square image
def preprocess_RIGA(x,
                      annotation_1,
                      annotation_2,
                      annotation_3,
                      annotation_4,
                      annotation_5,
                      annotation_6,
                      device_type,
                      x_id,
                      resize_canny_edge: int = 1000,
                      sigma_scale: int = 50,
                      circle_fit_steps: int = 100,
                      λ: float = 0.01,
                      fit_largest_contour: bool = False,):

    if "Magrabia" == device_type:
        λ = 0.1

    # Pad image to square.
    if "Magrabia" == device_type:
        x_square_padded = square_padding(x, add_pad=500)
        annotation_1_square_padded = square_padding(annotation_1, add_pad=500)
        annotation_2_square_padded = square_padding(annotation_2, add_pad=500)
        annotation_3_square_padded = square_padding(annotation_3, add_pad=500)
        annotation_4_square_padded = square_padding(annotation_4, add_pad=500)
        annotation_5_square_padded = square_padding(annotation_5, add_pad=500)
        annotation_6_square_padded = square_padding(annotation_6, add_pad=500)
    else:
        x_square_padded = square_padding(x)
        annotation_1_square_padded = square_padding(annotation_1)
        annotation_2_square_padded = square_padding(annotation_2)
        annotation_3_square_padded = square_padding(annotation_3)
        annotation_4_square_padded = square_padding(annotation_4)
        annotation_5_square_padded = square_padding(annotation_5)
        annotation_6_square_padded = square_padding(annotation_6)

    # Detect outer circle.
    height, width, _ = x_square_padded.shape
    # Resize image for canny edge detection (scales bad with image size).
    x_resized = np.array(
        Image.fromarray(x_square_padded).resize((resize_canny_edge,) * 2)
    )
    x_gray = x_resized.mean(axis=-1)
    edges = feature.canny(
        x_gray,
        sigma=resize_canny_edge / sigma_scale,
        low_threshold=0.99,
        high_threshold=1,
    )
    # If no edges are found, the image is skipped (e.g. for completely black images).
    if np.all(edges == 0):
        print(f"No edges found, skip image with id {x_id}.\n")
        return None, True
    # Restore original image size.
    edges_resized = np.array(Image.fromarray(edges).resize((height,) * 2))

    if fit_largest_contour:
        contours, _ = cv2.findContours(
            edges_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Sort contours by number of points.
        contours_sorted = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        points = contours_sorted[0].squeeze().T  # take the largest contour
    else:
        points = np.flip(np.array(edges_resized.nonzero()), axis=0)

    x_init = width / 2
    y_init = height / 2

    if "Magrabia" == device_type:
        r_init = x.shape[1] / 2
    else:
        r_init = height / 2 - 100
    # Initialize circle parameters (initial guess is close to the largest possible circle).
    circle_init = (x_init, y_init, r_init)
    x_c, y_c, r = circle_em(points, circle_init, circle_fit_steps, λ)

    if "Magrabia" == device_type:
        r += 80
        y_c += 150

    if math.isnan(x_c) or math.isnan(y_c) or math.isnan(r):
        print(f"Wrong circle found, skip id {x_id}.\n")
        return None, True

    # Create masked image.
    masked_img = x_square_padded.copy()

    # Tight crop around circle.
    masked_img = masked_img[
        int(y_c - r) : int(y_c + r) + 1, int(x_c - r) : int(x_c + r) + 1, :
    ]
    if masked_img.size == 0:
        print(f"Wrong circle found, skip id {x_id}.\n")
        return None, True

    # crop annotations
    annotation_1_square_padded = annotation_1_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :]
    annotation_2_square_padded = annotation_2_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :]
    annotation_3_square_padded = annotation_3_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :]
    annotation_4_square_padded = annotation_4_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :]
    annotation_5_square_padded = annotation_5_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :]
    annotation_6_square_padded = annotation_6_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :]

    return masked_img, annotation_1_square_padded, annotation_2_square_padded, annotation_3_square_padded,\
           annotation_4_square_padded, annotation_5_square_padded, annotation_6_square_padded


# preprocessing function - adapted from Sarah Müller; returns un-resized, cropped wrt. fundus, square image
def preprocess_Chaksu(x,
                      annotation_1,
                      annotation_2,
                      annotation_3,
                      annotation_4,
                      annotation_5,
                      consensus_annotation,
                      device_type,
                      x_id,
                      resize_canny_edge: int = 1000,
                      sigma_scale: int = 50,
                      circle_fit_steps: int = 100,
                      λ: float = 0.01,
                      fit_largest_contour: bool = False,):

    # Pad image to square.
    x_square_padded = square_padding(x, gray=False)
    annotation_1_square_padded = square_padding(annotation_1, gray=True)
    annotation_2_square_padded = square_padding(annotation_2, gray=True)
    annotation_3_square_padded = square_padding(annotation_3, gray=True)
    annotation_4_square_padded = square_padding(annotation_4, gray=True)
    annotation_5_square_padded = square_padding(annotation_5, gray=True)
    consensus_annotation_square_padded = square_padding(consensus_annotation, gray=True)

    # Detect outer circle.
    height, _, _ = x_square_padded.shape
    # Resize image for canny edge detection (scales bad with image size).
    x_resized = np.array(
        Image.fromarray(x_square_padded).resize((resize_canny_edge,) * 2)
    )
    x_gray = x_resized.mean(axis=-1)
    edges = feature.canny(
        x_gray,
        sigma=resize_canny_edge / sigma_scale,
        low_threshold=0.99,
        high_threshold=1,
    )
    # If no edges are found, the image is skipped (e.g. for completely black images).
    if np.all(edges == 0):
        print(f"No edges found, skip image with id {x_id}.\n")
        return None, True
    # Restore original image size.
    edges_resized = np.array(Image.fromarray(edges).resize((height,) * 2))

    if fit_largest_contour:
        contours, _ = cv2.findContours(
            edges_resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Sort contours by number of points.
        contours_sorted = sorted(contours, key=lambda x: x.shape[0], reverse=True)
        points = contours_sorted[0].squeeze().T  # take the largest contour
    else:
        points = np.flip(np.array(edges_resized.nonzero()), axis=0)

    # set r_init depending on device type
    orig_height, orig_width, _ = x.shape
    if device_type == "bosch":
        r_init = orig_width / 2
    else:
        r_init = orig_width / 2 - 50

    x_init = orig_height / 2
    # Initialize circle parameters (initial guess is close to the largest possible circle).
    circle_init = (x_init, x_init, r_init)
    x_c, y_c, r = circle_em(points, circle_init, circle_fit_steps, λ)

    # change r for bosch images
    if device_type == "Bosch":
        r += 80

    if math.isnan(x_c) or math.isnan(y_c) or math.isnan(r):
        print(f"Wrong circle found, skip id {x_id}.\n")
        return None, True
    # Create masked image.
    masked_img = x_square_padded.copy()

    # Tight crop around circle.
    masked_img = masked_img[
                 int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1, :
                 ]
    if masked_img.size == 0:
        print(f"Wrong circle found, skip id {x_id}.\n")
        return None, True

    # crop annotations
    annotation_1_square_padded = annotation_1_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1]
    annotation_2_square_padded = annotation_2_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1]
    annotation_3_square_padded = annotation_3_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1]
    annotation_4_square_padded = annotation_4_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1]
    annotation_5_square_padded = annotation_5_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1]
    consensus_annotation_square_padded = consensus_annotation_square_padded[int(y_c - r): int(y_c + r) + 1, int(x_c - r): int(x_c + r) + 1]

    return masked_img, annotation_1_square_padded, annotation_2_square_padded, annotation_3_square_padded,\
           annotation_4_square_padded, annotation_5_square_padded, consensus_annotation_square_padded


def crop_RIGA(prime_pillow, annos_pillow, image_input_pillow, idx, save_dir, device_type):
    # convert to numpy array
    prime = np.asarray(prime_pillow)
    prime_pillow.close()

    image_input = np.asarray(image_input_pillow)
    image_input_pillow.close()

    annos = []
    for anno in annos_pillow:
        annos.append(np.asarray(anno))
        anno.close()

    pad = 50
    # ------ annotation by U-Net ------
    # prepare for U-Net:
    image_input = (image_input - image_input.mean(axis=(0, 1))) / image_input.std(axis=(0, 1))
    # change shape from (size, size, 3) to (3, size, size)
    image_input = np.moveaxis(image_input, -1, 0)
    # Convert to torch tensor
    image_input = torch.from_numpy(image_input)
    # Convert uint8 to float tensors
    image_input = image_input.type(torch.FloatTensor)
    # add batch dimension
    image_input = torch.unsqueeze(image_input, 0)

    # load U-Net and predict
    unet = UNet.load_from_checkpoint("") # TODO: weights for U-Net
    softmax = unet.predict(image_input)
    unet_pred = harden_softmax_outputs(softmax, dim=1)
    # prepare as numpy array
    unet_pred = unet_pred.detach().numpy()
    unet_pred = np.argmax(unet_pred, axis=1)
    unet_pred = unet_pred[0]
    unet_pred = unet_pred.astype("uint8")

    # ------ preprocess original fundus image and annotations to get correct shape ------
    image, anno1, anno2, anno3, anno4, anno5, anno6 = preprocess_RIGA(prime, annos[0],
                                                                        annos[1], annos[2],
                                                                        annos[3], annos[4],
                                                                        annos[5], device_type, idx)

    # ------ calculate bounding box using the U-Net output ------
    X, Y, W, H = cv2.boundingRect(unet_pred)

    # ------ transform coordinates of the bounding box from 320x320 image space to original fundus image space ------
    # Define the scaling factors
    scale_factor_x = max(image.shape[1], image.shape[0]) / unet_pred.shape[1]
    scale_factor_y = image.shape[0] / unet_pred.shape[0]

    # Calculate the corresponding coordinates in the original image
    x_original = int(X * scale_factor_x)
    y_original = int(Y * scale_factor_y)
    w_original = int(W * scale_factor_x)
    h_original = int(H * scale_factor_y)

    # calculate the coordinates and dimensions for the cropped region with padding
    y_start = max(0, y_original - pad)
    x_start = max(0, x_original - pad)
    y_end = min(image.shape[0], y_original + h_original + pad)
    x_end = min(image.shape[1], x_original + w_original + pad)

    # ------ crop the image ------
    cropped_image = image[y_start:y_end, x_start:x_end, :]
    cropped_image = transform.resize(cropped_image, (320,) * 2, anti_aliasing=True)

    cropped_anno1 = anno1[y_start:y_end, x_start:x_end, :]
    cropped_anno1 = transform.resize(cropped_anno1, (320,) * 2, anti_aliasing=True)

    cropped_anno2 = anno2[y_start:y_end, x_start:x_end, :]
    cropped_anno2 = transform.resize(cropped_anno2, (320,) * 2, anti_aliasing=True)

    cropped_anno3 = anno3[y_start:y_end, x_start:x_end, :]
    cropped_anno3 = transform.resize(cropped_anno3, (320,) * 2, anti_aliasing=True)

    cropped_anno4 = anno4[y_start:y_end, x_start:x_end, :]
    cropped_anno4 = transform.resize(cropped_anno4, (320,) * 2, anti_aliasing=True)

    cropped_anno5 = anno5[y_start:y_end, x_start:x_end, :]
    cropped_anno5 = transform.resize(cropped_anno5, (320,) * 2, anti_aliasing=True)

    cropped_anno6 = anno6[y_start:y_end, x_start:x_end, :]
    cropped_anno6 = transform.resize(cropped_anno6, (320,) * 2, anti_aliasing=True)

    Image.fromarray((cropped_anno1 * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "-1.png")
    Image.fromarray((cropped_anno2 * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "-2.png")
    Image.fromarray((cropped_anno3 * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "-3.png")
    Image.fromarray((cropped_anno4 * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "-4.png")
    Image.fromarray((cropped_anno5 * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "-5.png")
    Image.fromarray((cropped_anno6 * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "-6.png")
    Image.fromarray((cropped_image * 255).astype(np.uint8)).save(save_dir + "/image" + str(idx) + "prime.png")

def crop_Chaksu(image_paths, input_file, device_type):
    for i in range(len(image_paths)):
        # ------ original fundus image and original annotations ------
        img_id = osp.splitext(osp.basename(image_paths[i]))[0]
        # load annotation images

        if device_type == "Forus":
            annotation_1_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/Expert_1/' + device_type
                                             + '/' + img_id + '.jpg')
        else:
            annotation_1_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/Expert_1/' + device_type
                                             + '/' + img_id + '.tif')

        annotation_2_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/Expert_2/' + device_type
                                         + '/' + img_id + '.tif')
        annotation_3_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/Expert_3/' + device_type
                                         + '/' + img_id + '.tif')
        annotation_4_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/Expert_4/' + device_type
                                         + '/' + img_id + '.tif')
        annotation_5_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/Expert_5/' + device_type
                                         + '/' + img_id + '.tif')
        annotation_1 = np.asarray(annotation_1_pillow)
        annotation_2 = np.asarray(annotation_2_pillow)
        annotation_3 = np.asarray(annotation_3_pillow)
        annotation_4 = np.asarray(annotation_4_pillow)
        annotation_5 = np.asarray(annotation_5_pillow)
        annotation_1_pillow.close()
        annotation_2_pillow.close()
        annotation_3_pillow.close()
        annotation_4_pillow.close()
        annotation_5_pillow.close()

        image_pillow = Image.open(image_paths[i])
        image = np.asarray(image_pillow)
        image_pillow.close()

        consensus_annotation_pillow = Image.open(input_file + '/4.0_OD_CO_Fusion_Images/' + device_type +
                                                 '/STAPLE/' + img_id + '.png')
        consensus_annotation = np.asarray(consensus_annotation_pillow)
        consensus_annotation_pillow.close()

        pad = 50
        # ------ annotation by U-Net ------
        # 320x320 image from H5 file
        image_input_pillow = Image.open(input_file + '/preprocessed_img_test/' + img_id + '.png') # here: no _test for train and val
        image_input = np.asarray(image_input_pillow)
        image_input_pillow.close()
        # prepare for U-Net:
        image_input = (image_input - image_input.mean(axis=(0, 1))) / image_input.std(axis=(0, 1))
        # change shape from (size, size, 3) to (3, size, size)
        image_input = np.moveaxis(image_input, -1, 0)
        # Convert to torch tensor
        image_input = torch.from_numpy(image_input)
        # Convert uint8 to float tensors
        image_input = image_input.type(torch.FloatTensor)
        # add batch dimension
        image_input = torch.unsqueeze(image_input, 0)

        # load U-Net and predict
        unet = UNet.load_from_checkpoint("") #TODO: Unet weights
        softmax = unet.predict(image_input)
        unet_pred = harden_softmax_outputs(softmax, dim=1)
        # prepare as numpy array
        unet_pred = unet_pred.detach().numpy()
        unet_pred = np.argmax(unet_pred, axis=1)
        unet_pred = unet_pred[0]
        unet_pred = unet_pred.astype("uint8")

        # ------ preprocess original fundus image and annotations to get correct shape ------
        image, anno1, anno2, anno3, anno4, anno5, consensus_annotation = preprocess_Chaksu(image, annotation_1,
                                                                                           annotation_2,
                                                                                           annotation_3,
                                                                                           annotation_4,
                                                                                           annotation_5,
                                                                                           consensus_annotation,
                                                                                           device_type, img_id)

        # ------ calculate bounding box using the U-Net output ------
        X, Y, W, H = cv2.boundingRect(unet_pred)

        # ------ transform coordinates of the bounding box from 320x320 image space to original fundus image space ------
        # Define the scaling factors
        scale_factor_x = max(image.shape[1], image.shape[0]) / unet_pred.shape[1]
        scale_factor_y = image.shape[0] / unet_pred.shape[0]

        # Calculate the corresponding coordinates in the original image
        x_original = int(X * scale_factor_x)
        y_original = int(Y * scale_factor_y)
        w_original = int(W * scale_factor_x)
        h_original = int(H * scale_factor_y)

        # calculate the coordinates and dimensions for the cropped region with padding
        y_start = max(0, y_original - pad)
        x_start = max(0, x_original - pad)
        y_end = min(image.shape[0], y_original + h_original + pad)
        x_end = min(image.shape[1], x_original + w_original + pad)

        # ------ crop the image ------
        cropped_image = image[y_start:y_end, x_start:x_end, :]
        cropped_image = transform.resize(cropped_image, (320,) * 2, anti_aliasing=True)

        cropped_anno1 = anno1[y_start:y_end, x_start:x_end]
        cropped_anno1 = transform.resize(cropped_anno1, (320,) * 2, anti_aliasing=True)

        cropped_anno2 = anno2[y_start:y_end, x_start:x_end]
        cropped_anno2 = transform.resize(cropped_anno2, (320,) * 2, anti_aliasing=True)

        cropped_anno3 = anno3[y_start:y_end, x_start:x_end]
        cropped_anno3 = transform.resize(cropped_anno3, (320,) * 2, anti_aliasing=True)

        cropped_anno4 = anno4[y_start:y_end, x_start:x_end]
        cropped_anno4 = transform.resize(cropped_anno4, (320,) * 2, anti_aliasing=True)

        cropped_anno5 = anno5[y_start:y_end, x_start:x_end]
        cropped_anno5 = transform.resize(cropped_anno5, (320,) * 2, anti_aliasing=True)

        cropped_consensus_annotation = consensus_annotation[y_start:y_end, x_start:x_end]
        cropped_consensus_annotation = transform.resize(cropped_consensus_annotation, (320,) * 2,
                                                        anti_aliasing=True)

    Image.fromarray((cropped_anno1 * 255).astype(np.uint8)).save(input_file + "/ROI_annotation1/" + img_id + ".png")
    Image.fromarray((cropped_anno2 * 255).astype(np.uint8)).save(input_file + "/ROI_annotation2/" + img_id + ".png")
    Image.fromarray((cropped_anno3 * 255).astype(np.uint8)).save(input_file + "/ROI_annotation3/" + img_id + ".png")
    Image.fromarray((cropped_anno4 * 255).astype(np.uint8)).save(input_file + "/ROI_annotation4/" + img_id + ".png")
    Image.fromarray((cropped_anno5 * 255).astype(np.uint8)).save(input_file + "/ROI_annotation5/" + img_id + ".png")
    Image.fromarray((cropped_image * 255).astype(np.uint8)).save(input_file+ "/ROI_img/" + img_id + ".png")
    Image.fromarray((cropped_consensus_annotation * 255).astype(np.uint8)).save(
        input_file + "/ROI_consensus_annotations/" + img_id + ".png")

#-----------------------------------------------------------------------------------------------------------------------

#crop all RIGA images

# preprocess MESSIDOR
messidor_path = "/RIGA-dataset/MESSIDOR"
messidor_path_anno = "/RIGA_preprocess/MESSIDOR"
save_dir_messidor = "/RIGA_ROI/MESSIDOR"

for i in range(1,461):
    prime = Image.open(messidor_path + "/image" + str(i) + "prime" + ".tif")
    image_input = Image.open("/RIGA_cropped/MESSIDOR"
                             + "/image" + str(i) + "prime" + ".png")
    annos = []
    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(messidor_path_anno + "/image" + str(i) + "-" + str(j) + ".png"))
    crop_RIGA(prime, annos, image_input, i, save_dir_messidor, "MESSIDOR")

    for anno in annos: anno.close()
    prime.close()
    annos.clear()

# preprocess BinRushed1-Corrected
BinRushed1_Corrected_path = "/RIGA-dataset/BinRushed/BinRushed1-Corrected"
BinRushed1_Corrected_path_anno = "/RIGA_preprocess/BinRushed/BinRushed1-Corrected"
save_dir_BinRushed1_Corrected = "/RIGA_ROI/BinRushed/BinRushed1-Corrected"
for i in range(1, 51):
    annos = []
    if osp.isfile(BinRushed1_Corrected_path + "/image" + str(i) + "prime" + ".jpg"):
        prime = Image.open(BinRushed1_Corrected_path + "/image" + str(i) + "prime" + ".jpg")
    else:
        prime = Image.open(BinRushed1_Corrected_path + "/image" + str(i) + "prime" + ".tif")
    image_input = Image.open("/RIGA_cropped/BinRushed/BinRushed1-Corrected"
                             + "/image" + str(i) + "prime" + ".png")
    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(BinRushed1_Corrected_path_anno + "/image" + str(i) + "-" + str(j) + ".png"))
    crop_RIGA(prime, annos, image_input, i, save_dir_BinRushed1_Corrected, "BinRushed1-Corrected")

    for anno in annos: anno.close()
    prime.close()
    annos.clear()

# preprocess BinRushed2
BinRushed2_path = "/RIGA-dataset/BinRushed/BinRushed2"
BinRushed2_path_anno = "/RIGA_preprocess/BinRushed/BinRushed2"
save_dir_BinRushed2 = "/RIGA_ROI/BinRushed/BinRushed2"

for i in range(1, 48):
    annos = []
    prime = Image.open(BinRushed2_path + "/image" + str(i) + "prime" + ".jpg")
    image_input = Image.open("/RIGA_cropped/BinRushed/BinRushed2"
                             + "/image" + str(i) + "prime" + ".png")
    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(BinRushed2_path_anno + "/image" + str(i) + "-" + str(j) + ".png"))
    crop_RIGA(prime, annos, image_input, i, save_dir_BinRushed2, "BinRushed2")

    for anno in annos: anno.close()
    prime.close()
    annos.clear()

# preprocess BinRushed3
BinRushed3 = "/RIGA-dataset/BinRushed/BinRushed3"
BinRushed3_anno = "/RIGA_preprocess/BinRushed/BinRushed3"
save_dir_BinRushed3 = "/RIGA_ROI/BinRushed/BinRushed3"

for i in range(1, 48):
    annos = []
    prime = Image.open(BinRushed3 + "/image" + str(i) + "prime" + ".jpg")
    image_input = Image.open("/RIGA_cropped/BinRushed/BinRushed3"
                             + "/image" + str(i) + "prime" + ".png")

    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(BinRushed3_anno + "/image" + str(i) + "-" + str(j) + ".png"))
    crop_RIGA(prime, annos, image_input, i, save_dir_BinRushed3, "BinRushed3")
    for anno in annos: anno.close()
    prime.close()
    annos.clear()

# preprocess BinRushed4
BinRushed4 = "/RIGA-dataset/BinRushed/BinRushed4"
BinRushed4_anno = "/RIGA_preprocess/BinRushed/BinRushed4"
save_dir_BinRushed4 = "/RIGA_ROI/BinRushed/BinRushed4"

for i in range(1, 48):
    annos = []
    prime = Image.open(BinRushed4 + "/image" + str(i) + "prime" + ".jpg")
    image_input = Image.open("/RIGA_cropped/BinRushed/BinRushed4"
                             + "/image" + str(i) + "prime" + ".png")

    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(BinRushed4_anno + "/image" + str(i) + "-" + str(j) + ".png"))
    crop_RIGA(prime, annos, image_input, i, save_dir_BinRushed4, "BinRushed4")
    for anno in annos: anno.close()
    prime.close()
    annos.clear()

# preprocess Magrabia Female
MagrabiaFemale = "/RIGA-dataset/Magrabia/MagrabiaFemale"
MagrabiaFemale_anno = "/RIGA_preprocess/Magrabia/MagrabiaFemale"
save_dir_MagrabiaFemale = "/RIGA_ROI/Magrabia/MagrabiaFemale"

for i in range(1, 48):
    annos = []
    prime = Image.open(MagrabiaFemale + "/image" + str(i) + "prime" + ".tif")
    image_input = Image.open("/RIGA_cropped/Magrabia/MagrabiaFemale"
                             + "/image" + str(i) + "prime" + ".png")

    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(MagrabiaFemale_anno + "/image" + str(i) + "-" + str(j) + ".png"))
    crop_RIGA(prime, annos, image_input, i, save_dir_MagrabiaFemale, "Magrabia")
    for anno in annos: anno.close()
    prime.close()
    annos.clear()

# preprocess Magrabia Male
MagrabiaMale = "/RIGA-dataset/Magrabia/MagrabiaMale"
MagrabiaMale_anno = "/RIGA_preprocess/Magrabia/MagrabiaMale"
save_dir_MagrabiaMale = "/RIGA_ROI/Magrabia/MagrabiaMale"

for i in range(1, 48): # ! attention: image 48 prime does not exist -> only processing until image 47
    annos = []
    if osp.isfile(MagrabiaMale + "/Image" + str(i) + "prime" + ".tif"):
        prime = Image.open(MagrabiaMale + "/Image" + str(i) + "prime" + ".tif")
    else:
        prime = Image.open(MagrabiaMale + "/image" + str(i) + "prime" + ".tif")

    image_input = Image.open("/Magrabia/MagrabiaMale"
                             + "/image" + str(i) + "prime" + ".png")
    # preprocess for all experts
    for j in range(1, 7):
        annos.append(Image.open(MagrabiaMale_anno + "/image" + str(i) + "-" + str(j) + ".png"))

    crop_RIGA(prime, annos, image_input, i, save_dir_MagrabiaMale, "Magrabia")
    for anno in annos: anno.close()
    prime.close()
    annos.clear()

#-----------------------------------------------------------------------------------------------------------------------
file_bosch = '1.0_Original_Fundus_Images/Bosch/*'
file_forus = '1.0_Original_Fundus_Images/Forus/*'
file_remidio = '1.0_Original_Fundus_Images/Remidio/*'
# crop all Chaksu train images

input_file = '/Chaksu/Train/Train'
# paths to the images
path_bosch = osp.join(input_file, file_bosch)
path_forus = osp.join(input_file, file_forus)
path_remidio = osp.join(input_file, file_remidio)

# add all image addresses
img_addr_bosch = glob.glob(path_bosch)
img_addr_forus = glob.glob(path_forus)
img_addr_remidio = glob.glob(path_remidio)

crop_Chaksu(img_addr_bosch, input_file, "Bosch")
crop_Chaksu(img_addr_forus, input_file, "Forus")
crop_Chaksu(img_addr_remidio, input_file, "Remidio")

# crop all Chaksu test images
input_file = '/Chaksu/Test'
# paths to the images
path_bosch = osp.join(input_file, file_bosch)
path_forus = osp.join(input_file, file_forus)
path_remidio = osp.join(input_file, file_remidio)

# add all image addresses
img_addr_bosch = glob.glob(path_bosch)
img_addr_forus = glob.glob(path_forus)
img_addr_remidio = glob.glob(path_remidio)

crop_Chaksu(img_addr_bosch, input_file, "Bosch")
crop_Chaksu(img_addr_forus, input_file, "Forus")
crop_Chaksu(img_addr_remidio, input_file, "Remidio")