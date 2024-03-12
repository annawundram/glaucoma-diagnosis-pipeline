import numpy as np
import h5py
import cv2
import math
from phiseg_pytorch.src.models import PHISeg
import matplotlib.pyplot as plt
import torch
import gc


def rim_thickness_func(anno, plot):
    """
    Compute the rim thickness given an annotated image.

    Parameters:
    - anno (torch.Tensor): Annotated image tensor.
    - plot (bool, optional): Whether to plot intermediate steps. Default is False.

    Returns:
    - list: List of distances representing the rim thickness at different angles.
    """
    anno = anno.detach().cpu().numpy()
    anno = np.ascontiguousarray(anno, dtype=np.uint8)
    mask = anno.copy()

    disk = mask.copy()
    cup = mask.copy()

    disk[disk == 2] = 1
    cup[cup == 1] = 0

    contours_disk, hierarchy_disk = cv2.findContours(
        disk, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE
    )
    contours_cup, hierarchy_cup = cv2.findContours(
        cup, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE
    )

    outer = np.zeros_like(mask)
    inner = np.zeros_like(mask)

    if len(contours_disk) == 1:
        cv2.drawContours(
            image=outer,
            contours=contours_disk,
            contourIdx=0,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    else:
        for i, d in enumerate(contours_disk):
            hull = cv2.convexHull(d)
            if len(hull) > 15:
                cv2.drawContours(
                    image=outer,
                    contours=contours_disk,
                    contourIdx=i,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    if len(contours_cup) == 1:
        cv2.drawContours(
            image=inner,
            contours=contours_cup,
            contourIdx=0,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    else:
        for j, c in enumerate(contours_cup):
            hull = cv2.convexHull(c)
            if len(hull) > 15:
                cv2.drawContours(
                    image=inner,
                    contours=contours_cup,
                    contourIdx=j,
                    color=(255, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

    # centre of outer contour
    M = cv2.moments(inner)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    x, y = (cX, cY + 180)  # for ROI 180, else 80

    distance = []

    for angle in np.arange(0, 360, 0.5):

        # Convert the angle from degrees to radians
        angle_rad = math.radians(angle)

        # Calculate the new coordinates after rotation
        x_new = cX + (x - cX) * math.cos(angle_rad) - (y - cY) * math.sin(angle_rad)
        y_new = cY + (x - cX) * math.sin(angle_rad) + (y - cY) * math.cos(angle_rad)

        line = np.zeros_like(anno)
        cv2.line(line, (cX, cY), (int(x_new), int(y_new)), (255, 0, 0), 1)

        intersect_outer = np.logical_and(outer, line)
        intersect_inner = np.logical_and(inner, line)

        # now sharpen contour pixels
        outer[outer < 200] = 0
        inner[inner < 200] = 0

        # find pixels that will be used for distance
        outer_not_exact = True
        inner_not_exact = True
        outer_relevant = np.argwhere(intersect_outer)
        inner_relevant = np.argwhere(intersect_inner)

        # 1. lines intersect exactly
        for i in range(len(outer_relevant)):
            if outer[outer_relevant[i, 0], outer_relevant[i, 1]] != 0:
                outer_pixel = outer_relevant[i]
                outer_not_exact = False

        for i in range(len(inner_relevant)):
            if inner[inner_relevant[i, 0], inner_relevant[i, 1]] != 0:
                inner_pixel = inner_relevant[i]
                inner_not_exact = False

        if outer_not_exact:
            # 2. lines intersect diagonally between pixels
            # find pixel that has pixels belonging to outer contour left and beneath it, then choose below pixel of contour
            for i in range(len(outer_relevant)):
                if (
                    outer[outer_relevant[i, 0] - 1, outer_relevant[i, 1]] != 0
                    and outer[outer_relevant[i, 0] + 1, outer_relevant[i, 1]] != 0
                ):
                    outer_pixel = outer_relevant[i]
                    outer_pixel[0] += 1
        if inner_not_exact:
            # find pixel that has pixels belonging to inner contour right and above it, then choose right pixel of contour
            for i in range(len(inner_relevant)):
                if (
                    inner[inner_relevant[i, 0], inner_relevant[i, 1] - 1] != 0
                    and inner[inner_relevant[i, 0] + 1, inner_relevant[i, 1]] != 0
                ):
                    inner_pixel = inner_relevant[i]
                    inner_pixel[1] += 1

        # calculate distance between the two pixels
        dx2 = (inner_pixel[0] - outer_pixel[0]) ** 2
        dy2 = (inner_pixel[1] - outer_pixel[1]) ** 2
        distance.append(math.sqrt(dx2 + dy2))

        if plot and angle % 50 == 0:
            fig, ax = plt.subplots()
            ax = plt.imshow(inner + outer + line, cmap="gray")
            ax = plt.plot(
                inner_pixel[1],
                inner_pixel[0],
                marker="o",
                markerfacecolor="green",
                markersize=2,
            )
            ax = plt.plot(
                outer_pixel[1],
                outer_pixel[0],
                marker="o",
                markerfacecolor="red",
                markersize=2,
            )
            plt.show()
    return distance


def create_dataset(hf5, no_images, n_samples, t, Chaksu):
    """
    Create datasets for rim thickness, area CDR, and segmentations in an HDF5 file.

    Parameters:
    - hf5 (h5py.File): HDF5 file object to create datasets in.
    - no_images (int): Number of images to process.
    - n_samples (int): Number of samples to generate per image.
    - t (str): Prefix for dataset names.
    - Chaksu (dict): Dictionary containing Chaksu data.

    Returns:
    None
    """
    rim_thickness_h5 = hf5.create_dataset(
        t + "rim_thickness", shape=(no_images, n_samples, 720), dtype="float32"
    )
    cdr_h5 = hf5.create_dataset(
        t + "area_cdr", shape=(no_images, n_samples), dtype="float32"
    )
    segmentation_h5 = hf5.create_dataset(
        t + "segmentations", shape=(no_images, n_samples, 320, 320), dtype="uint8"
    )

    # for each Chaksu train image sample twenty times from PHiSeg
    for i in range(no_images):
        print("image", i, "of", no_images)
        image = Chaksu[t + "images"][i]
        # normalise image
        image = (image - image.mean(axis=(0, 1))) / image.std(axis=(0, 1))

        # change shape from (size, size, 3) to (3, size, size)
        image = np.moveaxis(image, -1, 0)

        # Convert to torch tensor
        image = torch.from_numpy(image)
        # Convert uint8 to float tensors
        image = image.type(torch.FloatTensor)
        image = torch.unsqueeze(image, 0)

        phiseg_pred_sm = phiseg.predict_output_samples(image, N=n_samples)
        samples = torch.argmax(phiseg_pred_sm, dim=2)
        segmentation_h5[i] = samples.detach().cpu().numpy()
        # calculate rim thickness for each sample
        for j, sample in enumerate(samples[0]):
            area_disk = np.count_nonzero(sample.detach().cpu().numpy() != 0)
            area_cup = np.count_nonzero(sample.detach().cpu().numpy() == 2)
            cdr_h5[i, j] = area_cup / area_disk
            try:
                rim_thickness_h5[i, j] = rim_thickness_func(sample, False)
            except:
                np.save("PhiSeg_weird_array" + str(i) + "_" + str(j), sample)
                print(i, j)
        del phiseg_pred_sm, samples, image
        gc.collect()
    # save in H5 file
    hf5.create_dataset(
        t + "majority_vote",
        shape=(no_images,),
        data=np.array(Chaksu[t + "majority_diagnosis"]),
    )


# model weights
phiseg = PHISeg.load_from_checkpoint("path_to_your_model")

# -------------------------------------------------------
# files
Chaksu_file_tr = "ground_truth_train_file"
Chaksu_file_test = "ground_truth_test_file"
hf_Chaksu_tr = h5py.File(Chaksu_file_tr, "r")
hf_Chaksu_test = h5py.File(Chaksu_file_test, "r")

n_samples = 100
no_images_tr = hf_Chaksu_tr["train/images"].shape[0]
no_images_val = hf_Chaksu_tr["val/images"].shape[0]
no_images_test = hf_Chaksu_test["images"].shape[0]


hf5_tr = h5py.File(f"your_path_to/PHiSeg_dataset_ROI_{n_samples}.h5", "w")
hf5_test = h5py.File(f"your_path_to/PHiSeg_dataset_ROI_{n_samples}_test.h5", "w")
create_dataset(hf5_tr, no_images_tr, n_samples, "train/", hf_Chaksu_tr)
create_dataset(hf5_tr, no_images_val, n_samples, "val/", hf_Chaksu_tr)
create_dataset(hf5_test, no_images_test, n_samples, "", hf_Chaksu_test)
hf5_tr.close()
hf5_test.close()


hf_Chaksu_tr.close()
hf_Chaksu_test.close()
