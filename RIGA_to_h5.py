'''
1. create shuffle indices for train/val for all six datasets
2. 2 datasets Train/ Val
        (#, 320, 320, 3) for all prime images (! resizing needed)
        (#, 6, 320, 320) for all binary masks (! convert to 0,1,2)
        (#) MESSIDOR: 0, BinRushed1-Corrected: 1, BinRushed2: 2, BinRushed3: 3, BinRushed4: 4, MagrabiaFemale: 5, MagrabiaMale: 6
        (#, 6) vertical CDR
        (#, 6) area CDR
'''
import h5py
from PIL import Image
import numpy as np
import random

def write_range_to_hdf5(images, counter_from, counter_to, img_data, vendor_data, device_type, annotations,
                        annotations_1, annotations_2, annotations_3, annotations_4, annotations_5, annotations_6):
    """ writes range of 5 images to hdf5 file
                    Parameters:
                    ---------------
                    images        list of arrays (images)
                    counter_from  write from
                    counter_to    write to
                    img_data hdf5 dataset
                    vendor_data   hdf5 dataset
                    device_type   type of device: Bosch, Forus, Remidio
                    annotations   hdf5 dataset
                    annotations_1 array image
                    annotations_2 array image
                    annotations_3 array image
                    annotations_4 array image
                    annotations_5 array image
                    consensus_annotations array image
                    consensus_segmentation hdf5 dataset
    """
    # add images
    img_arr = np.asarray(images)
    img_data[counter_from:counter_to] = img_arr

    # add device type
    if "MESSIDOR" in device_type:
        vendor_data[counter_from:counter_to] = np.zeros((len(images), ))  # 0
    elif "BinRushed1-Corrected" in device_type:
        vendor_data[counter_from:counter_to] = np.ones((len(images), ))  # 1
    elif "BinRushed2" in device_type:
        vendor_data[counter_from:counter_to] = np.ones((len(images), )) + 1  # 2
    elif "BinRushed3" in device_type:
        vendor_data[counter_from:counter_to] = np.ones((len(images),)) + 2  # 4
    elif "BinRushed4" in device_type:
        vendor_data[counter_from:counter_to] = np.ones((len(images),)) + 3  # 5
    elif "MagrabiaFemale" in device_type:
        vendor_data[counter_from:counter_to] = np.ones((len(images),)) + 4  # 6
    else:
        vendor_data[counter_from:counter_to] = np.ones((len(images), )) + 5  # 7

    # add annotations
    annotations[counter_from:counter_to, 0] = np.asarray(annotations_1)
    annotations[counter_from:counter_to, 1] = np.asarray(annotations_2)
    annotations[counter_from:counter_to, 2] = np.asarray(annotations_3)
    annotations[counter_from:counter_to, 3] = np.asarray(annotations_4)
    annotations[counter_from:counter_to, 4] = np.asarray(annotations_5)
    annotations[counter_from:counter_to, 5] = np.asarray(annotations_6)

def convert_to_gt(mask):
    """ changes rgb values to 0,1,2 for background disk, cup respectively.
        Array consists of values ranging from zero to 255 (mostly 0, 127, 255).
        Zero values will receive label zero. Values closer to 127 will receive label 1 and labels closer to 255
        will receive label 2.
                Parameters:
                ---------------
                mask   numpy array of mask
    """
    mask[mask < 64] = 0
    a = np.zeros_like(mask[mask != 0]) + 255
    mask[mask != 0] = np.where(np.isclose(mask[mask != 0], a, atol=64, rtol=0), 2, 1)
    return mask[:, :, 0]

def add_images(input_file, ids, device_type, img_data, vendor_data, annotations, counter_from):
    """ preprocesses images and adds them to .
            Parameters:
            ---------------
            input_file  root directory
            ids        number of prime images in this folder
            device_type   sub path to data
            img_data      hdf5 dataset
            vendor_data   hdf5 dataset
            annotations   hdf5 dataset
    """
    max_write_buffer = 4
    write_buffer = 0
    images = []
    annotations_1 = []
    annotations_2 = []
    annotations_3 = []
    annotations_4 = []
    annotations_5 = []
    annotations_6 = []


    # go through all images, then preprocess them and write them to hdf5 files in batches
    for i in ids:
        # write to hdf5 file if write_buffer is full
        if write_buffer >= max_write_buffer:
            # write images to hdf5 file
            counter_to = counter_from + write_buffer
            write_range_to_hdf5(images, counter_from, counter_to, img_data, vendor_data, device_type, annotations,
                                annotations_1, annotations_2, annotations_3, annotations_4, annotations_5, annotations_6)
            # delete cash/lists
            images.clear()
            annotations_1.clear()
            annotations_2.clear()
            annotations_3.clear()
            annotations_4.clear()
            annotations_5.clear()
            annotations_6.clear()
            # reset stuff for next iteration
            write_buffer = 0
            counter_from += 4

        # load image

        image_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "prime.png")
        image = np.asarray(image_pillow).copy()
        images.append(image)
        image_pillow.close()


        # load annotation images
        annotation_1_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "-1.png")
        annotation_2_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "-2.png")
        annotation_3_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "-3.png")
        annotation_4_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "-4.png")
        annotation_5_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "-5.png")
        annotation_6_pillow = Image.open("/data/RIGA_ROI" + device_type + "/image" + str(i) + "-6.png")

        annotation_1 = np.asarray(annotation_1_pillow).copy()
        annotation_2 = np.asarray(annotation_2_pillow).copy()
        annotation_3 = np.asarray(annotation_3_pillow).copy()
        annotation_4 = np.asarray(annotation_4_pillow).copy()
        annotation_5 = np.asarray(annotation_5_pillow).copy()
        annotation_6 = np.asarray(annotation_6_pillow).copy()

        annotation_1_pillow.close()
        annotation_2_pillow.close()
        annotation_3_pillow.close()
        annotation_4_pillow.close()
        annotation_5_pillow.close()
        annotation_6_pillow.close()

        annotation_1 = convert_to_gt(annotation_1)
        annotation_2 = convert_to_gt(annotation_2)
        annotation_3 = convert_to_gt(annotation_3)
        annotation_4 = convert_to_gt(annotation_4)
        annotation_5 = convert_to_gt(annotation_5)
        annotation_6 = convert_to_gt(annotation_6)

        annotations_1.append(annotation_1)
        annotations_2.append(annotation_2)
        annotations_3.append(annotation_3)
        annotations_4.append(annotation_4)
        annotations_5.append(annotation_5)
        annotations_6.append(annotation_6)

        write_buffer += 1
    # write remaining images to hdf5 if images list still contains images
    if images:
        counter_to = counter_from + write_buffer
        write_range_to_hdf5(images, counter_from, counter_to, img_data,vendor_data, device_type, annotations,
                            annotations_1, annotations_2, annotations_3, annotations_4, annotations_5, annotations_6)

def convert_to_hdf5(input_file, output_file):
    """ main function ot convert the directory of images to a hdf5 file.
                Parameters:
                ---------------
                input_file   root directory
    """
    hdf5_file = h5py.File(output_file, "w")
    # ---- train val split ----
    # compute indices for train, val (80/20)
    MESSIDOR_ind_shuffled = list(range(1, 461))
    random.shuffle(MESSIDOR_ind_shuffled)
    MESSIDOR_train, MESSIDOR_val = np.split(np.asarray(MESSIDOR_ind_shuffled), [int(len(MESSIDOR_ind_shuffled) * 0.8)])
    MESSIDOR_train = MESSIDOR_train.tolist()
    MESSIDOR_val = MESSIDOR_val.tolist()

    B1_ind_shuffled = list(range(1, 51))
    random.shuffle(B1_ind_shuffled)
    B1_train, B1_val = np.split(np.asarray(B1_ind_shuffled), [int(len(B1_ind_shuffled) * 0.8)])
    B1_train = B1_train.tolist()
    B1_val = B1_val.tolist()

    B2_ind_shuffled = list(range(1, 48))
    random.shuffle(B2_ind_shuffled)
    B2_train, B2_val = np.split(np.asarray(B2_ind_shuffled), [int(len(B2_ind_shuffled) * 0.8)])
    B2_train = B2_train.tolist()
    B2_val = B2_val.tolist()

    B3_ind_shuffled = list(range(1, 48))
    random.shuffle(B3_ind_shuffled)
    B3_train, B3_val = np.split(np.asarray(B3_ind_shuffled), [int(len(B3_ind_shuffled) * 0.8)])
    B3_train = B3_train.tolist()
    B3_val = B3_val.tolist()

    B4_ind_shuffled = list(range(1, 48))
    random.shuffle(B4_ind_shuffled)
    B4_train, B4_val = np.split(np.asarray(B4_ind_shuffled), [int(len(B4_ind_shuffled) * 0.8)])
    B4_train = B4_train.tolist()
    B4_val = B4_val.tolist()

    MagrabiaF_ind_shuffled = list(range(1, 48))
    random.shuffle(MagrabiaF_ind_shuffled)
    MagrabiaF_train, MagrabiaF_val = np.split(np.asarray(MagrabiaF_ind_shuffled),
                                              [int(len(MagrabiaF_ind_shuffled) * 0.8)])
    MagrabiaF_train = MagrabiaF_train.tolist()
    MagrabiaF_val = MagrabiaF_val.tolist()

    MagrabiaM_ind_shuffled = list(range(1, 48))
    random.shuffle(MagrabiaM_ind_shuffled)
    MagrabiaM_train, MagrabiaM_val = np.split(np.asarray(MagrabiaM_ind_shuffled),
                                              [int(len(MagrabiaM_ind_shuffled) * 0.8)])
    MagrabiaM_train = MagrabiaM_train.tolist()
    MagrabiaM_val = MagrabiaM_val.tolist()

    # number of images in each set
    number_train_images = len(MESSIDOR_train) + len(B1_train) + len(B2_train) + len(B3_train) + len(B4_train)\
                          + len(MagrabiaM_train) + len(MagrabiaF_train)
    number_val_images = len(MESSIDOR_val) + len(B1_val) + len(B2_val) + len(B3_val) + len(B4_val)\
                          + len(MagrabiaM_val) + len(MagrabiaF_val)

    img_data_train = hdf5_file.create_dataset("train/images",shape=(number_train_images, 320, 320, 3), dtype="uint8")  # resize size = 320
    img_data_val = hdf5_file.create_dataset("val/images", shape=(number_val_images, 320, 320, 3), dtype="uint8")  # resize size = 320

    vendor_data_train = hdf5_file.create_dataset("train/vendor", shape=(number_train_images,), dtype="int")
    vendor_data_val = hdf5_file.create_dataset("val/vendor", shape=(number_val_images,), dtype="int")

    annotations_train = hdf5_file.create_dataset("train/annotations", shape=(number_train_images, 6, 320, 320), dtype="uint8")  # resize size = 320
    annotations_val = hdf5_file.create_dataset("val/annotations", shape=(number_val_images, 6, 320, 320), dtype="uint8")

    # ---- train ----
    add_images(input_file, MESSIDOR_train, "/MESSIDOR", img_data_train, vendor_data_train, annotations_train, counter_from=0)
    print("Finished MESSIDOR train")
    add_images(input_file, B1_train, "/BinRushed/BinRushed1-Corrected", img_data_train, vendor_data_train,
               annotations_train, counter_from=len(MESSIDOR_train))
    print("Finished BinRushed1-Corrected train")
    add_images(input_file, B2_train, "/BinRushed/BinRushed2", img_data_train, vendor_data_train, annotations_train,
               counter_from=len(MESSIDOR_train)+len(B1_train))
    print("Finished BinRushed2 train")
    add_images(input_file, B3_train, "/BinRushed/BinRushed3", img_data_train, vendor_data_train, annotations_train,
               counter_from=len(MESSIDOR_train)+len(B1_train)+len(B2_train))
    print("Finished BinRushed3 train")
    add_images(input_file, B4_train, "/BinRushed/BinRushed4", img_data_train, vendor_data_train, annotations_train,
               counter_from=len(MESSIDOR_train)+len(B1_train)+len(B2_train)+len(B3_train))
    print("Finished BinRushed4 train")
    add_images(input_file, MagrabiaF_train, "/Magrabia/MagrabiaFemale", img_data_train, vendor_data_train, annotations_train,
               counter_from=len(MESSIDOR_train)+len(B1_train)+len(B2_train)+len(B3_train)+len(B4_train))
    print("Finished MagrabiFemale train")
    add_images(input_file, MagrabiaM_train, "/Magrabia/MagrabiaMale", img_data_train, vendor_data_train, annotations_train,
               counter_from=len(MESSIDOR_train)+len(B1_train)+len(B2_train)+len(B3_train)+len(B4_train)+len(MagrabiaF_train))
    print("Finished MagrabiMale train")

    # ---- val ----
    add_images(input_file, MESSIDOR_val, "/MESSIDOR", img_data_val, vendor_data_val, annotations_val,
               counter_from=0)
    print("Finished MESSIDOR val")
    add_images(input_file, B1_val, "/BinRushed/BinRushed1-Corrected", img_data_val, vendor_data_val,
               annotations_val, counter_from=len(MESSIDOR_val))
    print("Finished BinRushed1-Corrected val")
    add_images(input_file, B2_val, "/BinRushed/BinRushed2", img_data_val, vendor_data_val, annotations_val,
               counter_from=len(MESSIDOR_val) + len(B1_val))
    print("Finished BinRushed2 val")
    add_images(input_file, B3_val, "/BinRushed/BinRushed3", img_data_val, vendor_data_val, annotations_val,
               counter_from=len(MESSIDOR_val) + len(B1_val) + len(B2_val))
    print("Finished BinRushed3 val")
    add_images(input_file, B4_val, "/BinRushed/BinRushed4", img_data_val, vendor_data_val, annotations_val,
               counter_from=len(MESSIDOR_val) + len(B1_val) + len(B2_val) + len(B3_val))
    print("Finished BinRushed4 val")
    add_images(input_file, MagrabiaF_val, "/Magrabia/MagrabiaFemale", img_data_val, vendor_data_val,
               annotations_val,
               counter_from=len(MESSIDOR_val) + len(B1_val) + len(B2_val) + len(B3_val) + len(B4_val))
    print("Finished MagrabiaFemale val")
    add_images(input_file, MagrabiaM_val, "/Magrabia/MagrabiaMale", img_data_val, vendor_data_val,
               annotations_val,
               counter_from=len(MESSIDOR_val) + len(B1_val) + len(B2_val) + len(B3_val) + len(B4_val) + len(
                   MagrabiaF_val))
    print("Finished MagrabiMale val")

    # ---- calculate CDR ----
    # vertical and area
    # go through all train/val annotations
    for type in ['train', 'val']:
        annotations = hdf5_file['train/annotations']
        vert_cdrs = []
        area_cdrs = []
        for i in range(annotations.shape[0]):
            vert_cdr = []
            area_cdr = []
            # for each expert
            for expert in range(6):
                annotation = annotations[i, expert, :, :]

                # calculate vertical CDR for each image
                temp = np.copy(annotation)
                temp[temp == 2] = 1
                vertical_disk = np.sum(temp, axis=0).max()
                temp = np.copy(annotation)
                temp[temp == 1] = 0
                vertical_cup = np.sum(temp, axis=0).max() / 2
                if vertical_disk == 0:
                    print("Error in Vertical CDR for " + type + ", image " + str(i) + ", expert " + str(expert))
                    vert_cdr.append(0)
                else:
                    ratio_vert = vertical_cup / vertical_disk
                    vert_cdr.append(ratio_vert)

                # calculate area CDR for each image
                area_disk = np.count_nonzero(annotation != 0)
                area_cup = np.count_nonzero(annotation == 2)
                if area_disk == 0:
                    print("Error in Area CDR for " + type + ", image " + str(i) + ", expert " + str(expert))
                    area_cdr.append(0)
                else:
                    ratio_area = area_cup / area_disk
                    area_cdr.append(ratio_area)

            vert_cdrs.append(vert_cdr)
            area_cdrs.append(area_cdr)

        hdf5_file.create_dataset(type + "/vertical_CDR", data=np.asarray(vert_cdrs, dtype="f"))
        print("Finished vertical CDR")
        hdf5_file.create_dataset(type + "/area_CDR", data=np.asarray(area_cdrs, dtype="f"))
        print("Finished area CDR")
    hdf5_file.close()

if __name__ == '__main__':
    resize_size = (320, 320)
    convert_to_hdf5(input_file='',  # add root directory where data folder is located
                    output_file='')  # add path and name of h5 file