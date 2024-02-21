'''
1. one dataset for all three vendors
2. 1 dataset
        (#, 320, 320, 3) for all prime images (! resizing needed)
        (#, 5, 320, 320) for all binary masks (! convert to 0,1,2)
        (#) Bosch: 0, Forus: 1, Remidio: 2
        (#, 6) vertical CDR
        (#, 6) area CDR
        (#, 6) diagnosis
        (#) majority vote
'''

import h5py
from PIL import Image
import numpy as np
import os.path as osp
import glob
import pandas as pd

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
    return mask

def write_range_to_hdf5(images, counter_from, counter_to, img_data, vendor_data, device_type, annotations,
                        annotations_1, annotations_2, annotations_3, annotations_4, annotations_5, id, ids, consensus_annotations,
                        consensus_segmentation):
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
    if device_type == "Bosch":
        vendor_data[counter_from:counter_to] = np.zeros((len(images), ))  # 0
    elif device_type == "Forus":
        vendor_data[counter_from:counter_to] = np.ones((len(images), ))  # 1
    else:
        vendor_data[counter_from:counter_to] = np.ones((len(images), )) + 1  # 2

    # add annotations
    annotations[counter_from:counter_to, 0] = np.asarray(annotations_1)
    annotations[counter_from:counter_to, 1] = np.asarray(annotations_2)
    annotations[counter_from:counter_to, 2] = np.asarray(annotations_3)
    annotations[counter_from:counter_to, 3] = np.asarray(annotations_4)
    annotations[counter_from:counter_to, 4] = np.asarray(annotations_5)

    # add ids
    dt = h5py.special_dtype(vlen=str)
    id[counter_from:counter_to] = np.asarray(ids, dtype=dt)

    # add consensus annotation
    consensus_segmentation[counter_from:counter_to] = np.asarray(consensus_annotations)


def add_images(input_file, image_ids, device_type, img_data, vendor_data, annotations, counter_from, id,
               consensus_segmentation):
    """ preprocesses images and adds them to .
            Parameters:
            ---------------
            image_paths   list of paths to all imamges
            orig          whether it's an original image and the mask must be used (boolean)
            device_type   Bosch, Forus or Remidio
            img_data      hdf5 dataset
            vendor_data   hdf5 dataset
            annotations   hdf5 dataset
            id            hdf5 dataset
            consensus_segmentation hdf5 dataset
    """
    max_write_buffer = 4
    write_buffer = 0
    images = []
    annotations_1 = []
    annotations_2 = []
    annotations_3 = []
    annotations_4 = []
    annotations_5 = []
    consensus_annotations = []
    ids = []

    # go through all images and write them to hdf5 files in batches
    for i in range(len(image_ids)):
        # write to hdf5 file if write_buffer is full
        if write_buffer >= max_write_buffer:
            # write images to hdf5 file
            counter_to = counter_from + write_buffer
            print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
            write_range_to_hdf5(images, counter_from, counter_to, img_data, vendor_data, device_type, annotations,
                                annotations_1, annotations_2, annotations_3, annotations_4, annotations_5, id, ids,
                                consensus_annotations, consensus_segmentation)
            # delete cash/lists
            images.clear()
            annotations_1.clear()
            annotations_2.clear()
            annotations_3.clear()
            annotations_4.clear()
            annotations_5.clear()
            ids.clear()
            consensus_annotations.clear()
            # reset stuff for next iteration
            write_buffer = 0
            counter_from += 4


        # get id of image
        img_id = image_ids[i]
        ids.append(img_id)

        # load preprocessed image
        preprocessed_image_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_img', img_id + ".png"))
        preprocessed_image = np.asarray(preprocessed_image_pillow, dtype="uint8").copy()
        preprocessed_image_pillow.close()
        images.append(preprocessed_image)

        # load preprocessed annotations
        # from expert 1
        preprocessed_annotations1_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_annotation1', img_id + ".png"))
        preprocessed_annotations1 = np.asarray(preprocessed_annotations1_pillow, dtype="uint8").copy()
        preprocessed_annotations1_pillow.close()
        preprocessed_annotations1 = convert_to_gt(preprocessed_annotations1)
        annotations_1.append(preprocessed_annotations1)

        # from expert 2
        preprocessed_annotations2_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_annotation2', img_id + ".png"))
        preprocessed_annotations2 = np.asarray(preprocessed_annotations2_pillow, dtype="uint8").copy()
        preprocessed_annotations2_pillow.close()
        preprocessed_annotations2 = convert_to_gt(preprocessed_annotations2)
        annotations_2.append(preprocessed_annotations2)

        # from expert 3
        preprocessed_annotations3_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_annotation3', img_id + ".png"))
        preprocessed_annotations3 = np.asarray(preprocessed_annotations3_pillow, dtype="uint8").copy()
        preprocessed_annotations3_pillow.close()
        preprocessed_annotations3 = convert_to_gt(preprocessed_annotations3)
        annotations_3.append(preprocessed_annotations3)

        # from expert 4
        preprocessed_annotations4_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_annotation4', img_id + ".png"))
        preprocessed_annotations4 = np.asarray(preprocessed_annotations4_pillow, dtype="uint8").copy()
        preprocessed_annotations4_pillow.close()
        preprocessed_annotations4 = convert_to_gt(preprocessed_annotations4)
        annotations_4.append(preprocessed_annotations4)

        # from expert 5
        preprocessed_annotations5_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_annotation5', img_id + ".png"))
        preprocessed_annotations5 = np.asarray(preprocessed_annotations5_pillow, dtype="uint8").copy()
        preprocessed_annotations5_pillow.close()
        preprocessed_annotations5 = convert_to_gt(preprocessed_annotations5)
        annotations_5.append(preprocessed_annotations5)

        # consensus annotation
        preprocessed_consensus_annotation_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI_test/ROI_consensus_annotations',
                                                                       img_id + ".png"))
        preprocessed_consensus_annotation = np.asarray(preprocessed_consensus_annotation_pillow, dtype="uint8").copy()
        preprocessed_consensus_annotation_pillow.close()
        preprocessed_consensus_annotation = convert_to_gt(preprocessed_consensus_annotation)
        consensus_annotations.append(preprocessed_consensus_annotation)

        write_buffer += 1
    # write remaining images to hdf5 if images list still contains images
    if images:
        counter_to = counter_from + write_buffer
        print("writing ids ", ids, " at indices from", counter_from, " to ", counter_to)
        write_range_to_hdf5(images, counter_from, counter_to, img_data,vendor_data, device_type, annotations,
                            annotations_1, annotations_2, annotations_3, annotations_4, annotations_5, id, ids, consensus_annotations,
                            consensus_segmentation)


def create_hdf5_dataset(input_file, hdf5_file_path, label_file):
    """ cerates dataset to add to hdf5 file.
                Parameters:
                ---------------
                input_file   root directory
                orig         whether its an orgiginal image (boolean)
                Returns:
                ----------
                images      filled images array to add ot hdf5 file
    """
    print("Converting images")
    # all ids
    id_bosch = ['P10_Image2', 'P11_Image1', 'P11_Image2', 'P12_Image1', 'P12_Image2', 'P13_Image1', 'P13_Image2', 'P14_Image1', 'P14_Image2', 'P15_Image1', 'P15_Image2', 'P16_Image1', 'P16_Image2', 'P17_Image1', 'P17_Image2', 'P18_Image1', 'P18_Image2', 'P19_Image1', 'P19_Image2', 'P20_Image1', 'P20_Image2', 'P21_Image1', 'P21_Image2', 'P22_Image1', 'P22_Image2', 'P23_Image1', 'P23_Image2', 'P24_Image1', 'P24_Image2', 'P25_Image1', 'P25_Image2', 'P26_Image1', 'P26_Image2', 'P27_Image1', 'P27_Image2', 'P28_Image1', 'P28_Image2', 'P29_Image1', 'P29_Image2', 'P30_Image1', 'P30_Image2']
    id_forus = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '96', '97', '98', '99']
    id_remidio = ['IMG_3709', 'IMG_3710', 'IMG_3711', 'IMG_3713', 'IMG_3714', 'IMG_3719', 'IMG_3720', 'IMG_3727', 'IMG_3728', 'IMG_3729', 'IMG_3730', 'IMG_3735', 'IMG_3740', 'IMG_3743', 'IMG_3744', 'IMG_3745', 'IMG_3746', 'IMG_3759', 'IMG_3760', 'IMG_3763', 'IMG_3766', 'IMG_3768', 'IMG_3770', 'IMG_3791', 'IMG_3793', 'IMG_3797', 'IMG_3798', 'IMG_3799', 'IMG_3800', 'IMG_3811', 'IMG_3812', 'IMG_3814', 'IMG_3815', 'IMG_3816', 'IMG_3818', 'IMG_3820', 'IMG_3823', 'IMG_3832', 'IMG_3833', 'IMG_3836', 'IMG_3837', 'IMG_3841', 'IMG_3844', 'IMG_3849', 'IMG_3852', 'IMG_3853', 'IMG_3854', 'IMG_3857', 'IMG_3859', 'IMG_3881', 'IMG_3882', 'IMG_3887', 'IMG_3888', 'IMG_3890', 'IMG_3891', 'IMG_3893', 'IMG_3894', 'IMG_3897', 'IMG_3898', 'IMG_3900', 'IMG_3903', 'IMG_3905', 'IMG_3907', 'IMG_3910', 'IMG_3911', 'IMG_3912', 'IMG_3913', 'IMG_3915', 'IMG_3919', 'IMG_3921', 'IMG_3922', 'IMG_3926', 'IMG_3929', 'IMG_3930', 'IMG_3936', 'IMG_3938', 'IMG_3939', 'IMG_3942', 'IMG_3944', 'IMG_3945', 'IMG_3949', 'IMG_3950', 'IMG_3952', 'IMG_3954', 'IMG_3955', 'IMG_3961', 'IMG_3962', 'IMG_3969', 'IMG_3971', 'IMG_3973', 'IMG_3974', 'IMG_3975', 'IMG_3977', 'IMG_3978', 'IMG_3981', 'IMG_3983', 'IMG_3984', 'IMG_3986', 'IMG_3987', 'IMG_3988', 'IMG_3989', 'IMG_3994', 'IMG_3995', 'IMG_3997', 'IMG_4002', 'IMG_4003', 'IMG_4004', 'IMG_4024', 'IMG_4025', 'IMG_4032', 'IMG_4034', 'IMG_4043', 'IMG_4046', 'IMG_4048', 'IMG_4049', 'IMG_4058', 'IMG_4059', 'IMG_4066', 'IMG_4070', 'IMG_4078', 'IMG_4079', 'IMG_4099', 'IMG_4100', 'IMG_4110', 'IMG_4111', 'IMG_4112', 'IMG_4114', 'IMG_4129', 'IMG_4131', 'IMG_4132', 'IMG_4133', 'IMG_4144', 'IMG_4145', 'IMG_4150', 'IMG_4153', 'IMG_4154', 'IMG_4155', 'IMG_4166', 'IMG_4167', 'IMG_4169', 'IMG_4170', 'IMG_4176', 'IMG_4177', 'IMG_4191', 'IMG_4192', 'IMG_4198', 'IMG_4201', 'IMG_4202', 'IMG_4203', 'IMG_4228', 'IMG_4239', 'IMG_4240', 'IMG_4244', 'IMG_4245', 'IMG_4253', 'IMG_4254', 'IMG_4255', 'IMG_4256', 'IMG_4257', 'IMG_4258', 'IMG_4396', 'IMG_4397', 'IMG_4398', 'IMG_4399', 'IMG_4400', 'IMG_4401', 'IMG_4402', 'IMG_4403', 'IMG_4404', 'IMG_4405', 'IMG_4407', 'IMG_4408', 'IMG_4409', 'IMG_4410', 'IMG_4411', 'IMG_4412', 'IMG_4413', 'IMG_4414', 'IMG_4415', 'IMG_4416', 'IMG_4418', 'IMG_4420', 'IMG_4421', 'IMG_4422', 'IMG_4423', 'IMG_4424', 'IMG_4425', 'IMG_4426', 'IMG_4427', 'IMG_4432', 'IMG_4433', 'IMG_4434', 'IMG_4435', 'IMG_4436', 'IMG_4437', 'IMG_4438', 'IMG_4439', 'IMG_4440', 'IMG_4441', 'IMG_4442', 'IMG_4443', 'IMG_4444', 'IMG_4445', 'IMG_4446', 'IMG_4447', 'IMG_4448', 'IMG_4449', 'IMG_4451', 'IMG_4452', 'IMG_4453', 'IMG_4454', 'IMG_4455', 'IMG_4456', 'IMG_4457', 'IMG_4458', 'IMG_4459', 'IMG_4461', 'IMG_4462', 'IMG_4464', 'IMG_4465', 'IMG_4466', 'IMG_4467', 'IMG_4468', 'IMG_4469', 'IMG_4470', 'IMG_4471', 'IMG_4472', 'IMG_4473', 'IMG_4474', 'IMG_4475', 'IMG_4476', 'IMG_4477', 'IMG_4478', 'IMG_4480', 'IMG_4481', 'IMG_4482', 'IMG_4483', 'IMG_4485', 'IMG_4486', 'IMG_4488', 'IMG_4489', 'IMG_4491', 'IMG_4492', 'IMG_4493', 'IMG_4494', 'IMG_4495', 'IMG_4497', 'IMG_4501', 'IMG_4502', 'IMG_4505', 'IMG_4506', 'IMG_4507', 'IMG_4518', 'IMG_4520', 'IMG_4542', 'IMG_4543', 'IMG_4544', 'IMG_4545', 'IMG_4546', 'IMG_4547', 'IMG_4558', 'IMG_4559', 'IMG_4560', 'IMG_4561']

    hdf5_file = h5py.File(hdf5_file_path, "w")


    # number of images in each set
    number_images = len(id_bosch) + len(id_forus) + len(id_remidio)

    img_data = hdf5_file.create_dataset("images", shape=(number_images, 320, 320, 3), dtype="uint8")  # resize size = 320

    vendor_data = hdf5_file.create_dataset("vendor", shape=(number_images,), dtype="int")

    annotations = hdf5_file.create_dataset("annotations", shape=(number_images, 5, 320, 320), dtype="uint8")  # resize size = 320

    consensus = hdf5_file.create_dataset("consensus", shape=(number_images, 320, 320), dtype="uint8")

    dt = h5py.special_dtype(vlen=str)
    id = hdf5_file.create_dataset("id", shape=(number_images,), dtype=dt)

    # ---- images ----
    add_images(input_file, id_bosch , "Bosch", img_data,
               vendor_data, annotations, counter_from=0, id=id,
               consensus_segmentation=consensus)
    add_images(input_file, id_forus, "Forus", img_data,
               vendor_data, annotations, counter_from=len(id_bosch), id=id,
               consensus_segmentation=consensus)
    add_images(input_file, id_remidio, "Remidio", img_data,
               vendor_data, annotations, counter_from=len(id_bosch)+len(id_forus), id=id,
               consensus_segmentation=consensus)

    # ---- diagnoses ----
    print("Converting diagnoses")
    diagnoses = hdf5_file.create_dataset("diagnosis", shape=(number_images, 5), dtype="i")

    for i, expert in enumerate(['Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5']):
        label_bosch, label_forus, label_remidio = create_labels_dataset('6.0_Glaucoma_Decision/' + expert +
                                                                        '/Bosch.csv', '6.0_Glaucoma_Decision/' +
                                                                        expert + '/Forus.csv',
                                                                        '6.0_Glaucoma_Decision/' + expert +
                                                                        '/Remidio.csv', label_file)

        expert_labels = [label_bosch[i] for i in list(range(len(id_bosch)))]
        expert_labels.extend([label_forus[i] for i in list(range(len(id_forus)))])
        expert_labels.extend([label_remidio[i] for i in list(range(len(id_remidio)))])
        diagnoses[:, i] = np.asarray(expert_labels, dtype='i')
        expert_labels.clear()

    hdf5_file.close()

    # ---- majority vote ----
    hdf5_file = h5py.File(hdf5_file_path, "r+")
    from scipy.stats import mode
    def get_majority_vote(diagnoses):
        diagnoses_majority_votes = []
        for i in range(diagnoses.shape[0]):
            diagnoses_array = diagnoses[i, :]
            vote = mode(diagnoses_array, keepdims=True)[0][0]
            diagnoses_majority_votes.append(vote)
        return diagnoses_majority_votes

    hdf5_file.create_dataset("majority_vote", shape=(number_images, ), dtype="i", data=get_majority_vote(hdf5_file["diagnosis"]))

    hdf5_file.close()

def create_labels_dataset(file_bosch, file_forus, file_remidio, label_file):
    """ main function ot convert the directory of images to a hdf5 file.
                Parameters:
                ---------------
                file_bosch   file to bosch csv
                file_forus   files to forus csv
                file_remidio files to remidio csv
                input_file   file where dataset is
    """
    # paths to the labels
    path_bosch = osp.join(label_file, file_bosch)
    path_forus = osp.join(label_file, file_forus)
    path_remidio = osp.join(label_file, file_remidio)

    print(path_bosch, path_forus, path_remidio)

    # read entire csv file
    csv_bosch = pd.read_csv(path_bosch)
    csv_forus = pd.read_csv(path_forus)
    csv_remidio = pd.read_csv(path_remidio)

    labels_bosch = []
    labels_forus = []
    labels_remidio = []

    csv_bosch['Glaucoma Decision'].apply(lambda x: labels_bosch.append(0) if x == 'NORMAL' else labels_bosch.append(1))
    csv_forus['Glaucoma Decision'].apply(lambda x: labels_forus.append(0) if x == 'NORMAL' else labels_forus.append(1))
    csv_remidio['Glaucoma Decision'].apply(lambda x: labels_remidio.append(0) if x == 'NORMAL' else labels_remidio.append(1))

    return labels_bosch, labels_forus, labels_remidio


def convert_to_hdf5(input_file, output_file, label_file):
    """ main function ot convert the directory of images to a hdf5 file.
                Parameters:
                ---------------
                input_file   root directory
    """
    create_hdf5_dataset(input_file, hdf5_file_path=output_file, label_file = label_file)

    # ---- calculate CDR ----
    hdf5_file = h5py.File(output_file, "r+")
    # vertical and area
    # go through all annotations
    annotations = hdf5_file['annotations']
    vert_cdrs = []
    area_cdrs = []
    for i in range(annotations.shape[0]):
        vert_cdr = []
        area_cdr = []
        # for each expert
        for expert in range(5):
            annotation = annotations[i, expert, :, :]

            # calculate vertical CDR for each image
            temp = np.copy(annotation)
            temp[temp == 2] = 1
            vertical_disk = np.sum(temp, axis=0).max()
            temp = np.copy(annotation)
            temp[temp == 1] = 0
            vertical_cup = np.sum(temp, axis = 0).max() / 2
            ratio_vert = vertical_cup / vertical_disk
            vert_cdr.append(ratio_vert)

            # calculate area CDR for each image
            area_disk = np.count_nonzero(annotation != 0)
            area_cup = np.count_nonzero(annotation == 2)
            ratio_area = area_cup / area_disk
            area_cdr.append(ratio_area)

        vert_cdrs.append(vert_cdr)
        area_cdrs.append(area_cdr)

    hdf5_file.create_dataset("vertical_CDR", data=np.asarray(vert_cdrs, dtype="f"))
    hdf5_file.create_dataset("area_CDR", data=np.asarray(area_cdrs, dtype="f"))


    annotations = hdf5_file['consensus']
    vert_cdrs = []
    area_cdrs = []
    for i in range(annotations.shape[0]):
        annotation = annotations[i]
        # calculate vertical CDR for each image
        temp = np.copy(annotation)
        temp[temp == 2] = 1
        vertical_disk = np.sum(temp, axis=0).max()
        temp = np.copy(annotation)
        temp[temp == 1] = 0
        vertical_cup = np.sum(temp, axis=0).max() / 2
        ratio_vert = vertical_cup / vertical_disk

        # calculate area CDR for each image
        area_disk = np.count_nonzero(annotation != 0)
        area_cup = np.count_nonzero(annotation == 2)
        ratio_area = area_cup / area_disk

        vert_cdrs.append(ratio_vert)
        area_cdrs.append(ratio_area)

    hdf5_file.create_dataset("consensus_vertical_CDR", data=np.asarray(vert_cdrs, dtype="f"))
    hdf5_file.create_dataset("consensus_area_CDR", data=np.asarray(area_cdrs, dtype="f"))


if __name__ == '__main__':
    convert_to_hdf5(input_file='',  # add root directory where data folder is located
                    output_file='', # add path and name of h5 file
                    label_file = '')  # add path to 6.0 Glaucoma Decision from original Chaksu dataset