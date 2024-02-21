import h5py
from PIL import Image
import numpy as np
import os.path as osp
import glob
import pandas as pd
import random

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
            image_ids     list of ids to all imamges
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
        preprocessed_image_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_img', img_id + ".png"))
        preprocessed_image = np.asarray(preprocessed_image_pillow, dtype="uint8").copy()
        preprocessed_image_pillow.close()
        images.append(preprocessed_image)

        # load preprocessed annotations
        # from expert 1
        preprocessed_annotations1_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_annotation1', img_id + ".png"))
        preprocessed_annotations1 = np.asarray(preprocessed_annotations1_pillow, dtype="uint8").copy()
        preprocessed_annotations1_pillow.close()
        preprocessed_annotations1 = convert_to_gt(preprocessed_annotations1)
        annotations_1.append(preprocessed_annotations1)

        # from expert 2
        preprocessed_annotations2_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_annotation2', img_id + ".png"))
        preprocessed_annotations2 = np.asarray(preprocessed_annotations2_pillow, dtype="uint8").copy()
        preprocessed_annotations2_pillow.close()
        preprocessed_annotations2 = convert_to_gt(preprocessed_annotations2)
        annotations_2.append(preprocessed_annotations2)

        # from expert 3
        preprocessed_annotations3_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_annotation3', img_id + ".png"))
        preprocessed_annotations3 = np.asarray(preprocessed_annotations3_pillow, dtype="uint8").copy()
        preprocessed_annotations3_pillow.close()
        preprocessed_annotations3 = convert_to_gt(preprocessed_annotations3)
        annotations_3.append(preprocessed_annotations3)

        # from expert 4
        preprocessed_annotations4_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_annotation4', img_id + ".png"))
        preprocessed_annotations4 = np.asarray(preprocessed_annotations4_pillow, dtype="uint8").copy()
        preprocessed_annotations4_pillow.close()
        preprocessed_annotations4 = convert_to_gt(preprocessed_annotations4)
        annotations_4.append(preprocessed_annotations4)

        # from expert 5
        preprocessed_annotations5_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_annotation5', img_id + ".png"))
        preprocessed_annotations5 = np.asarray(preprocessed_annotations5_pillow, dtype="uint8").copy()
        preprocessed_annotations5_pillow.close()
        preprocessed_annotations5 = convert_to_gt(preprocessed_annotations5)
        annotations_5.append(preprocessed_annotations5)

        # consensus annotation
        preprocessed_consensus_annotation_pillow = Image.open(osp.join(input_file, '/data/Chaksu_ROI/ROI_consensus_annotations',
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
    id_bosch = ['Image101', 'Image102', 'Image103', 'Image104', 'Image105', 'Image106', 'Image107', 'Image108', 'Image109', 'Image110', 'Image111', 'Image112', 'Image113', 'Image114', 'Image115', 'Image116', 'Image117', 'Image118', 'Image119', 'Image120', 'Image121', 'Image122', 'Image123', 'Image124', 'Image125', 'Image126', 'Image127', 'Image128', 'Image129', 'Image130', 'Image131', 'Image132', 'Image133', 'Image134', 'Image135', 'Image136', 'Image137', 'Image138', 'Image139', 'Image140', 'Image141', 'Image142', 'Image143', 'Image144', 'Image146', 'Image147', 'Image148', 'Image149', 'Image150', 'Image151', 'Image152', 'Image153', 'Image154', 'Image155', 'Image156', 'Image157', 'Image158', 'Image159', 'Image160', 'Image161', 'Image163', 'Image164', 'Image165', 'Image166', 'Image167', 'Image168', 'Image169', 'Image171', 'Image172', 'Image173', 'Image175', 'Image176', 'Image177', 'Image178', 'Image179', 'Image180', 'Image181', 'Image182', 'Image186', 'Image187', 'Image188', 'Image189', 'Image190', 'P10_Image1', 'P1_Image1', 'P1_Image2', 'P2_Image1', 'P2_Image2', 'P3_Image1', 'P3_Image2', 'P4_Image1', 'P4_Image2', 'P5_Image1', 'P5_Image2', 'P6_Image1', 'P6_Image2', 'P7_Image1', 'P7_Image2', 'P8_Image1', 'P8_Image2', 'P9_Image1', 'P9_Image2']
    id_forus = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95']
    id_remidio = ['17521', '18499', '35977', '37749', '39636', '46109', '51763', '64744', '66557', '67239', '69707', '70718', '81590', '82021', 'IMG_2431', 'IMG_2432', 'IMG_2433', 'IMG_2434', 'IMG_2438', 'IMG_2439', 'IMG_2440', 'IMG_2443', 'IMG_2444', 'IMG_2445', 'IMG_2446', 'IMG_2447', 'IMG_2448', 'IMG_2449', 'IMG_2450', 'IMG_2451', 'IMG_2452', 'IMG_2453', 'IMG_2454', 'IMG_2456', 'IMG_2457', 'IMG_2458', 'IMG_2459', 'IMG_2460', 'IMG_2463', 'IMG_2464', 'IMG_2465', 'IMG_2466', 'IMG_2467', 'IMG_2468', 'IMG_2469', 'IMG_2470', 'IMG_2471', 'IMG_2472', 'IMG_2475', 'IMG_2476', 'IMG_2477', 'IMG_2478', 'IMG_2479', 'IMG_2480', 'IMG_2481', 'IMG_2482', 'IMG_2483', 'IMG_2484', 'IMG_2485', 'IMG_2486', 'IMG_2487', 'IMG_2488', 'IMG_2489', 'IMG_2490', 'IMG_2491', 'IMG_2492', 'IMG_2493', 'IMG_2494', 'IMG_2495', 'IMG_2496', 'IMG_2497', 'IMG_2498', 'IMG_2499', 'IMG_2500', 'IMG_2501', 'IMG_2502', 'IMG_2503', 'IMG_2504', 'IMG_2505', 'IMG_2506', 'IMG_2507', 'IMG_2508', 'IMG_2509', 'IMG_2510', 'IMG_2511', 'IMG_2512', 'IMG_2516', 'IMG_2519', 'IMG_2520', 'IMG_2521', 'IMG_2522', 'IMG_2523', 'IMG_2524', 'IMG_2525', 'IMG_2526', 'IMG_2527', 'IMG_2528', 'IMG_2529', 'IMG_2530', 'IMG_2531', 'IMG_2532', 'IMG_2533', 'IMG_2534', 'IMG_2535', 'IMG_2536', 'IMG_2537', 'IMG_2538', 'IMG_2539', 'IMG_2540', 'IMG_2541', 'IMG_2542', 'IMG_2543', 'IMG_2544', 'IMG_2545', 'IMG_2546', 'IMG_2547', 'IMG_2548', 'IMG_2549', 'IMG_2550', 'IMG_2551', 'IMG_2552', 'IMG_2553', 'IMG_2554', 'IMG_2555', 'IMG_2556', 'IMG_2557', 'IMG_2558', 'IMG_2559', 'IMG_2560', 'IMG_2561', 'IMG_2562', 'IMG_2563', 'IMG_2564', 'IMG_2565', 'IMG_2566', 'IMG_2567', 'IMG_2568', 'IMG_2569', 'IMG_2570', 'IMG_2571', 'IMG_2572', 'IMG_2573', 'IMG_2574', 'IMG_2575', 'IMG_2576', 'IMG_2577', 'IMG_2578', 'IMG_2579', 'IMG_2580', 'IMG_2581', 'IMG_2582', 'IMG_2583', 'IMG_2584', 'IMG_2585', 'IMG_2586', 'IMG_2587', 'IMG_2588', 'IMG_2589', 'IMG_2590', 'IMG_2591', 'IMG_2592', 'IMG_2593', 'IMG_2594', 'IMG_2597', 'IMG_2598', 'IMG_2599', 'IMG_2600', 'IMG_2601', 'IMG_2604', 'IMG_2605', 'IMG_2606', 'IMG_2607', 'IMG_2608', 'IMG_2610', 'IMG_2611', 'IMG_2612', 'IMG_2613', 'IMG_2614', 'IMG_2615', 'IMG_2616', 'IMG_2617', 'IMG_2618', 'IMG_2619', 'IMG_2621', 'IMG_2622', 'IMG_2623', 'IMG_2626', 'IMG_2627', 'IMG_2628', 'IMG_2629', 'IMG_2630', 'IMG_2631', 'IMG_2632', 'IMG_2633', 'IMG_2634', 'IMG_2635', 'IMG_2636', 'IMG_2637', 'IMG_2638', 'IMG_2639', 'IMG_2640', 'IMG_2641', 'IMG_2642', 'IMG_2643', 'IMG_2645', 'IMG_2646', 'IMG_2647', 'IMG_2648', 'IMG_2649', 'IMG_2650', 'IMG_2652', 'IMG_2657', 'IMG_2658', 'IMG_2659', 'IMG_2660', 'IMG_2662', 'IMG_2663', 'IMG_2664', 'IMG_2665', 'IMG_2666', 'IMG_2667', 'IMG_2668', 'IMG_2669', 'IMG_2670', 'IMG_2671', 'IMG_2672', 'IMG_2673', 'IMG_2674', 'IMG_2677', 'IMG_2678', 'IMG_2679', 'IMG_2680', 'IMG_2681', 'IMG_2682', 'IMG_2683', 'IMG_2684', 'IMG_2685', 'IMG_2686', 'IMG_2687', 'IMG_2688', 'IMG_2689', 'IMG_2690', 'IMG_2692', 'IMG_2695', 'IMG_2696', 'IMG_2698', 'IMG_2699', 'IMG_2700', 'IMG_2701', 'IMG_2702', 'IMG_2703', 'IMG_2704', 'IMG_2705', 'IMG_2706', 'IMG_2707', 'IMG_2708', 'IMG_2709', 'IMG_2710', 'IMG_2711', 'IMG_2712', 'IMG_2713', 'IMG_2714', 'IMG_2715', 'IMG_2716', 'IMG_2717', 'IMG_2718', 'IMG_2719', 'IMG_2720', 'IMG_2721', 'IMG_2722', 'IMG_2723', 'IMG_2724', 'IMG_2725', 'IMG_2726', 'IMG_2727', 'IMG_2728', 'IMG_2730', 'IMG_2731', 'IMG_2732', 'IMG_2734', 'IMG_2735', 'IMG_2736', 'IMG_2737', 'IMG_2738', 'IMG_2739', 'IMG_2740', 'IMG_2741', 'IMG_2742', 'IMG_2743', 'IMG_2744', 'IMG_2745', 'IMG_2746', 'IMG_2747', 'IMG_2748', 'IMG_2749', 'IMG_2750', 'IMG_2751', 'IMG_2752', 'IMG_2753', 'IMG_2754', 'IMG_2755', 'IMG_2756', 'IMG_2757', 'IMG_2758', 'IMG_2759', 'IMG_2760', 'IMG_2761', 'IMG_2762', 'IMG_2763', 'IMG_2764', 'IMG_2769', 'IMG_2770', 'IMG_2773', 'IMG_2774', 'IMG_2775', 'IMG_2776', 'IMG_2777', 'IMG_2778', 'IMG_2779', 'IMG_2780', 'IMG_2781', 'IMG_2782', 'IMG_2783', 'IMG_2784', 'IMG_2785', 'IMG_2786', 'IMG_2788', 'IMG_2789', 'IMG_2790', 'IMG_2791', 'IMG_2792', 'IMG_2793', 'IMG_2794', 'IMG_2795', 'IMG_2796', 'IMG_2797', 'IMG_2798', 'IMG_2799', 'IMG_2800', 'IMG_2801', 'IMG_2802', 'IMG_2804', 'IMG_2805', 'IMG_2806', 'IMG_2807', 'IMG_3153', 'IMG_3154', 'IMG_3155', 'IMG_3156', 'IMG_3157', 'IMG_3158', 'IMG_3159', 'IMG_3160', 'IMG_3161', 'IMG_3162', 'IMG_3163', 'IMG_3164', 'IMG_3165', 'IMG_3166', 'IMG_3167', 'IMG_3168', 'IMG_3169', 'IMG_3170', 'IMG_3173', 'IMG_3174', 'IMG_3175', 'IMG_3176', 'IMG_3177', 'IMG_3178', 'IMG_3179', 'IMG_3180', 'IMG_3182', 'IMG_3183', 'IMG_3184', 'IMG_3185', 'IMG_3186', 'IMG_3188', 'IMG_3189', 'IMG_3190', 'IMG_3191', 'IMG_3192', 'IMG_3193', 'IMG_3194', 'IMG_3195', 'IMG_3196', 'IMG_3197', 'IMG_3198', 'IMG_3199', 'IMG_3202', 'IMG_3203', 'IMG_3204', 'IMG_3205', 'IMG_3206', 'IMG_3208', 'IMG_3209', 'IMG_3210', 'IMG_3211', 'IMG_3212', 'IMG_3213', 'IMG_3214', 'IMG_3215', 'IMG_3216', 'IMG_3217', 'IMG_3218', 'IMG_3219', 'IMG_3220', 'IMG_3221', 'IMG_3222', 'IMG_3223', 'IMG_3225', 'IMG_3226', 'IMG_3227', 'IMG_3228', 'IMG_3229', 'IMG_3230', 'IMG_3231', 'IMG_3232', 'IMG_3233', 'IMG_3234', 'IMG_3235', 'IMG_3236', 'IMG_3237', 'IMG_3238', 'IMG_3239', 'IMG_3240', 'IMG_3241', 'IMG_3242', 'IMG_3243', 'IMG_3244', 'IMG_3245', 'IMG_3246', 'IMG_3247', 'IMG_3248', 'IMG_3249', 'IMG_3250', 'IMG_3251', 'IMG_3252', 'IMG_3253', 'IMG_3254', 'IMG_3255', 'IMG_3256', 'IMG_3257', 'IMG_3258', 'IMG_3261', 'IMG_3262', 'IMG_3263', 'IMG_3264', 'IMG_3265', 'IMG_3266', 'IMG_3267', 'IMG_3268', 'IMG_3270', 'IMG_3271', 'IMG_3272', 'IMG_3274', 'IMG_3275', 'IMG_3276', 'IMG_3277', 'IMG_3278', 'IMG_3282', 'IMG_3283', 'IMG_3285', 'IMG_3286', 'IMG_3287', 'IMG_3288', 'IMG_3292', 'IMG_3293', 'IMG_3294', 'IMG_3295', 'IMG_3296', 'IMG_3297', 'IMG_3298', 'IMG_3299', 'IMG_3302', 'IMG_3303', 'IMG_3304', 'IMG_3305', 'IMG_3306', 'IMG_3307', 'IMG_3308', 'IMG_3309', 'IMG_3310', 'IMG_3312', 'IMG_3313', 'IMG_3314', 'IMG_3315', 'IMG_3316', 'IMG_3317', 'IMG_3318', 'IMG_3319', 'IMG_3320', 'IMG_3321', 'IMG_3322', 'IMG_3323', 'IMG_3324', 'IMG_3326', 'IMG_3327', 'IMG_3328', 'IMG_3331', 'IMG_3332', 'IMG_3335', 'IMG_3336', 'IMG_3337', 'IMG_3338', 'IMG_3339', 'IMG_3340', 'IMG_3341', 'IMG_3342', 'IMG_3345', 'IMG_3346', 'IMG_3347', 'IMG_3348', 'IMG_3349', 'IMG_3350', 'IMG_3351', 'IMG_3352', 'IMG_3353', 'IMG_3354', 'IMG_3355', 'IMG_3356', 'IMG_3357', 'IMG_3358', 'IMG_3359', 'IMG_3360', 'IMG_3361', 'IMG_3362', 'IMG_3363', 'IMG_3364', 'IMG_3365', 'IMG_3366', 'IMG_3367', 'IMG_3368', 'IMG_3369', 'IMG_3370', 'IMG_3371', 'IMG_3372', 'IMG_3373', 'IMG_3374', 'IMG_3375', 'IMG_3376', 'IMG_3377', 'IMG_3378', 'IMG_3379', 'IMG_3380', 'IMG_3381', 'IMG_3382', 'IMG_3383', 'IMG_3384', 'IMG_3385', 'IMG_3386', 'IMG_3387', 'IMG_3388', 'IMG_3389', 'IMG_3390', 'IMG_3391', 'IMG_3392', 'IMG_3393', 'IMG_3394', 'IMG_3395', 'IMG_3396', 'IMG_3397', 'IMG_3398', 'IMG_3399', 'IMG_3400', 'IMG_3401', 'IMG_3403', 'IMG_3404', 'IMG_3405', 'IMG_3406', 'IMG_3407', 'IMG_3411', 'IMG_3412', 'IMG_3413', 'IMG_3414', 'IMG_3415', 'IMG_3416', 'IMG_3417', 'IMG_3418', 'IMG_3419', 'IMG_3420', 'IMG_3421', 'IMG_3422', 'IMG_3423', 'IMG_3424', 'IMG_3425', 'IMG_3426', 'IMG_3427', 'IMG_3428', 'IMG_3429', 'IMG_3430', 'IMG_3431', 'IMG_3432', 'IMG_3434', 'IMG_3435', 'IMG_3436', 'IMG_3437', 'IMG_3438', 'IMG_3439', 'IMG_3440', 'IMG_3441', 'IMG_3442', 'IMG_3443', 'IMG_3444', 'IMG_3445', 'IMG_3446', 'IMG_3447', 'IMG_3448', 'IMG_3449', 'IMG_3450', 'IMG_3451', 'IMG_3452', 'IMG_3453', 'IMG_3454', 'IMG_3455', 'IMG_3456', 'IMG_3457', 'IMG_3458', 'IMG_3459', 'IMG_3460', 'IMG_3461', 'IMG_3462', 'IMG_3463', 'IMG_3464', 'IMG_3465', 'IMG_3466', 'IMG_3467', 'IMG_3468', 'IMG_3469', 'IMG_3470', 'IMG_3471', 'IMG_3472', 'IMG_3473', 'IMG_3474', 'IMG_3475', 'IMG_3476', 'IMG_3477', 'IMG_3478', 'IMG_3479', 'IMG_3480', 'IMG_3481', 'IMG_3482', 'IMG_3483', 'IMG_3484', 'IMG_3485', 'IMG_3486', 'IMG_3487', 'IMG_3488', 'IMG_3489', 'IMG_3490', 'IMG_3491', 'IMG_3492', 'IMG_3493', 'IMG_3494', 'IMG_3495', 'IMG_3496', 'IMG_3497', 'IMG_3498', 'IMG_3499', 'IMG_3500', 'IMG_3502', 'IMG_3503', 'IMG_3504', 'IMG_3505', 'IMG_3506', 'IMG_3507', 'IMG_3508', 'IMG_3509', 'IMG_3510', 'IMG_3511', 'IMG_3512', 'IMG_3513', 'IMG_3516', 'IMG_3517', 'IMG_3518', 'IMG_3519', 'IMG_3520', 'IMG_3521', 'IMG_3523', 'IMG_3524', 'IMG_3525', 'IMG_3526', 'IMG_3527', 'IMG_3528', 'IMG_3529', 'IMG_3530', 'IMG_3531', 'IMG_3532', 'IMG_3533', 'IMG_3534', 'IMG_3535', 'IMG_3536', 'IMG_3537', 'IMG_3538', 'IMG_3539', 'IMG_3540', 'IMG_3541', 'IMG_3542', 'IMG_3543', 'IMG_3544', 'IMG_3545', 'IMG_3546', 'IMG_3547', 'IMG_3548', 'IMG_3549', 'IMG_3550', 'IMG_3555', 'IMG_3556', 'IMG_3557', 'IMG_3558', 'IMG_3560', 'IMG_3561', 'IMG_3562', 'IMG_3563', 'IMG_3564', 'IMG_3567', 'IMG_3568', 'IMG_3569', 'IMG_3570', 'IMG_3571', 'IMG_3572', 'IMG_3573', 'IMG_3574', 'IMG_3575', 'IMG_3576', 'IMG_3577', 'IMG_3579', 'IMG_3580', 'IMG_3581', 'IMG_3582', 'IMG_3583', 'IMG_3584', 'IMG_3585', 'IMG_3586', 'IMG_3587', 'IMG_3588', 'IMG_3589', 'IMG_3590', 'IMG_3591', 'IMG_3592', 'IMG_3593', 'IMG_3594', 'IMG_3595', 'IMG_3596', 'IMG_3597', 'IMG_3600', 'IMG_3603', 'IMG_3604', 'IMG_3605', 'IMG_3606', 'IMG_3619', 'IMG_3622', 'IMG_3625', 'IMG_3626', 'IMG_3631', 'IMG_3632', 'IMG_3633', 'IMG_3634', 'IMG_3635', 'IMG_3636', 'IMG_3637', 'IMG_3638', 'IMG_3639', 'IMG_3640', 'IMG_3641', 'IMG_3642', 'IMG_3643', 'IMG_3644', 'IMG_3646', 'IMG_3650', 'IMG_3654', 'IMG_3655', 'IMG_3656', 'IMG_3657', 'IMG_3658', 'IMG_3659', 'IMG_3662', 'IMG_3663', 'IMG_3664', 'IMG_3665', 'IMG_3666', 'IMG_3667', 'IMG_3668', 'IMG_3669', 'IMG_3670', 'IMG_3671', 'IMG_3672', 'IMG_3673', 'IMG_3674', 'IMG_3675', 'IMG_3676', 'IMG_3677', 'IMG_3678', 'IMG_3679', 'IMG_3681', 'IMG_3687', 'IMG_3688', 'IMG_3689', 'IMG_3690', 'IMG_3691', 'IMG_3692', 'IMG_3693', 'IMG_3694', 'IMG_3702', 'IMG_3703', 'IMG_3704', 'IMG_3705', 'IMG_3706', 'IMG_3707', 'IMG_3708']

    hdf5_file = h5py.File(hdf5_file_path, "w")

    # ---- train val split ----
    # compute indices for train, val (80/20)
    bosch_ind_shuffled = list(range(len(id_bosch)))
    random.shuffle(bosch_ind_shuffled)
    bosch_train, bosch_val = np.split(np.asarray(bosch_ind_shuffled), [int(len(bosch_ind_shuffled) * 0.8)])
    bosch_train = bosch_train.tolist()
    bosch_val = bosch_val.tolist()

    forus_ind_shuffled = list(range(len(id_forus)))
    random.shuffle(forus_ind_shuffled)
    forus_train, forus_val = np.split(np.asarray(forus_ind_shuffled), [int(len(forus_ind_shuffled) * 0.8)])
    forus_train = forus_train.tolist()
    forus_val = forus_val.tolist()

    remidio_ind_shuffled = list(range(len(id_remidio)))
    random.shuffle(remidio_ind_shuffled)
    remidio_train, remidio_val = np.split(np.asarray(remidio_ind_shuffled), [int(len(remidio_ind_shuffled) * 0.8)])
    remidio_train = remidio_train.tolist()
    remidio_val = remidio_val.tolist()

    # number of images in each set
    number_train_images = len(bosch_train) + len(forus_train) + len(remidio_train)
    number_val_images = len(bosch_val) + len(forus_val) + len(remidio_val)

    img_data_train = hdf5_file.create_dataset("train/images",shape=(number_train_images, 320, 320, 3), dtype="uint8")  # resize size = 320
    img_data_val = hdf5_file.create_dataset("val/images", shape=(number_val_images, 320, 320, 3), dtype="uint8")  # resize size = 320

    vendor_data_train = hdf5_file.create_dataset("train/vendor", shape=(number_train_images,), dtype="int")
    vendor_data_val = hdf5_file.create_dataset("val/vendor", shape=(number_val_images,), dtype="int")

    annotations_train = hdf5_file.create_dataset("train/annotations", shape=(number_train_images, 5, 320, 320), dtype="uint8")  # resize size = 320
    annotations_val = hdf5_file.create_dataset("val/annotations", shape=(number_val_images, 5, 320, 320), dtype="uint8")  # resize size = 320

    consensus_train = hdf5_file.create_dataset("train/consensus", shape=(number_train_images, 320, 320), dtype="uint8")
    consensus_val = hdf5_file.create_dataset("val/consensus", shape=(number_val_images, 320, 320), dtype="uint8")

    dt = h5py.special_dtype(vlen=str)
    id_train = hdf5_file.create_dataset("train/id", shape=(number_train_images,), dtype=dt)
    id_val = hdf5_file.create_dataset("val/id", shape=(number_val_images,), dtype=dt)

    # ---- train ----
    add_images(input_file, [id_bosch[i] for i in bosch_train], "Bosch", img_data_train,
               vendor_data_train, annotations_train, counter_from=0, id=id_train,
               consensus_segmentation=consensus_train)
    add_images(input_file, [id_forus[i] for i in forus_train], "Forus", img_data_train,
               vendor_data_train, annotations_train, counter_from=len(bosch_train), id=id_train,
               consensus_segmentation=consensus_train)
    add_images(input_file, [id_remidio[i] for i in remidio_train], "Remidio", img_data_train,
               vendor_data_train, annotations_train, counter_from=len(bosch_train)+len(forus_train), id=id_train,
               consensus_segmentation=consensus_train)

    # ---- val ----
    add_images(input_file, [id_bosch[i] for i in bosch_val], "Bosch", img_data_val, vendor_data_val,
               annotations_val, counter_from=0, id=id_val, consensus_segmentation=consensus_val)
    add_images(input_file, [id_forus[i] for i in forus_val], "Forus", img_data_val, vendor_data_val,
               annotations_val, counter_from=len(bosch_val), id=id_val, consensus_segmentation=consensus_val)
    add_images(input_file, [id_remidio[i] for i in remidio_val], "Remidio", img_data_val,
               vendor_data_val, annotations_val, counter_from=len(bosch_val)+len(forus_val), id=id_val,
               consensus_segmentation=consensus_val)

    # ---- diagnoses ----
    print("Converting diagnoses")
    diagnoses_train = hdf5_file.create_dataset("train/diagnosis", shape=(number_train_images, 5), dtype="i")
    diagnoses_val = hdf5_file.create_dataset("val/diagnosis", shape=(number_val_images, 5), dtype="i")

    for i, expert in enumerate(['Expert_1', 'Expert_2', 'Expert_3', 'Expert_4', 'Expert_5']):
        label_bosch, label_forus, label_remidio = create_labels_dataset('6.0_Glaucoma_Decision/' + expert +
                                                                        '/Bosch.csv', '6.0_Glaucoma_Decision/' +
                                                                        expert + '/Forus.csv',
                                                                        '6.0_Glaucoma_Decision/' + expert +
                                                                        '/Remidio.csv', label_file)

        # train
        expert_labels = [label_bosch[i] for i in bosch_train]
        expert_labels.extend([label_forus[i] for i in forus_train])
        expert_labels.extend([label_remidio[i] for i in remidio_train])
        diagnoses_train[:, i] = np.asarray(expert_labels, dtype='i')
        expert_labels.clear()
        # val
        expert_labels = [label_bosch[i] for i in bosch_val]
        expert_labels.extend([label_forus[i] for i in forus_val])
        expert_labels.extend([label_remidio[i] for i in remidio_val])
        diagnoses_val[:, i] = np.asarray(expert_labels, dtype='i')
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

    # train
    hdf5_file.create_dataset("train/majority_vote", shape=(number_train_images,), dtype="i",
                             data=get_majority_vote(hdf5_file["train/diagnosis"]))
    # val
    hdf5_file.create_dataset("val/majority_vote", shape=(number_val_images,), dtype="i",
                             data=get_majority_vote(hdf5_file["val/diagnosis"]))

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
    create_hdf5_dataset(input_file, hdf5_file_path=output_file, label_file=label_file)

    # ---- calculate CDR ----
    hdf5_file = h5py.File(output_file, "r+")
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

        hdf5_file.create_dataset(type+"/vertical_CDR", data=np.asarray(vert_cdrs, dtype="f"))
        hdf5_file.create_dataset(type + "/area_CDR", data=np.asarray(area_cdrs, dtype="f"))

    for type in ['train', 'val']:
        annotations = hdf5_file[type+'/consensus']
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

        hdf5_file.create_dataset(type + "/consensus_vertical_CDR", data=np.asarray(vert_cdrs, dtype="f"))
        hdf5_file.create_dataset(type + "/consensus_area_CDR", data=np.asarray(area_cdrs, dtype="f"))




if __name__ == '__main__':
    convert_to_hdf5(input_file='',  # add root directory where data folder is located
                    output_file='',  # add path and name of h5 file
                    label_file='')  # add path to 6.0 Glaucoma Decision from original Chaksu dataset
