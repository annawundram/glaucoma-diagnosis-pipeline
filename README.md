# Pipeline Code
Code for "Leveraging Probabilistic Segmentation Models for Improved Glaucoma Diagnosis: A Clinical Pipeline Approach" MIDL 2024.
This clinical pipeline includes a region of interest (ROI) extraction of the original fundus image and a probabilistic segmentation model that segments the optic disc and cup. Followed by a rim thickness extraction that is fed into a classifier that outputs a probability of the fundus image belonging to a glaucoma suspect.

First, the data used in this study must be prepared. Therefore, load the original [Chákṣu](https://www.nature.com/articles/s41597-023-01943-4) dataset (Kumar et al., 2023). Additionally, load the preprocessed data used in this paper from here.

The ROI ectraction step can be skipped, by directly running ```Chaksu_to_h5.py, Chaksu_to_h5_test.py and RIGA_to_h5.py``` consecutively to create H5 files from the data. Alternatively, run ```ROI_from_unet.py``` to create ROI images. Note that this will overwrite previously loaded ROI images.

Use the public implementation of the [PHiSeg Code](https://github.com/baumgach/PHiSeg-code) (Baumgartner et al., 2019) for the PHiSeg, U-Net, probabilistic U-Net and MC Dropout implementation. Please refer to the original repository for further information on how to train the models. Note: Use the dataloader provided in ```data_loader.py```.

