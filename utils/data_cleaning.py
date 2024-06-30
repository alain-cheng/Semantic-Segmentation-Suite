from PIL import Image
import cv2
from glob import glob
import numpy as np
from natsort import natsorted
import os
import argparse


def snap_colors(im, targets, thresh):
    im_reshaped = im.reshape(-1, 3)
    dists = np.linalg.norm(im_reshaped[:, np.newaxis] - targets, axis=2)
    closest = targets[np.argmin(dists, axis=1)]

    # Turn colors black if outside threshold
    im_reshaped[np.min(dists, axis=1) >= thresh] = [0, 0, 0]

    return closest.reshape(im.shape)

target = {
    #BGR Format
    (64, 0, 192): 'StegaStamp', 
    (128, 192, 64): 'Normal',
    (0, 0, 0): 'Unlabelled'
}

parser = argparse.ArgumentParser()
parser.add_argument('--dirty_dir', type=str, default=None, help='Directory containing the dirty labels')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save cleaned labels')
args = parser.parse_args()

# Load all images from directory
lab_files_list = natsorted(glob(os.path.join(args.dirty_dir, '*.png')))
target_array = np.array(list((target.keys())))

# Threshold determines which colors to consider the same color.
thresh = 5

# Iterate over each image file
for lab_file in lab_files_list[:]:
    im = cv2.imread(lab_file)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    im = snap_colors(im, target_array, thresh)
    
    save_name =  lab_file.split('/')[-1].split('\\')[-1]

    cv2.imwrite(args.save_dir + '/' + save_name, im)
    print("Cleaned " + args.save_dir + '/' + save_name)

#####################
### Commands List ###
#####################

# Full Dataset
    # python utils/data_cleaning.py --dirty_dir "StegaStampV1/test_labels" --save_dir "StegaStampV1/test_labels"
    # python utils/data_cleaning.py --dirty_dir "StegaStampV1/val_labels" --save_dir "StegaStampV1/val_labels"
    # python utils/data_cleaning.py --dirty_dir "StegaStampV1/train_labels" --save_dir "StegaStampV1/train_labels"

# (1/3) Dataset
    