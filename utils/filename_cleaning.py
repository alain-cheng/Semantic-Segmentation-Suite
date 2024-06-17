import cv2
from glob import glob
from natsort import natsorted
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labels_dir', type=str, default=None, help='Directory containing the labels to fix')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save fixed labels')
args = parser.parse_args()

# Load all images from directory
lab_files_list = natsorted(glob(os.path.join(args.labels_dir, '*.png')))

# Iterate over each image file
for lab_file in lab_files_list[:]:
    im = cv2.imread(lab_file)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    save_name =  lab_file.split('/')[-1].split('\\')[-1].split('_')[0]

    cv2.imwrite(args.save_dir + '/' + save_name + '.png', im)
    print("Cleaned " + args.save_dir + '/' + save_name + '.png')


# python utils/filename_cleaning.py --labels_dir "StegaStampV1_sample/test_labels" --save_dir "StegaStampV1_sample/test_labels"
# del StegaStampV1_sample\test_labels\*_L.png
# python utils/filename_cleaning.py --labels_dir "StegaStampV1_sample/train_labels" --save_dir "StegaStampV1_sample/train_labels"
# del StegaStampV1_sample\train_labels\*_L.png
# python utils/filename_cleaning.py --labels_dir "StegaStampV1_sample/val_labels" --save_dir "StegaStampV1_sample/val_labels"
# del StegaStampV1_sample\val_labels\*_L.png