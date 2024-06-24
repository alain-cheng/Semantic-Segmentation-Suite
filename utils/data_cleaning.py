import cv2
from glob import glob
from natsort import natsorted
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', type=str, default=None, help='Directory containing the extensions to fix')
parser.add_argument('--save_dir', type=str, default=None, help='Directory to save fixed extensions')
args = parser.parse_args()

# Load all images from directory
im_files_list = natsorted(glob(os.path.join(args.images_dir, '*.jpg')))

# Iterate over each image file
for im_file in im_files_list[:]:
    im = cv2.imread(im_file)
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    save_name =  im_file.split('/')[-1].split('\\')[-1].split('.')[0]

    cv2.imwrite(args.save_dir + '/' + save_name + '.png', im)
    print("Cleaned " + args.save_dir + '/' + save_name + '.png')

###############################
### Commands List (WINDOWS) ###
###############################

# Full Dataset    
    # python utils/extension_cleaning.py --images_dir "StegaStampV1/test" --save_dir "StegaStampV1/test"
    # del StegaStampV1\test\*.jpg 
    # python utils/extension_cleaning.py --images_dir "StegaStampV1/train" --save_dir "StegaStampV1/train"
    # del StegaStampV1\train\*.jpg
    # python utils/extension_cleaning.py --images_dir "StegaStampV1/val" --save_dir "StegaStampV1/val"
    # del StegaStampV1\val\*.jpg

# (1/3) Dataset    
    # python utils/extension_cleaning.py --images_dir "StegaStampV1_sample/test" --save_dir "StegaStampV1_sample/test"
    # del StegaStampV1_sample\test\*.jpg 
    # python utils/extension_cleaning.py --images_dir "StegaStampV1_sample/train" --save_dir "StegaStampV1_sample/train"
    # del StegaStampV1_sample\train\*.jpg
    # python utils/extension_cleaning.py --images_dir "StegaStampV1_sample/val" --save_dir "StegaStampV1_sample/val"
    # del StegaStampV1_sample\val\*.jpg

#############################
### Commands List (LINUX) ###
#############################

# Full Dataset    
    # python utils/extension_cleaning.py --images_dir "StegaStampV1/test" --save_dir "StegaStampV1/test"
    # find "./StegaStampV1/test" -name "*.jpg" -type f -delete
    # python utils/extension_cleaning.py --images_dir "StegaStampV1/train" --save_dir "StegaStampV1/train"
    # find "./StegaStampV1/train" -name "*.jpg" -type f -delete
    # python utils/extension_cleaning.py --images_dir "StegaStampV1/val" --save_dir "StegaStampV1/val"
    # find "./StegaStampV1/val" -name "*.jpg" -type f -delete

# (1/3) Dataset    
    # python utils/extension_cleaning.py --images_dir "StegaStampV1_sample/test" --save_dir "StegaStampV1_sample/test"
    # find "./StegaStampV1_sample/test" -name "*.jpg" -type f -delete 
    # python utils/extension_cleaning.py --images_dir "StegaStampV1_sample/train" --save_dir "StegaStampV1_sample/train"
    # find "./StegaStampV1_sample/train" -name "*.jpg" -type f -delete 
    # python utils/extension_cleaning.py --images_dir "StegaStampV1_sample/val" --save_dir "StegaStampV1_sample/val"
    # find "./StegaStampV1_sample/val" -name "*.jpg" -type f -delete 