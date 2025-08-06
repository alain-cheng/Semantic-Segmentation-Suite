import numpy as np
from glob import glob
from natsort import natsorted
import os
import cv2
from PIL import Image

dataset1_imgs = "mixed/test"
dataset1_labels = "mixed/test_labels"
dataset2_imgs = "white-and-brown/test"
dataset2_labels = "white-and-brown/test_labels"

output_dir_imgs1 = "synthesized/val"
output_dir_labels1 = "synthesized/val_labels"
output_dir_imgs2 = "synthesized/train"
output_dir_labels2 = "synthesized/train_labels"

im_files_list1 = natsorted(glob(os.path.join(dataset1_imgs, '*.png')))
label_files_list1 = natsorted(glob(os.path.join(dataset1_labels, '*.png')))

im_files_list2 = natsorted(glob(os.path.join(dataset2_imgs, '*.png')))
label_files_list2 = natsorted(glob(os.path.join(dataset2_labels, '*.png')))

im_files_list = im_files_list1 + im_files_list2
label_files_list = label_files_list1 + label_files_list2

files_list = np.column_stack((im_files_list, label_files_list))

np.random.shuffle(files_list)

for filename, i in zip(files_list[:75], range(25001, 25001 + 75)):
    im = np.array(Image.open(filename[0]))
    label = np.array(Image.open(filename[1]))

    im_save_name = 'im' + f'{i}'
    label_save_name = 'im' + f'{i}'
        
    cv2.imwrite(output_dir_imgs1 + '/' + im_save_name + '.png', im)
    print("Saved " + filename[0] + " to " + output_dir_imgs1 + '/' + im_save_name + '.png')
    
    cv2.imwrite(output_dir_labels1 + '/' + label_save_name + '.png', label)
    print("Saved " + filename[1] + " to " + output_dir_labels1 + '/' + label_save_name + '.png')

for filename, j in zip(files_list[75:150], range(25076, 25076 + 75)):
    im = np.array(Image.open(filename[0]))
    label = np.array(Image.open(filename[1]))

    im_save_name = 'im' + f'{j}'
    label_save_name = 'im' + f'{j}'
        
    cv2.imwrite(output_dir_imgs2 + '/' + im_save_name + '.png', im)
    print("Saved " + filename[0] + " to " + output_dir_imgs2 + '/' + im_save_name + '.png')
    
    cv2.imwrite(output_dir_labels2 + '/' + label_save_name + '.png', label)
    print("Saved " + filename[1] + " to " + output_dir_labels2 + '/' + label_save_name + '.png')