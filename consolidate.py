import numpy as np
from glob import glob
from natsort import natsorted
import shutil
import os
import csv

dir_dataset_v1 = "synthesized-v2-1"
dir_dataset_v2 = "synthesized-v2-2"
dir_dataset_v3 = "synthesized-v2-3"
dir_dataset_v4 = "synthesized-v2-4"

dataset_list = [
    dir_dataset_v1,
    dir_dataset_v2,
    dir_dataset_v3,
    dir_dataset_v4
]

# dest
dir_out_dataset = "synthesized-v2-5"
dir_out_train = dir_out_dataset + "/train"
dir_out_train_labels = dir_out_dataset + "/train_labels"
dir_out_test = dir_out_dataset + "/test"
dir_out_test_labels = dir_out_dataset + "/test_labels"
dir_out_val = dir_out_dataset + "/val"
dir_out_val_labels = dir_out_dataset + "/val_labels"

save_dirs = {
    dir_out_train: dir_out_train_labels,
    dir_out_test: dir_out_test_labels,
    dir_out_val: dir_out_val_labels
}

dir_class_dict_csv = dir_out_dataset + '/' + 'class_dict.csv'

class_dict_data = [
    ["name", "r", "g", "b"],
    ["StegaStamp", 255, 255, 255],
    ["Unlabelled", 0, 0, 0]
]

# GLOBAL
cnt = 0 

def create(dataset_dir, limit1=250, limit2=50):
    global cnt
    
    files_gt = natsorted(glob(os.path.join(dataset_dir, 'train/*.png')))
    files_lab = natsorted(glob(os.path.join(dataset_dir, 'train_labels/*.png')))
    train_files_list = np.column_stack((files_gt, files_lab))
    
    files_gt = natsorted(glob(os.path.join(dataset_dir, 'test/*.png')))
    files_lab = natsorted(glob(os.path.join(dataset_dir, 'test_labels/*.png')))
    test_files_list = np.column_stack((files_gt, files_lab))
    
    files_gt = natsorted(glob(os.path.join(dataset_dir, 'val/*.png')))
    files_lab = natsorted(glob(os.path.join(dataset_dir, 'val_labels/*.png')))
    val_files_list = np.column_stack((files_gt, files_lab))

    # Train
    for file_gt, file_lab in train_files_list[:limit1]:
        save_name_gt = 'im' + str(cnt) + '.png'
        save_name_lab = 'im' + str(cnt) + '_L.png'
        print('Copying %s to %s as %s'%(file_gt,dir_out_train,save_name_gt))
        print('Copying %s to %s as %s'%(file_lab,dir_out_train_labels,save_name_lab))
        shutil.copy(file_gt, dir_out_train + '/' + save_name_gt)
        shutil.copy(file_lab, dir_out_train_labels + '/' + save_name_lab)
        cnt += 1
    # Test
    for file_gt, file_lab in test_files_list[:limit2]:
        save_name_gt = 'im' + str(cnt) + '.png'
        save_name_lab = 'im' + str(cnt) + '_L.png'
        print('Copying %s to %s as %s'%(file_gt,dir_out_test,save_name_gt))
        print('Copying %s to %s as %s'%(file_lab,dir_out_test_labels,save_name_lab))
        shutil.copy(file_gt, dir_out_test + '/' + save_name_gt)
        shutil.copy(file_lab, dir_out_test_labels + '/' + save_name_lab)
        cnt += 1
    # Val
    for file_gt, file_lab in val_files_list[:limit2]:
        save_name_gt = 'im' + str(cnt) + '.png'
        save_name_lab = 'im' + str(cnt) + '_L.png'
        print('Copying %s to %s as %s'%(file_gt,dir_out_val,save_name_gt))
        print('Copying %s to %s as %s'%(file_lab,dir_out_val_labels,save_name_lab))
        shutil.copy(file_gt, dir_out_val + '/' + save_name_gt)
        shutil.copy(file_lab, dir_out_val_labels + '/' + save_name_lab)
        cnt += 1


for im_dir, lab_dir in save_dirs.items():
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    if not os.path.exists(lab_dir):
        os.makedirs(lab_dir)

# Generate class_dict.csv
with open(dir_class_dict_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(class_dict_data)

for dataset_dir in dataset_list:
    create(dataset_dir)

print('')