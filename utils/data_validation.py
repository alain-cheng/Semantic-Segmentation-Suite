import os
import cv2
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None, help='Directory of your images to validate')
args = parser.parse_args()

def data_validation(dir):
    for root, _, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            img = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)
            if img is None:
                print(f"Corrupted/Unreadable image: {path}")
                sys.exit("Error")
            else:
                sys.stdout.write(f"\rImage checked: {path}")
                sys.stdout.flush()
                

data_validation(args.dir)

#####################
### Commands List ###
#####################

# python utils/data_validation.py --dir "./synthesized/train"
# python utils/data_validation.py --dir "./synthesized/test"
# python utils/data_validation.py --dir "./synthesized/val"