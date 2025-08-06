import os
from natsort import natsorted
from glob import glob

images_dir = "real-life-v2/test"
message_save_dir = "real-life-v2_test_secrets/"

if images_dir is not None:
    files_list = natsorted(glob(os.path.join(images_dir, '*')))

for i in range(len(files_list)):
    save_name = 'im' + str(i + 1)
    msg_path = os.path.join(message_save_dir, save_name + '.txt')
    with open(msg_path, "w") as file:
        secret_bits = "01010011011101000110010101100111011000010010000100100001"
        file.write(secret_bits)