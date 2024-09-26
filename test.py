import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from natsort import natsorted
from glob import glob

from utils import utils, helpers
from builders import model_builder
#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--image_root_dir', type=str, default='', required=False, help='Root Directory of the image_dir')
parser.add_argument('--image_dir', type=str, default='', required=False, help='Image directory we want to predict.')
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--save_dir', type=str, default="Predictions", required=False, help='Save directory')
args = parser.parse_args()

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

num_classes = len(label_values)

print("\n***** Begin prediction *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Classes -->", num_classes)
print("Image -->", args.image)
print("Image Directory -->", args.image_dir)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input,
                                        num_classes=num_classes,
                                        crop_width=args.crop_width,
                                        crop_height=args.crop_height,
                                        is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

if args.image is not None:
    files_list = [args.image]
elif args.image_dir is not None:
    files_list = glob(args.image_root_dir + args.image_dir, recursive=True)
    print("Detected files:")
    print(files_list)
else:
    print('Missing input image')
    sys.exit()    


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for file in files_list:
    print("Testing image " + file)
    
    loaded_image = utils.load_image(file)
    resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
    
    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    
    run_time = time.time()-st
    
    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    base_path = args.save_dir + os.path.dirname(file.replace(args.image_root_dir, ''))
    file_name = utils.filepath_to_name(file)

    img_path = "%s/%s_pred.png"%(base_path,file_name)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    

    if cv2.imwrite(img_path,cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)):
        print("Wrote image: " + img_path)
    else:
        print("Failed to write image: " + img_path)
    
print("Finished!")