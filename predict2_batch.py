import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
from itertools import zip_longest
from datetime import date

from utils import utils, helpers
from builders import model_builder

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default=None, required=False, help='The image you want to predict on. ')
parser.add_argument('--image_root_dir', type=str, default='', required=False, help='Root Directory of the image_dir')
parser.add_argument('--image_dir', type=str, default='*', required=False, help='Image directory we want to predict.')
parser.add_argument('--checkpoint_paths', nargs='*', type=str, default=None, required=False, help='The path(s) to the latest checkpoint weights for your model(s).')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--models', nargs='*', type=str, default=None, required=True, help='The models you are using. MUST BE SAME ORDER AS CHECKPOINT PATHS. Put StegaStamp last because it has no checkpoint_path.')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--save_dir', type=str, default="Predictions", required=False, help='Save directory')


args = parser.parse_args()

###
# Initialize the model
#
# Returns:
# - num_classes
# - sess
# - net_input
# - network
# - label_values
###
def __init_model__(model, checkpoint_path):
    class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
    
    num_classes = len(label_values)

    # Initializing network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Initializing netowork - StegaStamp
    if model == 'StegaStamp':
        detector_graph = tf.Graph()
    
        with detector_graph.as_default():
            sess = tf.Session()
            detector_model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], 'StegaStamp/detector_models/stegastamp_detector')
        
            detector_input_name = detector_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
            net_input = detector_graph.get_tensor_by_name(detector_input_name)
        
            detector_output_name = detector_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['detections'].name
            network = detector_graph.get_tensor_by_name(detector_output_name)
    
    # Initializing netowork - Semantic Segmentation Models
    else:
        sess=tf.Session(config=config)
        net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
        network, _ = model_builder.build_model(
            model, 
            net_input=net_input,
            num_classes=num_classes,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            is_training=True)
    
        sess.run(tf.global_variables_initializer())
    
        if checkpoint_path is not None:
            print('Loading model checkpoint weights')
            saver=tf.train.Saver(max_to_keep=1000)
            saver.restore(sess, checkpoint_path)


    return num_classes, sess, net_input, network, label_values


###
# Initialize the input files
###
def __init_files__():
    if args.image is not None:
        files_list = [args.image]
    elif args.image_dir is not None:
        files_list = glob(os.path.join(args.image_root_dir, args.image_dir))
    else:
        print('Missing input image')
        sys.exit()
    
    print("\n\n**************************")
    print("Detected files:")
    print(files_list)
    print("**************************\n\n")

    return files_list


###
# Run detector network
#
# Param model:
# - sess
# - net_input
# - network
# - label_values
#
# Returns:
# - Output mask
# - Input image
###
def __run_detector__(file, model):
    sess, net_input, network, label_values = model
    
    print("Testing image " + file)
        
    loaded_image = utils.load_image(file)
    resized_image =cv2.resize(loaded_image, (args.crop_width, args.crop_height))
    input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
    
    # st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    
    # run_time = time.time()-st
    
    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    # return np.uint8(output_image)

    out_mask = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
    input_image = cv2.cvtColor(np.uint8(resized_image), cv2.COLOR_RGB2BGR)

    return out_mask, input_image

    # return resized_image


def __draw_contours__(mask, out_images):
    contours, _ = cv2.findContours(cv2.cvtColor(np.float32(mask), cv2.COLOR_BGR2GRAY).astype(np.uint8),1,2)
    extrema = np.zeros((8,2))
    corners = np.zeros((4,2))
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        hull = cv2.convexHull(cnt)
        if len(hull) < 4:
            continue

        extrema[0,:] = hull[np.argmax(hull[:,0,0]),0,:]
        extrema[1,:] = hull[np.argmax(hull[:,0,0]+hull[:,0,1]),0,:]
        extrema[2,:] = hull[np.argmax(hull[:,0,1]),0,:]
        extrema[3,:] = hull[np.argmax(-hull[:,0,0]+hull[:,0,1]),0,:]
        extrema[4,:] = hull[np.argmax(-hull[:,0,0]),0,:]
        extrema[5,:] = hull[np.argmax(-hull[:,0,0]-hull[:,0,1]),0,:]
        extrema[6,:] = hull[np.argmax(-hull[:,0,1]),0,:]
        extrema[7,:] = hull[np.argmax(hull[:,0,0]-hull[:,0,1]),0,:]

        extrema_lines = extrema - np.roll(extrema, shift=1, axis=0)
        extrema_len = extrema_lines[:,0]**2 + extrema_lines[:,1]**2
        line_idx = np.sort(extrema_len.argsort()[-4:])
        for c in range(4):
            p1 = extrema[line_idx[(c-1)%4],:]
            p2 = extrema[(line_idx[(c-1)%4]-1)%8,:]
            p3 = extrema[line_idx[c],:]
            p4 = extrema[(line_idx[c]-1)%8,:]
            corners[c,:] = utils.get_intersect(p1, p2, p3, p4)

        new_area = utils.poly_area(corners)
        if new_area / area > 1.5:
            continue

        corners = utils.order_points(corners)

        color = (100,250,100)

        for out_img in out_images:
            cv2.polylines(out_img, np.int32([corners]), thickness=6, color=color, isClosed=True)

    


###
# Save image
###

## TTTT OOOOO DDDDDD OOOOOO
# TODO!
# Prepend current date MMDDYY to modelname
# -> MMDDYY_modelname
def __write_image__(image, modelname, file, suffix="pred"):

    today = date.today().strftime("%m%d%y")

    # Write image
    if args.image is not None:
        base_path = os.path.join(args.save_dir, today + "_" + modelname)
    else:
        base_path = os.path.join(args.save_dir, today + "_" + modelname) + os.path.dirname(file.replace(args.image_root_dir, ''))
    
    file_name = utils.filepath_to_name(file)

    img_path = "%s/%s_%s.png"%(base_path, file_name, suffix)

    # if not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    if cv2.imwrite(img_path, image):
        print("Wrote image: " + img_path)
    else:
        print("Failed to write image: " + img_path)

def main():

    
    
    print("\n***** Begin prediction *****")
    print("Dataset -->", args.dataset)
    print("Models -->", args.models)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("Image -->", args.image)
    print("Image Directory -->", args.image_dir)
    print("\n\n\n")
    
    files_list = __init_files__()

    for m, c in zip_longest(args.models, args.checkpoint_paths):
        num_classes, *model = __init_model__(m, c)

        for file in files_list:
            out_mask, input_image = __run_detector__(file, model)

            __draw_contours__(out_mask, [out_mask, input_image]);
            __write_image__(out_mask, m, file, "mask");
            __write_image__(input_image, m, file, "image");
        
        
        
    print("Finished!")


if __name__ == "__main__":
    main()