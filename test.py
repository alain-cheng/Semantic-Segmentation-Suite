import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="0" # if it refuses try switching between 0, 1, 2, or 3
from PIL import Image
from utils import utils, helpers
from builders import model_builder

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="synthesized", required=False, help='The dataset you are using')
parser.add_argument('--brightness', type=float, default=None, required=False, help='Change brightness scale of test images')
parser.add_argument('--out', type=str, default=None, required=False, help='Directory to save outputs')
parser.add_argument('--rescale', type=str2bool, default=False, help='Whether to rescale input sizes')
parser.add_argument('--rescale_size', type=int, default=512, help='Size of rescaled input image to network. Must explicitly set rescale to True.')
parser.add_argument('--noise_level', type=float, default=0, required=False, help='Noise intensity')
parser.add_argument('--compress', type=str2bool, default=False, help='apply JPEG compression before inference')
args = parser.parse_args()

def brightness(input_image):
    # ex: 0.25, 0.5, 1.0 (default), 1.25, 1.5 
    if args.brightness:
        scale = np.array([args.brightness] * 3) 
        input_image = input_image * scale
        input_image = np.clip(input_image, 0, 255).astype(np.uint8)
    return input_image

def gaussian_noise(input_image): #n
    # ex: 0, 0.5, 1.0
    mean = 0
    std = args.noise_level
    noise = np.random.normal(mean, std, input_image.shape).astype(np.uint8)
    input_image = cv2.add(input_image, noise)
    return input_image

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if args.out:
    if not os.path.isdir("%s"%(args.out)):
            os.makedirs("%s"%(args.out))

    target=open("%s/test_scores.csv"%(args.out),'w')
    target.write("test_name, test_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))
    
scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    
    input_image = cv2.cvtColor(cv2.imread(test_input_names[ind], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    input_image = brightness(input_image)

    #input_image = gaussian_noise(input_image)

    if args.compress: # JPEG COMPRESSION
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        result, encimg = cv2.imencode('.jpg', input_image, encode_params)
        input_image = cv2.cvtColor(cv2.imdecode(encimg, cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)

    if args.rescale:
        input_image = Image.fromarray(input_image)
        input_image = input_image.resize((args.rescale_size, args.rescale_size), Image.BILINEAR)
        save_im = np.float32(input_image.copy())
        input_image = np.expand_dims(np.float32(input_image) / 255.0, axis=0)

        gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_height]
        gt = Image.fromarray(gt).resize((args.rescale_size, args.rescale_size), Image.NEAREST)
        gt = np.array(gt)
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
    else:
        input_image = np.expand_dims(np.float32(input_image[:args.crop_height, :args.crop_width]),axis=0)/255.0
        save_im = brightness(utils.load_image(test_input_names[ind])) #
        gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
        
    #print("\nPrint GT: \n", gt)
    # Start Timer
    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    # Stop Timer
    run_times_list.append(time.time()-st)
    
    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    #print("\nOutput Image:\n", output_image)

    accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

    file_name = utils.filepath_to_name(test_input_names[ind])
    
    if args.out:
        target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
        
        for item in class_accuracies:
            target.write(", %f"%(item))
        target.write("\n")

    scores_list.append(accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)
    
    gt = helpers.colour_code_segmentation(gt, label_values)

    if args.out:
        cv2.imwrite("%s/%s_in.png"%(args.out, file_name),cv2.cvtColor(save_im, cv2.COLOR_RGB2BGR))
        cv2.imwrite("%s/%s_pred.png"%(args.out, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite("%s/%s_gt.png"%(args.out, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

if args.out:
    target.close()

avg_score = np.mean(scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)
print("Average test accuracy = ", avg_score)
print("Average per class test accuracies = \n")
for index, item in enumerate(class_avg_scores):
    print("%s = %f" % (class_names_list[index], item))
print("Average precision = ", avg_precision)
print("Average recall = ", avg_recall)
print("Average F1 score = ", avg_f1)
print("Average mean IoU score = ", avg_iou)
print("Average run time = ", avg_time)
