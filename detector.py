## Compute message recovery percentage
import os,time,cv2, sys, math
import bchlib
import glob
import tensorflow as tf
import argparse
import numpy as np
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from PIL import Image
from natsort import natsorted

from utils import utils
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def get_intersect(p1, p2, p3, p4):
    s = np.vstack([p1,p2,p3,p4])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        print('invalid')
        return (0,0)
    return (x/z, y/z)

def poly_area(poly):
    return 0.5*np.abs(np.dot(poly[:,0],np.roll(poly[:,1],1))-np.dot(poly[:,1],np.roll(poly[:,0],1)))

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector_model', type=str, required=True)
    parser.add_argument('--decoder_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True) # Semantic-Segmentation-Suite/synthesized/test/im12508.png
    parser.add_argument('--secrets', type=str, required=True)
    parser.add_argument('--brightness', type=float, default=None, required=False, help='Change brightness scale of test images')
    #parser.add_argument('--size', type=int, default=512, help='Size of input image to network.')
    parser.add_argument('--out', type=str, default=None, required=False, help='Save directory for predicted secrets')
    args = parser.parse_args()
    
    train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

    test_input_names = natsorted(test_input_names)

    total_accuracy = 0.0
    total_correct_bits = 0
    # total_num_recoverable_bits = 56 * len(test_input_names)
    num_recoverable_images = 0

    if args.secrets is not None:
        secrets_list = natsorted(glob.glob(os.path.join(args.secrets, '*')))

    if args.out is not None:
        if not os.path.exists(args.out):
            os.makedirs(args.out)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    detector_graph = tf.Graph()
    decoder_graph = tf.Graph()

    with detector_graph.as_default():
        detector_sess = tf.Session()
        detector_model = tf.saved_model.loader.load(detector_sess, [tag_constants.SERVING], args.detector_model)

        detector_input_name = detector_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        detector_input = detector_graph.get_tensor_by_name(detector_input_name)

        detector_output_name = detector_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['detections'].name
        detector_output = detector_graph.get_tensor_by_name(detector_output_name)

    with decoder_graph.as_default():
        decoder_sess = tf.Session()
        decoder_model = tf.saved_model.loader.load(decoder_sess, [tag_constants.SERVING], args.decoder_model)

        decoder_input_name = decoder_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        decoder_input = decoder_graph.get_tensor_by_name(decoder_input_name)

        decoder_output_name = decoder_model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
        decoder_output = decoder_graph.get_tensor_by_name(decoder_output_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    
    #print(test_input_names)
    #print(secrets_list, "\n")
    for i, im_file in enumerate(test_input_names):
        #print("Image", i, "/", len(test_input_names), end='\r')
        print("Image", i, "/", len(test_input_names))
        
        frame = cv2.imread(im_file)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if args.brightness:
            scale = np.array([args.brightness] * 3) 
            frame_rgb = frame_rgb * scale
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)
        #if args.size != 512:
        #    detector_image_input = Image.fromarray(frame_rgb).resize((args.size, args.size), Image.BILINEAR) # downscale, 
        #    detector_image_input = detector_image_input.resize((512, 512), Image.BILINEAR) # then re-upscale because the model expects 512x512 inputs.
        #    detector_image_input = np.expand_dims(np.float32(detector_image_input),axis=0)/255.0
        #else:
        detector_image_input = np.expand_dims(np.float32(frame_rgb),axis=0)/255.0

        output_image = detector_sess.run(detector_output,feed_dict={detector_input:detector_image_input})
        output_image = np.array(output_image[0,:,:,:])
        output_image = np.argmax(output_image,axis=-1)

        color_codes = np.array([[255,255,255],[0,0,0]])
        out_vis_image = color_codes[output_image.astype(int)]

        #mask_im = cv2.resize(np.float32(out_vis_image), (512,512))
        mask_im = np.float32(out_vis_image)

        contours, _ = cv2.findContours(cv2.cvtColor(mask_im, cv2.COLOR_BGR2GRAY).astype(np.uint8),1,2)
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
                corners[c,:] = get_intersect(p1, p2, p3, p4)

            new_area = poly_area(corners)
            if new_area / area > 1.5:
                continue

            corners = order_points(corners)
            corners_full_res = corners

            pts_dst = np.array([[0,0],[399,0],[399,399],[0,399]])
            h, status = cv2.findHomography(corners_full_res, pts_dst)
            try:
                warped_im = cv2.warpPerspective(frame_rgb, h, (400,400))
                w_im = warped_im.astype(np.float32)
                w_im /= 255.
            except:
                continue

            #best_num_correct_bits = 0
            #print("File: ", im_file)
            decoded_success_flag = False
            for im_rotation in range(4):
                w_rotated = np.rot90(w_im, im_rotation)
                recovered_secret = decoder_sess.run([decoder_output],feed_dict={decoder_input:[w_rotated]})[0][0]
                recovered_secret = list(recovered_secret)
                recovered_secret = [int(i) for i in recovered_secret]
                
                packet_binary = "".join([str(bit) for bit in recovered_secret[:96]])

                footer = recovered_secret[96:]
                
                recovered_packet_bits = packet_binary[:56]

                packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
                packet = bytearray(packet)
                data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]
                bitflips = bch.decode_inplace(data, ecc)
                
                #print('Num bits corrected: ', bitflips)
                if bitflips != -1:
                    #print('Num bits corrected: ', bitflips)
                    
                    try:
                        pred_code = data.decode("utf-8")
                    except:
                        continue
                
                    # decoded success check
                    if pred_code:
                        with open(secrets_list[i], 'r') as secret_file:
                            secret_msg = secret_file.read().strip()
                        ground_truth_bits = secret_msg
                        correct_bits = sum(1 for bit1, bit2 in zip(recovered_packet_bits, ground_truth_bits) if bit1 == bit2)
                        gt_byte_array = int(ground_truth_bits, 2).to_bytes(7, byteorder='big')
                        gt_code = gt_byte_array.decode('utf-8')
                        
                        print("Ground Truth Message: ", gt_code)
                        print("Predicted Message: ", pred_code, "\n")

                        if args.out:
                            save_name = im_file.split('/')[-1].split('.')[0]
                            gt_msg_path = os.path.join(args.out, save_name + '_gt.txt')
                            with open(gt_msg_path, "w") as file:
                                file.write(gt_code)

                            pred_msg_path = os.path.join(args.out, save_name + '_pred.txt')
                            with open(pred_msg_path, "w") as file:
                                file.write(pred_code)
                        
                        total_correct_bits += correct_bits
                        decoded_success_flag = True
                
            if decoded_success_flag:
                num_recoverable_images += 1
            else:
                with open(secrets_list[i], 'r') as secret_file:
                    secret_msg = secret_file.read().strip()
                ground_truth_bits = secret_msg
                correct_bits = sum(1 for bit1, bit2 in zip(recovered_packet_bits, ground_truth_bits) if bit1 == bit2)
                gt_byte_array = int(ground_truth_bits, 2).to_bytes(7, byteorder='big')
                gt_code = gt_byte_array.decode('utf-8')
                
                print("Ground Truth Message: ", gt_code)
                print("Predicted Message: ", "[RECOVERY FAILED!]", "\n")

                if args.out:
                    save_name = im_file.split('/')[-1].split('.')[0]
                    gt_msg_path = os.path.join(args.out, save_name + '_gt.txt')
                    with open(gt_msg_path, "w") as file:
                        file.write(gt_code)
    
                    pred_msg_path = os.path.join(args.out, save_name + '_pred.txt')
                    with open(pred_msg_path, "w") as file:
                        file.write("[RECOVERY FAILED!]")
                
    total_num_recoverable_bits = num_recoverable_images * 56
                        
    total_accuracy = total_correct_bits / total_num_recoverable_bits
    recovery_rate = num_recoverable_images / len(test_input_names)
    
    print("\n\nComplete.")
    print("Total Correct Bits: ", total_correct_bits)
    print("Recoverable Bits: ", total_num_recoverable_bits)
    print("Recovery Accuracy: ", total_accuracy)
    print("Recovery Rate: ", recovery_rate)

if __name__ == "__main__":
    main()