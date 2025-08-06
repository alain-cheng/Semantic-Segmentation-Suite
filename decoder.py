import os, sys, bchlib, glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

from utils import utils

BCH_POLYNOMIAL = 137
BCH_BITS = 5

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--secrets', type=str, required=False)
    args = parser.parse_args()

    if args.image_dir:
        test_input_names = glob.glob(args.image_dir + '/*')
    else:
        train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)
        
        secrets_list = glob.glob(os.path.join(args.secrets, '*'))
        

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    total_correct_bits = 0
    total_num_recoverable_bits = 56 * len(test_input_names)

    for i, filename in enumerate(test_input_names):
        sys.stdout.write("Image %d / %d"%(i, len(test_input_names)))
        sys.stdout.flush()
        
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(400, 400)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        secret = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        packet_binary = "".join([str(int(bit)) for bit in secret[:96]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        packet_binary = "".join([str(bit) for bit in secret[:96]])

        recovered_packet_bits = packet_binary[:56]

        if args.image_dir:
            ground_truth_bits = "01010011011101000110010101100111011000010010000100100001" # Stega!!
            correct_bits = sum(1 for bit1, bit2 in zip(recovered_packet_bits, ground_truth_bits) if bit1 == bit2)
        else:
            with open(secrets_list[i], 'r') as secret_file:
                secret_msg = secret_file.read().strip()
            ground_truth_bits = secret_msg
            correct_bits = sum(1 for bit1, bit2 in zip(recovered_packet_bits, ground_truth_bits) if bit1 == bit2)

        total_correct_bits += correct_bits

    total_accuracy = total_correct_bits / total_num_recoverable_bits

    print("\n\nComplete.")
    print("Total Correct Bits: ", total_correct_bits)
    print("Recoverable Bits: ", total_num_recoverable_bits)
    print("Total Recovery Accuracy: ", total_accuracy)

if __name__ == "__main__":
    main()
