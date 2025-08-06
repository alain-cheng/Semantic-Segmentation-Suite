import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, required=True, help='The model you want to covert to SavedModel format. ')
args = parser.parse_args()

if args.model == "BiSeNet":
    trained_checkpoint_prefix = "BiSeNet/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_bisenet")
elif args.model == "BiSeNet-ResNet50":
    trained_checkpoint_prefix = "BiSeNet-ResNet50/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_bisenet_resnet50")
elif args.model == "BiSeNet-MobileNetV2":
    trained_checkpoint_prefix = "BiSeNet-MobileNetV2/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_bisenet_mobilenetv2")
elif args.model == "MobileBiSeNet":
    trained_checkpoint_prefix = "MobileBiSeNet/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_mobilebisenet")
elif args.model == "MobileBiSeNet-ResNet50":
    trained_checkpoint_prefix = "MobileBiSeNet-ResNet50/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_mobilebisenet_resnet50")
elif args.model == "MobileBiSeNet-MobileNetV2":
    trained_checkpoint_prefix = "MobileBiSeNet-MobileNetV2/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_mobilebisenet_mobilenetv2")
elif args.model == "UNet":
    trained_checkpoint_prefix = "UNet/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_unet")
elif args.model == "MobileUNet":
    trained_checkpoint_prefix = "MobileUNet/checkpoints/latest_model.ckpt"
    export_dir = os.path.join("detector_models", "trained_mobileunet")

graph = tf.compat.v1.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
  # Restore from checkpoint
  loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
  loader.restore(sess, trained_checkpoint_prefix)

  input_tensor = graph.get_tensor_by_name('Placeholder:0') #
  output_tensor = graph.get_tensor_by_name('logits/BiasAdd:0') #

  # Define SignatureDef
  signature_def = tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
      inputs={'image': tf.compat.v1.saved_model.utils.build_tensor_info(input_tensor)},
      outputs={'detections': tf.compat.v1.saved_model.utils.build_tensor_info(output_tensor)},
      method_name=signature_constants.PREDICT_METHOD_NAME
  )

  # Export checkpoint to SavedModel
  builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
  builder.add_meta_graph_and_variables(sess,
                                       [tag_constants.SERVING],
                                       signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def},
                                       strip_default_attrs=True)
  builder.save()