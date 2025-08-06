echo "=====Starting Batch Tests====="

# BISENET

python test.py --checkpoint_path "BiSeNet/checkpoints/latest_model.ckpt" --model "BiSeNet" --dataset "datasets/synthesized-2c" --out "Tests/revision2c/bisenet-test-synthetic-2c"

python detector.py --detector_model "../StegaStamp/detector_models/trained_bisenet" --decoder_model "../StegaStamp/saved_models/stegastamp_pretrained" --dataset "datasets/synthesized-2c" --secrets "datasets/2c_test_secrets" --out "Tests/revision2c/bisenet-test-synthetic-2c-secrets"

# BISENET-R50

python test.py --checkpoint_path "BiSeNet-ResNet50/checkpoints/latest_model.ckpt" --model "BiSeNet-ResNet50" --dataset "datasets/synthesized-2c" --out "Tests/revision2c/bisenet-resnet50-test-synthetic-2c"

python detector.py --detector_model "../StegaStamp/detector_models/trained_bisenet_resnet50" --decoder_model "../StegaStamp/saved_models/stegastamp_pretrained" --dataset "datasets/synthesized-2c" --secrets "datasets/2c_test_secrets" --out "Tests/revision2c/bisenet-resnet50-test-synthetic-2c-secrets"

# MOBILEBISENET

python test.py --checkpoint_path "MobileBiSeNet/checkpoints/latest_model.ckpt" --model "MobileBiSeNet" --dataset "datasets/synthesized-2c" --out "Tests/revision2c/mobilebisenet-test-synthetic-2c"

python detector.py --detector_model "../StegaStamp/detector_models/trained_mobilebisenet" --decoder_model "../StegaStamp/saved_models/stegastamp_pretrained" --dataset "datasets/synthesized-2c" --secrets "datasets/2c_test_secrets" --out "Tests/revision2c/mobilebisenet-test-synthetic-2c-secrets"

# MOBILEBISENET-R50

python test.py --checkpoint_path "MobileBiSeNet-ResNet50/checkpoints/latest_model.ckpt" --model "MobileBiSeNet-ResNet50" --dataset "datasets/synthesized-2c" --out "Tests/revision2c/mobilebisenet-resnet50-test-synthetic-2c"

python detector.py --detector_model "../StegaStamp/detector_models/trained_mobilebisenet_resnet50" --decoder_model "../StegaStamp/saved_models/stegastamp_pretrained" --dataset "datasets/synthesized-2c" --secrets "datasets/2c_test_secrets" --out "Tests/revision2c/mobilebisenet-resnet50-test-synthetic-2c-secrets"

# MOBILEUNET

python test.py --checkpoint_path "MobileUNet/checkpoints/latest_model.ckpt" --model "MobileUNet" --dataset "datasets/synthesized-2c" --out "Tests/revision2c/mobileunet-test-synthetic-2c"

python detector.py --detector_model "detector_models/trained_mobileunet" --decoder_model "../StegaStamp/saved_models/stegastamp_pretrained" --dataset "datasets/synthesized-2c" --secrets "datasets/2c_test_secrets" --out "Tests/revision2c/mobileunet-test-synthetic-2c-secrets"

python test.py --checkpoint_path "UNet/checkpoints/latest_model.ckpt" --model "UNet" --dataset "datasets/synthesized-2c" --out "Tests/revision2c/unet-test-synthetic-2c"

python detector.py --detector_model "detector_models/trained_unet" --decoder_model "../StegaStamp/saved_models/stegastamp_pretrained" --dataset "datasets/synthesized-2c" --secrets "datasets/2c_test_secrets" --out "Tests/revision2c/unet-test-synthetic-2c-secrets"

######################################

zip -r revision2c.zip Tests/revision2c

echo "==========Tests Finished=========="