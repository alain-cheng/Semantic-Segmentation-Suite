import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def ConvBlock(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Builds the conv block for MobileNets
	Apply successivly a 2D convolution, BatchNormalization relu
	"""
	# Skip pointwise by setting num_outputs=Non
	net = slim.conv2d(inputs, n_filters, kernel_size=kernel_size, activation_fn=None)
	net = slim.batch_norm(net, fused=True)
	net = tf.nn.relu(net)
	return net

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3]):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	"""
	net = slim.conv2d_transpose(inputs, n_filters, kernel_size=kernel_size, stride=[2, 2], activation_fn=None)
	net = tf.nn.relu(slim.batch_norm(net))
	return net

def build_unet(inputs, preset_model, num_classes):

	has_skip = False
	if preset_model == "UNet":
		has_skip = False
	elif preset_model == "UNet-Skip":
		has_skip = True
	else:
		raise ValueError("Unsupported UNet model '%s'. This function only supports UNet and UNet-Skip" % (preset_model))

    ####################
	# Contracting path #
	####################
	""" The contracting path follows
        the typical architecture of a convolutional network. It consists of the repeated
        application of two 3x3 convolutions (unpadded convolutions), each followed by
        a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2
        for downsampling. At each downsampling step we double the number of feature
        channels. 
	"""
	net = ConvBlock(inputs, 64, kernel_size=[3, 3])
	net = ConvBlock(net, 64, kernel_size=[3, 3])
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_1 = net

	net = ConvBlock(net, 128, kernel_size=[3, 3])
	net = ConvBlock(net, 128, kernel_size=[3, 3])
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_2 = net

	net = ConvBlock(net, 256, kernel_size=[3, 3])
	net = ConvBlock(net, 256, kernel_size=[3, 3])
	net = ConvBlock(net, 256, kernel_size=[3, 3])
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_3 = net

	net = ConvBlock(net, 512, kernel_size=[3, 3])
	net = ConvBlock(net, 512, kernel_size=[3, 3])
	net = ConvBlock(net, 512, kernel_size=[3, 3])
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')
	skip_4 = net

	net = ConvBlock(net, 1024, kernel_size=[3, 3])
	net = ConvBlock(net, 1024, kernel_size=[3, 3])
	net = ConvBlock(net, 1024, kernel_size=[3, 3])
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	###################
	# Expansive path  #
	###################
	""" Every step in the expansive path consists of an upsampling of the
        feature map followed by a 2x2 convolution (“up-convolution”) that halves the
        number of feature channels, a concatenation with the correspondingly cropped
        feature map from the contracting path, and two 3x3 convolutions, each followed by a ReLU.
    """
	net = conv_transpose_block(net, 1024, kernel_size=[2, 2])
	net = ConvBlock(net, 1024, kernel_size=[3, 3])
	net = ConvBlock(net, 1024, kernel_size=[3, 3])
	net = ConvBlock(net, 512, kernel_size=[3, 3])
	if has_skip:
		net = tf.add(net, skip_4)

	net = conv_transpose_block(net, 512, kernel_size=[2, 2])
	net = ConvBlock(net, 512, kernel_size=[3, 3])
	net = ConvBlock(net, 512, kernel_size=[3, 3])
	net = ConvBlock(net, 256, kernel_size=[3, 3])
	if has_skip:
		net = tf.add(net, skip_3)

	net = conv_transpose_block(net, 256, kernel_size=[2, 2])
	net = ConvBlock(net, 256, kernel_size=[3, 3])
	net = ConvBlock(net, 256, kernel_size=[3, 3])
	net = ConvBlock(net, 128, kernel_size=[3, 3])
	if has_skip:
		net = tf.add(net, skip_2)

	net = conv_transpose_block(net, 128, kernel_size=[2, 2])
	net = ConvBlock(net, 128, kernel_size=[3, 3])
	net = ConvBlock(net, 64, kernel_size=[3, 3])
	if has_skip:
		net = tf.add(net, skip_1)

	net = conv_transpose_block(net, 64, kernel_size=[2, 2])
	net = ConvBlock(net, 64, kernel_size=[3, 3])
	net = ConvBlock(net, 64, kernel_size=[3, 3])

	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
	return net