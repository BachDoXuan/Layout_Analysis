#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:35:31 2019

@author: bachdx
"""

#!/usr/bin/env python3
# MODIFY CARND TO SEMANTICALLY SEGMENT PRImA DATASET FOR LAYOUT ANALYSIS
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
#import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
	 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(
			 tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train \
				  your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
	"""
	Load Pretrained VGG Model into TensorFlow.
	:param sess: TensorFlow Session
	:param vgg_path: Path to vgg folder, containing "variables/" and 
		"saved_model.pb"
	:return: Tuple of Tensors from VGG model 
	    (image_input, keep_prob,layer3_out, layer4_out, layer7_out)
	"""
    # TODO: Implement function
	# Use tf.saved_model.loader.load to load the model and weights
	tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
	
	# Get Tensors to be returned from graph
	graph = tf.get_default_graph()
	image_input = graph.get_tensor_by_name('image_input:0')
	keep_prob = graph.get_tensor_by_name('keep_prob:0')
	layer3 = graph.get_tensor_by_name('layer3_out:0')
	layer4 = graph.get_tensor_by_name('layer4_out:0')
	layer7 = graph.get_tensor_by_name('layer7_out:0')
	
	return image_input, keep_prob, layer3, layer4, layer7

#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  
	Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # Use a shorter variable name for simplicity
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # Apply 1x1 convolution in place of fully connected layer ???
	# ??? 1x1 convolution means convoluting through the depth, if image has 3
	# color channels, 1X1 convolution will convolute through color channels
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, 
							name="fcn8")

    # Upsample fcn8 with size depth=(4096?) to match size of layer 4 so that \
	# we can add skip connection with 4th layer
    fcn9 = tf.layers.conv2d_transpose(fcn8, 
									  filters=layer4.get_shape().as_list()[-1],
									  kernel_size=4, strides=(2, 2), 
									  padding='SAME', name="fcn9")

    # Add a skip connection between current final layer fcn8 and 4th layer
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # Upsample again
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, 
									   filters = \
									      layer3.get_shape().as_list()[-1], 
									   kernel_size=4, strides=(2, 2), 
									   padding='SAME', name="fcn10_conv2d")

    # Add skip connection
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # Upsample again
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, 
									   filters=num_classes,
									   kernel_size=16, strides=(8, 8), 
									   padding='SAME', name="fcn11")

    return fcn11
    
#tests.test_layers(layers)

def run():
	NUM_CLASSES = 6
	# We resize PRImA dataset into 320x224 images for our model
	IMAGE_SHAPE = (320, 224)  
	DATA_DIR = './Small_Data'
	TRAIN_DIR = './Small_Data/train/'
	TRAIN_GT_DIR = './Small_Data/train_gt/'
	DEV_DIR = './Small_Data/dev/'
	DEV_GT_DIR = './Small_Data/dev_gt/'
	RUNS_DIR = './runs'
	LOG_DIR = 'logs'
	EPOCHS = 20
	BATCH_SIZE = 2
#	DROPOUT = 0.75
	
	# CLEAR OLD VARIABLES
	tf.reset_default_graph()
	
	# BUILD VARIABLES FOR INPUTS OF COMPUTATION GRAPH MODEL
	correct_label = tf.placeholder(tf.float32, [None, IMAGE_SHAPE[0], 
												IMAGE_SHAPE[1], NUM_CLASSES])
	learning_rate = tf.placeholder(tf.float32)
	keep_prob = tf.placeholder(tf.float32)

	# BUILD SESSION
	sess = tf.Session()
	
	# BUILD COMPUTATION GRAPH MODEL	(named fcn model)
	# Download pretrained vgg model
	helper.maybe_download_pretrained_vgg(DATA_DIR)
	
	# Path to vgg model
	vgg_path = os.path.join(DATA_DIR, 'vgg')
	
	# Create function to generate batches of training data to train model
	get_batches_fn = helper.gen_batch_function(TRAIN_DIR, TRAIN_GT_DIR, 
											 IMAGE_SHAPE, NUM_CLASSES)
	
	# Load the vgg model and weights into tf.session sess and use 
	# image_input, keep_prob, layer3, layer4, and layer7 tensor and operations
	# to build following layers
	image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, 
															vgg_path)	
	# Build layers for computation graph model
	fcn11 = layers(layer3, layer4, layer7, NUM_CLASSES)
	
	# Build loss operation with layer fcn11 and correct label
	logits_op = tf.reshape(fcn11, (-1, NUM_CLASSES), 
					  name="logits_op")
	
#	class_eye_op = tf.eye(NUM_CLASSES, dtype = tf.uint8)
	predict_label_op = tf.argmax(logits_op, axis = 1)
	
	correct_label_reshaped = tf.reshape(correct_label, (-1, NUM_CLASSES))
	
	accuracy_op = tf.equal(predict_label_op, 
						tf.cast(tf.argmax(correct_label_reshaped, axis = 1), 
						  dtype = tf.int64))
	
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits_op, labels=correct_label_reshaped[:])
	loss_op = tf.reduce_mean(cross_entropy, name="loss_op")

	# Build minimize (optimize) operation 
	global_step = tf.Variable(0, trainable=False)
	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).\
					minimize(loss_op,  global_step=global_step, 
							  name="train_op")	
					
	# BUILD SUMMARY OPERATION
	tf.summary.scalar('loss', loss_op)
	summary_op = tf.summary.merge_all()
	# Write sess.graph into log_dir
	summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
	
	# CREATE A SAVER TO SAVE CHECKPOINTS OF TRAINED WEIGHTS   
	saver = tf.train.Saver(tf.trainable_variables())
	
	# Print this notice when done building model
	print("Model build successful, starting training")
	
	# INITIALIZE VARIABLE FOR SESSION
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	# TRAIN MODEL
	for epoch in range(EPOCHS):
		for X_batch, gt_batch in get_batches_fn(BATCH_SIZE):
			loss, _, summary_str, step = sess.run([loss_op, train_op, 
										   summary_op, global_step], 
										   feed_dict = {image_input: X_batch, 
									       correct_label: gt_batch, 
										   keep_prob: 0.5, 
										   learning_rate:0.001})
			print('Step:', step, "Epoch:", epoch + 1, 'Loss:', loss)
		
		# Calculate train accuracy and dev accuracy after each epoch
		train_acc = helper.calculate_accuracy(sess, accuracy_op, keep_prob, 
										image_input, correct_label,
										TRAIN_DIR, TRAIN_GT_DIR, 
										IMAGE_SHAPE, NUM_CLASSES)
		dev_acc = helper.calculate_accuracy(sess, accuracy_op, keep_prob, 
									  image_input, correct_label,
									  DEV_DIR, DEV_GT_DIR, 
									  IMAGE_SHAPE, NUM_CLASSES)
		print("(Epoch", epoch + 1, "/", EPOCHS ,")", "train_acc:", train_acc,\
				"; dev_acc:", dev_acc)
		
		summary_writer.add_summary(summary_str, global_step = step)
	
	
	# ASSESS THE TRAINED MODEL ON DEV DATASET
	helper.save_inference_samples(RUNS_DIR, DEV_DIR, sess, IMAGE_SHAPE, 
							   logits_op, keep_prob, image_input)
	
	print("All done!")


if __name__ == '__main__':
    run()
