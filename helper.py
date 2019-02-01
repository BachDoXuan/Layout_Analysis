'''
MODIFY THIS HELPER TO TRAIN SEMANTIC SEGMENTATION FOR LAYOUT ANALYSIS


This file is used primarily to download vgg if it has not yet been,
give you the progress of the download, get batches for your training,
as well as around generating and saving the image outputs.
'''

#import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
#from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    """
    Report download progress to the terminal.
    :param tqdm: Information fed to the tqdm library to estimate progress.
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        Store necessary information for tracking progress.
        :param block_num: current block of the download
        :param block_size: size of current block
        :param total_size: total download size, if known
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)  # Updates progress
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
	"""
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
	vgg_filename = 'vgg.zip'
	vgg_path = os.path.join(data_dir, 'vgg')
	vgg_files = [
		os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
		os.path.join(vgg_path, 'variables/variables.index'),
		os.path.join(vgg_path, 'saved_model.pb')]

	missing_vgg_files = [vgg_file for vgg_file in vgg_files 
					  if not os.path.exists(vgg_file)]
	if missing_vgg_files:
		# Clean vgg dir
		if os.path.exists(vgg_path):
			shutil.rmtree(vgg_path)
		os.makedirs(vgg_path)

		# Download vgg
		print('Downloading pre-trained vgg model...')
		with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
			urlretrieve(
				'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/ \
				vgg.zip',
				os.path.join(vgg_path, vgg_filename),
				pbar.hook)

		# Extract vgg
		print('Extracting model...')
		zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
		zip_ref.extractall(data_dir)
		zip_ref.close()

		# Remove zip file to save space
		os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(train_folder, train_gt_folder, image_shape, 
					   num_classes, image_list=None):
	"""
	Generate function to create batches of training data
	Params:
		- train_folder: Path to folder that contains all the train data,
			.tif files
		- train_gt_folder: Path to folder that contains all the groundtruth 
			data, .png files
		- image_shape: Tuple - Shape of image
		- num_classes: int - number of classes
		- image_list: if not None, then it is a subset of images in train_folder
	Return: a function that generates batches of image input and groundtruth 
		input.
	"""
	gt_type = "png"
	
	def get_batches_fn(batch_size):
		"""Create batches of training data.
        :param batch_size: Batch Size
        :return: Batches of training data
        """
		# Grab image and label paths - maybe a subset of images and labels
		if image_list is None:
			image_list = os.listdir(train_folder)

		# Shuffle training data
		random.shuffle(image_list)
		# Loop through batches and grab images, yielding each batch
		for batch_i in range(0, len(image_list), batch_size):
			images = []
			gts = []   # ground-truth for images
			
			for image_file in image_list[batch_i:batch_i+batch_size]:
				gt_file = image_file[:-3] + gt_type
				
				# Re-size to image_shape
				image = scipy.misc.imresize(scipy.misc.imread(
						os.path.join(train_folder, image_file)), image_shape)
				gt = scipy.misc.imresize(scipy.misc.imread(
						os.path.join(train_gt_folder, gt_file)), image_shape)
				
				# Create "one-hot" labels by class
				class_eye = np.eye(num_classes, dtype = np.uint8)
				gt = class_eye[gt, :]
                           
				images.append(image)
				gts.append(gt)
                
			yield np.array(images), np.array(gts)
	return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_input, data_dir, 
					image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_input: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in os.listdir(data_dir):
		image = scipy.misc.imresize(
				scipy.misc.imread(os.path.join(data_dir, image_file)),
				image_shape)

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_input: [image]})
		
#		# Splice out second column (road), reshape output back to image_shape
#		im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], 
#						 image_shape[1])
		# Use numpy.argmax to find out the class with the highest probability 
		# for each pixel, reshape output back to image_shape
		predicted_label = (np.argmax(im_softmax[0], axis=1)).reshape(
				image_shape[0], image_shape[1]) # bug here axis = 0
		
#		# If road softmax > 0.5, prediction is road
#		segmentation = (im_softmax > 0.5).reshape(image_shape[0], 
#				 image_shape[1], 1)
		
#		# Create mask based on segmentation to apply to original image
#		mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
#		mask = scipy.misc.toimage(mask, mode="RGBA")
#		street_im = scipy.misc.toimage(image)
#		street_im.paste(mask, box=None, mask=mask)
#
#		yield os.path.basename(image_file), np.array(street_im)
		
		yield image_file, predicted_label


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, 
						   keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to hard drive
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(sess, logits, keep_prob, input_image, 
								 data_dir, image_shape)
	for name, image in image_outputs:
		scipy.misc.imsave(os.path.join(output_dir, name[:-3] + "png"), 
					image)


def calculate_accuracy(sess, accuracy_op, keep_prob, image_input, input_dir, 
					   input_gt_dir, image_shape, num_classes, 
					   num_samples = None, batch_size = 16):
	"""
	Calculate accuracy (the number of pixels predicted correctly over 
	the number of pixels) and return result
	:param sess: tensorflow sess (containing variables, computation graph,
							   and weights)
	:param logits: tf tensor for logits of the last layer
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_input: TF Placeholder for image input
	:param input_dir: directory for image input
	:param input_gt_dir: directory for image groundtruth input
	:param num_samples: the number of samples to calculate accuracy
	:param batch_size: split input and input_groundtruth into batches of this
		size to avoid using too much memory
	"""
	image_list = os.listdir(input_dir)
	num_images = len(image_list)
	
	# Maybe calculate accuracy over a subset of the input data
	if num_samples is not None and num_samples < num_images:
		mask = np.random.choice(num_images, num_samples)
		num_images = num_samples
		image_list =  image_list[mask]
		
	# Compute predictions in batches
	num_batches = num_images // batch_size
	if num_images % batch_size != 0:
		num_batches += 1
	
	get_batches_fn = gen_batch_function(input_dir, input_gt_dir, image_shape,
									 num_classes, image_list)
	
	label_pred = []
	label_gt = []
	for X_batch, gt_batch in get_batches_fn(batch_size):
		# Run inference
		logits_value = sess.run([tf.nn.softmax(logits_op)], 
						 {keep_prob: 1.0, image_input: X_batch})
		predicted_label = np.argmax(logits_value[0], axis=1)