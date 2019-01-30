import numpy as np
import cv2
import tensorflow as tf
import os
# from preprocess import create_feed_dict, create_feed_dict_recover, get_datasets, create_eval_feed_dict
from preprocess_recover import create_feed_dict, create_feed_dict_recover, create_feed_dict_recover_eval, get_datasets, create_eval_feed_dict
import math
import random
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

# first_part_path = '../DIQA_Release_1.0_Part1'
# second_part_path = '../DIQA_Release_1.0_Part2/FineReader/'
#training_image_paths = '/Users/liemhd/sources/Recover_DQIA/images'
training_image_paths = '/home/recover/results'
#validation_image_paths = '/home/recover/recover/images'
#eval_image_paths = '/home/recover/recover/images'
training_eval_paths= '/home/recover/results'
validation_eval_paths= '/home/recover/results'
annotation_file_train='./results.csv'
annotation_file_validation='./results.csv'
#IMAGE_SIZE = 48
num_epoch = 10000
batch_size = 10

def conv_bn_relu(current, number, in_channels, out_channels, is_training, init):
    filters = tf.get_variable(name='conv' + str(number) + '_' + 'W',
            initializer=init, shape=(5, 5, in_channels, out_channels))
    bias = tf.get_variable(name='conv' + str(number) + '_' + 'b',
            initializer=init, shape=(out_channels))
    current = tf.nn.bias_add(tf.nn.conv2d(current, filters, strides=[1, 1, 1, 1], padding="VALID"), bias)

    batch_mean, batch_var = tf.nn.moments(current, [0, 1, 2], name='batch_moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    offset = tf.Variable(tf.constant(0.0, shape=[out_channels]), name='offset' + str(number), trainable=True)
    scale = tf.Variable(tf.constant(1.0, shape=[out_channels]), name='scale' + str(number), trainable=True)
    current = tf.nn.batch_normalization(current, mean, variance, offset, scale, 1e-5)
    # current = tf.nn.relu(current)
    return current

def forward_recover(image_patches, batch_size, sess, fc1, image_placeholder, is_training, keep_prob):
    nr_of_examples = len(image_patches)
    print(len(image_patches))
    nr_of_batches = math.ceil(nr_of_examples / batch_size)
    patch_scores = np.zeros((nr_of_examples, 8) )
    batch_index = -1
    for batch_index in range(nr_of_batches - 1):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        # print(image_patches)
        fc1_ = sess.run(fc1, feed_dict={image_placeholder: image_patches[start_index:end_index], is_training: False, keep_prob: 1.})
        print(fc1_)
        patch_scores[start_index:end_index] = fc1_
    batch_index += 1
    start_index = batch_index * batch_size
    fc1_ = sess.run(fc1, feed_dict={image_placeholder: image_patches[start_index:], is_training: False, keep_prob: 1.})
    patch_scores[start_index:] = fc1_
    return np.mean(patch_scores)

def forward(image_patches, batch_size, sess, fc3, image_placeholder, is_training, keep_prob):
    nr_of_examples = len(image_patches)
    nr_of_batches = math.ceil(nr_of_examples / batch_size)
    patch_scores = np.zeros(nr_of_examples)
    batch_index = -1
    for batch_index in range(nr_of_batches - 1):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        fc3_ = sess.run(fc3, feed_dict={image_placeholder: image_patches[start_index:end_index], is_training: False, keep_prob: 1.})
        patch_scores[start_index:end_index] = fc3_
    batch_index += 1
    start_index = batch_index * batch_size
    fc3_ = sess.run(fc3, feed_dict={image_placeholder: image_patches[start_index:], is_training: False, keep_prob: 1.})
    patch_scores[start_index:] = fc3_
    return np.mean(patch_scores)

def conv_bn_relu_no_exponential(current, number,filter_size, in_channels, out_channels, is_training, init):
    filters = tf.get_variable(name='conv' + str(number) + '_' + 'W',
            initializer=init, shape=(filter_size, filter_size, in_channels, out_channels))
    bias = tf.get_variable(name='conv' + str(number) + '_' + 'b',
            initializer=init, shape=(out_channels))
    current = tf.nn.bias_add(tf.nn.conv2d(current, filters, strides=[1, 1, 1, 1], padding="SAME"), bias)
    current = tf.nn.relu(current)
    return current
def conv_relu(current, number, filter_size, in_channels, out_channels,padding_type,strides, is_training, init):
    filters = tf.get_variable(name='conv' + str(number) + '_' + 'W',
            initializer=init, shape=(filter_size, filter_size, in_channels, out_channels))
    bias = tf.get_variable(name='conv' + str(number) + '_' + 'b',
            initializer=init, shape=(out_channels))
    current = tf.nn.bias_add(tf.nn.conv2d(current, filters, strides=[1, strides[0], strides[1], 1], padding=padding_type), bias)
    current = tf.nn.relu(current)
    return current
# def loss_functions(predicted, label):
#     for i in len(predicted):


def main():
    LR = 3e-4
    learning_rate_decay_epochs = 5
    INPUT_SIZE_W= 256 
    INPUT_SIZE_HEIGHT=384
    image_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE_W, INPUT_SIZE_HEIGHT
        , 3), name='image_placeholder')
    label_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 8), name='label_placeholder')
    learning_rate_placeholder = tf.placeholder(dtype=tf.float32, name='learning_rate_placeholder')
    is_training = tf.placeholder(tf.bool, name="is_training")
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

    learning_rate_decay_factor = 0.95

    init = tf.contrib.layers.xavier_initializer()

#   Layer 1
    conv_1 = conv_bn_relu_no_exponential(image_placeholder, 1, 5, 3, 64, is_training, init)
    max_pooled_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
# Layer 2
    conv_2 = conv_bn_relu_no_exponential(max_pooled_1, 2, 5, 64, 128, is_training, init)
#Layer 3
    conv_3 = conv_bn_relu_no_exponential(conv_2, 3, 3, 128, 256, is_training, init)
    max_pooled_3 = tf.nn.max_pool(conv_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#Layer 4
    conv_4 = conv_relu(max_pooled_3, 4, 3, 256, 384,"SAME",[2,2], is_training, init)
#Layer 5
    conv_5 = conv_relu(conv_4, 5, 3, 384, 384,"SAME",[1,1], is_training, init)
    max_pooled_5 = tf.nn.max_pool(conv_5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
#Layer 6
    conv_6 = conv_relu(max_pooled_5, 6, 3, 384, 512,"SAME",[2,2], is_training, init)

#Layer 7
    conv_7 = conv_relu(conv_6, 7, 3, 512, 512,"SAME",[1,1], is_training, init)
    max_pooled_7 = tf.nn.max_pool(conv_7, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
#Layer 8
    conv_8 = conv_relu(max_pooled_7, 8, 3, 512, 1024,"SAME",[2,2], is_training, init)
#Layer 9
    conv_9 = conv_relu(conv_8, 9, 3, 1024, 1024,"SAME",[1,1], is_training, init)
#Layer 10
    conv_10 = conv_relu(conv_9, 10, 3, 1024, 1024,"SAME",[1,1], is_training, init)
#Layer 11
    conv_11 = conv_relu(conv_10, 11, 1, 1024, 2048,"SAME",[1,1], is_training, init)
    keep_prob_final_layer = 0.5
    conv_11 = tf.nn.dropout(conv_11, keep_prob_final_layer)
#Layer Fully connected 
    conv_11 = tf.contrib.layers.flatten(conv_11)
    fc1= tf.contrib.layers.fully_connected(conv_11, 8)
    # print(fc1)
    loss = tf.reduce_mean(tf.abs(tf.subtract(fc1, label_placeholder)), name='loss')
    tf.summary.scalar('loss', loss)

    training_images, training_labels = create_feed_dict_recover(training_image_paths, annotation_file_train, training_eval_paths)

    print(training_images)
    nr_of_training_examples = len(training_images)
    nr_of_training_batches = math.ceil(nr_of_training_examples / batch_size)

    # # loss = tf.losses.absolute_difference(label_placeholder, fc3)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step, learning_rate_decay_epochs * nr_of_training_batches, learning_rate_decay_factor, staircase=True)
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimiser.minimize(loss, global_step=global_step)

    tf.summary.scalar('learning_rate', learning_rate)
    # training_image_paths, training_eval_paths, validation_image_paths, validation_eval_paths, test_image_paths, test_eval_paths = get_datasets(first_part_path, second_part_path)

    # # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    eval_training_images, eval_training_labels = create_feed_dict_recover_eval(training_image_paths, annotation_file_train, training_eval_paths)

    validation_images, validation_scores = create_feed_dict_recover_eval(training_image_paths ,annotation_file_validation,  validation_eval_paths)
    # test_patches, test_scores = create_eval_feed_dict(test_image_paths, test_eval_paths)

    # # Create a saver
    saver = tf.train.Saver(tf.trainable_variables())
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # saver.restore(sess, 'logs/model.ckpt-2784')
    log_dir = 'logs'
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # random.seed(3796)
    patch_indices = list(range(nr_of_training_examples))
    for epoch_index in range(num_epoch):
        random.shuffle(patch_indices)
        training_images = training_images[np.array(patch_indices)]
        training_labels = training_labels[np.array(patch_indices)]
        for batch_index in range(nr_of_training_batches - 1):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            loss_, _, summary_str, step_ = sess.run([loss, train_op, summary_op, global_step], feed_dict={learning_rate_placeholder: LR, image_placeholder: training_images[start_index:end_index], label_placeholder: training_labels[start_index:end_index], is_training: True, keep_prob: 1.})
            print('Step:', step_, "Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)
            summary_writer.add_summary(summary_str, global_step=step_)
        batch_index += 1
        start_index = batch_index * batch_size
        loss_, _, summary_str, step_ = sess.run([loss, train_op, summary_op, global_step], feed_dict={learning_rate_placeholder: LR, image_placeholder: training_images[start_index:], label_placeholder: training_labels[start_index:], is_training: True, keep_prob: 1.})
        print('Step:', step_, "Epoch:", epoch_index + 1, "Batch:", batch_index + 1, '/', nr_of_training_batches, 'Loss:', loss_)
        summary_writer.add_summary(summary_str, global_step=step_)

        predicted_training_labels = np.zeros_like(validation_scores)
        for i in range(len(validation_images)):
            predicted_training_labels[i] = forward_recover(validation_images[i], batch_size, sess, fc1, image_placeholder, is_training, keep_prob)
    #     training_lcc = pearsonr(predicted_training_labels, eval_training_labels)[0]
    #     training_srocc = spearmanr(predicted_training_labels, eval_training_labels)[0]
    #     print("Training LCC:", training_lcc)
    #     print("Training SROCC:", training_srocc)

        predicted_validation_scores = np.zeros_like(validation_scores)
        # for i in range(len(validation_images)):
        #     predicted_validation_scores[i] = forward_recover(validation_images[i], batch_size, sess, fc3, image_placeholder, is_training, keep_prob)
    #     validation_lcc = pearsonr(predicted_validation_scores, validation_scores)[0]
    #     validation_srocc = spearmanr(predicted_validation_scores, validation_scores)[0]
    #     print("Validation LCC:", validation_lcc)
    #     print("Validation SROCC:", validation_srocc)

        summary = tf.Summary()
    #     summary.value.add(tag='training_lcc', simple_value=training_lcc)
    #     summary.value.add(tag='training_srocc', simple_value=training_srocc)
    #     summary.value.add(tag='validation_lcc', simple_value=validation_lcc)
    #     summary.value.add(tag='validation_srocc', simple_value=validation_srocc)
        summary_writer.add_summary(summary, global_step=step_)

    #     saver.save(sess, log_dir + '/model.ckpt', global_step=step_)        

    # predicted_test_scores = np.zeros_like(test_scores)
    # for i in range(len(test_patches)):
    #     predicted_test_scores[i] = forward(test_patches[i], batch_size, sess, fc3, image_placeholder, is_training, keep_prob)
    # print("Test LCC:", pearsonr(predicted_test_scores, test_scores)[0])
    # print("Test SROCC:", spearmanr(predicted_test_scores, test_scores)[0])

if __name__ == '__main__':
    main()
