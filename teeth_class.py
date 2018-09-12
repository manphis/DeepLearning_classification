import argparse
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import tensorflow.contrib.slim as slim
import Utils as utils
from collections import namedtuple


IMAGE_SIZE = 224
FEATURE_CLASS = 4

part_list = ['in_down_left', 'in_down_right', 'in_down_center', 'in_up_left', 'in_up_right', 'in_up_center',
            'out_down_left', 'out_down_right', 'out_down_center', 'out_up_left', 'out_up_right', 'out_up_center']
LEARNING_RATE = 0.0001
MODEL_DIR = 'Model_zoo/'
MODEL_URL = 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz'
IMAGE_NET_MEAN = [103.939, 116.779, 123.68]

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

parser = argparse.ArgumentParser()
parser.add_argument("--feature", help="input the size of feature", type=int)
args = parser.parse_args()
print (args.feature)

def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))[None, :, :, :]   # shape [1, 224, 224, 3]
#    resized_img = skimage.transform.resize(crop_img, (32, 32))[None, :, :, :]   # shape [1, 32, 32, 3]
    return resized_img

def load_data(image_dir, part_dir_list):
    image_list = []
    label_list = []
    
    for k in range(len(part_dir_list)):
        part_name = part_dir_list[k]
        dir = image_dir + part_name
        
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            image_list.append(resized_img)    # [1, height, width, depth] * n
            
            tag = np.zeros((1, len(part_dir_list)))
            tag[0][k] = 1
            label_list.append(tag)

#            if len(imgs[k]) == 400:        # only use 400 imgs to reduce my memory load
#                break
    
    image_data = np.concatenate(image_list, axis=0)
    label_data = np.concatenate(label_list, axis=0)
    
    return image_data, label_data

def create_feature(image_dir, part_dir_list, feature_size):
    feat_list = []
    feat_array = np.linspace(0, 1, FEATURE_CLASS)
    
    for k in range(len(part_dir_list)):
    	count = 0
    	part_name = part_dir_list[k]
    	dir = image_dir + part_name

    	index = 0
    	if k==0 or k==3 or k==7 or k==10:   #category 1
    		index = 0
    	elif k==2:
    		index = 2
    	elif k==5:
    		index = 3
    	else:
    		index = 1

    	for file in os.listdir(dir):
    		if not file.lower().endswith('.jpg'):
    			continue
    		count = count + 1

    	feature = np.full((count, feature_size), feat_array[index])
    	feat_list.append(feature)

    feat_data = np.concatenate(feat_list, axis=0)
    
    return feat_data


def mobile_net(image, final_endpoint=None):
	print("setting up mobile initialized conv layers ...")
	with tf.variable_scope('MobilenetV1'):
		net = image
		with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
			for i, conv_def in enumerate(_CONV_DEFS):
				end_point_base = 'Conv2d_%d' % i
				if isinstance(conv_def, Conv):
					net = slim.conv2d(net, conv_def.depth, conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point_base)
				elif isinstance(conv_def, DepthSepConv):
					end_point = end_point_base + '_depthwise'
					net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1, 
                                                stride=conv_def.stride,
                                                rate=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point)

					end_point = end_point_base + '_pointwise'
					net = slim.conv2d(net, conv_def.depth, [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
#                    print("end_point_base="+end_point_base)
				if final_endpoint and final_endpoint == end_point_base:
					print("break end_point_base==final_endpoint="+end_point_base)
					break
	return net



class MyNet:
	def __init__(self, image_size, category_size, feature_size=0):
		self.x = tf.placeholder(tf.float32,shape=[None, image_size, image_size, 3])
		self.y_true = tf.placeholder(tf.float32,shape=[None, category_size])
		if feature_size != 0:
			self.x_feat = tf.placeholder(tf.float32,shape=[None, feature_size])
		self.hold_prob = tf.placeholder(tf.float32)

		self.feature_size = feature_size

		utils.get_model_data(MODEL_DIR, MODEL_URL)

#		Create layers
#		convo_1 = self.convolutional_layer(self.x, shape=[4,4,3,32])
#		convo_1_pooling = self.max_pool_2by2(convo_1)

#		convo_2 = self.convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
#		convo_2_pooling = self.max_pool_2by2(convo_2)

		
		
		mobilenet_net = self.get_mobile_net(self.x, final_endpoint="Conv2d_11")
		variable_to_restore = [v for v in slim.get_variables_to_restore() if v.name.split('/')[0] == 'MobilenetV1']
#		shape of mobilenet_net: (?, 14, 14, 512)

#		self.size = (int)(image_size/4)
#		convo_2_flat = tf.reshape(convo_2_pooling, [-1, self.size*self.size*64])
		self.size = 14*14*512
		convo_2_flat = tf.reshape(mobilenet_net, [-1, self.size])

		full_layer_one = tf.nn.relu(self.normal_full_layer(convo_2_flat, 1024))
		full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=self.hold_prob)

		print("feature size = ", feature_size)
		if feature_size != 0:
			full_feature = tf.concat( [full_one_dropout, self.x_feat], 1 )
			self.y_pred = self.normal_full_layer(full_feature, category_size)
		else:
			self.y_pred = self.normal_full_layer(full_one_dropout, category_size)
#		y_pred = normal_full_layer(full_feature, len(part_list))

		self.sess = tf.Session()

		self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,logits=self.y_pred))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
		self.train_op = self.optimizer.minimize(self.cross_entropy)

		self.matches = tf.equal(tf.argmax(self.y_pred,1),tf.argmax(self.y_true,1))
		self.acc = tf.reduce_mean(tf.cast(self.matches,tf.float32))

		self.c_matrix = tf.confusion_matrix(tf.argmax(self.y_pred,1), tf.argmax(self.y_true,1))

		self.sess.run(tf.global_variables_initializer())

#		restore pre-train mobilenet
		self.saver = tf.train.Saver(variable_to_restore)
		self.saver.restore(self.sess, 'Model_zoo/mobilenet_v1_1.0_224.ckpt')

		print("build MyNet done")

		return


	def train(self, x_dataset, y_dataset, f_dataset):
#		print("x_shape: ", x_dataset.shape, "y_shape: ", y_dataset.shape, "f_shape: ", f_dataset.shape)
		if self.feature_size != 0:
			loss, _ = self.sess.run([self.cross_entropy, self.train_op], feed_dict={self.x: x_dataset, self.y_true: y_dataset, self.x_feat: f_dataset, self.hold_prob: 0.5})
		else:
			loss, _ = self.sess.run([self.cross_entropy, self.train_op], feed_dict={self.x: x_dataset, self.y_true: y_dataset, self.hold_prob: 0.5})

		return loss

	def validate(self, test_dataset, test_label, test_feature):
		if self.feature_size != 0:
			result = self.sess.run(self.acc, feed_dict={self.x:test_dataset, self.y_true:test_label, self.x_feat:test_feature, self.hold_prob:1.0})
		else:
			result = self.sess.run(self.acc, feed_dict={self.x:test_dataset, self.y_true:test_label, self.hold_prob:1.0})

		return result

	def validate_matrix(self, test_dataset, test_label, test_feature):
		if self.feature_size != 0:
			result, matrix = self.sess.run([self.acc, self.c_matrix], feed_dict={self.x:test_dataset, self.y_true:test_label, self.x_feat:test_feature, self.hold_prob:1.0})
		else:
			result, matrix = self.sess.run([self.acc, self.c_matrix], feed_dict={self.x:test_dataset, self.y_true:test_label, self.hold_prob:1.0})

		return result, matrix


	def init_weights(self, shape):
		init_random_dist = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(init_random_dist)

	def init_bias(self, shape):
		init_bias_vals = tf.constant(0.1, shape=shape)
		return tf.Variable(init_bias_vals)

	def conv2d(self, x, W):
	    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2by2(self, x):
	    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                          strides=[1, 2, 2, 1], padding='SAME')

	def convolutional_layer(self, input_x, shape):
	    W = self.init_weights(shape)
	    b = self.init_bias([shape[3]])
	    return tf.nn.relu(self.conv2d(input_x, W) + b)

	def normal_full_layer(self, input_layer, size):
	    input_size = int(input_layer.get_shape()[1])
	    W = self.init_weights([input_size, size])
	    b = self.init_bias([size])
	    return tf.matmul(input_layer, W) + b

	def get_mobile_net(self, image, final_endpoint=None):
		mean = tf.constant(IMAGE_NET_MEAN)
		image -= mean
		net = mobile_net(image, final_endpoint)

		return net

	

def train(_feature_size):
	train_image_dir = 'train_img/'
	test_image_dir = 'test_img/'
	test_feature = np.array([])
	f_dataset = np.array([])

	train_dataset, label_dataset = load_data(train_image_dir, part_dir_list=part_list)
	test_dataset, test_label = load_data(test_image_dir, part_dir_list=part_list)
	if _feature_size != 0:
		feature_dataset = create_feature(train_image_dir, part_dir_list=part_list, feature_size=_feature_size)
		test_feature = create_feature(test_image_dir, part_dir_list=part_list, feature_size=_feature_size)

	batch_size = 2
	index = 0

	my_net = MyNet(image_size=IMAGE_SIZE, category_size=len(part_list), feature_size=_feature_size)

	for i in range(5000):
		x_dataset = train_dataset[index:index+batch_size].reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)
		y_dataset = label_dataset[index:index+batch_size].reshape(-1, len(part_list))
		
		if _feature_size != 0:
			f_dataset = feature_dataset[index:index+batch_size].reshape(-1, _feature_size)

		index = (index+batch_size) % len(train_dataset)

		loss = my_net.train(x_dataset, y_dataset, f_dataset)
#		print("loss = ", loss)
		if i%100 == 0:
			acc, matrix = my_net.validate_matrix(test_dataset, test_label, test_feature)
			print(matrix)
			print('step {}'.format(i), ' ; loss = ', loss, ' ; accuracy = ', acc)
            
#            saver.save(sess, save_path, write_meta_graph=False)


	return

def test_dataset():
	train_image_dir = 'train_img/'
	feature_dataset = create_feature(train_image_dir, part_dir_list=part_list, feature_size=args.feature)

	print(feature_dataset.shape)
	return

def test_confusion_matrix():
	matrix = tf.confusion_matrix([1,1,1], [1,2,4])
	sess = tf.Session()
	matrix_result = sess.run(matrix)

	print(matrix_result)

	return

if __name__ == '__main__':
	if args.feature == None:
		_feature_size = 0
	else:
		_feature_size = args.feature

#	train(_feature_size)
	test_confusion_matrix()
