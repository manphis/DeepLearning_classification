import argparse
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import tensorflow.contrib.slim as slim
import Utils as utils
from collections import namedtuple
import plot_utils as pu
import image_loader as iLoader
from random import shuffle



FEATURE_CLASS = 12

g_image_size = 224
g_feature_size = 0
g_pretrain_ckpt = None
g_mode = ''
g_data_size = 0
g_train_data_size = 0
g_test_data_size = 0
g_file_batch_size = 100
g_train_batch_size = 2
g_gap_layer = True
g_acc_save_threshold = 0.85
g_training_iteration = 5000

train_image_dir = 'image_data/101090840001/Q8H_mix_face/self_train_img/'
test_image_dir = 'image_data/101090840001/Q8H_mix_face/test_img/'
predict_image_dir = 'face_detection/138839393939mix_face_map/'


g_data_dir_list = ['image_data/000000000038/Q8H_origin_image/train_img/'
,'image_data/000000016400/Q8H_origin_image/train_img/'
,'image_data/101090840001/Q8H_origin_image/train_img/'
,'image_data/138839393939/Q8H_origin_image/train_img/'
]

#part_list = ['in_down_left', 'in_down_right', 'in_down_center', 'in_up_left', 'in_up_right', 'in_up_center',
#            'out_down_left', 'out_down_right', 'out_down_center', 'out_up_left', 'out_up_right', 'out_up_center']

TEETH_PART_LIST = ['InDownLeft', 'InDownRight', 'InDownCenter', 'InUpLeft', 'InUpRight', 'InUpCenter',
'OutDownLeft', 'OutDownRight', 'OutDownCenter', 'OutUpLeft', 'OutUpRight', 'OutUpCenter']
LEARNING_RATE = 0.0001
MODEL_DIR = 'Model_zoo/'
LOG_DIR = 'logs/'
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
parser.add_argument("--mode", help="input mode: train, predict, train-with-pretrain", type=str)
parser.add_argument("--checkpoint", help="input checkpoint number for prediction", type=int)
args = parser.parse_args()
print (args.feature)


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
    def __init__(self, image_size, category_size, feature_size=0, predict_ckpt=None, gap_layer=False):
        self.x = tf.placeholder(tf.float32,shape=[None, image_size, image_size, 3])
        self.y_true = tf.placeholder(tf.float32,shape=[None, category_size])
        if feature_size != 0:
            self.x_feat = tf.placeholder(tf.float32,shape=[None, feature_size])
        self.hold_prob = tf.placeholder(tf.float32)

        self.feature_size = feature_size

        utils.get_model_data(MODEL_DIR, MODEL_URL)

#		transfer learning from MobilenetV1        
        self.mobilenet_net = self.get_mobile_net(self.x, final_endpoint="Conv2d_11")
        variable_to_restore = [v for v in slim.get_variables_to_restore() if v.name.split('/')[0] == 'MobilenetV1']
#       shape of mobilenet_net: (?, 14, 14, 512)

#       self.size = (int)(image_size/4)
#       convo_2_flat = tf.reshape(convo_2_pooling, [-1, self.size*self.size*64])
        self.size = 14*14*512
        self.gap_layer = gap_layer

        if self.gap_layer:
            # GAP layer
            self.gap_weight = tf.Variable(tf.random_normal([512, len(TEETH_PART_LIST)]))
            self.gap_bias = tf.Variable(tf.random_normal([len(TEETH_PART_LIST)]))
            self.y_pred = self.gap_out_layer(self.mobilenet_net, self.gap_weight, self.gap_bias)
        else:
            self.convo_2_flat = tf.reshape(self.mobilenet_net, [-1, self.size])

            if feature_size != 0:
                self.convo_2_flat = tf.concat( [self.convo_2_flat, self.x_feat], 1 )

            self.full_layer_one = tf.nn.relu(self.normal_full_layer(self.convo_2_flat, 1024))
            self.full_one_dropout = tf.nn.dropout(self.full_layer_one, keep_prob=self.hold_prob)

            self.y_pred = self.normal_full_layer(self.full_one_dropout, category_size)


        self.sess = tf.Session()

        if predict_ckpt:
            print('=====> predict_ckpt = ', predict_ckpt)
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, predict_ckpt)

            self.position = tf.argmax(self.y_pred,1)

            self.classmaps = self.generate_heatmap(self.mobilenet_net, self.position, self.gap_weight, image_size)

        else:
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true,logits=self.y_pred))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            self.train_op = self.optimizer.minimize(self.cross_entropy)

            self.matches = tf.equal(tf.argmax(self.y_pred,1),tf.argmax(self.y_true,1))
            self.acc = tf.reduce_mean(tf.cast(self.matches,tf.float32))

            self.c_matrix = tf.confusion_matrix(tf.argmax(self.y_pred,1), tf.argmax(self.y_true,1))

            self.sess.run(tf.global_variables_initializer())

    #       restore pre-train mobilenet
            if g_pretrain_ckpt == None:
                self.saver = tf.train.Saver(variable_to_restore)
                self.saver.restore(self.sess, 'Model_zoo/mobilenet_v1_1.0_224.ckpt')
            else:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, 'pretrain_model/model.ckpt-'+str(g_pretrain_ckpt))

        self.print_params()

        return

    def print_params(self):
        print("===== build MyNet done =================")
        print('MyNet: pretrain_ckpt        ', 'pretrain_model/model.ckpt-'+str(g_pretrain_ckpt))
        print('MyNet: feature_size         ', self.feature_size)
        print('MyNet: gap_layer =          ', self.gap_layer)
        # print('MyNet: convo_2_flat shape = ', self.convo_2_flat.shape)
        print("========================================")
        return

    def train(self, x_dataset, y_dataset, f_dataset):
#       print("x_shape: ", x_dataset.shape, "y_shape: ", y_dataset.shape, "f_shape: ", f_dataset.shape)
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

    def save_checkpoint(self, path, iteration):
        self.saver = tf.train.Saver(max_to_keep=50)
        self.saver.save(self.sess, path, iteration)
        return

    def predict(self, predict_dataset, predict_feature):
        print('num for predict = ', len(predict_dataset))
#        for i in range(len(predict_dataset)):
        if self.feature_size != 0:
            idx = self.sess.run(self.position, feed_dict={self.x:predict_dataset, self.x_feat:predict_feature, self.hold_prob:1.0})
        else:
            idx = self.sess.run(self.position, feed_dict={self.x:predict_dataset, self.hold_prob:1.0})

        print('predict index = ', idx)
        return idx

    def activation_map(self, predict_dataset, predict_feature):
        print('num for activation_map = ', len(predict_dataset))
#        for i in range(len(predict_dataset)):
        if self.feature_size != 0:
            maps = self.sess.run(self.classmaps, feed_dict={self.x:predict_dataset, self.x_feat:predict_feature, self.hold_prob:1.0})
        else:
            maps = self.sess.run(self.classmaps, feed_dict={self.x:predict_dataset, self.hold_prob:1.0})

        return maps

    def gap_out_layer(self, conv, weight, biase):
        gap = tf.nn.avg_pool(conv, ksize=[1,14,14,1], strides=[1,14,14,1], padding="SAME")
        gap = tf.reshape(gap,[-1, 512])
        out = tf.add(tf.matmul(gap, weight), biase)

        return out

    def generate_heatmap(self, conv, label, weight, size):
        conv2_resized = tf.image.resize_images(conv, [size,size])
        label_w = tf.gather(tf.transpose(weight), label)
        label_w = tf.reshape(label_w, [-1,512,1])
        conv2_resized = tf.reshape(conv2_resized,[-1, size*size, 512])
        classmap = tf.matmul( conv2_resized, label_w )
        classmap = tf.reshape( classmap, [-1, size,size] )
        
        return classmap

    

def train():
    global g_data_size, g_test_data_size, g_train_data_size
    test_feature = np.array([])
    f_dataset = np.array([])

    filename_list = iLoader.load_file_from_dir_list(g_data_dir_list)
    g_data_size = len(filename_list)

    shuffle(filename_list)
    g_test_data_size = int(g_data_size*0.1)
    g_train_data_size = g_data_size - g_test_data_size
    test_filename_list = filename_list[0:g_test_data_size]
    train_filename_list = filename_list[g_test_data_size:]

    test_dataset, test_label, test_feature, test_name = iLoader.load_data_by_fullname(file_list=test_filename_list,
                    part_list=TEETH_PART_LIST, image_size=g_image_size, feature_size=g_feature_size, feature_category=FEATURE_CLASS)
    
    BATCH_LOAD = True if g_train_data_size > 400 else False

    if not BATCH_LOAD:
        train_dataset, label_dataset, feature_dataset, name_dataset = iLoader.load_data_by_fullname(file_list=train_filename_list,
            part_list=TEETH_PART_LIST, image_size=g_image_size, feature_size=g_feature_size, feature_category=FEATURE_CLASS)

    print('=====> test_dataset = ', test_dataset.shape)

    index = 0
    accuracy_list = []

    my_net = MyNet(image_size=g_image_size, category_size=len(TEETH_PART_LIST), feature_size=g_feature_size, gap_layer=g_gap_layer)

    print_params()

    file_index = 0
    global g_acc_save_threshold

    for i in range(g_training_iteration):
        if BATCH_LOAD:
            if i%100 == 0:
                index = 0
                partial_filename_list = iLoader.circular_sample(train_filename_list, file_index, g_file_batch_size)
                print('=====> train index ', file_index, ' to ', file_index+g_file_batch_size) 
                file_index += g_file_batch_size
                train_dataset, label_dataset, feature_dataset, name_dataset = iLoader.load_data_by_fullname(file_list=partial_filename_list,
                    part_list=TEETH_PART_LIST, image_size=g_image_size, feature_size=g_feature_size, feature_category=FEATURE_CLASS)
                # print('=====> train from ', name_dataset[0], ' to ', name_dataset[len(name_dataset)-1], ' size = ', len(train_dataset))


        x_dataset = train_dataset[index:index+g_train_batch_size].reshape(-1, g_image_size, g_image_size, 3)
        y_dataset = label_dataset[index:index+g_train_batch_size].reshape(-1, len(TEETH_PART_LIST))
        
        if g_feature_size != 0:
            f_dataset = feature_dataset[index:index+g_train_batch_size].reshape(-1, g_feature_size)

        index = (index+g_train_batch_size) % len(train_dataset)

        loss = my_net.train(x_dataset, y_dataset, f_dataset)
#       print("loss = ", loss)
        if i%100 == 0:
            acc, matrix = my_net.validate_matrix(test_dataset, test_label, test_feature)
            # print(matrix)
            print('step {}'.format(i), ' ; loss = ', loss, ' ; accuracy = ', acc)

            accuracy_list.append(acc)

            if acc > g_acc_save_threshold:
                print('save ckpt')
                my_net.save_checkpoint(LOG_DIR + "model.ckpt", i)
                # g_acc_save_threshold = acc

#    save_train_result(accuracy_list)
    print(accuracy_list)
    return

def predict(_feature_size, ckpt_num=4000):
    predict_feature = np.array([])
    predict_dataset, predict_label, predict_feature, predict_file_name = iLoader.load_data_with_name(predict_image_dir, 
    	part_list=TEETH_PART_LIST, image_size=IMAGE_SIZE, feature_size=_feature_size, feature_category=FEATURE_CLASS)

    print(predict_file_name)
    # if _feature_size != 0:
    #     predict_feature = create_feature(predict_image_dir, part_dir_list=TEETH_PART_LIST, feature_size=_feature_size)

    ckpt_file = 'logs/model.ckpt-' + str(ckpt_num)
    my_net = MyNet(image_size=IMAGE_SIZE, category_size=len(TEETH_PART_LIST), feature_size=_feature_size, predict_ckpt=ckpt_file, gap_layer=g_gap_layer)

    result = my_net.predict(predict_dataset, predict_feature)

    maps = my_net.activation_map(predict_dataset, predict_feature)

    error_list = []
    correct_list = []
    index_list = []
    for i in range(len(result)):
        if result[i] != np.argmax(predict_label[i]):
            print('error: ', TEETH_PART_LIST[result[i]], '   ', predict_file_name[i])
            error_list.append(TEETH_PART_LIST[result[i]])
            correct_list.append(TEETH_PART_LIST[np.argmax(predict_label[i])])
            index_list.append(i)

#    pu.plot_error_result(error_list, correct_list, predict_dataset, index_list)

    pu.plot_classmap(predict_dataset, maps)

    return

def print_params():
    print('========================================')
    print('|                                      |')
    print('|           parameter list             |')
    print('|                                      |')
    print('========================================')
    print('data DIR:            ', g_data_dir_list)
    print('predicting data DIR: ', predict_image_dir)
    print('feature class:       ', FEATURE_CLASS)

    print('g_image_size         ', g_image_size)
    print('g_feature_size       ', g_feature_size)
    print('g_pretrain_ckpt      ', g_pretrain_ckpt)
    print('g_mode               ', g_mode)
    print('g_data_size          ', g_data_size)
    print('g_train_data_size    ', g_train_data_size)
    print('g_test_data_size     ', g_test_data_size)
    print('g_file_batch_size    ', g_file_batch_size)
    print('g_train_batch_size   ', g_train_batch_size)
    print('g_gap_layer          ', g_gap_layer)
    print('g_acc_save_threshold ', g_acc_save_threshold)
    print('g_training_iteration ', g_training_iteration)
    print('g_has_gsensor        ', iLoader.g_has_gsensor)

    print('========================================')
    return


def test_confusion_matrix():
    matrix = tf.confusion_matrix([1,1,1], [1,2,4])
    sess = tf.Session()
    matrix_result = sess.run(matrix)

    print(matrix_result)

    return

def test_partial_file():
    test_dir = 'test_dir/'
    filename_list = iLoader.load_file_name(test_dir)
    print('total file name = ', filename_list)
    index = 0

    g_file_batch_size = 3
    file_index = 0

    for i in range(10):
        if i%2 == 0:
            index = 0
            partial_filename_list = iLoader.circular_sample(filename_list, file_index, g_file_batch_size)
            print('file name = ', partial_filename_list)

            file_index += g_file_batch_size
            train_dataset, label_dataset, feature_dataset, name_dataset = iLoader.load_data_by_name(test_dir, file_list=partial_filename_list,
                part_list=TEETH_PART_LIST, image_size=IMAGE_SIZE, feature_size=_feature_size, feature_category=FEATURE_CLASS)

            print('train data shape = ', train_dataset.shape)


    return


if __name__ == '__main__':
    g_feature_size = 0 if args.feature == None else args.feature

    if args.mode == None:
        # test_dataset, test_label, test_feature, test_name = iLoader.load_data_by_fullname(file_list=result,
        #             part_list=TEETH_PART_LIST, image_size=224, feature_size=0, feature_category=12)
        print('please input mode with --mode')
    else:
        g_mode = args.mode
        if 'train' in g_mode:
            if g_mode == 'train-with-pretrain':
                g_pretrain_ckpt = args.checkpoint
            train()

        elif g_mode == 'predict':
            print('predicting...')
            if args.checkpoint == None:
                predict(_feature_size)
            else:
                predict(_feature_size, args.checkpoint)
        else:
            print('please use -h to know how to use --mode')
