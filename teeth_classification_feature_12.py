
# coding: utf-8



import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# # Get data



IMG_SIZE = 32
FEATURE_SIZE = 10
FEATURE_CLASS = 4




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
    resized_img = skimage.transform.resize(crop_img, (IMG_SIZE, IMG_SIZE))[None, :, :, :]   # shape [1, 224, 224, 3]
#    resized_img = skimage.transform.resize(crop_img, (32, 32))[None, :, :, :]   # shape [1, 32, 32, 3]
    return resized_img




def load_data(image_dir, part_dir_list):
    image_list = []
    label_list = []
    feat_list = []
    feat_array = np.linspace(0, 1, FEATURE_CLASS)
    
    for k in range(len(part_dir_list)):
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
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            image_list.append(resized_img)    # [1, height, width, depth] * n
            
            tag = np.zeros((1, len(part_dir_list)))
            tag[0][k] = 1
            label_list.append(tag)
            
            feature = np.full((1, FEATURE_SIZE), feat_array[index])
            feat_list.append(feature)

#            if len(imgs[k]) == 400:        # only use 400 imgs to reduce my memory load
#                break
    
    image_data = np.concatenate(image_list, axis=0)
    label_data = np.concatenate(label_list, axis=0)
    feat_data = np.concatenate(feat_list, axis=0)
    
    return image_data, label_data, feat_data




train_image_dir = 'train_img/'
test_image_dir = 'test_img/'




#part_list = ['out', 'in']
part_list = ['in_down_left', 'in_down_right', 'in_down_center', 'in_up_left', 'in_up_right', 'in_up_center',
            'out_down_left', 'out_down_right', 'out_down_center', 'out_up_left', 'out_up_right', 'out_up_center']




train_dataset, label_dataset, feature_dataset = load_data(train_image_dir, part_dir_list=part_list)
test_dataset, test_label, test_feature = load_data(test_image_dir, part_dir_list=part_list)




train_dataset.shape




label_dataset
label_dataset.shape
label_dataset[2]




feature_dataset.shape
feature_dataset[8]




#plt.imshow(train_dataset[2])


# # Model



import tensorflow as tf




x = tf.placeholder(tf.float32,shape=[None, IMG_SIZE, IMG_SIZE, 3])
#x = tf.placeholder(tf.float32,shape=[None,32,32,3])
y_true = tf.placeholder(tf.float32,shape=[None,len(part_list)])
x_feat = tf.placeholder(tf.float32,shape=[None,FEATURE_SIZE])




hold_prob = tf.placeholder(tf.float32)




def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# # Create layers



convo_1 = convolutional_layer(x,shape=[4,4,3,32])
convo_1_pooling = max_pool_2by2(convo_1)




convo_2 = convolutional_layer(convo_1_pooling,shape=[4,4,32,64])
convo_2_pooling = max_pool_2by2(convo_2)




#convo_2_flat = tf.reshape(convo_2_pooling, [-1, 8*8*64])  #example
size = (int)(IMG_SIZE/4)
convo_2_flat = tf.reshape(convo_2_pooling, [-1, size*size*64])




#convo_2_flat_ext = tf.concat( [convo_2_flat, x_feat ], 1 )




full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))




full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)




full_feature = tf.concat( [full_one_dropout, x_feat], 1 )




#y_pred = normal_full_layer(full_one_dropout,len(part_list))
y_pred = normal_full_layer(full_feature, len(part_list))




cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))




optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)




init = tf.global_variables_initializer()


# # Graph session



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    index = 0
    saver = tf.train.Saver()
    save_path='./output/teeth_learn'

    for i in range(5000):
        batch_size = 2
        x_dataset = train_dataset[index:index+batch_size].reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        y_dataset = label_dataset[index:index+batch_size].reshape(-1, len(part_list))
        f_dataset = feature_dataset[index:index+batch_size].reshape(-1, FEATURE_SIZE)

        
        index = (index+batch_size) % len(train_dataset)
        
        sess.run(train, feed_dict={x: x_dataset, y_true: y_dataset, x_feat: f_dataset, hold_prob: 0.5})
#        sess.run(train, feed_dict={x: x_dataset, y_true: y_dataset, hold_prob: 0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

#            print(sess.run(acc,feed_dict={x:test_dataset, y_true:test_label, hold_prob:1.0}))
            print(sess.run(acc,feed_dict={x:test_dataset, y_true:test_label, x_feat:test_feature, hold_prob:1.0}))
            print('\n')
            
            saver.save(sess, save_path, write_meta_graph=False)






