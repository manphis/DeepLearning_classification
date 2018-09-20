import os
import numpy as np
import skimage.io
import skimage.transform

def load_img(path, image_size):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (image_size, image_size))[None, :, :, :]   # shape [1, 224, 224, 3]
#    resized_img = skimage.transform.resize(crop_img, (32, 32))[None, :, :, :]   # shape [1, 32, 32, 3]
    return resized_img


def load_data(image_dir, part_list, image_size, feature_size, feature_category):
    print('load_data: feature size = ', feature_size, ' category = ', feature_category)

    image_data, label_data, feat_data, name_data = load_data_with_name(image_dir, part_list, image_size, feature_size, feature_category)


    print('feature shape = ', feat_data.shape)
    print(feat_data)
    
    return image_data, label_data, feat_data

def load_data_with_name(image_dir, part_list, image_size, feature_size, feature_category):
    image_list = []
    label_list = []
    feat_list = []
    name_list = []
    
    dir = image_dir
        
    for file in os.listdir(dir):
        if not file.lower().endswith('.jpg'):
            continue

        file_prefix, rest = file.split("_", 1)
        k = -1
        try:
            k = part_list.index(file_prefix)
        except ValueError:
            print('File without part: ', file)
            continue

        try:
            resized_img = load_img(os.path.join(dir, file), image_size)
        except OSError:
            continue
        image_list.append(resized_img)    # [1, height, width, depth] * n
        
        tag = np.zeros((1, len(part_list)))
        tag[0][k] = 1
        label_list.append(tag)

        feature = create_feature(k, part_list, feature_size, feature_category)
        feat_list.append(feature)

        name_list.append(file)
    
    image_data = np.concatenate(image_list, axis=0)
    label_data = np.concatenate(label_list, axis=0)
    feat_data = np.concatenate(feat_list, axis=0)
    name_data = np.array(name_list)
    
    return image_data, label_data, feat_data, name_data

def create_feature(type, part_list, feature_size, feature_category):
    feat_array = np.linspace(0, 1, feature_category)

    index = 0
    if type==0 or type==3 or type==7 or type==10:   #category 1
        index = 0
    elif type==2:
        index = 2
    elif type==5:
        index = 3
    else:
        index = 1

    feature = np.full((1, feature_size), feat_array[index])

    return feature

def create_feature_foreach(type, part_list, feature_size, feature_category):
    feat_array = np.linspace(0, 1, feature_category)
    index = type

    feature = np.full((1, feature_size), feat_array[index])
    
    return feature