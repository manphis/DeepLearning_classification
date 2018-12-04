import os
import numpy as np
import skimage.io
import skimage.transform
from itertools import cycle, islice

g_has_gsensor = False

G_SENSOR_COLOR_LENGTH = 10
COLOR_LIST = [[15, 67, 121], [240, 70, 142], [117, 6, 87], [30, 209, 125], [230, 202, 136],
              [126, 255, 127], [239, 186, 145], [5, 158, 146], [128, 254, 139], [210, 222, 146], [64, 237, 143], [120, 254, 132]]

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

def load_file_name(image_dir):
    name_list = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

    return name_list

def load_file_from_dir_list(dir_list):
    name_list = []
    for i in range(len(dir_list)):
        name_list += [str(dir_list[i]+f) for f in os.listdir(dir_list[i]) if f.lower().endswith('.jpg')]

    return name_list

def circular_sample(lst, start_index, length):
    aux = islice(cycle(lst), start_index, start_index+length)

    return list(aux)

def load_data_by_fullname(file_list, part_list, image_size, feature_size, feature_category):
    image_list = []
    label_list = []
    feat_list = []
    name_list = []

    if feature_size != 0:
        print('=====> feature class: ', np.linspace(0, 1, feature_category))
        
    for file in file_list:
        # print('file: ', file)
        file_items = file.split('/')
        file_name = file_items[len(file_items) - 1]
        # print('file_name: ', file_name)

        if not file_name.lower().endswith('.jpg'):
            continue

        file_prefix, rest = file_name.split("_", 1)
        k = -1
        try:
            k = part_list.index(file_prefix)
        except ValueError:
            print('File without part: ', file)
            continue

        try:
            resized_img = load_img(file, image_size)
            # add g-sensor color to image
            if g_has_gsensor:
                assign_color(resized_img[0], COLOR_LIST[k][0], COLOR_LIST[k][1], COLOR_LIST[k][2])

            # test_img = resized_img[0][:,:,0]
            # print("img data = ", test_img[0][0], test_img[0][1], test_img[0][2], test_img[0][3], test_img[0][4])

        except OSError:
            continue
        image_list.append(resized_img)    # [1, height, width, depth] * n
        
        tag = np.zeros((1, len(part_list)))
        tag[0][k] = 1
        label_list.append(tag)

        if feature_category == 12:
            feature = create_feature_foreach(k, part_list, feature_size, feature_category)
#            print('feature = ', feature)
        else:
            feature = create_feature(k, part_list, feature_size, feature_category)
        feat_list.append(feature)

        name_list.append(file)
    
    image_data = np.concatenate(image_list, axis=0)
    label_data = np.concatenate(label_list, axis=0)
    feat_data = np.concatenate(feat_list, axis=0)
    name_data = np.array(name_list)
    
    return image_data, label_data, feat_data, name_data

# def load_data_by_name(image_dir, file_list, part_list, image_size, feature_size, feature_category):
#     image_list = []
#     label_list = []
#     feat_list = []
#     name_list = []
    
#     dir = image_dir

#     if feature_size != 0:
#         print('=====> feature class: ', np.linspace(0, 1, feature_category))
        
#     for file in file_list:
# #        print('file: ', os.path.join(dir, file))
#         if not file.lower().endswith('.jpg'):
#             continue

#         file_prefix, rest = file.split("_", 1)
#         k = -1
#         try:
#             k = part_list.index(file_prefix)
#         except ValueError:
#             print('File without part: ', file)
#             continue

#         try:
#             resized_img = load_img(os.path.join(dir, file), image_size)
#         except OSError:
#             continue
#         image_list.append(resized_img)    # [1, height, width, depth] * n
        
#         tag = np.zeros((1, len(part_list)))
#         tag[0][k] = 1
#         label_list.append(tag)

#         if feature_category == 12:
#             feature = create_feature_foreach(k, part_list, feature_size, feature_category)
#         else:
#             feature = create_feature(k, part_list, feature_size, feature_category)
#         feat_list.append(feature)

#         name_list.append(file)
    
#     image_data = np.concatenate(image_list, axis=0)
#     label_data = np.concatenate(label_list, axis=0)
#     feat_data = np.concatenate(feat_list, axis=0)
#     name_data = np.array(name_list)
    
#     return image_data, label_data, feat_data, name_data


# def load_data(image_dir, part_list, image_size, feature_size, feature_category):
#     print('=====> load_data: feature size = ', feature_size, ' category = ', feature_category)

#     image_data, label_data, feat_data, name_data = load_data_with_name(image_dir, part_list, image_size, feature_size, feature_category)


#     print('=====> feature shape = ', feat_data.shape)
# #    print(feat_data)
    
#     return image_data, label_data, feat_data

# def load_data_with_name(image_dir, part_list, image_size, feature_size, feature_category):
#     image_list = []
#     label_list = []
#     feat_list = []
#     name_list = []
    
#     dir = image_dir

#     if feature_size != 0:
#         print('=====> feature class: ', np.linspace(0, 1, feature_category))
        
#     for file in os.listdir(dir):
#         if not file.lower().endswith('.jpg'):
#             continue

#         file_prefix, rest = file.split("_", 1)
#         k = -1
#         try:
#             k = part_list.index(file_prefix)
#         except ValueError:
#             print('File without part: ', file)
#             continue

#         try:
#             resized_img = load_img(os.path.join(dir, file), image_size)
#         except OSError:
#             continue
#         image_list.append(resized_img)    # [1, height, width, depth] * n
        
#         tag = np.zeros((1, len(part_list)))
#         tag[0][k] = 1
#         label_list.append(tag)

#         if feature_category == 12:
#             feature = create_feature_foreach(k, part_list, feature_size, feature_category)
#         else:
#             feature = create_feature(k, part_list, feature_size, feature_category)
#         feat_list.append(feature)

#         name_list.append(file)
    
#     image_data = np.concatenate(image_list, axis=0)
#     label_data = np.concatenate(label_list, axis=0)
#     feat_data = np.concatenate(feat_list, axis=0)
#     name_data = np.array(name_list)
    
#     return image_data, label_data, feat_data, name_data

def create_feature(type, part_list, feature_size, feature_category):
    feat_array = np.linspace(0, 1, feature_category)
    
    index = 0
    if type==0 or type==3 or type==7 or type==10:   #category 1
        index = 0
    elif type==2:
        index = 2
    elif type==5 or type==8 or type==11:
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

# def create_feature_foreach(type, part_list, feature_size, feature_category):
#     index = type

#     feature = np.full((1, feature_size), 0)
#     feature[0][index] = 1
    
#     return feature

def assign_color(img, r, g, b):
    print('assign_color')
    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]
    for i in range(G_SENSOR_COLOR_LENGTH):
        for j in range(G_SENSOR_COLOR_LENGTH):
            imgR[i][j] = r/255.0

    for i in range(G_SENSOR_COLOR_LENGTH):
        for j in range(G_SENSOR_COLOR_LENGTH):
            imgG[i][j] = g/255.0

    for i in range(G_SENSOR_COLOR_LENGTH):
        for j in range(G_SENSOR_COLOR_LENGTH):
            imgB[i][j] = b/255.0
    return img