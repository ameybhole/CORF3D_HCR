import os
import cv2
import numpy as np
from sklearn import preprocessing


def load_images(dataset, resize):

    """

    Function to return an list of images and its labels

    :param resize: resize the image
    :param dataset: Folder with class name and all the images in the dataset
    :return: Lists of image in an array form

    """

    img_data_list = []
    labels = []
    list_of_image_paths = []
    labels_list = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        img_list = os.listdir(dataset + '/' + folder_name)
        for image in img_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + image
            images = cv2.imread(retrieve_dir, 3)
            images = cv2.resize(images, (resize, resize))
            list_of_image_paths.append(images)
        img_data_list.append(img_list)
        labels_list.append(folder_name)
        labels.append([folder_name] * len(img_list))

    return np.array(list_of_image_paths), labels, labels_list

def load_feature_maps(dataset, resize):

    """

    Function to return an list of feautre maps and its labels

    :param resize: resize the feature maps
    :param dataset: Folder with class name and all the feature maps in the dataset
    :return: Lists of feature maps in an array form

    """

    feature_data_list = []
    labels = []
    list_of_feature_paths = []
    labels_list = []

    data_dir_list = os.listdir(dataset)
    for folder_name in data_dir_list:
        feature_list = os.listdir(dataset + '/' + folder_name)
        for feature_maps in feature_list:
            retrieve_dir = dataset + "/" + folder_name + "/" + feature_maps
            feature_maps = np.load(retrieve_dir)
            images = cv2.resize(feature_maps, (resize, resize))
            list_of_feature_paths.append(images)
        feature_data_list.append(feature_list)
        labels_list.append(folder_name)
        labels.append([folder_name] * len(feature_list))

    return np.array(list_of_feature_paths), labels, labels_list

def binarize_labels(labels):

    """

    Function to return an list of binarized labels

    :param labels: List of labels
    :return: Lists of binarized labels

    """

    flattened_list = np.asarray([y for x in labels for y in x], dtype="str")
    lb = preprocessing.LabelBinarizer().fit(flattened_list)
    labels = lb.transform(flattened_list)

    return labels
