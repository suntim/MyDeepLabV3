# -*- coding: UTF-8 -*-MakeTFRecord.py
#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import scipy.io as spio
# from matplotlib import pyplot as plt
from scipy.misc import imread

def get_files_list(filename):
    file = open(filename, 'r')
    images_filename_list = [line for line in file]
    return images_filename_list

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_annotation_from_mat_file(annotations_dir, image_name):
    annotations_path = os.path.join(annotations_dir, (image_name.strip() + ".mat"))
    # print("annotations_path = ",annotations_path)
    assert os.path.exists(annotations_path),"annotations_path = {} not exited!!!".format(annotations_path)
    mat = spio.loadmat(annotations_path)
    img = mat['GTcls']['Segmentation'][0][0]
    return img


def create_tfrecord_dataset(images_dir_aug_voc,annotations_dir_aug_voc,filename_list, writer):
    # create training tfrecord
    for i, image_name in enumerate(filename_list):
        try:
            image_np = imread(os.path.join(images_dir_aug_voc, image_name.strip() + ".jpg"))
        except FileNotFoundError:
            # read from Pascal VOC path
            print(os.path.join(images_dir_voc, image_name.strip() + ".jpg"))

        try:
            annotation_np = imread(os.path.join(annotations_dir_voc, image_name.strip() + ".png"))
        except FileNotFoundError:
            # read from Pascal VOC path
            print(os.path.join(annotations_dir_voc, image_name.strip() + ".png"))

        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()
        annotation_raw = annotation_np.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_h),
            'width': _int64_feature(image_w),
            'image_raw': _bytes_feature(img_raw),
            'annotation_raw': _bytes_feature(annotation_raw)}))

        writer.write(example.SerializeToString())

    print("End of TfRecord. Total of image written:", i+1)
    writer.close()



if __name__ == '__main__':
    # define base paths for pascal the original VOC dataset training images
    images_dir_voc = r"D:\2345DownLoad\deeplab_v3-master\dataset\VOC_aug\img"#图片地址
    annotations_dir_voc = r"D:\2345DownLoad\deeplab_v3-master\dataset\VOC_aug\label_png"#标注图片地址
    NameTxt = r"D:\2345DownLoad\deeplab_v3-master\dataset\VOC_aug\custom_train.txt"
    SaveTxt_Dir = r"D:\2345DownLoad\deeplab_v3-master\dataset\VOC_aug"
    SaveTF_DIR = r"D:\2345DownLoad\deeplab_v3-master\dataset\VOC_aug"
    TRAIN_FILE = 'train.tfrecords'
    VALIDATION_FILE = 'validation.tfrecords'

    images_filename_list = get_files_list(NameTxt)
    print("Total number of training images:", len(images_filename_list))

    # shuffle array and separate 10% to validation
    np.random.shuffle(images_filename_list)
    val_images_filename_list = images_filename_list[:int(0.10 * len(images_filename_list))]
    with open(os.path.join(SaveTxt_Dir,"val.txt"),'w') as f:
        for line in val_images_filename_list:
            f.writelines(line)
    train_images_filename_list = images_filename_list[int(0.10 * len(images_filename_list)):]
    with open(os.path.join(SaveTxt_Dir,"train.txt"),'w') as f:
        for line in train_images_filename_list:
            f.writelines(line)

    print("train set size:", len(train_images_filename_list))
    print("val set size:", len(val_images_filename_list))


    train_writer = tf.python_io.TFRecordWriter(os.path.join(SaveTF_DIR, TRAIN_FILE))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(SaveTF_DIR, VALIDATION_FILE))

    # create training dataset
    create_tfrecord_dataset(images_dir_voc,annotations_dir_voc,train_images_filename_list, train_writer)

    # create validation dataset
    create_tfrecord_dataset(images_dir_voc,annotations_dir_voc,val_images_filename_list, val_writer)
