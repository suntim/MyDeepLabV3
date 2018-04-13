# MyDeepLabV3






# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib
matplotlib.use('pdf')
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
import network
slim = tf.contrib.slim
import os,re,cv2
import argparse
import json
import PIL.Image
from metrics import *
try:
    import io
except ImportError:
    import io as io

plt.interactive(False)

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Eval params')
envarg.add_argument("--model_id", type=int, help="Model id name to be loaded.")
envarg.add_argument("--gpu_id", type=int, help="Gpu id name to be used.")
envarg.add_argument("--batch_size", type=int, default = 10, help="Batch size of test.")
envarg.add_argument("--ResizeWidth", type=int, default = 513, help="Resize Width of pic.")
envarg.add_argument("--ResizeHeight", type=int, default = 513, help="Resize Height of pic.")
envarg.add_argument("--save_Result", type=bool, default = False, help="Want to save result?.")
envarg.add_argument("--Pic_Dir", help="Test Pic_Dir.")
input_args = parser.parse_args()

# best: 16645
model_name = str(input_args.model_id)

# uncomment and set the GPU id if applicable.
os.environ["CUDA_VISIBLE_DEVICES"]=str(input_args.gpu_id)

log_folder = './tboard_logs'

if not os.path.exists(os.path.join(log_folder, model_name, "test")):
    os.makedirs(os.path.join(log_folder, model_name, "test"))

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

def scale_image_with_crop_padding(image, shapes):
    image_croped = tf.image.resize_image_with_crop_or_pad(image,input_args.ResizeHeight,input_args.ResizeWidth)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    return image_croped, shapes

def _Resize_function(image_decoded,shapes):
    # image_string = tf.read_file(filename)
    # image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [input_args.ResizeWidth,input_args.ResizeHeight])
    print("image_resized = ",image_resized)
    return image_resized,shapes

def tf_record_parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64)
    }
    features = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # reshape input and annotation images
    image = tf.reshape(image, (height, width, 3), name="image_reshape")
    return tf.to_float(image), (height, width)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord_dataset(Test_images_dir,filename_list, writer):
    # create training tfrecord
    for i, image_name in enumerate(filename_list):
        try:
            image_np = imread(os.path.join(Test_images_dir, image_name.strip()))
        except FileNotFoundError:
            # read from Pascal VOC path
            print("{} not Exited!!".format(os.path.join(Test_images_dir, image_name.strip())))


        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_h),
            'width': _int64_feature(image_w),
            'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())

    print("End of TfRecord. Total of image written:", i+1)
    writer.close()

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap
# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.3, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        img_gray = PIL.Image.fromarray(img).convert('LA')
        img_gray = np.asarray(img_gray.convert('RGB'))
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)
    return lbl_viz

def draw_label(label, img, label_names, colormap=None):
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if colormap is None:
        colormap = label_colormap(len(label_names))

    label_viz = label2rgb(label, img, n_labels=len(label_names))
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_value, label_name in enumerate(label_names):
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    out_size = (img.shape[1], img.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def extract_classes(segm):
    cl = np.unique(segm)#cls
    n_cl = len(cl)
    return cl, n_cl

if __name__ == '__main__':
    Test_images_dir = input_args.Pic_Dir  # 图片地址
    test_images_filename_list = []
    for filName in os.listdir(Test_images_dir):
        if re.match(".*[.]jpg",filName):
            test_images_filename_list.append(filName)

    # print("test_images_filename_list = ",test_images_filename_list)
    test_filenames = os.path.join(Test_images_dir,'test.tfrecords')
    test_writer = tf.python_io.TFRecordWriter(test_filenames)
    create_tfrecord_dataset(Test_images_dir, test_images_filename_list, test_writer)
    test_dataset = tf.data.TFRecordDataset([test_filenames])
    test_dataset = test_dataset.map(tf_record_parser)  # Parse the record into tensors.
    test_dataset = test_dataset.map(_Resize_function)
    # test_dataset = test_dataset.map(scale_image_with_crop_padding)
    test_dataset = test_dataset.batch(input_args.batch_size)

    iterator = test_dataset.make_one_shot_iterator()
    batch_images_tf, batch_shapes_tf = iterator.get_next()

    logits_tf = network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=False)


    predictions_tf = tf.argmax(logits_tf, axis=3)
    # probabilities_tf = tf.nn.softmax(logits_tf)
    saver = tf.train.Saver()

    # test_folder = test_filenames
    train_folder = os.path.join(log_folder, model_name, "train")

    with tf.Session() as sess:
        index = 1
        # Create a saver.
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
        print("Model", model_name, "restored.")

        while True:
            try:
                batch_images_np, batch_predictions_np, batch_shapes_np = sess.run(
                    [batch_images_tf, predictions_tf, batch_shapes_tf])
                heights, widths = batch_shapes_np

                # loop through the images in the batch and extract the valid areas from the tensors
                for i in range(batch_predictions_np.shape[0]):
                    print("index/batch_predictions_np.shape[0] = {}/{}".format(index, batch_predictions_np.shape[0]))

                    pred_image = batch_predictions_np[i]
                    input_image = batch_images_np[i]

                    # remove scale_image_with_crop_padding == 255
                    # indices = np.where(pred_image != 255)
                    # pred_image = pred_image[indices]
                    # input_image = input_image[indices]

                    print("pred_image.shape[0]*pred_image[1] = {}*{}={}".format(pred_image.shape[0],pred_image.shape[1],pred_image.shape[0]*pred_image.shape[1]))
                    print("input_image.shape[0] = ",input_image.shape[0])
                    sizeofShape = pred_image.shape[0]*pred_image.shape[1]
                    if sizeofShape == input_args.ResizeWidth * input_args.ResizeHeight:
                        pred_image = np.reshape(pred_image, (input_args.ResizeWidth,input_args.ResizeHeight))
                        input_image = np.reshape(input_image, (input_args.ResizeWidth,input_args.ResizeHeight, 3))
                    else:
                        pred_image = np.reshape(pred_image, (heights[i], widths[i]))
                        input_image = np.reshape(input_image, (heights[i], widths[i], 3))

                    if (input_args.save_Result):
                        cl, n_cl = extract_classes(pred_image)
                        print("cl = ",cl)
                        print("n_cl = ",n_cl)
                        # save as image
                        # lbl_names = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog',
                        #              'horse','motorbike','person','plant','sheep','sofa','train','tv']
                        # captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
                        # print("input_image shape = ",input_image.shape)
                        # lbl_viz = draw_label(pred_image, input_image, captions)
                        # save_dir = "./ouput"
                        # if not os.path.exists(save_dir):
                        #     os.mkdir(save_dir)
                        # PIL.Image.fromarray(lbl_viz).save(os.path.join(save_dir, "Deploy_mask_" + str(index) + "_label_viz.png"))
                        # index += 1

            except tf.errors.OutOfRangeError:
                break
