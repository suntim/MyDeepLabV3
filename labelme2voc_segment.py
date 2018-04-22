# -*- coding: UTF-8 -*-
#!/usr/bin/env python
from __future__ import print_function

import argparse
import glob
import json
import os
import io
import os.path as osp

import numpy as np
import PIL.Image
import PIL.ImageDraw

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

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
    import matplotlib.pyplot as plt
    backend_org = plt.rcParams['backend']
    plt.switch_backend('agg')

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
        if label_value not in label:
            continue
        if label_name.startswith('_'):
            continue
        fc = colormap[label_value]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    plt.switch_backend(backend_org)

    out_size = (img.shape[1], img.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out
def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap
def shapes_to_label(img_shape, shapes, label_name_to_value, type='class'):
    assert type in ['class', 'instance']

    cls = np.zeros(img_shape[:2], dtype=np.uint8)
    print("img_shape[:2] = ",img_shape[:2])
    if type == 'instance':
        ins = np.zeros(img_shape[:2], dtype=np.uint8)
        instance_names = ['_background_']
    for shape in shapes:
        polygons = shape['points']
        label = shape['label']
        if type == 'class':
            cls_name = label
        elif type == 'instance':
            cls_name = label.split('-')[0]
            if label not in instance_names:
                instance_names.append(label)
            ins_id = len(instance_names) - 1
        cls_id = label_name_to_value[cls_name]
        mask = polygons_to_mask(img_shape[:2], polygons)
        cls[mask] = cls_id
        if type == 'instance':
            ins[mask] = ins_id

    if type == 'instance':
        return cls, ins
    return cls
def main(argslabels_file,argsin_dir,argsout_dir,):
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('labels_file')
    # parser.add_argument('in_dir')
    # parser.add_argument('out_dir')
    # args = parser.parse_args()

    if not osp.exists(argsout_dir):
        os.makedirs(argsout_dir)
    else:
        print('Output directory already exists:', argsout_dir)
    os.makedirs(osp.join(argsout_dir, 'JPEGImages'))
    os.makedirs(osp.join(argsout_dir, 'SegmentationClass'))
    os.makedirs(osp.join(argsout_dir, 'SegmentationClassVisualization'))
    print('Creating dataset:', argsout_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(argslabels_file).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(argsout_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)

    colormap = label_colormap(255)

    for label_file in glob.glob(osp.join(argsin_dir, '*.json')):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            base = osp.splitext(osp.basename(label_file))[0]
            out_img_file = osp.join(
                argsout_dir, 'JPEGImages', base + '.jpg')
            out_lbl_file = osp.join(
                argsout_dir, 'SegmentationClass', base + '.png')
            out_viz_file = osp.join(
                argsout_dir, 'SegmentationClassVisualization', base + '.jpg')

            data = json.load(f)

            img_file = osp.join(osp.dirname(label_file), data['imagePath'])
            img = np.asarray(PIL.Image.open(img_file))
            PIL.Image.fromarray(img).save(out_img_file)

            lbl = shapes_to_label(
                img_shape=img.shape,
                shapes=data['shapes'],
                label_name_to_value=class_name_to_id,
            )

            print("img.shape = ",img.shape)
            lbl_pil = PIL.Image.fromarray(lbl)
            # Only works with uint8 label
            # lbl_pil = PIL.Image.fromarray(lbl, mode='P')
            # lbl_pil.putpalette((colormap * 255).flatten())
            lbl_pil.save(out_lbl_file)

            label_names = ['%d: %s' % (cls_id, cls_name)
                           for cls_id, cls_name in enumerate(class_names)]
            viz = draw_label(
                lbl, img, label_names, colormap=colormap)
            PIL.Image.fromarray(viz).save(out_viz_file)


if __name__ == '__main__':
    labels_file = r"D:\JD\JPG\Scale\by5\by5_train\labels.txt"#r"D:\JD\JPG\Scale\by5\by5_train\labels.txt"#
    jason_dir = r"D:\JD\JPG\Scale\by5\by5_train\data_annoted"#r"D:\JD\JPG\Scale\by5\by5_train\data_annoted"#
    save_dir = r"D:\JD\JPG\Scale\by5\by5_train\data_dataset_voc"#r"D:\JD\JPG\Scale\by5\by5_train\data_dataset_voc"
    main(labels_file,jason_dir,save_dir)
