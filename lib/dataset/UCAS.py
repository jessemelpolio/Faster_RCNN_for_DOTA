# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

"""
Pascal VOC database
This class loads ground truth notations from standard Pascal VOC XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the Pascal VOC format. Evaluation is based on mAP
criterion.
"""

import cPickle
import os
import numpy as np

from imdb import IMDB
import cv2
import zipfile
from bbox.bbox_transform import bbox_overlaps, bbox_transform, bbox_transform_quadrangle, bbox_pred_quadrangle
from PIL import Image
import codecs


# the target of this class is to get UCAS roidb
class UCAS(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None, mask_size=-1, binary_thresh=None):
        """
        fill basic information to initialize imdb
        :param image_set: train, test etc.
        :param root_path: 'selective_search_data' and 'cache'
        :param data_path: data and results
        :return: imdb object
        """
        self.image_set = image_set
        super(UCAS, self).__init__('UCAS', self.image_set, root_path, data_path, result_path)  # set self.name

        self.root_path = root_path
        self.data_path = data_path

        self.classes = ['__background__',  # always index 0
                        'plane', 
                        'small-vehicle']
        self.num_classes = len(self.classes)
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images
        self.mask_size = mask_size
        self.binary_thresh = binary_thresh

        self.config = {'comp_id': 'comp4',
                       'use_diff': False,
                       'min_size': 2}

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file, 'r') as f:
            lines = f.readlines()
        image_lists = [line.strip() for line in lines]
        return image_lists

    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param image_name: image name in the data dir
        :return: full path of this image
        """
        # hint: self.image_set means 'train' or 'test'
        # TODO: when data ready, the entrance here should be changed
        image_file = os.path.join(self.data_path, self.image_set, index)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def gt_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self.load_annotation(index) for index in self.image_set_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def load_annotation(self, index):
        """
        for a given index, load image and bounding boxes info from XML file
        :param image_name: image name in the data dir
        :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        # import xml.etree.ElementTree as ET
        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        # roi_rec['image_name'] = 'img_' + index + '.jpg'

        filename = os.path.join(self.data_path, 'wordlabel', os.path.splitext(os.path.basename(index))[0] + '.txt')
        # tree = ET.parse(filename)
        img_path = self.image_path_from_index(index)
        w, h = Image.open(img_path).size
        # size = tree.find('size')
        roi_rec['height'] = float(h)
        roi_rec['width'] = float(w)
        # roi_rec['height'] = float(size.find('height').text)
        # roi_rec['width'] = float(size.find('width').text)
        # roi_rec['index'] = int(index)
        # im_size = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION).shape
        # assert im_size[0] == roi_rec['height'] and im_size[1] == roi_rec['width']

        f = codecs.open(filename, 'r', 'utf-16')
        objs = f.readlines()
        objs = [obj.strip().split(' ') for obj in objs]
        # objs = tree.findall('object')
        if not self.config['use_diff'] and len(objs[0]) == 8:
            non_diff_objs = [obj for obj in objs if obj[9] != '1']
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.int16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        class_to_index = dict(zip(self.classes, range(self.num_classes)))
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj
            # Make pixel indexes 0-based
            x1 = float(bbox[0]) - 1
            y1 = float(bbox[1]) - 1
            x2 = float(bbox[2]) - 1
            y2 = float(bbox[3]) - 1
            x3 = float(bbox[4]) - 1
            y3 = float(bbox[5]) - 1
            x4 = float(bbox[6]) - 1
            y4 = float(bbox[7]) - 1
            xmin = max(min(x1, x2, x3, x4), 0)
            xmax = max(x1, x2, x3, x4)
            ymin = max(min(y1, y2, y3, y4), 0)
            ymax = max(y1, y2, y3, y4)
            cls = class_to_index[obj[8].lower().strip()]
            boxes[ix, :] = [xmin, ymin, xmax, ymax]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
        roi_rec.update({'boxes': boxes,
                        'gt_classes': gt_classes,
                        'gt_overlaps': overlaps,
                        'max_classes': overlaps.argmax(axis=1),
                        'max_overlaps': overlaps.max(axis=1),
                        'flipped': False})
        return roi_rec

    def evaluate_detections(self, detections):
        """
        :param detections: [cls][image] = N x [x1, y1, x2, y2, x3, y3, x4, y4, score]
        :return:
        """
        detection_results_path = os.path.join(self.result_path, 'test_results')
        info = ''
        if not os.path.isdir(detection_results_path):
            os.mkdir(detection_results_path)
        self.write_ucas_results(detections, threshold=0.0)
        return info

    def write_ucas_results(self, all_boxes, threshold=0.2):
        """
        write results files in pascal devkit path
        :param all_boxes: boxes to be processed [bbox, confidence]
        :return: None
        """
        path = os.path.join(self.result_path, 'test_results')
        if os.path.isdir(path):
            print "delete original test results files!"
            os.system("rm -r {}".format(path))
            os.mkdir(path)
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            for im_ind, index in enumerate(self.image_set_index):
                dets = all_boxes[cls_ind][im_ind]
                # if dets.shape[0] == 0:
                #     print "no detection results in {}".format(index)
                f = open(os.path.join(self.result_path, 'test_results', 'res_{}'.format(os.path.splitext(os.path.basename(index))[0] + '.txt')), 'a')
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    if dets[k, 4] <= threshold:
                        continue
                    f.write('{},{},{},{},{},{}\n'.format(int(dets[k, 0]), int(dets[k, 1]), int(dets[k, 2]),
                                                                     int(dets[k, 3]),dets[k, 4],self.classes[cls_ind]))

