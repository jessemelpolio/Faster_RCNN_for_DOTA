# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Guodong Zhang
# --------------------------------------------------------		  

import argparse
import pprint
import logging
import time
import os
import mxnet as mx
import numpy as np

from symbols import *
from dataset import *
from core.loader import TestLoader, QuadrangleTestLoader
from core.tester import Predictor, pred_eval, pred_eval_quadrangle, pred_eval_quadrangle_multiscale, pred_eval_dota, pred_eval_dota_quadrangle
from utils.load_model import load_param

import shapely
from shapely.geometry import Polygon, MultiPoint


def test_rcnn(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

def test_rcnn_dota(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    # get test data iter
    test_data = TestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

    # load model
    arg_params, aux_params = load_param(prefix, epoch, process=True)

    # infer shape
    data_shape_dict = dict(test_data.provide_data_single)
    sym_instance.infer_shape(data_shape_dict)

    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

    # decide maximum shape
    data_names = [k[0] for k in test_data.provide_data_single]
    label_names = None
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
    if not has_rpn:
        max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

    # create predictor
    predictor = Predictor(sym, data_names, label_names,
                          context=ctx, max_data_shapes=max_data_shape,
                          provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # start detection
    pred_eval_dota(predictor, test_data, imdb, cfg, vis=vis, ignore_cache=ignore_cache, thresh=thresh, logger=logger)

def merge_dets_to_one_file(path_prefix, scales):
    dst_path = os.path.join(path_prefix, 'test_results')
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    for index in range(1, 501):
        dst_file_path = os.path.join(dst_path, 'res_img_{}.txt'.format(index))
        df = open(dst_file_path, 'w')
        content = ''
        for scale in scales:
            src_path = os.path.join(path_prefix, 'test_{}_results'.format(scale))
            src_file_path = os.path.join(src_path, 'res_img_{}.txt'.format(index))
            sf = open(src_file_path, 'r')
            content += sf.read()
            sf.close()
        df.write(content)
        df.close()

def list_from_str(st):
    line = st.split(',')
    new_line = [float(a) for a in line[0:8]] + [float(line[-1])]
    return new_line


def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    # polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def nms(boxes, overlap):
    rec_scores = [b[-1] for b in boxes]
    indices = sorted(range(len(rec_scores)), key=lambda k: -rec_scores[k])
    box_num = len(boxes)
    nms_flag = [True] * box_num
    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue
        for j in range(box_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_flag[jj]:
                continue
            box1 = boxes[ii]
            box2 = boxes[jj]
            box1_score = rec_scores[ii]
            box2_score = rec_scores[jj]
            # str1 = box1[9]
            # str2 = box2[9]
            box_i = [box1[0], box1[1], box1[4], box1[5]]
            box_j = [box2[0], box2[1], box2[4], box2[5]]
            poly1 = polygon_from_list(box1[0:8])
            poly2 = polygon_from_list(box2[0:8])
            iou = polygon_iou(box1[0:8], box2[0:8])
            thresh = overlap

            if iou > thresh:
                if box1_score > box2_score:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_flag[jj] = False
                if box1_score == box2_score and poly1.area <= poly2.area:
                    nms_flag[ii] = False
                    break
            '''
            if abs((box_i[3]-box_i[1])-(box_j[3]-box_j[1]))<((box_i[3]-box_i[1])+(box_j[3]-box_j[1]))/2:
                if abs(box_i[3]-box_j[3])+abs(box_i[1]-box_j[1])<(max(box_i[3],box_j[3])-min(box_i[1],box_j[1]))/3:
                    if box_i[0]<=box_j[0] and (box_i[2]+min(box_i[3]-box_i[1],box_j[3]-box_j[1])>=box_j[2]):
                        nms_flag[jj] = False
            '''
    return nms_flag

def test_rcnn_dota_quadrangle(cfg, dataset, image_set, root_path, dataset_path,
              ctx, prefix, epoch,
              vis, ignore_cache, shuffle, has_rpn, proposal, thresh, logger=None, output_path=None):
    if not logger:
        assert False, 'require a logger'

    # print cfg
    pprint.pprint(cfg)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(cfg)))

    # load symbol and testing data
    if has_rpn:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        roidb = imdb.gt_roidb()
    else:
        sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
        sym = sym_instance.get_symbol_rcnn(cfg, is_train=False)
        imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=output_path)
        gt_roidb = imdb.gt_roidb()
        roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb)

    if cfg.TEST.DO_MULTISCALE_TEST:
        print "multiscale test!"
        multiscales = np.array(cfg.TEST.MULTISCALE)
        original_scales = cfg.SCALES
        for scale in multiscales:
            print "scale: {}".format(scale)
            cfg.SCALES[0] = (int(original_scales[0][0] * scale), int(original_scales[0][1] * scale))
            # get test data iter
            test_data = QuadrangleTestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

            # load model
            arg_params, aux_params = load_param(prefix, epoch, process=True)

            # infer shape
            data_shape_dict = dict(test_data.provide_data_single)
            sym_instance.infer_shape(data_shape_dict)

            sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

            # decide maximum shape
            data_names = [k[0] for k in test_data.provide_data_single]
            label_names = None
            max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
            if not has_rpn:
                max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

            # create predictor
            predictor = Predictor(sym, data_names, label_names,
                                  context=ctx, max_data_shapes=max_data_shape,
                                  provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                                  arg_params=arg_params, aux_params=aux_params)

            # start detection
            pred_eval_quadrangle_multiscale(scale, predictor, test_data, imdb, cfg, vis=vis, draw=True, ignore_cache=ignore_cache,
                                 thresh=thresh, logger=logger)
        # merge all different test scale results to one file
        merge_dets_to_one_file(imdb.result_path, multiscales)
        # do polygon nms then in evaluation script

    else:
        # get test data iter
        test_data = QuadrangleTestLoader(roidb, cfg, batch_size=len(ctx), shuffle=shuffle, has_rpn=has_rpn)

        # load model
        arg_params, aux_params = load_param(prefix, epoch, process=True)

        # infer shape
        data_shape_dict = dict(test_data.provide_data_single)
        sym_instance.infer_shape(data_shape_dict)

        sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict, is_train=False)

        # decide maximum shape
        data_names = [k[0] for k in test_data.provide_data_single]
        label_names = None
        max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES])))]]
        if not has_rpn:
            max_data_shape.append(('rois', (cfg.TEST.PROPOSAL_POST_NMS_TOP_N + 30, 5)))

        # create predictor
        predictor = Predictor(sym, data_names, label_names,
                              context=ctx, max_data_shapes=max_data_shape,
                              provide_data=test_data.provide_data, provide_label=test_data.provide_label,
                              arg_params=arg_params, aux_params=aux_params)

        # start detection
        pred_eval_dota_quadrangle(predictor, test_data, imdb, cfg, vis=False, draw=False, ignore_cache=ignore_cache,
                             thresh=thresh, logger=logger)

