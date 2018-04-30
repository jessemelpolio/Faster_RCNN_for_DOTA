# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
import cPickle
from core.rcnn import sample_rois_quadrangle

DEBUG = False


class ProposalTargetQuadrangleOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction):
        super(ProposalTargetQuadrangleOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._cfg = cfg
        # FG_FRACTION = 0.25
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois == -1 or self._batch_rois % self._batch_images == 0, \
            'batchimages {} must devide batch_rois {}'.format(self._batch_images, self._batch_rois)
        all_rois = in_data[0].asnumpy()
        gt_boxes = in_data[1].asnumpy()

        # here all_rois are [batch_idx, x1, y1, x2, y2], but we have to change it to [batch_idx, x1, y1, x2, y1, x2, y2, x1, y2]
        temp = np.zeros((all_rois.shape[0], 9), dtype=all_rois.dtype)
        temp[:, 0] = all_rois[:, 0]
        temp[:, 1] = all_rois[:, 1]
        temp[:, 2] = all_rois[:, 2]
        temp[:, 3] = all_rois[:, 3]
        temp[:, 4] = all_rois[:, 2]
        temp[:, 5] = all_rois[:, 3]
        temp[:, 6] = all_rois[:, 4]
        temp[:, 7] = all_rois[:, 1]
        temp[:, 8] = all_rois[:, 4]
        all_rois = temp

        if self._batch_rois == -1:
            rois_per_image = all_rois.shape[0] + gt_boxes.shape[0]
            fg_rois_per_image = rois_per_image
        else:
            rois_per_image = self._batch_rois / self._batch_images
            fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        gt_boxes_bbox = np.zeros(gt_boxes.shape,dtype=gt_boxes.dtype)
        ex_x = np.vstack((gt_boxes[:, 0], gt_boxes[:, 2], gt_boxes[:, 4], gt_boxes[:, 6]))
        ex_y = np.vstack((gt_boxes[:, 1], gt_boxes[:, 3], gt_boxes[:, 5], gt_boxes[:, 7]))
        x1 = np.amin(ex_x, axis=0)
        y1 = np.amin(ex_y, axis=0)
        x2 = np.amax(ex_x, axis=0)
        y2 = np.amax(ex_y, axis=0)
        gt_boxes_bbox[:, 0] = x1
        gt_boxes_bbox[:, 1] = y1
        gt_boxes_bbox[:, 2] = x2
        gt_boxes_bbox[:, 3] = y1
        gt_boxes_bbox[:, 4] = x2
        gt_boxes_bbox[:, 5] = y2
        gt_boxes_bbox[:, 6] = x1
        gt_boxes_bbox[:, 7] = y2
        gt_boxes_bbox[:, 8] = gt_boxes[:, 8]
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes_bbox[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights = \
            sample_rois_quadrangle(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, self._cfg, gt_boxes=gt_boxes)

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))

        for ind, val in enumerate([rois, labels, bbox_targets, bbox_weights]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('proposal_target_quadrangle')
class ProposalTargetQuadrangleProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, cfg, fg_fraction='0.25'):
        super(ProposalTargetQuadrangleProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._cfg = cPickle.loads(cfg)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]

        rois = rpn_rois_shape[0] + gt_boxes_shape[0] if self._batch_rois == -1 else self._batch_rois

        output_rois_shape = (rois, 5)
        label_shape = (rois, )
        bbox_target_shape = (rois, self._num_classes * 8)
        bbox_weight_shape = (rois, self._num_classes * 8)

        return [rpn_rois_shape, gt_boxes_shape], \
               [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalTargetQuadrangleOperator(self._num_classes, self._batch_images, self._batch_rois, self._cfg, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
