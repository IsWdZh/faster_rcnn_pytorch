# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
from model.rpn.bbox import bbox_transform_batch, bbox_overlaps_batch
from logger import get_logger

logger = get_logger()

class _ProposalTargetLayer(nn.Module):
    """
    对应GT目标的检测proposals 分类label和bbox回归目标
    选择128个ROIs用以训练
    """

    def __init__(self, num_classes):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = num_classes
        # Normalize the targets using "precomputed" (or made up) means and stdevs
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor((0.0, 0.0, 0.0, 0.0))
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor((0.1, 0.1, 0.2, 0.2))
        # Deprecated (inside weights)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor((1.0, 1.0, 1.0, 1.0))

    def forward(self, all_rois, gt_boxes, num_boxes, batch_size):

        self.batch_size = batch_size
        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        # 创建一个和gt_boxes形状一致的全零矩阵 [1, 20, 5]
        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]  # 将gt_boxes第三维前三列赋值给gt_boxes_append第三维1~4列

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)  # 如果输入的all_rois=[1,2000,5],输出[1,2020,5]

        num_images = 1
        rois_per_image = int(self.batch_size / num_images)
        # Fraction of minibatch that is labeled foreground
        fg_rois_per_image = int(np.round(0.25 * rois_per_image))  # if batch_size=1, fg_rois_per_image=0
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding box 回归目标(bbox_target_data) 以紧凑的形式 b*N*(类别, tx, ty, tw, th)
        该函数将目标扩展为使用4-of-4*k表示(即：只有一个类具有非零目标)
        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """计算单张图片的bounding box 回归目标"""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                   / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # all_rois=[1,2000,5], gt_boxes=[1,20,5], fg_rois_per_image=1,
        # rois_per_image=1, self._num_classes=21

        # overlaps: (rois x gt_boxes)
        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)
        logger.debug("proposal_target_layer->_sample_rois_pytorch: overlaps = {}".format(overlaps.size()))

        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        logger.debug("proposal_target_layer->_sample_rois_pytorch: max_overlaps = {}, "
              "gt_assignment = {}".format(overlaps.size(), gt_assignment))

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)  # offset=tensor([0])
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        logger.debug("proposal_target_layer->_sample_rois_pytorch: offset = {}, "
              "offset.view(-1)={}".format(offset, offset.view(-1)))

        # labels = gt_boxes[:, :, 4].contiguous().view(-1).index(offset.view(-1)) \
        #     .view(batch_size, -1)
        labels = gt_boxes[:, :, 4].contiguous().view(-1)[offset.view(-1)] \
            .view(batch_size, -1)
        logger.debug("labels = {}".format(labels))

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):
            # ROI判定为foreground的阈值
            fg_inds = torch.nonzero(max_overlaps[i] >= 0.5).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < 0.5) &
                                    (max_overlaps[i] >= 0.1)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                # rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                # rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                # rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i, :, 0] = i

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
            rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])

        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
