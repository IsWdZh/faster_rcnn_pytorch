import torch.nn as nn
import torch
import numpy as np
from .generate_anchors import generate_anchors
from .bbox import bbox_overlaps_batch, bbox_transform_batch
from logger import get_logger

logger = get_logger()

class _AnchorTargetLayer(nn.Module):
    '''
    输入：gt_boxes的信息，特征图的尺寸
    step1.在特征图每个点上生成9个anchors
    step2.确定所有anchors中顶点都在输入图片内的inside_anchors
    step3.计算anchors与ground-truth的IOU矩阵
    step4.确定inside_anchors的labels值，策略如下（按顺序执行）：
          1. 与所有的gt的IOU都<0.3,则为bg_anchor
          2. 与某个gt的IOU > 0.7 则fg_anchor
          3. 对每个gt，与其重叠比例最大的anchor的fg_anchor
    step5.得到fg_anchor的总个数sum_fg
          如果sum_fg > num_fg(128,人为规定)，则随机遗弃多的部分
    step6.根据剩下fg_anchor数量确定bg_anchor数量，总和为256
    step7.返回labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
          labels: [batch_size, 1, A * height, width]
          bbox_targets: [batch_size, A * 4, height, width]
          bbox_inside_weight: [batch_size, A * 4, height, width]
          bbox_outside_weight: [batch_size, A * 4, height, width]
    '''
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()
        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(
            generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        '''input is a tuple (rpn_cls_score.data, gt_boxes, im_info, num_boxes)'''
        rpn_cls_score = input[0]   # [batch_size, 2*9, H, W]
        gt_boxes = input[1]        # [batch_size, 20, 5(x1,y1,x2,y2,cls)]
        im_info = input[2]         # [batch_size, 3(h,w,ratio)]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride  # [0, 1, ..., W]*16
        shift_y = np.arange(0, feat_height) * self._feat_stride  # [0, 1, ..., H]*16
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                             shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()

        A = self._num_anchors
        K = shifts.size(0)

        self._anchors = self._anchors.type_as(gt_boxes)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)  # all_anchors为[n*m*9, 4]
        logger.debug("all_anchors = {}".format(all_anchors))

        total_anchors = int(K * A)
        # each anchors:（xmin，ymin，xmax，ymax）
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < int(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < int(im_info[0][0]) + self._allowed_border))
        inds_inside = torch.nonzero(keep).view(-1)
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        logger.debug("keep only inside anchors = {}".format(anchors))

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        logger.debug(labels)
        #input: anchors: (N, 4) ndarray of float
        #       gt_boxes: (b, K, 5) ndarray of float
        #output: overlaps: (b, N, K) ndarray of overlap between boxes and query_boxes
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        # max_overlaps[batch_size,N]第i行j列的元素代表第i张图片第j个推荐框与所有ground_truth最大的IOU值
        # element(i,j) represent the max IOU of the jth recommendation box with all the ground
        # truth in the ith picture
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)

        # gt_max_overlaps (batch_size,K)的元素（i，j）是第i张照片的第j个ground_truth与所有的推荐框IOU最大值
        # element(i,j) represent the max IOU of the jth ground truth with all the
        # recommendation box  in the ith picture
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5

        #for each ground-truth, the label of the recommendation box which has the maximum IOU is 1
        #对每个标定的真值候选区域，与其重叠比例最大的anchor记为前景样本
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
        if torch.sum(keep) > 0 :
            labels[keep>0] = 1

        # 如果其与某个gt重叠比例大于0.7，则其label 也为1, 小于0.3则其label为0
        labels[max_overlaps >= 0.7] = 1
        labels[max_overlaps < 0.3] = 0

        num_fg = int(0.5 * 256)  # 0.5*256    max_num_of_fg_example * total_num_of_examples

        #sum_bg, sum_fg:[batch]
        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            if sum_fg[i] > num_fg :
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = 256 - torch.sum((labels ==1).int(), 1)[i]
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)

        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1, 5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        bbox_inside_weights[labels == 1] = (1.0, 1.0, 1.0, 1.0)[0]


        num_examples = torch.sum(labels[i] >= 0)
        positive_weight = 1.0 / num_examples
        negative_weight = 1.0 / num_examples


        bbox_outside_weights[labels == 1] = positive_weight
        bbox_outside_weights[labels == 0] = negative_weight

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        # _unmap的作用是，构造一个[batch_size， totoal_anchors，4]矩阵，将inds_inside为True的位置对应填上bbox_targets值
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        # _unmap的作用是，构造一个[batch_size， totoal_anchors]矩阵，将inds_inside为True的位置对应填上bbox_targets值
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)
        logger.debug("In anchor_target_layer, labels = {}".format(labels.size()))

        bbox_targets = bbox_targets.view(batch_size, height, width, A * 4).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_targets)
        logger.debug("In anchor_target_layer, bbox_targets = {}".format(bbox_targets.size()))

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                            4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()

        outputs.append(bbox_inside_weights)
        logger.debug("In anchor_target_layer, bbox_inside_weights = {}".format(bbox_inside_weights.size()))

        bbox_outside_weights = bbox_outside_weights.view(batch_size, anchors_count, 1).expand(batch_size, anchors_count,
                                                                                              4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4 * A) \
            .permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_outside_weights)
        logger.debug("In anchor_target_layer, bbox_outside_weights = {}".format(bbox_outside_weights.size()))

        return outputs

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret

def _compute_targets_batch(ex_rois, gt_rois):
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])