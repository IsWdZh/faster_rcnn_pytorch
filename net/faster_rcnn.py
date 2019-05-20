import torch
import torch.nn as nn

from model.rpn import _RPN
from model.proposal_target_layer import _ProposalTargetLayer
from model.roi import _RoIPooling2d

class Faster_RCNN(nn.Module):
    '''Faster_RCNN model'''
    def __init__(self, classes):
        super(Faster_RCNN, self).__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # size of the pooled region after ROI pooling.
        self.RCNN_roi_pool = _RoIPooling2d(7, 7, 1.0/16.0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        self.init_module()
        self.batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        print(base_feat)

        roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)












    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            权重初始化: 截断正态分布/随机正态分布
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01)
        normal_init(self.RCNN_cls_score, 0, 0.01)
        normal_init(self.RCNN_bbox_pred, 0, 0.001)


    def init_model(self):
        self._init_weights()
        self._init_modules()
