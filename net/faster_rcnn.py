import torch
import torch.nn as nn
from model.rpn import _RPN
from model.proposal_target_layer import _ProposalTargetLayer

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
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        self.init_module()
        self.batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        print(base_feat.)
