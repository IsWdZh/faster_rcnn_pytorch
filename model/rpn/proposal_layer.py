import torch
import torch.nn as nn
import numpy as np
from model.rpn.generate_anchors import generate_anchors
from model.rpn.bbox import bbox_transform_inv, clip_boxes
from model.rpn.nms import nms
from logger import get_logger

logger = get_logger()

class _ProposalLayer(nn.Module):
    '''
    这个类的作用是
    step1.在feature_map上产生的基础推荐框anchores: [batch_size, K*A, 4]
    step2.与RPN训练过后的基础推荐框的偏移参数 相运算得到RPN网络最后真正的推荐框集合。
    step3.通过将推荐框内有物体的得分排序，按分数从高到低取12000个推荐框
    step4.通过nms将每个feature_map上的推荐框缩小至2000个推荐框，并输出
    '''
    def __init__(self, feat_stride, scales, ratios, use_gpu=False):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride   # [16]
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.nms_gpu = True
        else:
            self.nms_gpu = False

    def forward(self, input):
        '''对于每个location上的(H,W),生成以该位置为中心的k个anchor boxe
        并讲预测到的bbox deltas应用到k个anchor上
        在图像上截取该box,移除宽或者高不满足要求的预测框
        按score从高到低对proposal排序
        使用NMS筛选一定数量的proposal，然后再从选出的里面取N个'''
        # input[0]: rpn_cls_prob.data [batch_size, 18, H, W]    18: 2(bg/fg)×9(anchor)
        # input[1]: rpn_bbox_pred.data [batch_size, 36, H, W]   36: 4(窗口中心点坐标+宽和高)×9(anchor)
        # input[2]: im_info [batch_size, 3([h,w,ratio])]
        # input[3]: cfg_key
        scores = input[0][:, self._num_anchors: , :, :] # [batch_size, 2*9, H, W]取后9个
        bbox_deltas = input[1]
        im_info = input[2]
        '''
        print("In proposal: \n")
        print("Anchor = {}".format(self._anchors))
        print("_num_anchors = {}".format(self._num_anchors))
        print("scores: rpn_cls_prob.data[:, 9: , :, :] = {}".format(scores.size()))
        print("bbox_deltas: rpn_bbox_pred.data = {}".format(bbox_deltas.size()))
        print("im_info : im_info = {}".format(im_info))
        '''

        pre_nms_topN = 12000
        post_nms_topN = 2000
        nms_thresh = 0.7
        min_size = 8

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)
        logger.debug("feat_hegiht = {}, feat_width = {}\n".format(feat_height, feat_width))

        #shift_x: [W]: 生成0-feat_width(base_feat的width)数量的一维向量,每个元素乘16(但不是恢复图像原尺寸W)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        logger.debug("shift_x = {}".format(shift_x.shape))

        #shift_Y: [H]: 生成0-feat_height(base_feat的height)数量的一维向量,每个元素乘16(但不是恢复图像原尺寸H)
        shift_y = np.arange(0, feat_height) * self._feat_stride
        logger.debug("shift_y = {}".format(shift_y.shape))

        #shift_x:[H, W]->[[0, 16, 32, 48...,(W-1)*16],
        #                 [0, 16, 32, 48...,(W-1)*16],
        #                        ..............       ]
        #shift_y:[H, W]->[[0, 0, 0, 0...]
        #                 [16,16,16,16..]
        #                   ...........
        #                 [(H-1)*16,....]]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)   # 生成网格点坐标矩阵
        logger.debug("After np.meshgrid, shift_x = {}, shift_y = {}\n".format(shift_x.shape, shift_y.shape))

        #shifts:[H*W, 4]->[[0,  0,  0, 0],
        #                   ..........
        #                  [(W-1)*16, 0, (W-1)*16, 0],
        #                  [ 0 ,16, 0, 16],
        #                    ............
        #                  [(W-1)*16, 16, (W-1)*16, 16],
        #                     ............
        #                  [(W-1)*16, (H-1)*16, (W-1)*16, (H-1)*16]]

        # 矩阵按行叠加,转置,转为torch.Tensor
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        logger.debug("按行叠加，转置，再转为torch.Tensor后, shifts = {}\n".format(shifts.size()))

        # 把tensor变成在内存中连续分布的形式
        shifts = shifts.contiguous().type_as(scores).float()
        logger.debug("将其在内存内连续, shifts = {}\n".format(shifts.size()))

        A = self._num_anchors       # 9
        K = shifts.size(0)          # feature_map->(H, W) -> H * W = K   37*50=1850
        logger.debug("K = {}".format(K))

        self._anchors = self._anchors.type_as(scores)

        #anchors:[K, A, 4] 《=》[H*W, A, 4]
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        # [batch_size, H*W*A, 4(xmin,ymin,xmax,ymax)]
        anchors = anchors.view(1, K*A, 4).expand(batch_size, K*A, 4)

        # 调换维度顺序
        #bbox_delta:[batch_size, 36, H, W] => [batch_size, H, W, 36(9 anchors * 4)]
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        #bbox_delta:[batch_size, H, W, 36(9 anchors * 4)] => [batch_size, H*W*9, 4]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)  # 偏移量   each anchor have 4 coordinate

        #scores:[batch_size, 9, H, W] => [batch_szie, H, W, 9]
        scores = scores.permute(0, 2, 3, 1).contiguous()
        #scores:[batch_szie, H, W, 9] => [batch_size, H*W*9]
        scores = scores.view(batch_size, -1)   # each coordinate have 9 anchor

        #1.convert anchors into proposals
        proposals = bbox_transform_inv(anchors, bbox_deltas)
        #2.clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info)

        scores_keep = scores
        proposals_keep = proposals
        # 维度1, True代表降序
        _, order = torch.sort(scores_keep, 1, True)  # _是降序的排列,order是按其所在列显示

        output = scores.new(batch_size, post_nms_topN, 5).zero_()

        # 删除宽度或者高度小于阈值的预测框
        for i in range(batch_size):
            # 3. 按照排序顺序,选择前12000个
            proposals_single = proposals_keep[i]
            scores_single = scores[i]
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores.numel():  # numel统计数量
                order_single = order_single[:pre_nms_topN]


            # 4. proposal_single:[batch_size, pre_nms_topN, 4]
            # 5. scores_single : [batch_size, pre_nms_topN, 1]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1, 1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1),
                             nms_thresh, force_gpu=self.nms_gpu)  # torch.cat((a,b),dim)按维度拼接
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0 :
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            #output[i,:,0]是为了区分一个batch中的不同图片，
            #因为这些推荐框是在不同的feature_map上进行后续的选取
            output[i, :, 0] = i
            output[i, :num_proposal, 1:] = proposals_single

        return output




