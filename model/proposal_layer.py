import torch
import torch.nn as nn
import numpy as np
from model.generate_anchors import generate_anchors
from model.bbox import bbox_transform_inv, clip_boxes
from model.nms import nms

class _ProposalLayer(nn.Module):
    '''
    这个类的作用是
    step1.在feature_map上产生的基础推荐框anchores: [batch_size, K*A, 4]
    step2.与RPN训练过后的基础推荐框的偏移参数 相运算得到RPN网络最后真正的推荐框集合。
    step3.通过将推荐框内有物体的得分排序，按分数从高到低取12000个推荐框
    step4.通过nms将每个feature_map上的推荐框缩小至2000个推荐框，并输出
    (step1. get base recommendation box generated on the feature graph
     step2. After operation of boxes and bbox_deltas，all the anchors of feature_map
            perhaps have the different width，height，center_x and center_y.The result is
            the truely recommendation-boxes of the RPN-net
     step3. By sorting the score of the recommendation box, 12000 recommendation boxes
            are selected according to the scores from high to low.
     step4. By NMS, the num of recommended boxes on each feature_map is reduced to 2000)
    '''
    def __init__(self, feat_stride, scales, ratios):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales),
                                                          ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)
        self.nms_gpu = True

    def forward(self, input):
        '''对于每个location上的(H,W),生成以该位置为中心的k个anchor boxe
        并讲预测到的bbox deltas应用到k个anchor上
        在图像上截取该box,移除宽或者高不满足要求的预测框
        按score从高到低对proposal排序
        使用NMS筛选一定数量的proposal，然后再从选出的里面取N个'''
        # input[0]: rpn_cls_prob.data [batch_size, 18, H, W]
        # input[1]: rpn_bbox_pred.data [batch_size, 36, H, W]
        # input[2]: im_info [h,w,ratio]
        # input[3]: cfg_key
        scores = input[0][:,self._num_anchors: , :, :]
        bbox_deltas = input[1]
        im_info = input[2]

        cfg_key = input[3]
        pre_nms_topN = 12000
        post_nms_topN = 2000
        nms_thresh = 0.7
        min_size = 8

        batch_size = bbox_deltas.size(0)

        feat_hegiht, feat_width = scores.size(2), scores.size(3)

        #shift_x:[W]->[0, 16, 32, 48...,(W-1)*16]  缩小的16
        shift_x = np.arange(0, feat_width) * self._feat_stride

        #shift_Y:[H]->[0, 16, 32, 48...,(H-1)*16]
        shift_y = np.arange(0, feat_hegiht) * self._feat_stride

        #shift_x:[H, W]->[[0, 16, 32, 48...,(W-1)*16],
        #                 [0, 16, 32, 48...,(W-1)*16],
        #                        ..............       ]
        #shift_y:[H, W]->[[0, 0, 0, 0...]
        #                 [16,16,16,16..]
        #                   ...........
        #                 [(H-1)*16,....]]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)   # 生成网格点坐标矩阵

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
        # 把tensor变成在内存中连续分布的形式
        shifts = shifts.contiguous().type_as(scores).float()

        A = self._num_anchors       # 9
        K = shifts.size(0)          # feature_map->(H, W) -> H * W = K

        self._anchors = self._anchors.type_as(scores)

        #anchors:[K, A, 4] 《=》[H*W, A, 4]
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K*A, 4).expand(batch_size, K*A, 4)

        # 调换维度顺序
        #bbox_delta:[batch_size, 36, H, W] => [batch_size, H, W, 36(9 anchors * 4)]
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        #bbox_delta:[batch_size, H, W, 36(9 anchors * 4)] => [batch_size, H*w*9, 4]
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)  # 偏移量

        #scores:[batch_size, 9, H, W] => [batch_szie, H, W, 9]
        scores = scores.permute(0, 2, 3, 1).contiguous()
        #scores:[batch_szie, H, W, 9] => [batch_size, H*W*9]
        scores = scores.view(batch_size, -1)

        #1.convert anchors into proposals
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        #2.clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)

        scores_keep = scores
        proposals_keep = proposals
        # 维度1, True代表降序
        _, order = torch.sort(scores_keep, 1, True)

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # 3. 删除宽度或者高度小于阈值的预测框
            proposals_single = proposals_keep[i]
            scores_single = scores[i]
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores.numel():
                order_single = order_single[:pre_nms_topN]


            # 4. proposal_single:[batch_size, pre_nms_topN, 4]
            # 5. scores_single : [batch_size, pre_nms_topN, 1]
            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1),
                             nms_thresh, force_cpu=not self.nms_gpu)  # torch.cat((a,b),dim)按维度拼接
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0 :
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            #output[i,:,0]是为了区分一个batch中的不同图片，
            #因为这些推荐框是在不同的feature_map上进行后续的选取
            output[i,:,0] = i
            output[i,:num_proposal, 1:] = proposals_single

        return output




