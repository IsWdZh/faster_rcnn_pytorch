import numpy as np
import torch
def generate_anchors(base_size = 16, ratios =[0.5, 1, 2],
                     scales = 2**np.arange(3,6)):
    '''
    输入：参考anchor，ratios，scales
    输出：9个anchor：
             array([[ -83.,  -39.,  100.,   56.],
                    [-175.,  -87.,  192.,  104.],
                    [-359., -183.,  376.,  200.],
                    [ -55.,  -55.,   72.,   72.],
                    [-119., -119.,  136.,  136.],
                    [-247., -247.,  264.,  264.],
                    [ -35.,  -79.,   52.,   96.],
                    [ -79., -167.,   96.,  184.],
                    [-167., -343.,  184.,  360.]])
    (input:reference anchor, ratios, scales
     output:a set of nine anchors)
     通过枚举宽和高的比例X,生成anchor 与参考(0,0,15,15)窗口等比例缩放
    base_anchor 的大小为 16×16的, 其坐标为(0,0,15,15)
    '''

    #构造一个基础的anchor，面积为16*16
    # （generate a reference anchor which area is 16*16）
    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    ratio_anchors = _ratio_enum(base_anchor, ratios)

    anchors = np.vstack([_scale_enum(ratio_anchors[i,:], scales)
                        for i in range(ratio_anchors.shape[0])])

    return anchors

def _whctrs(anchor):
    '''
    输入的是一个anchor的[xmin, ymin, xmax, ymax]
    输出的是一个anchor的[width, height, x_center, y_center]
    返回一个anchor的宽, 高, 以及中心点的(x,y)坐标值
    给定一组围绕中心点(x_ctr, y_ctr) 的 widths(ws) 和 heights(hs) 序列, 输出对应的 anchors
    '''
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)

    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    '''
    输入的是一系列矩阵，它们weight，height不同，center_x，center_y相同
    输出的是一系列矩阵，格式为[[xmin, ymin, xmax, ymax],[...]...]
    (input：a set of anchors which have different weight 、height
            and have the same center_x 、center_y)
      output: a set if anchors which format is [[xmin, ymin, xmax, ymax],[...]...]
    )
    '''
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    '''
    输入：参考anchor（16*16）与缩放比集合ratios
    输出：三个anchor，它们的面积分为参考anchor面积的0.5，1，2倍
    在给定anchor下, 根据scale的值枚举所有可能的anchor box
    (注意, base_anchor的size只是作用一个过渡使用, 后面的语句会利用scales参数将其覆盖)
    '''
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratio = size / ratios
    ws = np.round(np.sqrt(size_ratio))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    '''根据给定的anchor(box), 枚举其所有可能scales的anchors(boxes)'''
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

'''
if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors(scales=np.array([8, 16, 32]), ratios=np.array([0.5, 1, 2]))
    anchors = torch.from_numpy(a)
    print(anchors)
    print(anchors.size(0))
    # from IPython import embed; embed()

'''