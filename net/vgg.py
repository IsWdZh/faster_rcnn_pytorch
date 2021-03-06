import torch
import torchvision.models as models
import torch.nn as nn
from net.faster_rcnn import Faster_RCNN


class VGG16(Faster_RCNN):
    '''VGG16 model'''
    def __init__(self, classes, use_gpu=False):
        self.use_gpu = use_gpu
        self.base_feat_out_dim = 512
        super(VGG16, self).__init__(classes)

    def _init_modules(self):
        vgg = models.vgg16(pretrained=True)

        # not using the last classify layer
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last Maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # fix the layers' parameters
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad=False

        if self.use_gpu:
            self.RCNN_top = vgg.classifier.cuda()
        else:
            self.RCNN_top = vgg.classifier
        # self.RCNN_top = vgg.classifier.cuda()

        self.RCNN_cls_score = nn.Linear(4096, self.num_classes)

        self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.num_classes)





