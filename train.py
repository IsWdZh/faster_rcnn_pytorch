import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import torch.optim as optim
from torch.autograd import Variable
from data.roidb import combined_roidb
from data.roi_batch_load import roibatchLoader
from data.sampler import sampler
from net.vgg import VGG16
# from faster_rcnn import VGG16
from IPython import embed

batch_size = 1
lr = 0.001
weight_decay = 0.0005
USE_WEIGHT_DECAY_ON_BIAS = False
DOUBLE_LR_ON_BIAS = True

imdb_name = "voc_2007_trainval"
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdb_name)
train_size = len(roidb)
print(imdb.classes)

sampler_batch = sampler(train_size, batch_size)

dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size,
                         imdb.num_classes, training=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                        sampler=sampler_batch, num_workers=0)

im_data = Variable(torch.FloatTensor(1))
im_info = Variable(torch.FloatTensor(1))
num_boxes = Variable(torch.LongTensor(1))
gt_boxes = Variable(torch.FloatTensor(1))

data_iter = iter(dataloader)
data = next(data_iter)

im_data.data.resize_(data[0].size()).copy_(data[0])
im_info.data.resize_(data[1].size()).copy_(data[1])
gt_boxes.data.resize_(data[2].size()).copy_(data[2])
num_boxes.data.resize_(data[3].size()).copy_(data[3])

faster_rcnn = VGG16(imdb.classes)
faster_rcnn.forward(im_data, im_info, gt_boxes, num_boxes)
embed()





'''
params = []
    for key, value in dict(faster_rcnn.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(DOUBLE_LR_ON_BIAS + 1),
                            'weight_decay': USE_WEIGHT_DECAY_ON_BIAS and weight_decay or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': weight_decay}]

optimizer = torch.optim.SGD(params, momentum=0.9)


print(len(roidb))
'''