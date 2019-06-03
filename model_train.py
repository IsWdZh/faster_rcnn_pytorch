import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
import torch.optim as optim
from torch.autograd import Variable
from data.roidb import PrepareData
from data.roi_batch_load import roibatchLoader
from data.sampler import sampler
from net.vgg import VGG16
import time
from logger import get_logger
import tqdm
import os
from IPython import embed
import PIL.Image as Image

max_iter = 100
epoch_save = 1
batch_size = 1
lr = 0.001
lr_decay_step = 50     # step to do lr decay (epoch)
lr_decay_gamma = 0.5    # learning rate decay ratio
momentum = 0.9
weight_decay = 0.0005
display_iter_num = 10    # 多少个batch展示一次
USE_WEIGHT_DECAY_ON_BIAS = False
DOUBLE_LR_ON_BIAS = True
USE_GPU = False


# now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

output_path = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_path):
    os.makedirs(output_path)
model_path = os.path.join(output_path, "model")
if not os.path.exists(model_path):
    os.makedirs(model_path)


logger = get_logger()

if torch.cuda.is_available():
    if not USE_GPU:
        logger.warning("CUDA is available, you can set USE_GPU to True to use gpu training")
    else:
        logger.info("Using GPU training model !")
else:
    if USE_GPU:
        USE_GPU = False
        logger.warning("CUDA is not available, Will use thr CPU training model by default!")


imdb_name = "voc_2007_trainval"
preparedata = PrepareData(imdb_name)
imdb, roidb, ratio_list, ratio_index = preparedata.combine()  # get each pic info
train_size = len(roidb)  # roidb is a list
logger.info("Classes: {}".format(imdb.classes))

sampler_batch = sampler(train_size, batch_size)
dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size,
                         imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch)

data_iter = iter(dataloader)
data = next(data_iter)

if USE_GPU:
    im_data = Variable(data[0]).cuda()
    im_info = Variable(data[1]).cuda()
    gt_boxes = Variable(data[2]).cuda()
    num_boxes = Variable(data[3]).cuda()
else:
    im_data, im_info = Variable(data[0]), Variable(data[1])
    gt_boxes, num_boxes = Variable(data[2]), Variable(data[3])
    

faster_rcnn = VGG16(imdb.classes, use_gpu=USE_GPU)
faster_rcnn.init_model()
logger.info(faster_rcnn)


optimizer = torch.optim.SGD(faster_rcnn.parameters(), lr=lr, momentum=momentum,
                            weight_decay=weight_decay)

# faster_rcnn.forward(im_data, im_info, gt_boxes, num_boxes)

params = []
for key, value in dict(faster_rcnn.named_parameters()).items():
    if value.requires_grad:
        if 'bias' in key:
            params += [{'params':[value], 'lr':lr*(DOUBLE_LR_ON_BIAS + 1),
                        'weight_decay': USE_WEIGHT_DECAY_ON_BIAS and weight_decay or 0}]
        else:
            params += [{'params':[value], 'lr':lr, 'weight_decay': weight_decay}]

if USE_GPU:
    faster_rcnn = faster_rcnn.cuda()

optimizer = torch.optim.SGD(params, momentum=0.9)

iters_per_epoch = int(train_size/batch_size)
for epoch in range(1, max_iter+1):
    faster_rcnn.train()
    loss_temp, epoch_loss = 0, 0
    start = time.time()

    if epoch % lr_decay_step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_decay_gamma * param_group['lr']
        lr *= lr_decay_gamma


    # bar = tqdm.tqdm(dataloader, total=len(dataloader))
    # for step, data in enumerate(bar):
    #     step += 1
    # data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
        logger.info("train sets percentage: {} / {}".format((step+1)*batch_size,train_size))

        data = next(data_iter)      # data: batch_size, so iteration iters_per_epoch
        
        if USE_GPU:
            im_data = Variable(data[0]).cuda()
            im_info = Variable(data[1]).cuda()
            gt_boxes = Variable(data[2]).cuda()
            num_boxes = Variable(data[3]).cuda()
        else:
            im_data, im_info = Variable(data[0]), Variable(data[1])
            gt_boxes, num_boxes = Variable(data[2]), Variable(data[3])

        faster_rcnn.zero_grad()
        # optimizer.zero_grad()

        rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, rois_label \
            = faster_rcnn(im_data, im_info, gt_boxes, num_boxes)

        logger.debug("rois={}, cls_prob={}, bbox_pred={}, \
                    rpn_loss_cls={}".format(rois.size(), cls_prob.size(), 
                                            bbox_pred.size(), rpn_loss_cls.size()))

        logger.debug("rpn_loss_box={}, RCNN_loss_cls={}, \n"
              "RCNN_loss_bbox={}, rois_label={}".format(rpn_loss_box, RCNN_loss_cls,
                                                        RCNN_loss_bbox, rois_label))


        loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + \
               RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
        logger.info("In a batch data, loss = {}".format(loss))

        # loss_temp += loss.data[0]
        loss_temp += loss.item()
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        faster_rcnn.clip_gradient(faster_rcnn, 10.)
        optimizer.step()

        if step % display_iter_num == 0:
            if step > 0:
                loss_temp /= (display_iter_num + 1)
            loss_rpn_cls = rpn_loss_cls.item()
            loss_rpn_box = rpn_loss_box.item()
            loss_rcnn_cls = RCNN_loss_cls.item()
            loss_rcnn_box = RCNN_loss_bbox.item()

            fg_cnt = torch.sum(rois_label.data.ne(0))
            bg_cnt = rois_label.data.numel() - fg_cnt

            logger.info("Average for every {} batches, loss = {}".format(display_iter_num, loss_temp))

            loss_temp = 0
            # bar.set_description("epoch{:2d} lr:{:.2e} loss:{:.4f} "
            #                     ":rpn_cls:{:.4f},rpn_box:{:.4f}, "
            #                     "rcnn_cls:{:.4f},rcnn_box{:.4f}".format(epoch, lr,
            #                                                             loss_temp,
            #                                                             loss_rpn_cls,
            #                                                             loss_rpn_box,
            #                                                             loss_rcnn_cls,
            #                                                             loss_rcnn_box))

    if epoch % epoch_save == 0:
        save_name = os.path.join(output_path,
                                 'faster_rcnn_epoch{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch + 1,
            'model': faster_rcnn.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name)
        logger.info('save model: {}'.format(save_name))

    end = time.time()

    logger.info("Each eopch mean loss = {}".format(float(epoch_loss / train_size)))
    logger.info("each epoch cost {}'s.".format(end - start))




