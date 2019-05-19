import torch
from torch.utils.data.sampler import Sampler

class sampler(Sampler):
    '''
        采样类：根据训练集总数量与batch-size来划分index, 构成index的集合
        例如：step1. train_size = 6, batch_size = 2 =>数据集的索引集合Index为[0,1,2,3,4,5]
             step2. 将Indexs随机打乱并且分成 train_size/batch_size 个集合：[1,3],[0,4],[2,5]
             step3. 每次迭代时,dataloader会调用该类的iter方法拿到一个index的集合--：[1,3]
    '''
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long() # 生成0-31的一维向量,再转换为1行32列的二位矩阵
        self.leftover_flag = False
        if train_size % batch_size:
            # 总样本数以mini-batch分,最后余下的样本部分    有剩余标志位改为True
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data