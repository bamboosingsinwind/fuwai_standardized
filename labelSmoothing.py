import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# https://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        # print(target.data)
        bs,class_num = x.shape
        # weight = torch.Tensor([ 4 if i>0.5 else 1 for i in target])#1  target*0.9#判定为阵发loss小一些(只用于二分类) 0.9
        weight = torch.tensor([[1,3] for _ in range(bs)]) #3 6
        if target.is_cuda and not weight.is_cuda:
            weight = weight.to(target.device)#weight.cuda()
        logprobs = F.log_softmax(x, dim=-1)
        target = F.one_hot(target,2)
        target = torch.clamp(target.float(),min=self.smoothing,max=self.confidence)
        target *= weight
        loss = -1*torch.sum(target*logprobs,1)

        return loss.mean()

# https://blog.csdn.net/weixin_41811314/article/details/115863126
class LabelSmoothing1(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
       

    def forward(self, x, target):
        # print(target.data)
        weight = torch.Tensor([ 3 if i>0.5 else 1 for i in target])
        if target.is_cuda and not weight.is_cuda:
            weight = weight.to(target.device)#weight.cuda()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))

        nll_loss = nll_loss*weight
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

