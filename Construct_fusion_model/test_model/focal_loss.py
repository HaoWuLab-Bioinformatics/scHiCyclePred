import torch
import torch.nn as nn

# https://zhuanlan.zhihu.com/p/562641889
# 首先，明确一下loss函数的输入：
# 一个pred，shape为 (bs, num_classes)，并且未经过softmax ；
# 一个target，shape为 (bs)，也就是一个向量，并且未经过one_hot编码。
# 通过前面的公式可以得出，我们需要在loss实现是做三件事情：
#
# 找到当前batch内每个样本对应的类别标签，然后根据预先设置好的alpha值给每个样本分配类别权重
# 计算当前batch内每个样本在类别标签位置的softmax值，作为公式里的
#  ，因为不管是focal loss还是cross_entropy_loss，每个样本的n个概率中不属于真实类别的都是用不到的
# 计算原始的cross_entropy_loss ，但不能求平均，需要得到每个样本的cross_entropy_loss ，因为需要对每个样本施加不同的权重

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，四分类中每一类的权重取得都是该类别出现频率的倒数
        :param gamma: 困难样本挖掘的gamma,gamma是给分错的样本的权重调大。
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        device = try_gpu(3)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 4)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss