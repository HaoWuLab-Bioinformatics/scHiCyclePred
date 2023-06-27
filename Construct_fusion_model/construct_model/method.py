import os, random

import pandas as pd
import numpy as np
from methods import generate_bin, replace_linetodot, replace_dottoline
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from pytorch_tools import EarlyStopping

from torch.optim import Adam
from focal_loss import MultiClassFocalLossWithAlpha

# 该函数用于固定所有的随机种子，包括PyTorch框架的随机种子，cuda的随机种子等，
# 从而保证每次运行网络的时候相同输入的输出是固定的
def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 现在我们定义了两个方便的函数， 这两个函数允许我们在不存在所需所有GPU的情况下运行代码。
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 读取BCP特征集
def load_BCP_dict():
    file_path = "../../Data/Bin_contact_strength/Full_chr/Multi_channel/Nor/Bin_contact_strength(chr).npy"
    Data = np.load(file_path, allow_pickle=True).item()
    return Data

# 读取CDP特征集
def load_CDP_dict():
    CDP_file = "../../Data/MyCDD/CDD.txt"
    Data = pd.read_table(CDP_file, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,nrows=None)
    return Data

# 读取SBCP特征集
def load_SBCP_dict():
    file_path = '../../Data/Small_Domain_Struct_Contact_pro/Small_Domain_Struct_Contact_pro_scale(up_tran)(1).npy'
    Data = np.load(file_path, allow_pickle=True).item()
    return Data


def load_BCP_data(BCP, idX, Y):
    # generate_bin()函数用于返回一个字典，字典中存的是每条染色体能够切出的块的数目，形如：
    # {'chr1': 198, 'chr2': 183, .... 'chrX': 172, 'chrY': 92}
    index = generate_bin()
    chr_list = sorted(index.keys())
    # print('BCP',idX[:5])
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0]) + "_reads"
        sbcp = []
        for chr in chr_list:
            # 不考虑Y染色体的接触信息
            if chr == "chrY":
                continue
            sbcp.append(BCP[cell_name][chr])
        X.append(np.concatenate(sbcp).tolist())
    # print(len(X))输出749。X是一个列表，其中的每一个元素也是一个一维列表，一个一维列表存的是一个细胞的全部bcp特征。
    print(np.array(X).shape)  # 如果输出(748, 2660)，也就是说当前的idX集合共有748个细胞，每个细胞对应一个包含有2660个特征数据的列表。一个细胞就对应一行数据
    # print('BCP', X[:1])
    # torch.from_numpy()方法把数组转换成张量
    # 与torch.tensor()的区别：使用torch.from_numpy更加安全，使用tensor.Tensor在非float类型下会与预期不符。
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]
    # np.array(X).shape[0]返回的是当前当前的idX集合中细胞的个数

def load_CDP_data(CDP, idX, Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = CDP.loc[CDP['cell_nm'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        # 如果不在最后那个[0]的话value是
        # [[0.0063179558030927, 0.0056127625585562, ..., 0.0018277457562478]]
        # 加上以后就是[0.0063179558030927, 0.0056127625585562, ..., 0.0018277457562478]
        X.append(value)
    # print('CDP', X[:1])
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    # print(deal_dataset[:1])
    return deal_dataset, np.array(X).shape[0]

def load_SBCP_data(SBCP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0]) + "_reads"
        sbcp = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            sbcp.append(SBCP[cell_name][chr])
        X.append(np.concatenate(sbcp).tolist())
    print(np.array(X).shape)
    # print('SBCP', X[:3])
    # print('SBCP',X)
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]

def com_linearsize(linear_size,Con_layer,kernel_size):
    for i in range(Con_layer):
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size


# [kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer]
# [7, 32, 0.3, 0, [2, 2, 2], 3]
# montage_model(
#   (conv1_BCP): Conv1d(1, 32, kernel_size=(7,), stride=(1,), padding=(1,))
#   (bn1_BCP): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (rule1_BCP): ReLU()
#   (pool_BCP): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (Con_BCP): Sequential(
#     (conv2): Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(1,))
#     (bach2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu2): ReLU()
#     (maxp2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv1_CDP): Conv1d(1, 32, kernel_size=(7,), stride=(1,), padding=(1,))
#   (bn1_CDP): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (rule1_CDP): ReLU()
#   (pool_CDP): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (dropout_CDP): Dropout(p=0.3, inplace=False)
#   (Con_CDP): Sequential(
#     (conv2): Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(1,))
#     (bach2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu2): ReLU()
#     (maxp2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (conv1_SBCP): Conv1d(1, 32, kernel_size=(7,), stride=(1,), padding=(1,))
#   (bn1_SBCP): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (rule1_SBCP): ReLU()
#   (pool_SBCP): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (Con_SBCP): Sequential(
#     (conv2): Conv1d(32, 32, kernel_size=(7,), stride=(1,), padding=(1,))
#     (bach2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu2): ReLU()
#     (maxp2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (fc1): Linear(in_features=43040, out_features=0, bias=True)
#   (relu2): ReLU()
#   (dropout): Dropout(p=0.3, inplace=False)
#   (Linear): Sequential(
#     (linear2): Linear(in_features=0, out_features=0, bias=True)
#     (linear_relu2): ReLU()
#     (linear_dropout2): Dropout(p=0.3, inplace=False)
#   )
#   (fc2): Linear(in_features=0, out_features=4, bias=True)
# )
# 看网络结构说明
class montage_model(nn.Module):
    def __init__(self, model_para):
        super(montage_model, self).__init__()
        # (652-5+2)/1 + 1 = 652
        kernel_size, cnn_feature,  dp, out_feature, Con_layer, linear_layer = model_para
        self.linear_layer = linear_layer
        linear_size_BCP_init = 2660
        linear_size_CDP_init = 98
        linear_size_SBCP_init = 2660
        self.Con_layer_BCP, self.Con_layer_CDP, self.Con_layer_SBCP = Con_layer
        linear_size_BCP = com_linearsize(linear_size_BCP_init,self.Con_layer_BCP,kernel_size)
        linear_size_CDP = com_linearsize(linear_size_CDP_init, self.Con_layer_CDP, kernel_size)
        linear_size_SBCP = com_linearsize(linear_size_SBCP_init, self.Con_layer_SBCP, kernel_size)
        self.linear_size_BCP = linear_size_BCP
        self.linear_size_CDP = linear_size_CDP
        self.linear_size_SBCP = linear_size_SBCP
        self.cnn_feature = cnn_feature
        if self.Con_layer_BCP!=0:
            self.conv1_BCP = nn.Conv1d(
                in_channels=1,
                out_channels=cnn_feature,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_BCP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_BCP = nn.ReLU()
            self.pool_BCP = nn.MaxPool1d(kernel_size=2)
            self.dropout_BCP = nn.Dropout(dp)
            self.Con_BCP = nn.Sequential()
            for i in range(self.Con_layer_BCP-1):
                layer_id = str(i+2)
                self.Con_BCP.add_module("conv%s" % layer_id,nn.Conv1d(in_channels=cnn_feature,out_channels=cnn_feature,kernel_size=kernel_size,stride=1,padding=1))
                self.Con_BCP.add_module("bach%s" % layer_id,nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_BCP.add_module("relu%s" % layer_id,nn.ReLU())
                self.Con_BCP.add_module("maxp%s" % layer_id,nn.MaxPool1d(kernel_size=2))
                self.Con_BCP.add_module("drop%s" % layer_id,nn.Dropout(dp))
        if self.Con_layer_CDP != 0:
            self.conv1_CDP = nn.Conv1d(
                in_channels=1,
                out_channels=cnn_feature,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_CDP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_CDP = nn.ReLU()
            self.pool_CDP = nn.MaxPool1d(kernel_size=2)
            self.dropout_CDP = nn.Dropout(dp)
            self.Con_CDP = nn.Sequential()
            for i in range(self.Con_layer_CDP - 1):
                layer_id = str(i + 2)
                self.Con_CDP.add_module("conv%s" % layer_id, nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                                       kernel_size=kernel_size, stride=1, padding=1))
                self.Con_CDP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_CDP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_CDP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_CDP.add_module("drop%s" % layer_id, nn.Dropout(dp))
        if self.Con_layer_SBCP != 0:
            self.conv1_SBCP = nn.Conv1d(
                in_channels=1,
                out_channels=cnn_feature,
                kernel_size=kernel_size,
                stride=1,
                padding=1
            )
            self.bn1_SBCP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_SBCP = nn.ReLU()
            self.pool_SBCP = nn.MaxPool1d(kernel_size=2)
            self.dropout_SBCP = nn.Dropout(dp)
            self.Con_SBCP = nn.Sequential()
            for i in range(self.Con_layer_SBCP - 1):
                layer_id = str(i + 2)
                self.Con_SBCP.add_module("conv%s" % layer_id, nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                                       kernel_size=kernel_size, stride=1, padding=1))
                self.Con_SBCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_SBCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_SBCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                self.Con_SBCP.add_module("drop%s" % layer_id, nn.Dropout(dp))

        if linear_layer == 1:
            self.fc = nn.Linear(in_features=cnn_feature * (linear_size_BCP+linear_size_CDP+linear_size_SBCP), out_features=4)
        else:
            self.fc1 = nn.Linear(in_features=cnn_feature * (linear_size_BCP+linear_size_CDP+linear_size_SBCP), out_features=out_feature)
            self.relu2 = nn.ReLU()
            self.dropout = nn.Dropout(dp)
            self.Linear = nn.Sequential()
            for i in range(linear_layer-2):
                l_layer_id = str(i + 2)
                self.Linear.add_module("linear%s" % l_layer_id,nn.Linear(in_features=out_feature, out_features=out_feature))
                self.Linear.add_module("linear_relu%s" % l_layer_id,nn.ReLU())
                self.Linear.add_module("linear_dropout%s" % l_layer_id, nn.Dropout(dp))
            self.fc2 = nn.Linear(in_features=out_feature, out_features=4)

        # Decay:488  Inscore:489

    # # [kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer]
    # # [7, 32, 0.3, 0, [2, 2, 2], 3]
    def forward(self, x1,x2,x3):
        if self.Con_layer_BCP != 0:
            x1 = self.rule1_BCP(self.bn1_BCP(self.conv1_BCP(x1)))
            x1 = self.pool_BCP(x1)
            x1 = self.dropout_BCP(x1)
            x1 = self.Con_BCP(x1)
            x1 = x1.view(-1, self.cnn_feature * self.linear_size_BCP)
        if self.Con_layer_CDP != 0:
            x2 = self.rule1_CDP(self.bn1_CDP(self.conv1_CDP(x2)))
            x2 = self.pool_CDP(x2)
            x2 = self.dropout_CDP(x2)
            x2 = self.Con_CDP(x2)
            x2 = x2.view(-1, self.cnn_feature * self.linear_size_CDP)
        if self.Con_layer_SBCP != 0:
            x3 = self.rule1_SBCP(self.bn1_SBCP(self.conv1_SBCP(x3)))
            x3 = self.pool_SBCP(x3)
            x3 = self.dropout_SBCP(x3)
            x3 = self.Con_SBCP(x3)
            x3 = x3.view(-1, self.cnn_feature * self.linear_size_SBCP)
        if self.Con_layer_BCP != 0 and self.Con_layer_CDP!=0 and self.Con_layer_SBCP!=0:
            x = torch.cat((x1, x2, x3), 1)
        elif self.Con_layer_BCP != 0 and self.Con_layer_CDP!=0 and self.Con_layer_SBCP==0:
            x = torch.cat((x1, x2), 1)
        elif self.Con_layer_BCP != 0 and self.Con_layer_CDP == 0 and self.Con_layer_SBCP != 0:
            x = torch.cat((x1, x3), 1)
        elif self.Con_layer_BCP == 0 and self.Con_layer_CDP != 0 and self.Con_layer_SBCP != 0:
            x = torch.cat((x2, x3), 1)
        if self.linear_layer == 1:
            x = self.fc(x)
        else:# Decay:492  Inscore:489
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu2(x)
            x = self.Linear(x)
            x = self.fc2(x)
        # x = nn.functional.log_softmax(x, dim=1)
        return x

def CNN_train(epoch, model, optimizer, train_loader, loss_fn,device):
    train_loader_BCP, train_loader_CDP, train_loader_SBCP = train_loader
    i = 0
    for(images_BCP, labels_BCP),(images_CDP, labels_CDP),(images_SBCP, labels_SBCP) in zip(train_loader_BCP, train_loader_CDP, train_loader_SBCP):
        # print('images_BCP',images_BCP[:1])
        # print('labels_BCP',labels_BCP)
        # print('images_CDP', images_CDP[:1])
        # print('labels_CDP',labels_CDP)
        # print('images_SBCP', images_SBCP[:3])
        # print('labels_SBCP', labels_SBCP)
        # 清空过往梯度
        optimizer.zero_grad()
        labels = torch.Tensor(labels_BCP.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        # torch.unsqueeze()这个函数主要是对数据维度进行扩充.
        # 这里就是相当于加了一维通道，形成[批次，通道，特征]三个维度。
        images_BCP = torch.unsqueeze(images_BCP.type(torch.FloatTensor), dim=1)
        images_CDP = torch.unsqueeze(images_CDP.type(torch.FloatTensor), dim=1)
        images_SBCP = torch.unsqueeze(images_SBCP.type(torch.FloatTensor), dim=1)
        images_BCP = images_BCP.to(device)
        images_CDP = images_CDP.to(device)
        images_SBCP = images_SBCP.to(device)
        size1 = images_BCP.shape   # torch.Size([64, 1, 2660])
        size2 = images_CDP.shape   # torch.Size([64, 1, 98])
        size3 = images_SBCP.shape  # torch.Size([64, 1, 2660])
        outputs = model(images_BCP, images_CDP, images_SBCP)
        #  loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        # 这个地方为什么是train_loss.cpu().data * images_BCP.size(0)
        # 因为loss_fn函数返回的是整个批64条数据的一个平均损失，*images_BCP.size(0)就是总的损失
        # 这个地方乘不乘都可以。
        # images_BCP.size(0)就是批次大小64
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # ！！！！！！！！！！！！！可以把下面这个累加的代码去掉！！！！！！！！！！！！！！
        # ！！！！！！！！！！！！！把*images_BCP.size(0)也去掉！！！！！！！！！！！！！！
        # ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        # train_loss += train_loss.cpu().data * images_BCP.size(0)

        # torch.max的第一个参数outputs.data
        # 是把
        # tensor([[-0.8726, -1.6201, -1.5376, -1.7758],
        #         ....................................,
        # [-1.4242, -0.8394, -1.5071, -2.2466]], grad_fn=<LogSoftmaxBackward0>)
        # 变成
        # tensor([[-0.8726, -1.6201, -1.5376, -1.7758],
        #         ....................................,
        #         # [-1.4242, -0.8394, -1.5071, -2.2466]])
        # 把值取出来。
        # 第二个参数1指的是按照行进行选最大值，
        # 此时torch.max返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        # torch.max共返回两个tensor，第一个返回值_返回的是每一行中的最大值所组成的tensor。
        # 第二个返回值prediction是每一行中的最大元素在这一行的列索引所组成的tensor。
        # 我们只需要第二个返回值就够了
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_BCP), len(train_loader_BCP.dataset),
                   100. * i / len(train_loader_BCP), train_loss.cpu().data))
        # 之前没有下面这一行代码，而且i=0也放在了for循环的内部
        i = i + 1
    return model, optimizer

def CNN_val(epoch, model, test_loader, loss_fn, test_size, device):
    val_loader_BCP, val_loader_CDP, val_loader_SBCP= test_loader
    i = 0
    for (images_BCP, labels_BCP),(images_CDP, labels_CDP),(images_SBCP, labels_SBCP) in zip(val_loader_BCP, val_loader_CDP, val_loader_SBCP):
        images_BCP = torch.unsqueeze(images_BCP.type(torch.FloatTensor), dim=1)
        images_CDP = torch.unsqueeze(images_CDP.type(torch.FloatTensor), dim=1)
        images_SBCP = torch.unsqueeze(images_SBCP.type(torch.FloatTensor), dim=1)
        images_BCP = images_BCP.to(device)
        images_CDP = images_CDP.to(device)
        images_SBCP = images_SBCP.to(device)
        labels = torch.Tensor(labels_BCP.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        outputs = model(images_BCP, images_CDP, images_SBCP)
        # loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
        val_loss = loss_fn(outputs, labels)
        # 下面这一句有没有都可以，因为验证集就一个批次，封装验证集的dataloader的时候，批次大小直接=val_size
        # 也就是load_loader函数中batch_size=val_size
        # val_loss += val_loss.cpu().data * images_BCP.size(0)
        _, prediction = torch.max(outputs.data, 1)
        label_pred = prediction.cpu().numpy()
        label = labels.data.cpu().numpy()
        prediction_num = int(torch.sum(prediction == labels.data))
        val_accuracy = prediction_num / test_size
        print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_BCP), len(val_loader_SBCP.dataset),
                   100. * i / len(val_loader_SBCP), val_loss.cpu().data))
        i = i+1
    return label_pred, label, val_loss, model, val_accuracy


def load_loader(train_dataset, val_dataset, val_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_size,
                            shuffle=False)
    return train_loader, val_loader


# tr_x, val_x都是细胞代号的集合，形如[['1CDX1_1'] ['1CDX1_317']...['1CDX3_272'] ['1CDX3_334']]
# tr_y, val_y是真实的类别标签，形如：[['G1'] ['G1']...['mid_S'] ['mid_S']]
def CNN_1D_montage(BCP, CDP, SBCP, tr_x, tr_y, val_x, val_y, lr, fold, model_para, alpha, gamma):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seed_torch()函数用于固定所有的随机种子，从而保证每次运行网络的时候相同输入的输出是固定的
    seed_torch()
    # 封装成DataSet
    train_dataset_BCP, train_size_BCP = load_BCP_data(BCP, tr_x, tr_y)
    val_dataset_BCP, val_size_BCP = load_BCP_data(BCP, val_x, val_y)
    train_dataset_CDP, train_size_CDP = load_CDP_data(CDP, tr_x, tr_y)
    val_dataset_CDP, val_size_CDP = load_CDP_data(CDP, val_x, val_y)
    train_dataset_SBCP, train_size_SBCP = load_SBCP_data(SBCP, tr_x, tr_y)
    val_dataset_SBCP, val_size_SBCP = load_SBCP_data(SBCP, val_x, val_y)

    # 封装成DataLoader
    # train_loader = DataLoader(dataset=train_dataset,
    #                               batch_size=64,
    #                               shuffle=False)
    #     val_loader = DataLoader(dataset=val_dataset,
    #                             batch_size=val_size,
    #                             shuffle=False)
    train_loader_BCP, val_loader_BCP = load_loader(train_dataset_BCP, val_dataset_BCP, val_size_BCP)
    train_loader_CDP, val_loader_CDP = load_loader(train_dataset_CDP, val_dataset_CDP, val_size_CDP)
    train_loader_SBCP, val_loader_SBCP = load_loader(train_dataset_SBCP, val_dataset_SBCP, val_size_SBCP)
    model = montage_model(model_para)
    device = try_gpu(3)
    model.to(device)
    print(model)
    train_loader = [train_loader_BCP, train_loader_CDP, train_loader_SBCP]
    val_loader = [val_loader_BCP, val_loader_CDP, val_loader_SBCP]
    # 早停机制是一种正则化的手段，用于避免训练数据集上的过拟合。早期停止会跟踪验证损失（val_loss），
    # 如果损失连续几个 epoch 停止下降，训练就会停止。
    # pytorchtool.py 中的 EarlyStopping 类用于创建一个对象，
    # 以便在训练 PyTorch 模型时跟踪验证损失。每次验证丢失减少时，它都会保存模型的一个检查点
    # 我们在EarlyStopping类中设置了patience参数，即在最后一次验证损失改善后，
    # 我们希望在中断训练循环之前等待多少个epochs。
    # verbose参数如果是true，则为每个验证损失改进打印一条消息
    early_stopping = EarlyStopping(patience=10, verbose=True)
    num_epochs = 150
    min_loss = 100000.0
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
    path = "../model/model_construct_lr=000001/Cross%s/" % fold
    for epoch in range(num_epochs):
        model.train()
        model, optimizer = CNN_train(epoch, model, optimizer, train_loader, loss_fn,device)
        model.eval()
        val_label, label, val_loss, model, val_accuracy = CNN_val(epoch, model, val_loader,
                                                                          loss_fn, val_size_BCP,device)
        #  def __call__(self, val_loss, model_testupdate5, path, inter_fold):
        # 看一下这个early_stopping
        early_stopping(val_loss, model, path, fold)
        if early_stopping.early_stop:
            print(val_accuracy)
            model_path = path + "checkpoint%s.pt" % fold
            # torch.load("path路径")表示加载已经训练好的模型
            # 而model.load_state_dict（torch.load(PATH)）表示将训练好的模型参数重新加载至网络模型中
            # model_testupdate5.load_state_dict(torch.load(model_path))只保存和恢复模型中的参数
            model.load_state_dict(torch.load(model_path))
            val_label, label, val_loss, model, val_accuracy = CNN_val(100000, model,  val_loader,
                                                                              loss_fn, val_size_BCP,device)
            break
    torch.cuda.empty_cache()

    return val_accuracy

