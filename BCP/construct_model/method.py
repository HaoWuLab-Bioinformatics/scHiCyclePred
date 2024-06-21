import os, random

import pandas as pd
import numpy as np
from methods import generate_bin, replace_linetodot
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from pytorch_tools import EarlyStopping

from torch.optim import Adam
#网络结构以及训练方法

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_BCP_dict():
    file_path = "./Data/BCP_Nor.npy"
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
        bcp = []
        for chr in chr_list:
            # 不考虑Y染色体的接触信息
            if chr == "chrY":
                continue
            bcp.append(BCP[cell_name][chr])
        X.append(np.concatenate(bcp).tolist())
    # print(len(X))输出749。X是一个列表，其中的每一个元素也是一个一维列表，一个一维列表存的是一个细胞的全部bcp特征。
    # print(np.array(X).shape)  # 如果输出(748, 2660)，也就是说当前的idX集合共有748个细胞，每个细胞对应一个包含有2660个特征数据的列表。一个细胞就对应一行数据
    # print('BCP', X[:1])
    # torch.from_numpy()方法把数组转换成张量
    # 与torch.tensor()的区别：使用torch.from_numpy更加安全，使用tensor.Tensor在非float类型下会与预期不符。
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]
    # np.array(X).shape[0]返回的是当前当前的idX集合中细胞的个数


class montage_model(nn.Module):
    def __init__(self, model_para):
        super(montage_model, self).__init__()
        # (652-5+2)/1 + 1 = 652
        kernel_size, cnn_feature, linear_size, dp, out_feature, Con_layer, linear_layer = model_para
        self.linear_size = linear_size
        self.linear_layer = linear_layer
        linear_size = 2660
        for i in range(Con_layer):
            linear_size = int(((linear_size+2*1-kernel_size)/1+1)//2)
        self.linear_size = linear_size
        self.cnn_feature = cnn_feature
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=cnn_feature,
            kernel_size=kernel_size,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(num_features=cnn_feature)
        self.rule1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dp)
        self.Con = nn.Sequential()

        for i in range(Con_layer-1):
            layer_id = str(i+2)
            self.Con.add_module("conv%s" % layer_id,nn.Conv1d(in_channels=cnn_feature,out_channels=cnn_feature,kernel_size=kernel_size,stride=1,padding=1))
            self.Con.add_module("bach%s" % layer_id,nn.BatchNorm1d(num_features=cnn_feature))
            self.Con.add_module("relu%s" % layer_id,nn.ReLU())
            self.Con.add_module("maxp%s" % layer_id,nn.MaxPool1d(kernel_size=2))
            self.Con.add_module("drop%s" % layer_id,nn.Dropout(dp))

        if linear_layer == 1:
            self.fc = nn.Linear(in_features=cnn_feature * linear_size, out_features=4)
        else:
            self.fc1 = nn.Linear(in_features=cnn_feature * linear_size, out_features=out_feature)
            self.relu2 = nn.ReLU()
            self.Linear = nn.Sequential()
            for i in range(linear_layer-2):
                l_layer_id = str(i + 2)
                self.Linear.add_module("linear%s" % l_layer_id,nn.Linear(in_features=out_feature, out_features=out_feature))
                self.Linear.add_module("linear_relu%s" % l_layer_id,nn.ReLU())
                self.Linear.add_module("linear_dropout%s" % l_layer_id, nn.Dropout(dp))
            self.fc2 = nn.Linear(in_features=out_feature, out_features=4)

        # Decay:488  Inscore:489


    def forward(self, x):
        x = self.rule1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.Con(x)

        x = x.view(-1, self.cnn_feature  * self.linear_size)
        if self.linear_layer == 1:
            x = self.fc(x)
        else:# Decay:492  Inscore:489
            x = self.fc1(x)
            x = self.dropout(x)
            x = self.relu2(x)
            x = self.Linear(x)
            x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

def CNN_train(epoch, model, optimizer, train_loader, loss_fn):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
        images = images.type(torch.FloatTensor)
        images = torch.unsqueeze(images, dim=1)
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = model(images)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        train_loss += train_loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images), len(train_loader.dataset),
                   100. * i / len(train_loader), train_loss.cpu().data * images.size(0)))
    return model, optimizer, train_loader

def CNN_val(epoch,model, test_loader, loss_fn, test_size):
    for i, (images, labels) in enumerate(test_loader):
        images = images.type(torch.FloatTensor)
        images = torch.unsqueeze(images, dim=1)
        # images = images.to(device)
        # labels = labels.to(device)
        labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
        outputs = model(images)
        val_loss = loss_fn(outputs, labels)
        _, prediction = torch.max(outputs.data, 1)
        label_pred = prediction.cpu().numpy()
        label = labels.data.cpu().numpy()
        prediction_num = int(torch.sum(prediction == labels.data))
        val_accuracy = prediction_num / test_size
        print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images), len(test_loader.dataset),
                   100. * i / len(test_loader), val_loss.cpu().data * images.size(0)))
    return label_pred, label, val_loss, model, val_accuracy


def CNN_1D_montage(BCP, tr_x, tr_y, val_x, val_y, lr, fold,model_para):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch()
    train_dataset, train_size = load_BCP_data(BCP, tr_x, tr_y)
    val_dataset, val_size = load_BCP_data(BCP, val_x, val_y)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_size,
                            shuffle=False)
    model = montage_model(model_para)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    num_epochs = 150
    min_loss = 100000.0
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    path = "./BCP/model/model_construct/Cross%s/" % fold
    for epoch in range(num_epochs):
        model.train()
        model, optimizer, train_loader = CNN_train(epoch, model, optimizer, train_loader, loss_fn)
        model.eval()
        val_label, label, val_loss, model, val_accuracy = CNN_val(epoch, model, val_loader,
                                                                          loss_fn, val_size)
        early_stopping(val_loss, model, path, fold)
        if early_stopping.early_stop:

            print(val_accuracy)
            model_path = path + "checkpoint%s.pt" % fold
            model.load_state_dict(torch.load(model_path))
            val_label, label, val_loss, model, val_accuracy = CNN_val(100000, model,  val_loader,
                                                                              loss_fn, val_size)
            break
    torch.cuda.empty_cache()

    return val_accuracy

