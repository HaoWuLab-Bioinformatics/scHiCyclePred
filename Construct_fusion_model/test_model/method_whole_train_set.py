import os, random

import pandas as pd
import numpy as np
from methods import generate_bin, replace_linetodot, replace_dottoline, padding, matrix_list
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from pytorch_tools import EarlyStopping

from torch.optim import Adam

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_BCP_dict():
    file_path = "../../Data/BCP/Full_chr/Multi_channel/Nor/Bin_contact_strength(chr).npy"
    Data = np.load(file_path, allow_pickle=True).item()
    return Data

def load_CDP_dict():
    CDP_file = "../../Data/CDD/CDD.txt"
    Data = pd.read_table(CDP_file, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,
                         nrows=None)
    return Data

def load_SBCP_dict():
    file_path = '../../Data/SICP/Small_Domain_Struct_Contact_pro_scale(up_tran)(1).npy'
    Data = np.load(file_path, allow_pickle=True).item()

    return Data


def load_BCP_data(BCP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    # print('BCP',idX[:5])
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0]) + "_reads"
        sbcp = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            sbcp.append(BCP[cell_name][chr])
        X.append(np.concatenate(sbcp).tolist())
    print(np.array(X).shape)
    # print('BCP', X[:1])
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]

def load_CDP_data(CDP,idX,Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = CDP.loc[CDP['cell_nm'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        X.append(value)
    # print('CDD', X[:1])
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
    # print('SICP', X[:3])
    # print('SICP',X)
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]

def com_linearsize(linear_size,Con_layer,kernel_size):
    for i in range(Con_layer):
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size

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


    def forward(self, x1,x2,x3):
        if self.Con_layer_BCP!=0:
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
            x3 = x3.view(-1, self.cnn_feature  * self.linear_size_SBCP)
        if self.Con_layer_BCP!=0 and self.Con_layer_CDP!=0 and self.Con_layer_SBCP!=0:
            x = torch.cat((x1, x2, x3), 1)
        elif self.Con_layer_BCP!=0 and self.Con_layer_CDP!=0 and self.Con_layer_SBCP==0:
            x = torch.cat((x1, x2), 1)
        elif self.Con_layer_BCP != 0 and self.Con_layer_CDP == 0 and self.Con_layer_SBCP != 0:
            x = torch.cat((x1, x3), 1)
        elif self.Con_layer_BCP == 0 and self.Con_layer_CDP != 0 and self.Con_layer_SBCP != 0:
            x = torch.cat((x2, x3), 1)
        elif self.Con_layer_BCP == 0 and self.Con_layer_CDP == 0 and self.Con_layer_SBCP != 0:
            x = x3
        elif self.Con_layer_BCP != 0 and self.Con_layer_CDP == 0 and self.Con_layer_SBCP == 0:
            x = x1
        elif self.Con_layer_BCP == 0 and self.Con_layer_CDP != 0 and self.Con_layer_SBCP == 0:
            x = x2
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

def CNN_train(epoch, model, optimizer, train_loader, loss_fn, device):
    i = 0
    train_loader_BCP, train_loader_CDP, train_loader_SBCP = train_loader
    for  (images_BCP, labels_BCP),(images_CDP, labels_CDP),(images_SBCP, labels_SBCP) in zip(train_loader_BCP, train_loader_CDP, train_loader_SBCP):
        # print('images_BCP',images_BCP[:1])
        # print('labels_BCP',labels_BCP)
        # print('images_CDP', images_CDP[:1])
        # print('labels_CDP',labels_CDP)
        # print('images_SBCP', images_SBCP[:3])
        # print('labels_SBCP', labels_SBCP)
        optimizer.zero_grad()
        labels = torch.Tensor(labels_BCP.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        images_BCP = torch.unsqueeze(images_BCP.type(torch.FloatTensor), dim=1)
        images_CDP = torch.unsqueeze(images_CDP.type(torch.FloatTensor), dim=1)
        images_SBCP = torch.unsqueeze(images_SBCP.type(torch.FloatTensor), dim=1)
        images_BCP = images_BCP.to(device)
        images_CDP = images_CDP.to(device)
        images_SBCP = images_SBCP.to(device)
        outputs = model(images_BCP, images_CDP, images_SBCP)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        train_loss = train_loss.cpu().data
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_BCP), len(train_loader_BCP.dataset),
                   100. * i / len(train_loader_BCP), train_loss.cpu().data))
        i += 1
    return model, optimizer

def CNN_val(epoch,model, test_loader, loss_fn, test_size, device):
    val_loader_BCP, val_loader_CDP, val_loader_SBCP= test_loader
    i = 0
    for(images_BCP, labels_BCP),(images_CDP, labels_CDP),(images_SBCP, labels_SBCP) in zip(val_loader_BCP, val_loader_CDP, val_loader_SBCP):
        images_BCP = torch.unsqueeze(images_BCP.type(torch.FloatTensor), dim=1)
        images_CDP = torch.unsqueeze(images_CDP.type(torch.FloatTensor), dim=1)
        images_SBCP = torch.unsqueeze(images_SBCP.type(torch.FloatTensor), dim=1)
        images_BCP = images_BCP.to(device)
        images_CDP = images_CDP.to(device)
        images_SBCP = images_SBCP.to(device)
        labels = torch.Tensor(labels_BCP.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        outputs = model(images_BCP, images_CDP, images_SBCP)
        val_loss = loss_fn(outputs, labels)
        _, prediction = torch.max(outputs.data, 1)
        label_pred = prediction.cpu().numpy()
        label = labels.data.cpu().numpy()
        prediction_num = int(torch.sum(prediction == labels.data))
        val_accuracy = prediction_num / test_size
        print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_BCP), len(val_loader_SBCP.dataset),
                   100. * i / len(val_loader_SBCP), val_loss.cpu().data ))
        i = i+1
    return label_pred, label, val_loss, model, val_accuracy

def load_loader(train_dataset,test_dataset,test_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_size,
                            shuffle=False)
    return train_loader, test_loader

def CNN_1D_montage(BCP_train, CDP_train, SBCP_train,BCP_test_update5, CDP_test_update5, SBCP_test_update5, tr_x, tr_y, X_test, y_test, lr, model_para, alpha, gamma):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = 'test'
    seed_torch()
    train_dataset_BCP, train_size_BCP = load_BCP_data(BCP_train, tr_x, tr_y)
    test_dataset_BCP, test_size_BCP = load_BCP_data(BCP_test_update5, X_test, y_test)
    train_dataset_CDP, train_size_CDP = load_CDP_data(CDP_train, tr_x, tr_y)
    test_dataset_CDP, test_size_CDP = load_CDP_data(CDP_test_update5, X_test, y_test)
    train_dataset_SBCP, train_size_SBCP = load_SBCP_data(SBCP_train, tr_x, tr_y)
    test_dataset_SBCP, test_size_SBCP = load_SBCP_data(SBCP_test_update5, X_test, y_test)

    train_loader_BCP, test_loader_BCP = load_loader(train_dataset_BCP,test_dataset_BCP, test_size_BCP)
    train_loader_CDP,  test_loader_CDP = load_loader(train_dataset_CDP, test_dataset_CDP, test_size_CDP)
    train_loader_SBCP, test_loader_SBCP = load_loader(train_dataset_SBCP, test_dataset_SBCP, test_size_SBCP)
    model = montage_model(model_para)
    device = try_gpu(3)
    model.to(device)
    print(model)
    train_loader = [train_loader_BCP, train_loader_CDP,train_loader_SBCP]
    test_loader = [test_loader_BCP, test_loader_CDP, test_loader_SBCP]
    num_epochs = 100
    min_loss = 100000.0
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
    for epoch in range(num_epochs):
        model.train()
        model, optimizer = CNN_train(epoch, model, optimizer, train_loader, loss_fn,device)
        model.eval()
        test_label, label, test_loss, model, test_accuracy = CNN_val(epoch, model, test_loader,
                                                                          loss_fn, test_size_BCP,device)
    torch.cuda.empty_cache()

    return test_accuracy, test_label, label
