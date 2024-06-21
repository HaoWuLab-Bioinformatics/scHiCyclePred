# coding:utf-8
import numpy as np
from itertools import chain
import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import SVC
from torch.optim import Adam
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from torchvision import transforms,datasets
import pandas as pd
import random
from scipy import stats
import numpy as np
from sklearn import tree
import collections
resolution = 1000000

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_bin():
    f = open("./mm10.main.nochrM.chrom.sizes")
    index= {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def mkdir(path):
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def str_reverse(s):
    s = list(s)
    s.reverse()
    return "".join(s)

def replace_linetodot(S):
    S = str_reverse(S)
    S = S.replace('_','.',1)
    return str_reverse(S)

def replace_dottoline(S):
    S = str_reverse(S)
    S = S.replace('.','_',1)
    return str_reverse(S)


def padding(list,max_size):
    list = list.tolist()
    if len(list)>max_size:
        list = list[:max_size]
    if len(list)<max_size:
        for i in range(max_size-len(list)):
            list.append(0)
    return list

def read_pair(path):
    file = open(path)
    file.readline()
    a = []
    for line in file.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a

def generate_contact_matrix(index,pair_list):
    contact_matrix = np.zeros((index, index))
    for pair in pair_list:
        bin1, bin2, num = pair
        contact_matrix[int(bin1), int(bin2)] += int(num)
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(num)
    return contact_matrix

def matrix_list(matrix):
    return list(chain.from_iterable(matrix))

def find_label(group):
    if group == 'late-S/G2':
        label = 0
    elif group == "early-S":
        label = 1
    elif group == "G1":
        label = 2
    elif group == "post-M" or group == "pre-M":
        label = 3
    return label



class Cnn_decay(nn.Module):
    def __init__(self,num_classes=4):
        super(Cnn_decay,self).__init__()
        #(652-3+2)/1 + 1 = 652
        self.conv1 = nn.Conv1d(
            in_channels=20,
            out_channels=32,
            kernel_size=3,
            stride = 1,
            padding=1
        )
        #shape(64, 12 652, 652)
        self.bn1 = nn.BatchNorm1d(num_features= 32)
        # shape(64, 12 652, 652)
        self.rule1 = nn.ReLU()
        # shape(64, 12 326, 326)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # shape(64, 20 326, 326)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # shape(64, 20, 326, 326)
        self.rule2 = nn.ReLU()
        # shape(64, 32, 326, 326)
        self.conv3 = nn.Conv1d(
            in_channels=20,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=32*98, out_features=num_classes)
        # Decay:488  Inscore:489


    def forward(self,x):
        x = self.rule1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.rule2(self.conv2(x))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = x.view(-1,32*98) # Decay:492  Inscore:489
        x = self.fc1(x)
        return x

def load_decay_data(idX,Y):
    index = generate_bin()
    file_path = "./Data/Decay/Decay.npy"
    chr_list = sorted(index.keys())
    Dict = np.load(file_path, allow_pickle=True).item()
    max_size = max(index.values())-1
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0])+"_reads"
        decay_matrix = []
        for chr in chr_list:
            if chr =="chrY":
                continue
            Dict[cell_name][chr]= padding(np.array(Dict[cell_name][chr]),max_size)
            decay_matrix.append(Dict[cell_name][chr])
        X.append(np.array(decay_matrix))
    print(np.array(X).shape)
    print(np.array(Y).shape)
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]

def CNN_1D_Decay(inter_tr_x, inter_tr_y, inter_val_x, inter_val_y, val_x, val_y, outer_fold, inter_fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("#####################################################")
    inter_train_dataset, inter_train_size = load_decay_data(inter_tr_x, inter_tr_y)
    inter_test_dataset, inter_test_size = load_decay_data(inter_val_x, inter_val_y)
    test_dataset, test_size = load_decay_data(val_x, val_y)
    train_loader = DataLoader(dataset=inter_train_dataset,
                              batch_size=64,
                              shuffle=True)
    interior_test_loader = DataLoader(dataset=inter_test_dataset,
                              batch_size=inter_test_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=test_size,
                              shuffle=False)
    model = Cnn_decay().to(device)
    model_path = "./model/model_%s/Decay_model_10/model%s.model" % (outer_fold, inter_fold)
    if os.path.exists(model_path):
        print("trained")
        prediction_num = 0
        interior_test_label = []
        model.load_state_dict(torch.load(model_path))
        for i, (images, labels) in enumerate(interior_test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            prediction_num += int(torch.sum(prediction == labels.data))
            interior_test_label.append(prediction.cpu().numpy().tolist())
            test_accuracy = prediction_num/inter_test_size
        best_accuracy = test_accuracy
        temp_test_data = []
        new_train_data = interior_test_label
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            temp_test_data.append(prediction.cpu().numpy().tolist())
        print(best_accuracy)
    else:
        num_epochs = 150
        best_accuracy = 0.0
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.001)
        loss_fn = nn.CrossEntropyLoss()
        min_loss = 100000.0
        for epoch in range(num_epochs):
            model.train()
            train_accuracy = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                optimizer.zero_grad()
                labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                images = images.type(torch.FloatTensor)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_accuracy += int(torch.sum(prediction == labels.data))
            if train_loss < min_loss:
                min_loss = train_loss
                prediction_num = 0
                interior_test_label = []
                for i, (images, labels) in enumerate(interior_test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    prediction_num += int(torch.sum(prediction == labels.data))
                    interior_test_label.append(prediction.cpu().numpy().tolist())
                test_accuracy = prediction_num / inter_test_size
                best_accuracy = test_accuracy
                es = 0
                new_train_data = interior_test_label
                path = "./model/model_%s/Decay_model_10/model%s.model" % (outer_fold,inter_fold)
                torch.save(model.state_dict(), path)
                temp_test_data = []
                for i, (images, labels) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    temp_test_data.append(prediction.cpu().numpy().tolist())
            else:
                es+=1
                if es > 15:
                    print("test_acc:"+str(best_accuracy))
                    print("epoch:"+str(epoch))
                    break
            model.eval()
    torch.cuda.empty_cache()
    return matrix_list(new_train_data), matrix_list(temp_test_data), best_accuracy


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)

def find_path(cell):
    if "1CDES" in cell:
        input_path = "./Data/CNN_chr_data/1CDES/" + cell+".npy"
    elif "1CDU" in cell:
        input_path = "./Data/CNN_chr_data/1CDU/" + cell+".npy"
    elif "1CDX1" in cell:
        input_path = "./Data/CNN_chr_data/1CDX1/" + cell+".npy"

    elif "1CDX2" in cell:
        input_path = "./Data/CNN_chr_data/1CDX2/" + cell+".npy"
    elif "1CDX3" in cell:
        input_path = "./Data/CNN_chr_data/1CDX3/" + cell+".npy"
    elif "1CDX4" in cell:
        input_path = "./Data/CNN_chr_data/1CDX4/" + cell+".npy"
    return input_path

def load_contact_matrix_data(idX,Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        cell_path = find_path(cell[0])
        cell_Dict = np.load(cell_path, allow_pickle=True).item()
        cell_matrix = []
        for chr in chr_list:
            if chr =="chrY":
                continue
            matrix = cell_Dict[chr]
            cell_matrix.append(matrix)
        X.append(cell_matrix)
    print(np.array(X).shape)
    print(len(Y))
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]

class Cnn_CM(nn.Module):
    def __init__(self,num_classes=4):
        super(Cnn_CM,self).__init__()
        #(652-3+2)/1 + 1 = 652
        self.conv1 = nn.Conv2d(
            in_channels=20,
            out_channels=32,
            kernel_size=3,
            stride = 1,
            padding=1
        )
        #shape(64, 12 652, 652)
        self.bn1 = nn.BatchNorm2d(num_features= 32)
        # shape(64, 12 652, 652)
        self.rule1 = nn.ReLU()
        # shape(64, 12 326, 326)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # shape(64, 20 326, 326)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # shape(64, 20, 326, 326)
        self.rule2 = nn.ReLU()
        # shape(64, 32, 326, 326)
        self.conv3 = nn.Conv2d(
            in_channels=20,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc = nn.Linear(in_features=32*65*65, out_features=num_classes)


    def forward(self,x):
        x = self.rule1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.rule2(self.conv2(x))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = x.view(-1,32*65*65)
        return self.fc(x)

def CNN_2D_CM(inter_tr_x, inter_tr_y, inter_val_x, inter_val_y, val_x, val_y, outer_fold, inter_fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("#####################################################")
    inter_train_dataset, inter_train_size = load_contact_matrix_data(inter_tr_x, inter_tr_y)
    inter_test_dataset, inter_test_size = load_contact_matrix_data(inter_val_x, inter_val_y)
    test_dataset, test_size = load_contact_matrix_data(val_x, val_y)
    train_loader = DataLoader(dataset=inter_train_dataset,
                              batch_size=32,
                              shuffle=True)
    interior_test_loader = DataLoader(dataset=inter_test_dataset,
                                      batch_size=32,
                                      shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=32,
                             shuffle=False)
    model = Cnn_CM().to(device)
    model_path = "./model/model_%s/CM_model_10/model%s.model" % (outer_fold, inter_fold)
    if os.path.exists(model_path):
        print("trained")
        prediction_num = 0
        interior_test_label = []
        model.load_state_dict(torch.load(model_path))
        for i, (images, labels) in enumerate(interior_test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            prediction_num += int(torch.sum(prediction == labels.data))
            interior_test_label.append(prediction.cpu().numpy().tolist())
            test_accuracy = prediction_num / inter_test_size
        best_accuracy = test_accuracy
        temp_test_data = []
        new_train_data = interior_test_label
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            temp_test_data.append(prediction.cpu().numpy().tolist())
        print(best_accuracy)
    else:
        num_epochs = 150
        best_accuracy = 0.0
        optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
        loss_fn = nn.CrossEntropyLoss()
        min_loss = 100000
        for epoch in range(num_epochs):
            model.train()
            train_accuracy = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                optimizer.zero_grad()
                labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                images = images.type(torch.FloatTensor)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)

                train_accuracy += int(torch.sum(prediction == labels.data))
            if train_loss < min_loss:
                min_loss = train_loss
                prediction_num = 0
                interior_test_label = []
                for i, (images, labels) in enumerate(interior_test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    prediction_num += int(torch.sum(prediction == labels.data))
                    interior_test_label.append(prediction.cpu().numpy().tolist())
                test_accuracy = prediction_num / inter_test_size
                best_accuracy = test_accuracy
                es = 0
                new_train_data = interior_test_label
                path = "./model/model_%s/CM_model_10/model%s.model" % (outer_fold, inter_fold)
                torch.save(model.state_dict(), path)
                temp_test_data = []
                for i, (images, labels) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    temp_test_data.append(prediction.cpu().numpy().tolist())
            else:
                es+=1
                if es > 15:
                    print("test_acc:"+str(best_accuracy))
                    print("epoch:"+str(epoch))
                    break
            model.eval()
    torch.cuda.empty_cache()
    return matrix_list(new_train_data), matrix_list(temp_test_data),best_accuracy

def load_MCM_data(idX,Y):
    MCM_file = "./Data/MCM/MCM.txt"
    Data = pd.read_table(MCM_file, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,nrows=None)
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = Data.loc[Data['cell_nm'] == cell_name].values[:, 1:].tolist()[0]
        X.append(value)
    print(np.array(X).shape)
    print(np.array(Y).shape)
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]


class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(in_features=8,out_features=6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=6, out_features=4)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.relu(out1)
        out = self.fc2(out2)
        return out


def FCNN_MCM(inter_tr_x, inter_tr_y, inter_val_x, inter_val_y, val_x, val_y, outer_fold, inter_fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("#####################################################")
    inter_train_dataset, inter_train_size = load_MCM_data(inter_tr_x, inter_tr_y)
    inter_test_dataset, inter_test_size = load_MCM_data(inter_val_x, inter_val_y)
    test_dataset, test_size = load_MCM_data(val_x, val_y)
    train_loader = DataLoader(dataset=inter_train_dataset,
                              batch_size=64,
                              shuffle=True)
    interior_test_loader = DataLoader(dataset=inter_test_dataset,
                                      batch_size=64,
                                      shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=False)
    model = FCNN().to(device)
    model_path = "./model/model_%s/MCM_model_10/model%s.model" % (outer_fold, inter_fold)
    if os.path.exists(model_path):
        print("trained")
        prediction_num = 0
        interior_test_label = []
        model.load_state_dict(torch.load(model_path))
        for i, (images, labels) in enumerate(interior_test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            prediction_num += int(torch.sum(prediction == labels.data))
            interior_test_label.append(prediction.cpu().numpy().tolist())
            test_accuracy = prediction_num / inter_test_size
        best_accuracy = test_accuracy
        temp_test_data = []
        new_train_data = interior_test_label
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            temp_test_data.append(prediction.cpu().numpy().tolist())
        print(best_accuracy)
    else:
        num_epochs = 150
        best_accuracy = 0.0
        optimizer = Adam(model.parameters(), lr = 0.05, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()
        min_loss = 100000
        for epoch in range(num_epochs):
            model.train()
            train_accuracy = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                optimizer.zero_grad()
                labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                images = images.type(torch.FloatTensor)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_accuracy += int(torch.sum(prediction == labels.data))
            if train_loss < min_loss:
                min_loss = train_loss
                prediction_num = 0
                interior_test_label = []
                for i, (images, labels) in enumerate(interior_test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    prediction_num += int(torch.sum(prediction == labels.data))
                    interior_test_label.append(prediction.cpu().numpy().tolist())
                test_accuracy = prediction_num / inter_test_size
                best_accuracy = test_accuracy
                es = 0
                new_train_data = interior_test_label
                path = "./model/model_%s/MCM_model_10/model%s.model" % (outer_fold, inter_fold)
                torch.save(model.state_dict(), path)
                temp_test_data = []
                for i, (images, labels) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    temp_test_data.append(prediction.cpu().numpy().tolist())
            else:
                es+=1
                if es > 25:
                    print("test_acc:"+str(best_accuracy))
                    print("epoch:"+str(epoch))
                    break
            model.eval()
    torch.cuda.empty_cache()
    return matrix_list(new_train_data), matrix_list(temp_test_data), best_accuracy



def load_BCS_data(idX,Y):
    index = generate_bin()
    file_path = "./Data/BCP/BCP.npy"
    chr_list = sorted(index.keys())
    Dict = np.load(file_path, allow_pickle=True).item()
    max_size = max(index.values())
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0]) + "_reads"
        decay_matrix = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            Dict[cell_name][chr] = padding(np.array(Dict[cell_name][chr]), max_size)
            decay_matrix.append(Dict[cell_name][chr])
        X.append(np.array(decay_matrix))
    print(np.array(X).shape)
    print(np.array(Y).shape)
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]


class Cnn_BCS(nn.Module):
    def __init__(self,num_classes=4):
        super(Cnn_BCS,self).__init__()
        #(652-5+2)/1 + 1 = 652
        self.conv1 = nn.Conv1d(
            in_channels=20,
            out_channels=32,
            kernel_size=5,
            stride = 1,
            padding=1
        )
        #shape(64, 12 652, 652)
        self.bn1 = nn.BatchNorm1d(num_features= 32)
        # shape(64, 12 652, 652)
        self.rule1 = nn.ReLU()
        # shape(64, 12 326, 326)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # shape(64, 20 326, 326)
        self.conv2 = nn.Conv1d(
            in_channels=32,
            out_channels=20,
            kernel_size=5,
            stride=1,
            padding=1
        )
        # shape(64, 20, 326, 326)
        self.rule2 = nn.ReLU()
        # shape(64, 32, 326, 326)
        self.conv3 = nn.Conv1d(
            in_channels=20,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=1
        )

        self.bn3 = nn.BatchNorm1d(num_features=32)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=32*94, out_features=4)
        # self.relu4 = nn.ReLU()
        # self.fc2 = nn.Linear(in_features=16*94, out_features=8*94)
        # self.relu5 = nn.ReLU()
        # self.fc3 = nn.Linear(in_features=8*94, out_features=94)
        # self.relu6 = nn.ReLU()
        # self.fc4 = nn.Linear(in_features=94, out_features=4)


        # Decay:488  Inscore:489


    def forward(self,x):
        x = self.rule1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.rule2(self.conv2(x))
        x = self.relu3(self.bn3(self.conv3(x)))

        x = x.view(-1,32*94) # Decay:492  Inscore:489
        x = self.fc1(x)
        # x = self.relu4(x)
        # x = self.fc2(x)
        # x = self.relu5(x)
        # x = self.fc3(x)
        # x = self.relu6(x)
        # x = self.fc4(x)

        return x

def CNN_1D_BCS(inter_tr_x, inter_tr_y, inter_val_x, inter_val_y, val_x, val_y,outer_fold, inter_fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("#####################################################")
    inter_train_dataset, inter_train_size = load_BCS_data(inter_tr_x, inter_tr_y)
    inter_test_dataset, inter_test_size = load_BCS_data(inter_val_x, inter_val_y)
    test_dataset, test_size = load_BCS_data(val_x, val_y)
    train_loader = DataLoader(dataset=inter_train_dataset,
                              batch_size=64,
                              shuffle=True)
    interior_test_loader = DataLoader(dataset=inter_test_dataset,
                                      batch_size=64,
                                      shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=64,
                             shuffle=False)
    model = Cnn_BCS().to(device)
    model_path = "./model/model_%s/BCS_model_10/model%s.model" % (outer_fold, inter_fold)
    if os.path.exists(model_path):
        print("trained")
        prediction_num = 0
        interior_test_label = []
        model.load_state_dict(torch.load(model_path))
        for i, (images, labels) in enumerate(interior_test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())
            labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            prediction_num += int(torch.sum(prediction == labels.data))
            interior_test_label.append(prediction.cpu().numpy().tolist())
            test_accuracy = prediction_num / inter_test_size
        best_accuracy = test_accuracy
        temp_test_data = []
        new_train_data = interior_test_label
        for i, (images, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                images = Variable(images.cuda())
            images = images.type(torch.FloatTensor)
            images = images.to(device)
            outputs = model(images)
            _, prediction = torch.max(outputs.data, 1)
            temp_test_data.append(prediction.cpu().numpy().tolist())
        print(best_accuracy)
    else:
        num_epochs = 150
        best_accuracy = 0.0
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()
        min_loss = 100000
        for epoch in range(num_epochs):
            model.train()
            train_accuracy = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                optimizer.zero_grad()
                labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                images = images.type(torch.FloatTensor)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.cpu().data * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_accuracy += int(torch.sum(prediction == labels.data))
            if train_loss < min_loss:
                min_loss = train_loss
                prediction_num = 0
                interior_test_label = []
                for i, (images, labels) in enumerate(interior_test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                        labels = Variable(labels.cuda())
                    labels = torch.Tensor(labels.type(torch.FloatTensor)).long()
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    prediction_num += int(torch.sum(prediction == labels.data))
                    interior_test_label.append(prediction.cpu().numpy().tolist())
                best_accuracy = prediction_num / inter_test_size
                es = 0
                new_train_data = interior_test_label
                path = "./model/model_%s/BCS_model_10/model%s.model" % (outer_fold, inter_fold)
                torch.save(model.state_dict(), path)
                temp_test_data = []
                for i, (images, labels) in enumerate(test_loader):
                    if torch.cuda.is_available():
                        images = Variable(images.cuda())
                    images = images.type(torch.FloatTensor)
                    images = images.to(device)
                    outputs = model(images)
                    _, prediction = torch.max(outputs.data, 1)
                    temp_test_data.append(prediction.cpu().numpy().tolist())
            else:
                es+=1
                if es > 20:
                    print("test_acc:"+str(best_accuracy))
                    print("epoch:"+str(epoch))
                    break
            model.eval()
    torch.cuda.empty_cache()
    return matrix_list(new_train_data), matrix_list(temp_test_data), best_accuracy


def vote(temp_test, valid_length, f_num):
    New_Test_Data = np.zeros((valid_length, f_num))
    for j in range(valid_length):
        for k in range(f_num):
            pr = []
            for l in temp_test.keys():
                pr.append(temp_test[l][j, k])
            New_Test_Data[j, k] = stats.mode(pr)[0][0]
    return New_Test_Data

def SVM(kernel,train_x,train_y,test_x):
    if kernel=="linear":
        c = 2.0 ** np.arange(1, 20)
        # gama =2.0 * np.arange(1, 500)
        parameters = {'C': c}
        optimizer = GridSearchCV(SVC(max_iter=5000), parameters)
        optimizer = optimizer.fit(train_x, train_y)
        params = optimizer.best_params_
        clf = SVC(kernel="linear", C=params['C'],max_iter=5000)
        clf.fit(train_x, train_y)
        label_pred = clf.predict(test_x)
    else:
        c = 2.0 ** np.arange(1, 20)
        gama =2.0 * np.arange(1, 500)
        parameters = {'C': c, 'gamma':gama}
        optimizer = GridSearchCV(SVC(max_iter=5000), parameters)
        optimizer = optimizer.fit(train_x, train_y)
        params = optimizer.best_params_
        clf = SVC(kernel=kernel, C=params['C'], gamma=params['gama'])
        clf.fit(train_x, train_y)
        label_pred = clf.predict(test_x)
    return label_pred

def write_result(result, path):
    with open(path, 'w') as wf:
        for i in result:
            wf.write(str(i)+"\n")
        wf.write(str(sum(result)/5)+'\n')


def Logistic(train_x,train_y,test_x):
    c = 2.0 ** np.arange(1, 20)
    parameters = {'C': c}
    optimizer = GridSearchCV(LogisticRegression(max_iter=2500), parameters)
    optimizer = optimizer.fit(train_x, train_y)
    params = optimizer.best_params_
    clf = LogisticRegression(max_iter=2500, C=params['C'])
    clf.fit(train_x, train_y)
    label_pred = clf.predict(test_x)
    return label_pred

def DCTree(criterion,train_x,train_y,test_x,):
    clf = tree.DecisionTreeClassifier(criterion=criterion)  #
    clf = clf.fit(train_x, train_y)
    label_pred = clf.predict(test_x)
    return label_pred


def Normalization(list):
    Min = min(list)
    Max = max(list)
    list = [(i-Min)/(Max-Min) for i in list]
    return list
def class_acc(label, pred, cla):
    count = 0
    for i in range(len(label)):
        if label[i] == pred[i] and label[i] == cla:
            count += 1
    return count / collections.Counter(label)[cla]
