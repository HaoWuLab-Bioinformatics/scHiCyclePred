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