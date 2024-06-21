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
