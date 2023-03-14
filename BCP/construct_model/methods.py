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
#一些常用的方法

def generate_bin():
    f = open("mm10.main.nochrM.chrom.sizes")
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


