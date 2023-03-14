from collections import Counter

import pandas as pd
import xlsxwriter
from scipy import stats
import numpy as np
from method_whole_train_set import load_BCP_dict, load_CDP_dict, load_SBCP_dict, CNN_1D_montage
from sklearn.model_selection import train_test_split, StratifiedKFold
import random, os, torch
from sklearn.metrics import f1_score, precision_score

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    path = "../../Data/new_cell_inf.txt"
    cell_inf = pd.read_table(path, sep=' ', header='infer', names=None, index_col=None, dtype=None, engine=None,
                             nrows=None)
    cell_inf = cell_inf.sort_values(by='cycle', ascending=True)
    X_index = np.array(cell_inf.values.tolist())[:, 0:1]
    Y = []
    label = np.array(cell_inf.values.tolist())[:, 1:]
    for l in label:
        if l == 'G1':
            Y.append(0)
        elif l == "early_S":
            Y.append(1)
        elif l == "mid_S":
            Y.append(2)
        elif l == "late_S":
            Y.append(3)
    Y = np.array(Y)
    BCP = load_BCP_dict()
    CDP = load_CDP_dict()
    SBCP = load_SBCP_dict()
    Con_layer_BCP = 2
    Con_layer_CDP = 2
    Con_layer_SBCP = 2
    # [2,2,2] 7 32 0 0.3 0.0001
    # [0,2,2] 7,32,0,0.3,0.0001

    Con_layer = [Con_layer_BCP,Con_layer_CDP,Con_layer_SBCP]
    linear_layer = 1
    X_train, X_test, y_train, y_test = train_test_split(X_index, Y, test_size=0.2, random_state=4096, stratify=Y)

    kernel_size = 7
    cnn_feature = 32
    out_feature = 0
    dp = 0.3
    lr = 0.0001
    model_para = [kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer]
    test_acc, test_label, real_label = CNN_1D_montage(BCP, CDP, SBCP, X_train, y_train, X_test, y_test, lr, model_para)
    print('test_acc: ', test_acc)

    label_count = []
    for i, j in zip(test_label, real_label):
        if i == j:
            label_count.append(i)
    print("预测结果：", Counter(label_count))

    # The F1-score of each category is calculated from several values of TP, FP, and FN of each category.
    # micro then means that no distinction is made between categories, either label0 or label1.
    # as long as the TP, FP, and FN of each category are added up to a single F1-score value.
    # macro, in contrast to micro, macro first calculates the F1-score for each category
    # then directly calculates the arithmetic mean of the F1-score of each category that is the final F1-score shown.
    micro_F1 = f1_score(real_label, test_label, average='micro')
    print("micro_F1：", micro_F1)
    macro_F1 = f1_score(real_label, test_label, average='macro')
    print("macro_F1：", macro_F1)
    micro_Precision = precision_score(real_label, test_label, average='micro')
    print("micro_Precision：", micro_Precision)
    macro_Precision = precision_score(real_label, test_label, average='macro')
    print("macro_Precision：", macro_Precision)

if __name__ == '__main__':
    main()