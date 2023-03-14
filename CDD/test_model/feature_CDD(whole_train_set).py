import pandas as pd
import xlsxwriter
from scipy import stats
import numpy as np
from method_whole_train_set import load_CDP_dict, CNN_1D_montage
from sklearn.model_selection import train_test_split, StratifiedKFold
import random,os, torch
from collections import Counter
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
    CDP = load_CDP_dict()
    Con_layer = 2
    linear_layer = 1
    X_train, X_test, y_train, y_test = train_test_split(X_index, Y, test_size=0.2, random_state=4096, stratify=Y)
    kernel_size = 5
    cnn_feature = 64
    out_feature = 0
    dp = 0.2
    lr = 0.0001
    model_para = [kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer]
    test_acc, test_label, real_label = CNN_1D_montage(CDP, X_train, y_train, X_test, y_test, lr, model_para)

    print('test_acc: ', test_acc)
    label_count = []
    for i, j in zip(test_label, real_label):
        if i == j:
            label_count.append(i)
    print("预测结果：",Counter(label_count))

if __name__ == '__main__':
    main()