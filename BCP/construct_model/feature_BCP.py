import pandas as pd
import xlsxwriter
from scipy import stats
import numpy as np
from method import load_BCP_dict, CNN_1D_montage
from sklearn.model_selection import train_test_split, StratifiedKFold
import random, os, torch

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    path = "./Data/new_cell_inf.txt"
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
    Con_layer = 2
    linear_layer = 4
    file = './BCP/val_result/val_result.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet1 = workbook.add_worksheet('model')
    worksheet1.write(0, 1, 'kernel_size')
    worksheet1.write(0, 2, 'cnn_feature')
    worksheet1.write(0, 3, 'out_feature')
    worksheet1.write(0, 4, 'dp')
    worksheet1.write(0, 5, 'lr')
    worksheet1.write(0, 6, 'Acc')
    X_train, X_test, y_train, y_test = train_test_split(X_index, Y, test_size=0.2, random_state=4096, stratify=Y)
    ave_ACC = 0
    if Con_layer == 1:
        linear_size = 2709
    elif Con_layer == 2:
        linear_size = 2000
    elif Con_layer == 3:
        linear_size = 3000
    elif Con_layer == 4:
        linear_size = 4000
    row = 0
    for kernel_size in [7]:
        for cnn_feature in [16]:
            for out_feature in [16,32,64,128]:
                for dp in [0.2,0.3,0.4,0.5]:
                    for lr in [0.01,0.001,0.0001]:
                        sum_acc = 0
                        row+=1
                        Folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021).split(X_train, y_train)
                        for fold, (tr_idx, val_idx) in enumerate(Folds):
                            tr_x, tr_y, val_x, val_y = X_train[tr_idx], y_train[tr_idx], X_train[val_idx], y_train[val_idx]
                            model_para = [kernel_size, cnn_feature, linear_size, dp, out_feature, Con_layer, linear_layer]
                            print(model_para)
                            sum_acc += CNN_1D_montage(BCP,tr_x, tr_y, val_x, val_y, lr, fold, model_para)
                        worksheet1.write(row, 1, kernel_size)
                        worksheet1.write(row + 1, 2, cnn_feature)
                        worksheet1.write(row + 1, 3, out_feature)
                        worksheet1.write(row + 1, 4, dp)
                        worksheet1.write(row + 1, 5, lr)
                        worksheet1.write(row + 1, 6, sum_acc/5)
    workbook.close()

if __name__ == '__main__':
    main()
