import pandas as pd
import xlsxwriter
import numpy as np
from method_whole_train_set import load_BCP_dict, load_CDP_dict, load_SBCP_dict, CNN_1D_montage
from sklearn.model_selection import train_test_split
import random, os, torch
from sklearn.metrics import f1_score, precision_score, balanced_accuracy_score

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
    # num用于统计每种类型细胞的数量
    num = {0: 0, 1: 0, 2: 0, 3: 0}
    label = np.array(cell_inf.values.tolist())[:, 1:]
    # label为：[['G1'] ['G1']...['mid_S'] ['mid_S']]
    for l in label:
        if l == 'G1':
            Y.append(0)
            num[0] = num[0] + 1
        elif l == "early_S":
            Y.append(1)
            num[1] = num[1] + 1
        elif l == "mid_S":
            Y.append(2)
            num[2] = num[2] + 1
        elif l == "late_S":
            Y.append(3)
            num[3] = num[3] + 1
    Y = np.array(Y)
    print(num)
    # alpha用于计算focal_loss
    # 每个类别对应的alpha=该类别出现频率的倒数
    alpha = []
    for value in num.values():
        ds = 1 / value
        alpha.append(ds)
    print(alpha)
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
    file = './Construct_fusion_model/test_result/test_result.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet1 = workbook.add_worksheet('model')
    worksheet1.write(0, 0, 'random_seed')
    worksheet1.write(0, 1, 'test_acc')
    worksheet1.write(0, 2, 'macro_F1')
    worksheet1.write(0, 3, 'macro_Precision')
    worksheet1.write(0, 4, 'bacc')
    row = 0
    linear_layer = 1
    kernel_size = 7
    cnn_feature = 32
    out_feature = 0
    dp = 0.2
    lr = 0.0001
    gamma = 1
    rand_array0 = [3900, 8849, 9261, 8659, 258, 2796, 4547, 953, 4526, 2361, 2625, 8299, 8468,
                   5980, 4340, 4858, 7814, 2272, 7804, 2375, 9896, 974, 7737, 9225, 1442, 7343,
                   4955, 1474, 3916, 8026, 638, 7111, 6987, 245, 8635, 7396, 6605,
                   5639, 5063, 7408, 7971, 4366, 618, 7575, 6390, 4622, 7676, 6571, 9194, 9643]
    for rs in rand_array0:
        X_train, X_test, y_train, y_test = train_test_split(X_index, Y, test_size=0.2, random_state=rs, stratify=Y)
        row += 1
        model_para = [kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer]
        test_acc, test_label, real_label = CNN_1D_montage(BCP, CDP, SBCP, X_train, y_train, X_test, y_test, lr, model_para, alpha, gamma)
        label_count = []
        for i, j in zip(test_label, real_label):
            if i == j:
                label_count.append(i)
        micro_F1 = f1_score(real_label, test_label, average='micro')
        macro_F1 = f1_score(real_label, test_label, average='macro')
        micro_Precision = precision_score(real_label, test_label, average='micro')
        macro_Precision = precision_score(real_label, test_label, average='macro')
        bacc =  balanced_accuracy_score(real_label, test_label)
        worksheet1.write(row, 0, rs)
        worksheet1.write(row, 1, test_acc)
        worksheet1.write(row, 2, macro_F1)
        worksheet1.write(row, 3, macro_Precision)
        worksheet1.write(row, 4, bacc)
    workbook.close()

if __name__ == '__main__':
    main()
