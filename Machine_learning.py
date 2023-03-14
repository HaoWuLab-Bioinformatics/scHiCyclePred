'''
The code on this page declares a number of machine learning methods for comparison
'''
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from Feature_sets_extraction.methods import generate_bin, replace_linetodot
from sklearn import svm,metrics


def load_BCP_dict():
    file_path = "../Data/BCP/Full_chr/Multi_channel/Nor/BCP(chr).npy"
    Data = np.load(file_path, allow_pickle=True).item()
    return Data

def load_CDD_dict():
    CDD_file = "../Data/CDD/CDD.txt"
    Data = pd.read_table(CDD_file, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,
                         nrows=None)
    return Data

def load_SICP_dict():
    file_path = '../Data/SICP/Small_Domain_Struct_Contact_pro_scale(up_tran)(1).npy'
    Data = np.load(file_path, allow_pickle=True).item()

    return Data

# Load BCP dataset according to the index (cell name) of the input training set, test set
def load_BCP_data(BCP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0]) + "_reads"
        bcp = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            bcp.append(BCP[cell_name][chr])
        X.append(np.concatenate(bcp).tolist())
    print(np.array(X).shape)
    return X, Y, np.array(X).shape[0]

# Load the CDD dataset according to the index (cell name) of the input training set, test set
def load_CDD_data(CDD,idX,Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = CDD.loc[CDD['cell_nm'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        X.append(value)

    return X, Y, np.array(X).shape[0]

# Load the SICP dataset according to the index (cell name) of the input training set, test set
def load_SICP_data(SICP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        cell_name = replace_linetodot(cell[0]) + "_reads"
        sicp = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            sicp.append(SICP[cell_name][chr])
        X.append(np.concatenate(sicp).tolist())
    return X, Y, np.array(X).shape[0]

# Three feature sets tested separately and three feature sets tested together, machine learning methods are SVM, logistic regression, random forest
def Multi_Classfier(Data, tr_x, tr_y, te_X, te_Y, type):
    if type == 'SICP':
        Train_X, Train_Y, Train_Size = load_SICP_data(Data, tr_x, tr_y)
        Test_X, Test_Y, Test_Size = load_SICP_data(Data, te_X, te_Y)
    elif type == 'BCP':
        Train_X, Train_Y, Train_Size = load_BCP_data(Data, tr_x, tr_y)
        Test_X, Test_Y, Test_Size = load_BCP_data(Data, te_X, te_Y)
    elif type == 'CDD':
        Train_X, Train_Y, Train_Size = load_CDD_data(Data, tr_x, tr_y)
        Test_X, Test_Y, Test_Size = load_CDD_data(Data, te_X, te_Y)
    elif type == 'SICP-CDD-BCP':
        SICP,CDD, BCP = Data
        CDD_Train_X, CDD_Train_Y, CDD_Train_Size = load_CDD_data(CDD, tr_x, tr_y)
        CDD_Test_X, CDD_Test_Y, CDD_Test_Size = load_CDD_data(CDD, te_X, te_Y)
        SICP_Train_X, SICP_Train_Y, SICP_Train_Size = load_SICP_data(SICP, tr_x, tr_y)
        SICP_Test_X, SICP_Test_Y, SICP_Test_Size = load_SICP_data(SICP, te_X, te_Y)
        BCP_Train_X, BCP_Train_Y, BCP_Train_Size = load_BCP_data(BCP, tr_x, tr_y)
        BCP_Test_X, BCP_Test_Y, BCP_Test_Size = load_BCP_data(BCP, te_X, te_Y)
        Train_X = np.hstack((SICP_Train_X, CDD_Train_X, BCP_Train_X))
        Train_Y = CDD_Train_Y
        Test_X = np.hstack((SICP_Test_X, CDD_Test_X, BCP_Test_X))
        Test_Y = CDD_Test_Y

    svm_train_acc,  svm_test_acc = SVM(Train_X, Train_Y,  Test_X, Test_Y)
    svm = [svm_train_acc, svm_test_acc]
    log_train_acc,  log_test_acc = logistic_Reg(Train_X, Train_Y, Test_X, Test_Y)
    logistic = [log_train_acc,  log_test_acc]
    rf_train_acc,rf_test_acc = randomForest(Train_X, Train_Y, Test_X, Test_Y)
    randomFor = [rf_train_acc, rf_test_acc]

    return svm, logistic, randomFor

# Training and testing of grid finding
def Search(train_model,params, Train_X,Train_Y, Test_X, Test_Y):
    train_model = GridSearchCV(estimator=train_model, param_grid=params, cv=5)
    train_model.fit(Train_X, Train_Y)
    test_label_pred = train_model.predict(Test_X)
    test_acc = metrics.accuracy_score(Test_Y, test_label_pred)
    return train_model.best_score_, test_acc


def SVM(Train_X, Train_Y, Test_X, Test_Y):
    train_model = svm.SVC(probability=True)
    params = [
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
        {'kernel': ['poly'], 'C': [1, 10], 'degree': [2, 3]},
        {'kernel': ['rbf'], 'C': [1, 10, 100, 1000],
         'gamma': [1, 0.1, 0.01, 0.001]}]
    train_acc,  test_acc = Search(train_model, params, Train_X, Train_Y, Test_X, Test_Y)
    return train_acc,  test_acc

def logistic_Reg(Train_X, Train_Y, Test_X, Test_Y):
    train_model = LogisticRegression()
    params = [{'C': [0.01,0.1,1,10,100]}]
    train_acc, test_acc = Search(train_model, params, Train_X, Train_Y,Test_X, Test_Y)
    return train_acc,  test_acc

def randomForest(Train_X, Train_Y, Test_X, Test_Y):
    train_model = RandomForestClassifier()
    params = {"n_estimators":[10,50,100,200,300,400],"max_depth":[2,4,6,10,30]}
    train_acc,  test_acc = Search(train_model,params, Train_X,Train_Y, Test_X, Test_Y)
    return train_acc, test_acc