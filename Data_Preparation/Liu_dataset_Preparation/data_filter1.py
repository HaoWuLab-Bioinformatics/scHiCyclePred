import os, sys, re
from collections import Counter
from math import sqrt
import numpy as np
import os
import numpy as np
import os
import math
import xlsxwriter
import shutil

def mkdir(path):
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def write_reads(out_path, chr_pair_list):
    print("start writing")
    file = open(out_path + "/" + "cell_1.txt" , "w")
    # file.write("chrome1    pos1    chrome2    pos2\n")
    for pair in chr_pair_list:
        if pair[1] == pair[3]:
            file.write(str(pair[1])+"\t"+str(pair[2])+"\t  " +str(pair[3])+"\t  "+ str(pair[4])+ "\n")
    file.close()

def contains_substring(file_name, substrings):
    for substring in substrings:
        if substring in file_name:
            return True
    return False

def move_file(source_path, destination_folder):
    # 确保目标文件夹存在
    os.makedirs(destination_folder, exist_ok=True)
    # 构造目标文件路径
    destination_path = os.path.join(destination_folder, os.path.basename(source_path))
    # 移动文件
    shutil.move(source_path, destination_path)
    print(f"File moved from '{source_path}' to '{destination_path}'.")

def main():
    file_list = np.loadtxt('./Data_Preparation/Liu_dataset_Preparation/cell_cycle.txt', dtype=str)   #
    need_cell = []  # 选出需要的细胞
    for file in file_list:
        need_cell.append(file[0])
    # print(need_cell)
    folder_path = './Data_Preparation/Liu_dataset_Preparation/GSE223917'  # 解压出来的数据文件夹
    files = os.listdir(folder_path)
    print(files)
    for cell in files:
        print(cell.split('_')[1].split('.')[0])
        if contains_substring(cell.split('_')[1].split('.')[0], need_cell):
            source_file_path = folder_path + '/' + cell  # 替换为你的源文件路径
            destination_folder = './Data_Preparation/Liu_dataset_Preparation/GSE_process1'  # 替换为你的目标文件夹路径
            move_file(source_file_path, destination_folder)
            print('over')

if __name__ == '__main__':
    main()