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
    file = open(out_path + "/" + "cell.txt", "w")
    # file.write("chrome1    pos1    chrome2    pos2\n")
    for pair in chr_pair_list:
        if pair[1] == pair[3]:
            file.write(str(pair[1]) + "\t" + str(pair[2]) + "\t  " + str(pair[3]) + "\t  " + str(pair[4]) + "\n")
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
    folder_path = './Data_Preparation/Liu_dataset_Preparation/GSE_process1'  # 替换为你的文件夹路径
    files = os.listdir(folder_path)
    for file in files:
        mkdir('./Data_Preparation/Liu_dataset_Preparation/GSE_process2/' + file.split('.')[0].split('_')[1])
        # print(file.split('.')[0].split('_')[1])  # GaseE752226
        path = './Data_Preparation/Liu_dataset_Preparation/GSE_process1' + '/' + file
        f = open(path)
        a = []
        count = 1
        for line in f.readlines():
            if count >= 25:
                a.append(line.split())
            count += 1
        write_reads(out_path='./Data_Preparation/Liu_dataset_Preparation/GSE_process2/' + file.split('.')[0].split('_')[1], chr_pair_list=a)
    print(files)

if __name__ == '__main__':
    main()