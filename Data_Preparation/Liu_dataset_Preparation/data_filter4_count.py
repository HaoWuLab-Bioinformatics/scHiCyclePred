import os, sys, re
from collections import Counter
from math import sqrt
import numpy as np
import os
import numpy as np
import os
import math
import xlsxwriter

resolution = 1000000
def mkdir(path):
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def write_reads(out_path, chr_name,chr_pair_list):
    print("start writing")
    file = open(out_path + "/" +chr_name , "w")
    for pair in chr_pair_list:
        file.write(str(pair[0])+" "+str(pair[1])+" "+str(pair[2])+"\n")
    file.close()

def define_bins(chromsizes, resolution):
    bins = {}  # hash the bins
    valid_chroms = {}  # hash the chromosome sizes
    lines = chromsizes.readlines()
    bins[resolution] = {}
    for line in lines:
        hindex = 0
        mindex = 0
        chromname, length = line.split()
        valid_chroms[chromname] = True

        for i in range(0, int(length), resolution):
            bins[resolution][(chromname, i)] = hindex  # bins{1000000:{(name,1000000*n):n}}
            hindex += 1
    return bins, valid_chroms  # bin:{resolution:{(chromname, i):index}}

def read_pair(path):
    #读接触文件中的接触列表
    file = open(path)
    file.readline()
    a = []
    for line in file.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a

def generate_chrdict(valid_chroms):
    dict = {}
    for chrname in valid_chroms.keys():
        if valid_chroms[chrname]:
            dict[chrname] = []
    return dict


def remove_duplicate_lists(main_list):
    unique_lists = []
    for sublist in main_list:
        if sublist not in unique_lists:
            unique_lists.append(sublist)
    return unique_lists

def main():
    chromsizes = open('./mm10.main.nochrM.chrom.sizes')  # positional argument 1 --> chromosome sizes
    folder = os.listdir('./GSE_process3_bin')
    folder.sort()
    for file1 in folder:
        folder_list = os.listdir('./GSE_process3_bin/' + file1)
        out_path = './GSE_final_bin_count/' + file1  # 这可能出问题了，可以重新跑一遍data_filter3
        mkdir(out_path)
        for folder in folder_list:  # 每个染色体
            path = './GSE_process3_bin/' + file1 + '/' + folder
            file = open(path)
            infor = []
            for line in file.readlines():
                infor.append(line.split())
                # print(infor)  # [['3', '3']]
                # break
            # print(infor)
            # 3 3 16
            for i in range(len(infor)): # 每一行
                count = 0
                for j in range( 0,len(infor)):
                    if infor[i][0] == infor[j][0]:
                    # print(infor[i][0],infor[j][0],infor[i][1],infor[j][1])
                        if infor[i][1] == infor[j][1]:
                            count += 1
                infor[i].append(count)
            every_chr_list = remove_duplicate_lists(infor)
            # print(every_chr_list) # 不重复bin的染色体 [['3', '3', 16], ['3', '27', 5], ['3', '40', 2], ['3', '26', 4],
            write_reads(out_path=out_path,chr_name=folder,chr_pair_list=every_chr_list)


if __name__ == '__main__':
    main()