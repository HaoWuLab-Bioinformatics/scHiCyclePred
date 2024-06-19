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
    file = open(out_path + "/" +chr_name+".txt" , "w")
    # file.write("bin1    bin2\n")
    for pair in chr_pair_list:
        file.write(str(pair[0])+"\t"+str(pair[1])+"\n")
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

def allocation_pair(bins,pair_list,chr_dict,out_path):
    # bins 就是每个染色体在1mb的分辨率下能分多少段
    # pair_list 是一个矩阵，每一行里面为[fend1_chr,fend1,fend2_chr,fend2,count]
    # chr_dict是字典每个染色体对应一个列表 {chr1:[],chr2:[],chr3:[]....,chrX:[],chrY:[]}
    # out_path是要输出的路径
    for pair in pair_list:
        fend1_chr,fend1,fend2_chr,fend2 = pair
        if fend1_chr == fend2_chr and fend1_chr!= "Y" and fend2_chr!= "Y":
            # 排除自己和自己的交互信息、排除Y染色体上的信息
            # pos1 pos2 转换到bin上  因为bins中是1mb增加的，所以要把pos转换为int随后映射到bin上
            # 循环这个矩阵，把每一行加到chr_dict中的染色体对应的列表当中
            pos1 = int(int(fend1) / resolution) * resolution
            pos2 = int(int(fend2) / resolution) * resolution
            bin1 = bins[resolution][(str(fend1_chr), pos1)]
            bin2 = bins[resolution][(str(fend2_chr), pos2)]
            if bin1 <= bin2:
                key = (bin1, bin2)
                chr_dict[str(fend1_chr)].append(key)
            else:
                key = (bin2, bin1)
                chr_dict[str(fend1_chr)].append(key)
    #{'chr1': [(161, 187, '5')], 'chr2': [], 'chr3': [], 'chr4': [], 'chr5': [], 'chr6': [], 'chr7': [], 'chr8': [], 'chr9': [], 'chr10': [], 'chr11': [(74, 74, '2')], 'chr12': [], 'chr13': [], 'chr14': [], 'chr15': [], 'chr16': [], 'chr17': [], 'chr18': [], 'chr19': [], 'chrX': [], 'chrY': []}
    for chr_name in chr_dict.keys():
        mkdir(out_path)
        write_reads(out_path,chr_name,chr_dict[chr_name])

def main():
    chromsizes = open('./mm10.main.nochrM.chrom.sizes')  # positional argument 1 --> chromosome sizes
    folder = os.listdir('./GSE_process2')
    bins, valid_chroms = define_bins(chromsizes, resolution)
    for file in folder:
        cell_path = './GSE_process2/' + file + '/cell.txt'  # positional argument 3 --> BEDPE file
        out_path = './GSE_process3_bin/' + file
        pair_list = read_pair(cell_path)
        chr_dict = generate_chrdict(valid_chroms)
        allocation_pair(bins,pair_list,chr_dict,out_path)
#     print(bins)  # 每个染色体pos对应的是第几个bin

if __name__ == '__main__':
    main()