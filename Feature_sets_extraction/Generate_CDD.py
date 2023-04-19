'''
The code on this page is used to generate the CDD feature set
The meaning of this feature set is to count a distribution of the distance between all the two contacts in each cell that are in contact.
Counts the probability of different distances between two contacts in contact
'''
# coding:utf-8
import numpy as np
import os
import math
import xlsxwriter
import pandas as pd

# This function is used to read the list of contacts in the cell contact file
def read_pair(path):
    file = open(path)
    file.readline()
    a = []
    for line in file.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a

# Convert xlsx files to txt files
def Excel_to_Txt(input_path, output_path):
    df = pd.read_excel(input_path, header=None)
    print('开始写入txt文件')
    df.to_csv(output_path, header=None, sep='\t', index=False, na_rep=0)  # sep指定分隔符，分隔单元格
    print('写入成功')

# Create TXT file with same path and same name
def creat_txt(input_path):
    length = len(input_path)
    output_path = ''
    for i in range(length - 1, -1, -1):
        if input_path[i] == '.':
            break
    for j in range(0, i + 1):
        output_path = output_path + input_path[j]
    output_path = output_path + 'txt'
    # file = open(output_path,)
    return output_path

file = '../Data/CDD.xlsx'
workbook = xlsxwriter.Workbook(file)
worksheet1 = workbook.add_worksheet('result')
worksheet1.write(0, 0, 'cell_name')
for i in range(1, 99):
    s = "bin_%s" % str(i+34)
    worksheet1.write(0, i, s)
types = ['1CDES','1CDU','1CDX1','1CDX2','1CDX3','1CDX4']
cell_id = 0
for type in types:
    outer_path = './Data/mapped_data/%s'%(type)
    folderlist = os.listdir(outer_path)
    folderlist.sort()
    for folder in folderlist:
        cell_name = folder
        cell_id += 1
        worksheet1.write(cell_id, 0, cell_name)
        cell_path = outer_path+"/"+folder
        print(cell_path)
        pair_list = read_pair(cell_path)
        bin_dict = {}
        count_sum = 0
        # A pair_list is a contact information within a cell
        for pair in pair_list:
            fend1_chr = pair[0]
            fend1 = int(pair[1])
            fend2_chr = pair[2]
            fend2 = int(pair[3])
            count = int(pair[4])
            # Filter out contacts where the distance between two contact points is less than or equal to 20,000
            if abs(fend2-fend1) <= 20000:
                continue
            # Filter out two contacts on separate chromosomes
            if fend1_chr!=fend2_chr:
                continue
            gene_distance = abs(fend2-fend1)/1000
            bin = math.floor((math.log(gene_distance, 2)+(0.125))/0.125)
            #math.floor : round down to the nearest bin based on the genetic distance bin should be within 132
            # Why should the bin value be within [35,132]?
            # After counting the first cell and the first folder for example, it has 377863 bins in total.
            # The number of bins <35 is 0, and the number of bins >132 is only 5852.
            # >132 bins are far less than the total number of bins, and the distribution of >125 is more scattered, so it is possible to discard
            if bin > 132:
                continue
            count_sum += count
            if bin in bin_dict.keys():
                bin_dict[bin] += count
            else:
                bin_dict[bin] = count
        print(bin_dict)
        # 98个 {118: 164, 37: 577, 60: 842, 119: 132, 88: 268, 64: 954, 111: 186, 41: 578, 61: 932, 67: 821, 50: 690, 69: 808, 52: 768, 74: 677, 77: 734, 91: 214, 57: 767, 59: 779, 47: 651, 84: 410, 44: 613, 87: 250, 51: 692, 104: 161, 45: 595, 40: 582, 66: 821, 54: 771, 36: 607, 76: 587, 72: 889, 43: 637, 75: 733, 73: 719, 115: 243, 116: 198, 63: 799, 83: 411, 78: 622, 53: 669, 68: 833, 129: 23, 58: 856, 103: 201, 89: 215, 93: 192, 42: 529, 71: 882, 48: 682, 46: 609, 81: 484, 35: 240, 65: 794, 39: 593, 56: 775, 113: 199, 49: 608, 82: 462, 85: 362, 38: 582, 96: 165, 114: 254, 95: 191, 90: 165, 55: 783, 120: 190, 70: 865, 99: 163, 100: 176, 62: 898, 128: 27, 80: 514, 97: 189, 86: 297, 109: 157, 79: 540, 117: 164, 108: 231, 112: 227, 101: 156, 131: 8, 94: 170, 92: 162, 107: 209, 110: 193, 105: 143, 98: 157, 121: 139, 106: 125, 102: 170, 122: 116, 130: 48, 123: 110, 126: 42, 127: 42, 125: 65, 124: 95, 132: 30}
        for key in sorted(bin_dict.keys()):
            bin_dict[key] = bin_dict[key]/count_sum
            worksheet1.write(cell_id, key-34, bin_dict[key])
            # 这个就是CDD.txt
workbook.close()

input_path = r'../Data/CDD.xlsx'
output_path = creat_txt(input_path)
Excel_to_Txt(input_path, output_path)