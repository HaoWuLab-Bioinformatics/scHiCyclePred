# coding:utf-8
import numpy as np
import os
import math
import xlsxwriter
#计算CDD特征集
def read_pair(path):
    #读取细胞接触文件中的接触列表
    file = open(path)
    file.readline()
    a = []
    for line in file.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a


file = './Data_Preparation/Liu_dataset_Preparation/Liu_CDD[35,132]_test.xlsx'
workbook = xlsxwriter.Workbook(file)
worksheet1 = workbook.add_worksheet('result')
worksheet1.write(0, 0, 'cell_name')
for i in range(1,140):
    s = "bin_%s" % str(i+34)
    worksheet1.write(0, i, s)
file_list = np.loadtxt('./Data_Preparation/Liu_dataset_Preparation/cell_cycle.txt', dtype=str)
need_cell = []  # 选出需要的细胞
for file in file_list:
    need_cell.append(file[0])
# print(need_cell)
types = need_cell[1:]
print(len(types)) # 6288
cell_id = 0
folderlist = []
for type in types:
    outer_path = './Data_Preparation/Liu_dataset_Preparation/GSE_pos/%s.txt' %(type)
    folderlist.append(outer_path)

folderlist.sort()
# print(folderlist)
# exit()
for folder in folderlist:
    cell_name = folder.split('/')[2].split('.')[0]
    print(cell_name)
    cell_id += 1
    print(cell_id)
    worksheet1.write(cell_id, 0, cell_name)
    # cell_path = outer_path+"/"+folder
    # print(cell_path)
    pair_list = read_pair(folder)  # 读取每一行
    bin_dict = {}
    count_sum = 0
    for pair in pair_list:  # 一个pair_list是一个细胞内的接触信息
        fend1_chr = pair[0]
        fend1 = int(pair[1])
        fend2_chr = pair[2]
        fend2 = int(pair[3])
        count = int(pair[4])
        if abs(fend2-fend1) <= 20000:
            continue
        if fend1_chr!=fend2_chr:
            continue
        gene_distance = abs(fend2-fend1)/1000
        bin = math.floor((math.log(gene_distance, 2)+(0.125))/0.125)
        # math.floor ：向下取整     根据基因距离算出bin  bin 要在132之内
        if bin>132:
            continue
        count_sum += count
        if bin in bin_dict.keys():
            bin_dict[bin]+=count
        else:
            bin_dict[bin] = count
    print(bin_dict)  # 是一个字典，通过不断地循环，把bin:count 加了进去
    # 98个 {118: 164, 37: 577, 60: 842, 119: 132, 88: 268, 64: 954, 111: 186, 41: 578, 61: 932, 67: 821, 50: 690, 69: 808, 52: 768, 74: 677, 77: 734, 91: 214, 57: 767, 59: 779, 47: 651, 84: 410, 44: 613, 87: 250, 51: 692, 104: 161, 45: 595, 40: 582, 66: 821, 54: 771, 36: 607, 76: 587, 72: 889, 43: 637, 75: 733, 73: 719, 115: 243, 116: 198, 63: 799, 83: 411, 78: 622, 53: 669, 68: 833, 129: 23, 58: 856, 103: 201, 89: 215, 93: 192, 42: 529, 71: 882, 48: 682, 46: 609, 81: 484, 35: 240, 65: 794, 39: 593, 56: 775, 113: 199, 49: 608, 82: 462, 85: 362, 38: 582, 96: 165, 114: 254, 95: 191, 90: 165, 55: 783, 120: 190, 70: 865, 99: 163, 100: 176, 62: 898, 128: 27, 80: 514, 97: 189, 86: 297, 109: 157, 79: 540, 117: 164, 108: 231, 112: 227, 101: 156, 131: 8, 94: 170, 92: 162, 107: 209, 110: 193, 105: 143, 98: 157, 121: 139, 106: 125, 102: 170, 122: 116, 130: 48, 123: 110, 126: 42, 127: 42, 125: 65, 124: 95, 132: 30}
    for key in sorted(bin_dict.keys()):
        bin_dict[key] = bin_dict[key]/count_sum
        worksheet1.write(cell_id, key-34, bin_dict[key])
        # 这个就是CDD.txt

workbook.close()




