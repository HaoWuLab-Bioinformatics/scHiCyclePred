'''
The first step of Data_Preparation: the data in the original dataset raw_data,
according to the specified mapping rules, will be in the original format:
[[fend1	fend2	count],
[fend1	fend2	count],
....
[fend1	fend2	count]]

Mapped to the following new format:
[[fend1_chr, new_fend1, fend2_chr, new_fend2, new_count],
[fend1_chr, new_fend1, fend2_chr, new_fend2, new_count],
....
[fend1_chr, new_fend1, fend2_chr, new_fend2, new_count]]
'''

import numpy as np
import os

def mkdir(path):
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def list_of_groups(init_list, children_list_len):
  list_of_groups = zip(*(iter(init_list),) *children_list_len)
  end_list = [list(i) for i in list_of_groups]
  count = len(init_list) % children_list_len
  end_list.append(init_list[-count:]) if count !=0 else end_list
  return end_list

# This function is used to read the rows and columns of the path file into the program as a matrix
def read_pair3(path1, path2, path3):
    file1 = open(path1)
    file1.readline()
    a = []
    for line in file1.readlines():
        a.append(line.split())
    file2 = open(path2)
    for line in file2.readlines():
        a.append(line.split())
    file3 = open(path3)
    for line in file3.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a

def read_pair(path):
    file = open(path)
    file.readline()
    a = []
    for line in file.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a
def find_scale(pair_list):
    p = pair_list.astype(int)
    L1 = np.concatenate(p[:,0:1])
    L2 = np.concatenate(p[:,1:2])
    return min(min(L1),min(L2)), max(max(L1),max(L2))

def write_reads(input_path,new_pair_list):
    print("start write")
    file = open(input_path + "_" + "reads", "w")
    file.write("fend1_chr    fend1    fend2_chr fend2    count\n")
    for pair in new_pair_list:
        file.write(str(pair[0])+" "+str(pair[1])+" "+str(pair[2])+" "+str(pair[3])+" "+str(pair[4])+"\n")
    file.close()

def map_reads(pair_list):
    # The format of pair_list is:
    # [[ 1295317  1295322        1]
    #  [10988304 10988301        2]
    #  [11573185 11573342        1]
    #  ...
    #  [10895336 10895335        1]
    #  [ 4104570  4104569        2]
    #  [    5449     5239        3]]
    new_pair_list = []
    print("start map")
    for pair in pair_list:
        fend1 = pair[0]
        fend2 = pair[1]
        count = pair[2]
        if fend1 <= 12815563:
            index1 = fend1
        elif fend1 >= 12815566:
            index1 = fend1-2
        else:
            new_pair_list.append(["nan", "nan", "nan", "nan", "nan"])
            continue
        if fend2 <= 12815563:
            index2 = fend2
        elif fend2 >= 12815566:
            index2 = fend2-2
        else:
            new_pair_list.append(["nan", "nan", "nan", "nan", "nan"])
            continue
        new_fend1 = fends_matrix[index1, 2]
        fend1_chr = fends_matrix[index1, 1]
        new_fend2 = fends_matrix[index2, 2]
        fend2_chr = fends_matrix[index2, 1]
        new_count = count
        new_pair_list.append([fend1_chr, new_fend1, fend2_chr, new_fend2, new_count])
    return new_pair_list

# 1, The first column in the GATC.friends file is the sequence number corresponding to the contact point
# 2, The second column is the sequence number of the chromosome to which the contact point belongs
# 3, The third column is the real coordinates of the contact point
fend_path1 = "./Data_Preparation/Data/GATC_fend/GATC.fends1"
fend_path2 = "./Data_Preparation/Data/GATC_fend/GATC.fends2"
fend_path3 = "./Data_Preparation/Data/GATC_fend/GATC.fends3"
fends_matrix = np.array(read_pair3(fend_path1, fend_path2, fend_path3))
print(fends_matrix)
print("matrix over")
more_fends = {}


types = ['1CDX1','1CDX2','1CDX3','1CDX4'] #
# cell_id = 0
for type in types:
    input_path = './Raw_data/%s'%(type)
    folderlist = os.listdir(input_path)
    # folderlist is a list where each element is the name of a folder, and each folder corresponds to a cell. Inside each folder are stored the number of contacts of fend1 and fend2
    # ['1CDES_p7.H10', '1CDES_p4.C12', '1CDES_p5.D2', '1CDES_p4.F1', '1CDES_p3.E8', '1CDES_p8.C5', '1CDES_p7.D5', '1CDES_p10.H8', '1CDES_p7.B4', '1CDES_p3.F3', '1CDES_p7.B6', '1CDES_p4.E10', '1CDES_p9.B7', '1CDES_p5.G9', '1CDES_p7.F10', '1CDES_p6.A3', '1CDES_p8.C11', '1CDES_p3.A3', '1CDES_p9.A12', '1CDES_p4.E8', '1CDES_p4.B10', '1CDES_p9.E6', '1CDES_p8.D2', '1CDES_p4.E11', '1CDES_p3.E11', '1CDES_p9.F3', '1CDES_p4.D11', '1CDES_p1.E3', '1CDES_p8.H9', '1CDES_p5.G2', '1CDES_p4.F2', '1CDES_p6.E5', '1CDES_p8.A7', '1CDES_p4.D9', '1CDES_p3.C11', '1CDES_p9.B11', '1CDES_p10.H11', '1CDES_p4.H1', '1CDES_p8.C3', '1CDES_p4.F9', '1CDES_p8.B7', '1CDES_p7.E11', '1CDES_p10.H6', '1CDES_p10.F1', '1CDES_p8.E8', '1CDES_p3.A9', '1CDES_p10.A11', '1CDES_p3.D6', '1CDES_p4.F11', '1CDES_p5.E8', '1CDES_p3.G10', '1CDES_p7.G10', '1CDES_p6.D11', '1CDES_p10.C2', '1CDES_p8.G6', '1CDES_p1.C6', '1CDES_p8.F7', '1CDES_p9.C11', '1CDES_p7.A7', '1CDES_p10.C10', '1CDES_p8.H5', '1CDES_p9.C7', '1CDES_p9.B2', '1CDES_p5.D7', '1CDES_p6.D3', '1CDES_p7.A3', '1CDES_p7.C10', '1CDES_p4.G3', '1CDES_p9.A10', '1CDES_p6.E1', '1CDES_p10.H1', '1CDES_p4.A8', '1CDES_p5.F11', '1CDES_p6.A2', '1CDES_p1.D5', '1CDES_p9.E10', '1CDES_p7.H4', '1CDES_p8.B9', '1CDES_p10.A5', '1CDES_p5.E3', '1CDES_p8.B3', '1CDES_p5.F6', '1CDES_p6.F9', '1CDES_p6.H6', '1CDES_p9.A1', '1CDES_p7.H3', '1CDES_p8.B5', '1CDES_p4.E5', '1CDES_p6.H12', '1CDES_p4.D4', '1CDES_p7.G6', '1CDES_p10.G11', '1CDES_p5.F4', '1CDES_p5.G7', '1CDES_p6.G4', '1CDES_p1.D4', '1CDES_p9.G7', '1CDES_p6.A5', '1CDES_p3.E7', '1CDES_p4.H10'...
    cell_num = len(folderlist)
    folderlist.sort()
    for folder in folderlist:
        cell_name = folder
        # cell_id += 1
        cell_path = input_path+"/"+folder+"/adj"
        output_path = "./Data_Preparation/Data/mapped_data/"+type+"/"+folder
        print(output_path)
        mkdir("./Data_Preparation/Data/mapped_data/"+type)
        print(cell_path)
        # Return information on the number of contacts of a cell as a matrix
        pair_list = read_pair(cell_path).astype(int)
        # A list of contact numbers
        # The format of pair_list is.
        # [[ 1295317  1295322        1]
        #  [10988304 10988301        2]
        #  [11573185 11573342        1]
        #  ...
        #  [10895336 10895335        1]
        #  [ 4104570  4104569        2]
        #  [    5449     5239        3]]
        new_pair_list = map_reads(pair_list)
        # The format of the new_pair_list is
        # [[fend1_chr, new_fend1, fend2_chr, new_fend2, new_count],
        # [fend1_chr, new_fend1, fend2_chr, new_fend2, new_count],
        # ....
        # [fend1_chr, new_fend1, fend2_chr, new_fend2, new_count]]
        write_reads(output_path, new_pair_list)


