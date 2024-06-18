'''
The code on this page defines the functions used to calculate the SICP feature set
'''

from methods import generate_bin, read_pair, generate_contact_matrix
import numpy as np


# This function is used to calculate the contact probability of the first-order bin neighborhood of a single chromatin in a single cell
# The input contact_matrix is the contact matrix corresponding to chromosome number chr under the current cell, index is the number of blocks that chromosome number chr can cut out
# scale is the size of the neighborhood to be calculated, this algorithm only calculates the first-order neighborhood
# AS: (bin-1，bin-1)   (bin-1，bin)   (bin-1,bin+1)
#                     (bin,bin)     (bin,bin+1)
#                                   (bin+1,bin+1)
def Small_Domain_Struct_Contact_pro(contact_matrix, index, scale):
    contact_matrix = np.array(contact_matrix)
    new_matrix = np.zeros((index+2*scale, index+2*scale))
    SICP = []
    chr_total = sum_matrix(contact_matrix)
    for i in range(index):
        for j in range(index):
            new_matrix[i+scale, j+scale] = contact_matrix[i, j]
    for i in range(index):
        bin = i+scale
        a = sum_matrix(new_matrix[bin-scale:bin+scale+1, bin-scale:bin+scale+1])
        if a == 0:
            SICP.append(float(0))
        else:
            SICP.append(float(a/chr_total))
    return SICP


# This function is used to calculate the total number of contacts in the input matrix
def sum_matrix(matrix):
    U = np.triu(matrix, 1)
    D = np.diag(np.diag(matrix))
    return sum(sum(U + D))


# Generate_BCPandSICP.py file will call this function, passing in a list of cell paths, in the shape of.
# ['. /Data/chr_matrix/1CDX1/1.1XDC11G_reads', '. /Data/chr_matrix/1CDX1/2.1XDC11G_reads', ...]
# 1.1XDC11G_reads is a folder with 21 files holding the contact information of the 21 chromosomes after the two-step mapping
def calculate_Small_Domain_Struct_Contact_pro(file_list):
    # generate_bin() function is used to return a dictionary in which the number of blocks that can be cut out of each chromosome is stored, in the shape of.
    # {'chr1': 198, 'chr2': 183, 'chr3': 161, 'chr4': 157, 'chr5': 153, 'chr6': 150, 'chr7': 153, 'chr8': 132, 'chr9': 125, 'chr10': 131, 'chr11': 123, 'chr12': 122, 'chr13': 121, 'chr14': 126, 'chr15': 105, 'chr16': 99, 'chr17': 96, 'chr18': 91, 'chr19': 62, 'chrX': 172, 'chrY': 92}
    index = generate_bin()
    print("start")
    # scale is the size of the neighborhood to be calculated, and this algorithm only calculates the first-order neighborhood
    # (bin-1，bin-1)   (bin-1，bin)   (bin-1,bin+1)
    #                  (bin,bin)     (bin,bin+1)
    #                                (bin+1,bin+1)
    scale = 1
    cell_SICP_dict = {}
    for i in file_list:
        cell_SICP_dict[i.split('/')[4]] = {}
        # i is. /Data/chr_matrix/1CDX4/1CDX1.1_reads"
        # i.split('/')[4] returns: 1CDX1.1_reads
        # i.split('/') returns: ['.' , 'Data', 'chr_matrix', '1CDX1', '1CDX1.1_reads']
        # 1CDX1.1_reads can be used as a code for a cell
        for chr in index.keys():
            cell_path = i+"/"+chr+".txt"
            print(cell_path)
            # The returned pair_list is a two-dimensional matrix with three columns
            pair_list = read_pair(cell_path)
            # generate_contact_matrix is the number of blocks that can be cut out of chromosome chr, and pair_list is the contact information of chromosome chr.
            # The returned contact_matrix is the contact matrix of the current cell with chromosome chr
            contact_matrix = generate_contact_matrix(index[chr], pair_list)
            SICP = Small_Domain_Struct_Contact_pro(contact_matrix, index[chr], scale)
            cell_SICP_dict[i.split('/')[4]][chr] = SICP
    out_path1 = './Data/SICP.npy'
    np.save(out_path1, cell_SICP_dict)

