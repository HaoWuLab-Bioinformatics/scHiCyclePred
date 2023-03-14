'''
The code on this page defines the functions used to calculate the BCP feature set
'''
from methods import generate_bin,read_pair, generate_contact_matrix,matrix_list
import numpy as np

# This function is used to return all non-zero elements of each row
# The input list is one of the rows of the chromosome contact matrix
def find_contact_list(list):
    con = []
    for i in list:
        if i != 0:
            con.append(i)
    return con

# This function is used to calculate the probability of bin contact for each chromosome
# The input contact_matrix is the contact matrix corresponding to chromosome number chr under the current cell, and index is the number of blocks chromosome number chr can cut out
def Bin_contact_pro(contact_matrix, index):
    Bc = []
    # chr_total is the total number of contacts in the contact matrix corresponding to chromosome chr number under the previous cell.
    # In short it is to find the total number of contacts on the current chromosome.
    chr_total = sum_matrix(contact_matrix)
    for i in range(index):
        # contact_matrix[i] is the i-th row of the chromosome contact matrix
        con_list = find_contact_list(contact_matrix[i])
        # The returned con_list is actually all the non-zero elements in row i of the contact matrix, that is, the number of contacts between the i-th chromosome segment and all segments
        if len(con_list) == 0:
            Bc.append(0)
        else:
            con_pro = sum(con_list)/chr_total
            # Each con_pro is the contact probability corresponding to each chromosome segment
            Bc.append(con_pro)
    return Bc

# This function is used to calculate the total number of contacts in the input matrix
def sum_matrix(matrix):
    # U is the upper triangular part of the matrix, excluding the diagonal
    U = np.triu(matrix, 1)
    # The matrix D contains only the elements on the diagonal of the original matrix matrix, the rest of the elements are 0
    D = np.diag(np.diag(matrix))
    return sum(sum(U + D))


# Generate_BCP&SICP.py file will call this function, passing in a list of cell paths, in the form of
# ['./Data/chr_matrix/1CDX1/1.1XDC11G_reads', './Data/chr_matrix/1CDX1/2.1XDC11G_reads', ...]
def calculate_Bin_contact_pro(file_list):
    # generate_bin() function is used to return a dictionary in which the number of blocks that can be cut out of each chromosome is stored, in the shape of.
    # {'chr1': 198, 'chr2': 183, 'chr3': 161, 'chr4': 157, 'chr5': 153, 'chr6': 150, 'chr7': 153, 'chr8': 132, 'chr9': 125, 'chr10': 131, 'chr11': 123, 'chr12': 122, 'chr13': 121, 'chr14': 126, 'chr15': 105, 'chr16': 99, 'chr17': 96, 'chr18': 91, 'chr19': 62, 'chrX': 172, 'chrY': 92}
    index = generate_bin()
    print("start")
    chr_list = sorted(index.keys())
    cell_dict = {}
    # each i is a folder, under the folder is the contact information of 21 chromosomes but the Y chromosome is not considered in it, fend_to_bin.py does not consider Y when mapping
    for i in file_list:
        cell_dict[i.split('/')[4]] = {}
        # i is. /Data/chr_matrix/1CDX4/1CDX1.1_reads"
        # i.split('/')[4] returns: 1CDX1.1_reads
        # i.split('/') returns: ['.' , 'Data', 'chr_matrix', '1CDX1', '1CDX1.1_reads']
        # 1CDX1.1_reads can be used as a code for a cell
        for chr in chr_list: # 循环21条染色体
            chr_path = i+"/"+chr+".txt"
            print(chr_path)
            # The returned pair_list is a two-dimensional matrix with three columns
            pair_list = read_pair(chr_path)
            # generate_contact_matrix is the number of blocks that can be cut out of chromosome chr, and pair_list is the contact information of chromosome chr.
            # The returned contact_matrix is the contact matrix of the current cell with chromosome chr
            contact_matrix = generate_contact_matrix(index[chr], pair_list)
            Bcp = Bin_contact_pro(contact_matrix, index[chr])
            # Bcp is the contact probability for each segment on chromosome chr (contact probability = total number of contacts for that segment / total number of contacts on the chromosome = sum of a row in the contact matrix / sum of all elements in the matrix)
            cell_dict[i.split('/')[4]][chr] = Bcp
    out_path1 = '../Data/BCP/Full_chr/BCP(chr).npy'
    np.save(out_path1, cell_dict)
# calculate_Decay()