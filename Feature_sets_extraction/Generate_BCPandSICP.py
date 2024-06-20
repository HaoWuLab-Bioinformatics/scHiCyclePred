'''
The code on this page is used to generate the BCP feature set and the SICP feature set
'''
from Methods_for_calculating_BCP import calculate_Bin_contact_pro
from Methods_for_calculating_SICP import calculate_Small_Domain_Struct_Contact_pro
import numpy as np


# This function is used to change the file name
# Enter 1CDX1_1, return 1_1XDC1
def str_reverse(s):
    s = list(s)
    s.reverse()
    return "".join(s)

# This function is used to change the file name, replacing the horizontal line in the file name with a dot
def replace(S):
    S = str_reverse(S)
    S = S.replace('_','.',1)
    return str_reverse(S)

def main():
    file_names = np.loadtxt("./Data/new_cell_inf.txt", dtype=str)
    # 1172 list with small lists inside, where each row of the file corresponds to each row in the matrix
    #[['cell' 'cycle'], ['1CDX1_1' 'G1'], ['1CDX1_2' 'G1'], ['1CDX1_3' 'G1'], ['1CDX1_4' 'G1'], ['1CDX1_5' 'G1'], ['1CDX2_1' 'early_S'], ['1CDX2_2' 'early_S'], ['1CDX2_3' 'early_S'], ['1CDX2_4' 'early_S'], ['1CDX2_5' 'early_S'], ['1CDX2_6' 'early_S'], ['1CDX3_1' 'mid_S'], ['1CDX3_2' 'mid_S'], ['1CDX3_3' 'mid_S'], ['1CDX3_5' 'mid_S'], ['1CDX3_6' 'mid_S'], ['1CDX4_1' 'late_S'], ['1CDX4_2' 'late_S'], ['1CDX4_3' 'late_S'], ['1CDX4_4' 'late_S'], ['1CDX4_5' 'late_S'], ['1CDX4_6' 'late_S'], ['1CDX1_12' 'G1'], ['1CDX1_13' 'G1'], ['1CDX1_14' 'G1'], ['1CDX1_17' 'G1'], ['1CDX1_22' 'G1'], ['1CDX1_23' 'G1'], ['1CDX1_24' 'G1'], ['1CDX1_25' 'G1'], ['1CDX1_26' 'G1'], ['1CDX1_27' 'G1'], ['1CDX1_32' 'G1'], ['1CDX1_33' 'G1'], ['1CDX1_34' 'G1'], ['1CDX1_35' 'G1'], ['1CDX1_36' 'G1'], ['1CDX1_37' 'G1'], ['1CDX1_38' 'G1'], ['1CDX1_41' 'G1'], ['1CDX1_42' 'G1'], ['1CDX1_43' 'G1'], ['1CDX1_44' 'G1'], ['1CDX1_45' 'G1'], ['1CDX1_46' 'G1'], ['1CDX1_47' 'G1'], ['1CDX1_51' 'G1'], ['1CDX1_52' 'G1'], ['1CDX1_53' 'G1'], ['1...
    file_list = []
    for i in range(len(file_names)):
        print(file_list)
        # replace function turns 1CDX1_1 into 1CDX1.1
        print(replace(file_names[i][0]))
        if "1CDX1" in file_names[i][0]:
            file_list.append("./Data_Preparation/Data/chr_matrix/1CDX1/"+replace(file_names[i][0])+"_reads")
        elif "1CDX2"  in file_names[i][0]:
            file_list.append("./Data_Preparation/Data/chr_matrix/1CDX2/"+replace(file_names[i][0])+"_reads")
        elif "1CDX3" in file_names[i][0]:
            file_list.append("./Data_Preparation/Data/chr_matrix/1CDX3/" + replace(file_names[i][0])+"_reads")
        elif "1CDX4" in file_names[i][0]:
            file_list.append("./Data_Preparation/Data/chr_matrix/1CDX4/" + replace(file_names[i][0])+"_reads")

    # file_list is a list of cell paths, each of which points to the file generated after the two-step mapping of each cell, i.e. the contact information after cutting into bin.
    calculate_Bin_contact_pro(file_list) # BCP feature set
    calculate_Small_Domain_Struct_Contact_pro(file_list) # SICP feature set

if __name__ == "__main__":
    main()



