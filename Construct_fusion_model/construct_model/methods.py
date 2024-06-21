resolution = 1000000
#一些常用的方法

#  generate_bin()函数用于返回一个字典，字典中存的是每条染色体能够切出的块的数目，形如：
# {'chr1': 198, 'chr2': 183, .... 'chrX': 172, 'chrY': 92}
def generate_bin():
    f = open("./mm10.main.nochrM.chrom.sizes")
    index= {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index

def str_reverse(s):
    s = list(s)
    s.reverse()
    return "".join(s)

def replace_linetodot(S):
    S = str_reverse(S)
    S = S.replace('_','.',1)
    return str_reverse(S)

def replace_dottoline(S):
    S = str_reverse(S)
    S = S.replace('.','_',1)
    return str_reverse(S)


