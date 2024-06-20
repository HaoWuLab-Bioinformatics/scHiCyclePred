1、You need to unzip the data package ‘GSE223917_RAW.tar’ downloaded from the https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223917.

2、Extract the files ending with ".pairs.gz" from the compressed package and place them in the GSE223917 folder.

3、Generate the data format required by Liu_dataset by executing the following statement.
```
python .\Data_Preparation\data_filter1.py   # Select the required cell file and generate folder GSE_process1
```
```
python .\Data_Preparation\data_filter2.py  # Generate folder GSE_process2 
```
```
python .\Data_Preparation\data_filter3.py  # Generate folder GSE_process3_bin
```
```
python .\Data_Preparation\data_filter4_count.py  # Generate folder GSE_final_bin_count
```
```
python .\Data_Preparation\data_filter_pos.py  # Generate folder GSE_pos
```

### Taking a few cells as an example, the following table shows the file formats in each generated folder.
| Folder Name  | Files |
| ------------- | ------------- |
| GSE_process1  | GSM6998595_GasaE751001.pairs、GSM6998596_GasaE751002.pairs  |
| GSE_process2  | GasaE751001/cell.txt、GasaE751002/cell.txt  |
| GSE_process3_bin  |  GasaE751001/chr1.txt-chrY.txt、GasaE751002/chr1.txt-chrY.txt |
| GSE_final_bin_count  | GasaE751001/chr1.txt-chrY.txt、GasaE751002/chr1.txt-chrY.txt  |
| GSE_pos  | GasaE751001.txt、GasaE751002.txt  |

### The data format in 'GSE_process2/GasaE751001/cell.txt' is as follows:
| chr_start  | loc_start | chr_end | loc_end |
| ------------- | ------------- | ------------- | ------------- |
| chr1  | 3052398 | chr1 | 4307873 |
| chr1  | 3070049 | chr1 | 3075376 |

### The data format in 'GSE_process3_bin/GasaE751001/chr1.txt' is as follows:
| bin_1  | bin_2 |
| ------------- | ------------- |
| 3  | 3 |
| 3  | 27 |

### The data format in 'GSE_final_bin_count/GasaE751001/chr1.txt' is as follows:
| bin_1  | bin_2 | count |
| ------------- | ------------- | ------------- |
| 3  | 3 | 16 |
| 3  | 27 | 4 |

### The data format in 'GSE_process2/GasaE751001/cell.txt' is as follows:
| chr_start  | loc_start | chr_end | loc_end | count |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| chr1  | 3052398 | chr1 | 4307873 | 1 |
| chr1  | 3070049 | chr1 | 3075376 | 1 |

### Note: You need to ensure that the folder (GSE223917) you are reading does not contain any other unnecessary files.

