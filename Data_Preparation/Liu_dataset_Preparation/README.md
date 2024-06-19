1、You need to unzip the data package ‘GSE223917_RAW.tar’ downloaded from the https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223917.

2、Extract the files ending with ".pairs.gz" from the compressed package and place them in the GSE_process1 folder.

3、Generate the data format required by Liu_dataset by executing the following statement.
```
python .\Data_Preparation\data_filter1.py   # Select the required cell file.
```
```
python .\Data_Preparation\data_filter2.py  
```
```
python .\Data_Preparation\data_filter3.py  
```
```
python .\Data_Preparation\data_filter4_count.py  
```
```
python .\Data_Preparation\data_filter_pos.py  
```

### Note: You need to ensure that the folder you are reading does not contain any other unnecessary files.
