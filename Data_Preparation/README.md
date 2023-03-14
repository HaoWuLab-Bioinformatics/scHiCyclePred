## This folder holds the code for Data_preparation

### The ‘Data’ folder holds the data generated or used in Data_preparation
### fend_map_coord.py
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

### fend_to_bin.py
The second step of data preparation：Mapping the real coordinates of the contact point to the corresponding bin
