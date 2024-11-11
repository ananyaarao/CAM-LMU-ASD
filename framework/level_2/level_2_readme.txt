—-LEVEL 2—-
Script 1: EEG preproc
Currently being done by other student at Tristan lab
Therefore for now, It will just have all the code I had implemented while trying to extract sleep measures directly from raw data. 

Script 2:
Any remaining processing with TET. Currently nothing. All can be taken to analyses directly. Most of what is required was already done in level 1.
Binning value logic (15 minutes and 1 hour)
Done for all available day/nights
Optionally additional visualisations with aggr scl, temp and acc and save figures
Done for each day/night separately as per user input

Script 3: EDA preproc
Extraction of scl and scr. Currently using flirt package which claims that raw data can directly be supplied to extract scl and scr. 
This can be replaced/added with any other verified preprocessing strategy/package if deemed necessary
The extracted scr and scl could maybe be stored as numpy arrays and saved 

As in level 1, All these scripts should be in terms of functions such that they are callable from another common script and the data returned can be used for further analyses
—-END OF LEVEL 2—---
