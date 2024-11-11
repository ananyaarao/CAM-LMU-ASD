—-LEVEL 1—-
Script 1:
Creating the required folders on local system - code-enabled
Downloading files directly from lrz onto the required storage folder (e.g lab hard drive) - manually
Sending them to the correct newly created folders as per required organization - code-enabled

Script 2: TET
TET data extractions - semi-automatic (partly manual, mostly code-enabled)
Plotting and saving the figures - code-enabled
Extraction one day/night at a time

Script 3: Extraction of dreem eeg measures (not to be confused with the extraction of sleep parameters directly from raw eeg via preprocessing and signal processing strategies and various packages)
Extracting the required sleep measures and putting them in dictionaries
Extraction done all available days/nights at once in one single dictionary
Subsequent visualization


Script 4: wristband data (mainly: electrodermal activity. Can be modified later on to add other measures if required)
Reading the raw files from avro to csv
Extraction done all available days/nights at once and stored in the correct directories

All these scripts should be in terms of functions such that they are callable from another common script (Script 5) and the data returned can be used for further analyses -> need to put up such a script and tie all the scripts together
—-END OF LEVEL 1—--
