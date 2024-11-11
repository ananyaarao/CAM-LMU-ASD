#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
level 2
Script 2:
Any remaining processing with TET. Most of what is required was already done in level 1.
Binning value logic (15 minutes and 1 hour)
Optionally additional visualisations with aggr scl, temp and acc and save figures to system
"""


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


"""
turning the x and y values obtaining for each day into a function as well
"""
def giv_x_y_vals(mainfolder, qnum, ger):
    #ger = True if subject recruited in Germany. Else False
    
    dict_TET_x = {}
    dict_TET_y = {}
    if ger:
        expr = 'TET – Tag';
        # to avoid duplicates in question indices - that could happen if the subject erased graph data instead of submitting the graph
        expr2 = 'Submit rating'; 
        # to make sure at least one data point has been entered by the subject
        expr3 = 'Saved drawing data'
    else:
        expr = 'Daily Experience';
        # to avoid duplicates in question indices - that could happen if the subject erased graph data instead of submitting the graph
        expr2 = 'Submit rating'; 
        # to make sure at least one data point has been entered by the subject
        expr3 = 'Saved drawing data'


    for folder in sorted(os.listdir(mainfolder)):
        print(folder)
        if folder.endswith('.png') or folder.endswith('.csv') or folder.endswith('.txt') or folder.endswith('.xlsx') or folder.endswith('.pdf')or folder.endswith('.edf'):
            continue
        for file in os.listdir(os.path.join(mainfolder, folder)):
            if file.endswith('_corrected.xlsx'):
                print(file)
                df_tet = pd.read_excel(os.path.join(mainfolder, folder, file))
                if 'MENÜ' in df_tet.columns:
                    pl1 = 'MENÜ'
                    pl2 = 'Press button: Tägliche Erfahrungen'
                    pl3 = 'Unnamed: 3'
                else:
                    pl1 = 'Category'
                    pl2 = 'Action'
                    pl3 = 'Question'
            
                #where expr and expr2 are found in the same row i.e, the rows containing the questions
                mask_tet = (df_tet[pl1] == expr) & (df_tet[pl2] == expr2)
                indices_tet = df_tet.index[mask_tet].tolist()
                ind = 0
                for i in range(0,len(indices_tet)):
                    #print(i)
                    if i == len(indices_tet)-1:
                        i1 = len(df_tet)
                        print('yes', i)
                    #if indices_tet[i]+1>=len(df_tet): #commented out due to unpredictable behaviour, re-introduce after correction
                     #   print("Warning! Below this last index, no x and y data may have been submitted! -> ", i)
                      #  break
                    else:
                        i1 = indices_tet[i+1]
                    if df_tet[pl3].iloc[indices_tet[i]] == qnum:
                        if ind !=0:
                            print("ind has already been assigned, assign this new index to another variable: ", indices_tet[i], " this is its position in the indices list: ", i)
                        else:
                            ind = indices_tet[i]
                            if qnum == 'q7':
                                if df_tet[pl2].iloc[ind+1] != expr3 and df_tet[pl2].iloc[ind+2] != expr3:
                                    print("Warning! Below this index, no x and y data may have been submitted therefore no data! -> ", ind)
                                else:
                                    x = df_tet['x_val'].iloc[indices_tet[i]:i1]
                                    y = df_tet['y_val'].iloc[indices_tet[i]:i1]
                                    print("x and y have been assigned and the correct index for q is ind which is: ", ind)
                
                                    dict_TET_x[folder] = np.array(x)
                                    dict_TET_y[folder] = np.array(y)
                            else:
                                if df_tet[pl2].iloc[ind+1] != expr3:
                                    print("Warning! Below this index, no x and y data may have been submitted therefore no data! -> ", ind)
                                else:
                                    x = df_tet['x_val'].iloc[indices_tet[i]:i1]
                                    y = df_tet['y_val'].iloc[indices_tet[i]:i1]
                                    print("x and y have been assigned and the correct index for q is ind which is: ", ind)
                
                                    dict_TET_x[folder] = np.array(x)
                                    dict_TET_y[folder] = np.array(y)
    return (dict_TET_x, dict_TET_y)


# In[3]:


"""
#making the whole bin_dict and bin_dict_mean generation code into a function so that it can be run multiple times for the required dimnesions of the required days

def give_binned_vals(x_val, y_val):
    bin_dict = {}
    bin_arr = np.arange(0,24.25,0.25)
    for i in range(0, len(bin_arr) - 1):
        #Create the key for the dictionary
        key = str(bin_arr[i]) + '_' + str(bin_arr[i+1])
    
        #Initialize an empty list for this key
        templst = []
        bin_dict[key] = templst
    
        #Iterate over x_val, append to templst if condition is met
        for j in range(0, len(x_val)):
            if x_val[j] >= bin_arr[i] and x_val[j] < bin_arr[i+1]:
                #Append y_val[j] directly to the list in the dictionary
                bin_dict[key].append(y_val[j])

    #for conversion of lists to numpy arrays
    for key in bin_dict:
        bin_dict[key] = np.array(bin_dict[key])
    #print(bin_dict)

    bin_dict_mean = {}
    for key in bin_dict:
        if len(bin_dict[key])!=0:
            bin_dict_mean[key] = np.nanmean(bin_dict[key]) #modified to nanmean from mean even though there isn't a chance of getting nan values (just safety)
        else:
            bin_dict_mean[key] = -5000

    return bin_dict_mean # bin_dict,
    
    
#binning and aggregating stress values by hour instead of by 15 minutes

def give_binned_vals_hour(x_val, y_val):
    bin_dict = {}
    bin_arr = np.arange(0,25)
    for i in range(0, len(bin_arr) - 1):
        #Create the key for the dictionary
        key = str(bin_arr[i]) + '_' + str(bin_arr[i+1])
    
        #Initialize an empty list for this key
        templst = []
        bin_dict[key] = templst
    
        #Iterate over x_val, append to templst if condition is met
        for j in range(0, len(x_val)):
            if x_val[j] >= bin_arr[i] and x_val[j] < bin_arr[i+1]:
                #Append y_val[j] directly to the list in the dictionary
                bin_dict[key].append(y_val[j])

    #for conversion of lists to numpy arrays
    for key in bin_dict:
        bin_dict[key] = np.array(bin_dict[key])
    #print(bin_dict)

    bin_dict_mean = {}
    for key in bin_dict:
        if len(bin_dict[key])!=0:
            bin_dict_mean[key] = np.nanmean(bin_dict[key]) #modified to nanmean from mean even though there isn't a chance of getting nan values (just safety)
        else:
            bin_dict_mean[key] = -5000

    return bin_dict_mean # bin_dict,
"""


# In[ ]:


#making the whole bin_dict and bin_dict_mean generation code into a function so that it can be run multiple times for the required dimnesions of the required days
"""
def give_binned_vals(x_val, y_val):
    bin_dict = {}
    bin_arr = np.arange(0,24.25,0.25)
    for i in range(0, len(bin_arr) - 1):
        #Create the key for the dictionary
        key = str(bin_arr[i]) + '_' + str(bin_arr[i+1])
    
        #Initialize an empty list for this key
        templst = []
        bin_dict[key] = templst
    
        #Iterate over x_val, append to templst if condition is met
        for j in range(0, len(x_val)):
            if x_val[j] >= bin_arr[i] and x_val[j] < bin_arr[i+1]:
                #Append y_val[j] directly to the list in the dictionary
                bin_dict[key].append(y_val[j])

    #for conversion of lists to numpy arrays
    for key in bin_dict:
        bin_dict[key] = np.array(bin_dict[key])
    #print(bin_dict)

    bin_dict_mean = {}
    for key in bin_dict:
        if len(bin_dict[key])!=0:
            bin_dict_mean[key] = np.nanmean(bin_dict[key]) #modified to nanmean from mean even though there isn't a chance of getting nan values (just safety)
        else:
            bin_dict_mean[key] = -5000

    return bin_dict_mean # bin_dict,

#binning and aggregating stress values by hour instead of by 15 minutes
"""
def give_binned_vals(x_val, y_val, time_bin_val):
    #time_bin_val is a string variable that should say either '15' or '30' or '60' to indicate 15 minute time bins or half hour time bins or one hour time bins
    bin_dict = {}
    if time_bin_val == '60':
        bin_arr = np.arange(0,25)
    elif time_bin_val == '30':
        bin_arr = np.arange(0,24.5,0.5)
    elif time_bin_val == '15': #did elif instead of else because this allows adding more time bin lengths in future if needed instead of else defaulting to 15 minute time bins. This will force the user to explicitly state the value of time bin
        bin_arr = np.arange(0,24.25,0.25)
    
    for i in range(0, len(bin_arr) - 1):
        #Create the key for the dictionary
        key = str(bin_arr[i]) + '_' + str(bin_arr[i+1])
    
        #Initialize an empty list for this key
        templst = []
        bin_dict[key] = templst
    
        #Iterate over x_val, append to templst if condition is met
        for j in range(0, len(x_val)):
            if x_val[j] >= bin_arr[i] and x_val[j] < bin_arr[i+1]:
                #Append y_val[j] directly to the list in the dictionary
                bin_dict[key].append(y_val[j])

    #for conversion of lists to numpy arrays
    for key in bin_dict:
        bin_dict[key] = np.array(bin_dict[key])
    #print(bin_dict)

    bin_dict_mean = {}
    for key in bin_dict:
        if len(bin_dict[key])!=0:
            bin_dict_mean[key] = np.nanmean(bin_dict[key]) #modified to nanmean from mean even though there isn't a chance of getting nan values (just safety)
        else:
            bin_dict_mean[key] = -5000

    return bin_dict_mean # bin_dict,



def give_binned_vals_category(x_val, y_val):
    bin_dict = {}
    bin_arr = np.arange(0,25,6)
    for i in range(0, len(bin_arr) - 1):
        #Create the key for the dictionary
        key = str(bin_arr[i]) + '_' + str(bin_arr[i+1])
    
        #Initialize an empty list for this key
        templst = []
        bin_dict[key] = templst
    
        #Iterate over x_val, append to templst if condition is met
        for j in range(0, len(x_val)):
            if x_val[j] >= bin_arr[i] and x_val[j] < bin_arr[i+1]:
                #Append y_val[j] directly to the list in the dictionary
                bin_dict[key].append(y_val[j])

    #for conversion of lists to numpy arrays
    for key in bin_dict:
        bin_dict[key] = np.array(bin_dict[key])
    #print(bin_dict)

    bin_dict_mean = {}
    for key in bin_dict:
        if len(bin_dict[key])!=0:
            bin_dict_mean[key] = np.nanmean(bin_dict[key]) #modified to nanmean from mean even though there isn't a chance of getting nan values (just safety)
        else:
            bin_dict_mean[key] = -5000

    return bin_dict_mean # bin_dict,



# In[4]:


"""
Definition Block
"""
ger = True
"""
Question list
"""
if ger:
    question_dict = {
    'q1' : 'Wie wach fühlten Sie sich im Tagesverlauf?',
    'q2' : 'Wie gelangweilt fühlten Sie sich im Tagesverlauf?',
    'q3' : 'Wie sehr haben Sie gezielt versucht  einen oder verschiedene Sinneseindrücke zu vermeiden (z.B. Geruch  Geschmack  Geräusche)?',
    'q4' : 'Wie sehr haben Sie versucht  soziale Interaktion zu vermeiden (virtuell und/oder persönlich)?',
    'q5' : 'Wie sehr haben Sie sich über den Tag hinweg körperlich angespannt gefühlt?',
    'q6' : 'Wie sehr haben Sie sich Sorgen gemacht über gegenwärtige oder zukünftige Ereignisse/Erfahrungen?',
    'q7' : 'Wie sehr haben Sie sich Sorgen oder Gedanken über vergangene Erfahrungen / Ereignisse gemacht?',
    'q8' : 'Wie sehr haben Sie sich im Tagesverlauf gestresst gefühlt?',
    'q9' : 'Wie sehr haben Sie körperliche Schmerzen im Tagesverlauf gehabt?',
    'q10' : 'Falls Sie eine zusätzliche individuelle Erfahrung und ihre Dynamik ergänzen möchten  haben Sie hier Platz. Bitte ergänzen Sie den Namen der Erfahrung als Überschrift  sowie die Ausprägung der Intensität im Graphen links. '
    }    


else:
    question_dict = {
    'q1' : 'Question 1: How alert did you feel during the day?',
    'q2' : 'Question 2: How bored did you feel during the day?',
    'q3' : 'Question 3: Were you avoiding stimulation of your senses (touch',
    'q4' : 'Question 4: Were you avoiding social interactions (virtual and/or in person)?',
    'q5' : 'Question 5: How physically tense did you feel throughout the day? ',
    'q6' : 'Question 6: How much were you lost in thoughts worrying about present and future events (e.g.',
    'q7' : 'Question 7: How much were you lost in thoughts worrying about past events (e.g.',
    'q8' : 'Question 8: How stressed did you feel during the day?',
    'q9' : 'Question 9: How strong did you feel physical pain?'
    }


# In[91]:


mainfolder = input('enter subject folder: ')
ger = True
dict_TET_x1, dict_TET_y1 = giv_x_y_vals(mainfolder, question_dict['q1'], ger)
dict_TET_x2, dict_TET_y2 = giv_x_y_vals(mainfolder, question_dict['q2'], ger)
dict_TET_x3, dict_TET_y3 = giv_x_y_vals(mainfolder, question_dict['q3'], ger)
dict_TET_x4, dict_TET_y4 = giv_x_y_vals(mainfolder, question_dict['q4'], ger)
dict_TET_x5, dict_TET_y5 = giv_x_y_vals(mainfolder, question_dict['q5'], ger)
dict_TET_x6, dict_TET_y6 = giv_x_y_vals(mainfolder, question_dict['q6'], ger)
dict_TET_x7, dict_TET_y7 = giv_x_y_vals(mainfolder, question_dict['q7'], ger)
dict_TET_x8, dict_TET_y8 = giv_x_y_vals(mainfolder, question_dict['q8'], ger)
dict_TET_x9, dict_TET_y9 = giv_x_y_vals(mainfolder, question_dict['q9'], ger)


# In[92]:


"""
Definition Block
"""
dict_TET_map = {
    "dict_TET_x1": dict_TET_x1,
    "dict_TET_x2": dict_TET_x2,
    "dict_TET_x3": dict_TET_x3,
    "dict_TET_x4": dict_TET_x4,
    "dict_TET_x5": dict_TET_x5,
    "dict_TET_x6": dict_TET_x6,
    "dict_TET_x7": dict_TET_x7,
    "dict_TET_x8": dict_TET_x8,
    "dict_TET_x9": dict_TET_x9,
    "dict_TET_y1": dict_TET_y1,
    "dict_TET_y2": dict_TET_y2,
    "dict_TET_y3": dict_TET_y3,
    "dict_TET_y4": dict_TET_y4,
    "dict_TET_y5": dict_TET_y5,
    "dict_TET_y6": dict_TET_y6,
    "dict_TET_y7": dict_TET_y7,
    "dict_TET_y8": dict_TET_y8,
    "dict_TET_y9": dict_TET_y9
}


# In[93]:


ip_dict_TET_x = input('enter the dictionary of x values you need binned values for: ')
if ip_dict_TET_x in dict_TET_map:
    dict_TET_x = dict_TET_map[ip_dict_TET_x]
    print("dictionary for x assigned")
else:
    print("invalid input, enter correct dictionary name")

ip_dict_TET_y = input('enter the dictionary of y values you need binned values for: ')
if ip_dict_TET_y in dict_TET_map:
    dict_TET_y = dict_TET_map[ip_dict_TET_y]
    print("dictionary for y assigned")
else:
    print("invalid input, enter correct dictionary name")
    
stream_sub_dict = {}
stream_sub_dict_hour = {}
for key in dict_TET_x:
    x_val = (dict_TET_x[key])*6
    y_val = dict_TET_y[key]
    stream_sub_dict[key] = give_binned_vals(x_val, y_val)
    stream_sub_dict_hour[key] = give_binned_vals_hour(x_val, y_val)


# In[ ]:


"""
enter the dictionary of x values you need binned values for: dict_TET_x8
dictionary for x assigned
enter the dictionary of y values you need binned values for: dict_TET_y8
dictionary for y assigned
"""


# In[ ]:


"""
Till now, all days were processed together.
From now, the below visualisations will be done day by day
"""


# In[94]:


"""
Definition Block
"""
parentfolder = input("enter path to subject folder for the required day: ")
"""
names of folders to be used
"""
folder1 = 'empatica'
folder2 = 'saved_figures'

folder11 = 'aggr_p_min'
folder12 = 'avro_files'
folder13 = 'avro2csv'


# In[95]:


#Optional visualisations of TET with  aggr scl, temp and acc and save figures

"""
code block to record the start and end timestamps (in utc/unix as well as cet) in a nested dictionary format. The key would be the filename (with or without .csv), and the values would be the first and last timestamps as entered in the excel sheet. All the files generated for that day are to be included in order to obtain a quick overview.
"""
def get_start_end_timestamp(ipfileDir):
    for file in os.listdir(ipfileDir):
        if file.endswith('eda_cet.csv'):
            df_eda = pd.read_csv(os.path.join(ipfileDir, file))
            dict_time = {}
            dict_time['start'] = [df_eda['unix_timestamp'].iloc[0], df_eda['CET_timestamp'].iloc[0]]
            dict_time['end'] = [df_eda['unix_timestamp'].iloc[-1], df_eda['CET_timestamp'].iloc[-1]]
            filenm = file
    return dict_time, filenm
    
dict_file = {}
for folder in os.listdir(os.path.join(parentfolder, folder1, folder13)):
    ipfileDir = os.path.join(parentfolder, folder1, folder13, folder)
    dict_time, file = get_start_end_timestamp(ipfileDir)
    dict_file[file] = dict_time #took out file.split('.')[0] and replaced it with file instead because it might be easier to iterate over when doing os.listdir


# In[96]:


dict_file


# In[97]:


#Extracting aggr eda, temp and acc (these can be used for plotting or other analyses as required)
"""
now to gather all scaled values of empatica eda start and end times in 2 separate lists
"""
start_times = []
end_times = []
for key in dict_file:
    time_val_start = int(dict_file[key]['start'][1].split()[1].split(':')[0])+(int(dict_file[key]['start'][1].split()[1].split(':')[1]))/60
    start_times.append(time_val_start)
    time_val_end = int(dict_file[key]['end'][1].split()[1].split(':')[0])+(int(dict_file[key]['end'][1].split()[1].split(':')[1]))/60
    end_times.append(time_val_end)

"""
the above code line seemed to work fine. Now to discard any values that pertain to the day after the date in question (which is the day for which the TET data is plotted here so far)
"""
date_num = eval(input('enter the date required: '))
start_times = []
end_times = []
for key in dict_file:
    if int(dict_file[key]['start'][1].split()[0].split('-')[2])==date_num:
        time_val_start = int(dict_file[key]['start'][1].split()[1].split(':')[0])+(int(dict_file[key]['start'][1].split()[1].split(':')[1]))/60
        start_times.append(time_val_start)
    if int(dict_file[key]['end'][1].split()[0].split('-')[2])==date_num:
        time_val_end = int(dict_file[key]['end'][1].split()[1].split(':')[0])+(int(dict_file[key]['end'][1].split()[1].split(':')[1]))/60
        end_times.append(time_val_end)



# In[98]:


"""
Definition Block
"""
ys = np.full(len(start_times), -0.2)
ye = np.full(len(end_times), -0.4)


# In[99]:


"""
Definition block: function
"""
def get_req_ips(parentfolder, folder2, dict_TET_map, question_dict):
    key = parentfolder.split('\\')[-1]
    
    #extract x
    ip_dict_TET_x = input('enter the main dictionary from which the required x values should be extracted: ')
    if ip_dict_TET_x in dict_TET_map:
        dict_TET_x = dict_TET_map[ip_dict_TET_x]
        x = dict_TET_x[key]
        print("x assigned")
    else:
        print("invalid input, enter correct dictionary name")
        
    #extract y
    ip_dict_TET_y = input('enter the main dictionary from which the required y values should be extracted: ')
    if ip_dict_TET_y in dict_TET_map:
        dict_TET_y = dict_TET_map[ip_dict_TET_y]
        y = dict_TET_y[key]
        print("y assigned")
    else:
        print("invalid input, enter correct dictionary name")
        
    #get question number for the figure name when saving to folder
    q = input('enter the question number (in the form q1, q2, q3, etc): ')
    #from q, obtaining the question and hence the title for the figure
    title = question_dict[eval(q)]
    #get folder name in which to save 
    fig_folder = os.path.join(parentfolder, folder2)
    
    
    return x, y, q, title, fig_folder   
       
    
    
    
        


# In[100]:


"""
now to plot TET data, start and end times all in one graph
"""
x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

#Plotting
plt.figure(figsize=(35, 16))  
plt.plot(x*6, y, label='TET data')  # Plot using x_val as x-axis and y_val as y-axis
plt.scatter(start_times, ys, color = 'orange', label='Start times')
plt.scatter(end_times, ye, color = 'green', label='End times')
plt.xlim(0, 24)
plt.xticks(range(0, 25, 1))

#title and labels
plt.title(title)
plt.xlabel('hours since start of day')
plt.ylabel('intensity')

#legend
plt.legend()

#saving and displaying figure
#plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_rawdata_timeavailability_new.png')))
plt.savefig(os.path.join(fig_folder, ('TET_q8_rawdata_timeavailability_new.png')))
plt.show()


# In[ ]:


"""
enter the main dictionary from which the required x values should be extracted: dict_TET_x8
x assigned
enter the main dictionary from which the required y values should be extracted: dict_TET_y8
y assigned
enter the question number (in the form q1, q2, q3, etc): q8
"""


# In[101]:


"""
Now, need to aggregate and align the eda aggregated per minute which has the scl values throughout the day aggregated by minute
Since this file has timestamps in ISO, need to readjust it to align with CET timezone
"""
"""
#insert file name via 'copy as path' on windows. Therefore no need to replace with 'os.path.join', etc
filepath_eda = input('enter filepath to the required eda aggr file (via "copy as path" on windows): ')
df_eda_aggr = pd.read_csv(filepath_eda)
"""
"""
To incorporate temperature and acc data as well. Starting out with what is given in the per minute aggregated datasets and averaging that for every 15 minutes.
"""
#same as aggr eda, enter file name via "copy as path" for windows
filepath_temp = input('enter filepath to the required temp aggr file (via "copy as path" on windows): ')
df_temp_aggr = pd.read_csv(filepath_temp)

#same as aggr eda, enter file name via "copy as path" for windows
filepath_devrec = input('enter filepath to the required device recording file (via "copy as path" on windows): ')
df_devrec_aggr = pd.read_csv(filepath_devrec)

"""
filepath_acc = input('enter filepath to the required acc aggr file (via "copy as path" on windows): ')
df_acc_aggr = pd.read_csv(filepath_acc)
"""


# In[48]:


"""
Definition Block
"""
aggr_dict = {}
for i in range(0,len(df_eda_aggr['timestamp_iso']), 15):
    #1 added to convert the iso(utc) to cet
    aggr_dict[1 + int(df_eda_aggr['timestamp_iso'][i].split('T')[1].split(':')[0]) + (int(df_eda_aggr['timestamp_iso'][i].split('T')[1].split(':')[1]))/60] = df_eda_aggr['eda_scl_usiemens'][i:(i+15)].mean(skipna=True)

#aggr_dict
x_aggr = list(aggr_dict.keys())
y_aggr = list(aggr_dict.values())


# In[49]:


"""
now to plot TET data, start and end times, 15 minute aggr scl (obtained from per minute aggr scl) all in one graph
"""
x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

# Plotting
plt.figure(figsize=(35, 26))  
plt.plot(x*6, y, label='TET data')  # Plot using x_val as x-axis and y_val as y-axis
plt.scatter(start_times, ys, color = 'orange', label='Start times')
plt.scatter(end_times, ye, color = 'green', label='End times')
plt.scatter(x_aggr, y_aggr, color = 'red', label='SCL aggregated per 15 minutes')
plt.xlim(0, 24)
plt.xticks(range(0, 25, 1))

# Adding title and labels
plt.title(title)
plt.xlabel('hours since start of day')
plt.ylabel('intensity')

# Show legend
plt.legend()

plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_aggr_scl_noscale.png')))
plt.show()


# In[50]:


"""
Definition Block
Calculating the standardised version of the aggregate values of scl from empatica
"""
mean_val = np.nanmean(y_aggr)
std_dev = np.nanstd(y_aggr)
standardized_values = [(x - mean_val) / std_dev for x in y_aggr] 


# In[51]:


"""
now to plot them all but this time with the standardised version of the aggregate values
"""

x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

"""
now to plot TET data, start and end times, 15 minute aggr scl (obtained from per minute aggr scl) all in one graph
"""
ys = np.full(len(start_times), -1.2) #this line and the next retained here because the values have been changed
ye = np.full(len(end_times), -1.4)

# Plotting
plt.figure(figsize=(35, 26))  
plt.plot(x*6, y, label='TET data')  # Plot using x_val as x-axis and y_val as y-axis
plt.scatter(start_times, ys, color = 'orange', label='Start times')
plt.scatter(end_times, ye, color = 'green', label='End times')
plt.scatter(x_aggr, standardized_values, color = 'red', label='SCL aggregated per 15 minutes, standardised')
plt.xlim(0, 24)
plt.xticks(range(0, 25, 1))

# Adding title and labels
plt.title(title)
plt.xlabel('hours since start of day')
plt.ylabel('intensity')

# Show legend
plt.legend()

plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_aggr_scl_standardised.png')))
plt.show()


# In[52]:


"""
Definition Block
Calculating the min-max normalised version of the aggregate values of scl from empatica
"""
min_val = np.nanmin(y_aggr)
max_val = np.nanmax(y_aggr)
if min_val == max_val:
    print("Warning! Possible division by 0! Do not plot")
else:
    normalized_val = [(x - min_val) / (max_val - min_val) for x in y_aggr]


# In[53]:


"""
now to plot them all but this time with min-max normalised version of the aggregate values
"""
x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

"""
now to plot TET data, start and end times, 15 minute aggr scl (obtained from per minute aggr scl) all in one graph
"""
ys = np.full(len(start_times), -0.5) #retained because value changed
ye = np.full(len(end_times), -0.7)

# Plotting
plt.figure(figsize=(35, 26))  
plt.plot(x*6, y, label='TET data')  # Plot using x_val as x-axis and y_val as y-axis
plt.scatter(start_times, ys, color = 'orange', label='Start times')
plt.scatter(end_times, ye, color = 'green', label='End times')
plt.scatter(x_aggr, normalized_val, color = 'red', label='SCL aggregated per 15 minutes, normalised')
plt.xlim(0, 24)
plt.xticks(range(0, 25, 1))

# Adding title and labels
plt.title(title)
plt.xlabel('hours since start of day')
plt.ylabel('intensity')

# Show legend
plt.legend()
plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_aggr_scl_minmaxnorm.png')))
plt.show()


# In[102]:


"""
Definition Block
"""
aggr_dict_temp = {}
for i in range(0,len(df_temp_aggr['timestamp_iso']), 15):
    #1 added to convert the iso(utc) to cet. The '1' can be changed keeping in mind daylight savings time and so on
    aggr_dict_temp[1 + int(df_temp_aggr['timestamp_iso'][i].split('T')[1].split(':')[0]) + (int(df_temp_aggr['timestamp_iso'][i].split('T')[1].split(':')[1]))/60] = df_temp_aggr['temperature_celsius'][i:(i+15)].mean(skipna=True)/10

aggr_dict_devrec = {}
for i in range(0,len(df_devrec_aggr['timestamp_iso']), 15):
    #1 added to convert the iso(utc) to cet. The '1' can be changed keeping in mind daylight savings time and so on
    aggr_dict_devrec[1 + int(df_devrec_aggr['timestamp_iso'][i].split('T')[1].split(':')[0]) + (int(df_devrec_aggr['timestamp_iso'][i].split('T')[1].split(':')[1]))/60] = df_devrec_aggr['wearing_detection_percentage'][i:(i+15)].mean(skipna=True)/100


"""
aggr_dict_acc = {}
for i in range(0,len(df_acc_aggr['timestamp_iso']), 15):
    #1 added to convert the iso(utc) to cet
    aggr_dict_acc[1 + int(df_acc_aggr['timestamp_iso'][i].split('T')[1].split(':')[0]) + (int(df_acc_aggr['timestamp_iso'][i].split('T')[1].split(':')[1]))/60] = df_acc_aggr['accelerometers_std_g'][i:(i+15)].mean(skipna=True)
"""

x_aggr_temp = list(aggr_dict_temp.keys())
y_aggr_temp = list(aggr_dict_temp.values())
#y_aggr_temp

x_aggr_devrec = list(aggr_dict_devrec.keys())
y_aggr_devrec = list(aggr_dict_devrec.values())
#y_aggr_temp

"""
x_aggr_acc = list(aggr_dict_acc.keys())
y_aggr_acc = list(aggr_dict_acc.values())
#y_aggr_acc
"""

"""
min-max for temp and acc

min_val_temp = np.nanmin(y_aggr_temp)
max_val_temp = np.nanmax(y_aggr_temp)
normalized_val_temp = [(x - min_val_temp) / (max_val_temp - min_val_temp) for x in y_aggr_temp]

min_val_acc = np.nanmin(y_aggr_acc)
max_val_acc = np.nanmax(y_aggr_acc)
normalized_val_acc = [(x - min_val_acc) / (max_val_acc - min_val_acc) for x in y_aggr_acc]
"""


# In[64]:


"""
adding temp and devrec values from the aggregated files
"""

x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

"""
now need to pair up the obtained scr and scl means with the right x values and plot their normalised version along with everything else
chosen x values to pair up with: start_times of the raw files from which they were obtained. 
"""

"""
now trying to plot them all but this time with min-max normalised version of the aggregate values
"""

"""
min-max for scl and scr
"""

"""
min-max for temp and acc
"""

"""
now to plot TET data, start and end times, 15 minute aggr scl (obtained from per minute aggr scl) all in one graph and scl,scr from flirt
"""

# Plotting
plt.figure(figsize=(25, 16))  
plt.plot(x*6, y, color = 'blue', label='TET data')  # Plot using x_val as x-axis and y_val as y-axis
plt.scatter(start_times, ys, color = 'orange', label='Start times')
plt.scatter(end_times, ye, color = 'green', label='End times')

#plt.plot(x_aggr, normalized_val, marker='o', linestyle='-', color = 'red', label='SCL aggregated per 15 minutes, normalised')
plt.plot(x_aggr_temp, y_aggr_temp, marker='o', linestyle='-', color = 'red', label='Temperature averaged per 15 minutes; scaled down by 10')
plt.plot(x_aggr_devrec, y_aggr_devrec, marker='o', linestyle='-', color = 'magenta', label='Device recording status averaged per 15 minutes; 1 for valid recording status')
#plt.plot(x_aggr_acc, normalized_val_acc, marker='o', linestyle='-', color = 'magenta', label='Standard deviation of the accelerometer magnitude aggregated per 15 minutes, normalised')


plt.xlim(0, 24)
plt.xticks(range(0, 25, 1))

# Adding title and labels
plt.title(title)
plt.xlabel('hours since start of day')
plt.ylabel('intensity')

# Show legend
plt.legend()

#plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_aggr_scl_temp_acc.png')))
plt.savefig(os.path.join(fig_folder, ('TET_q8_aggr_temp_devrec.png')))
plt.show()


# In[103]:


"""
adding temp and devrec values from the aggregated files
"""

x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

"""
now need to pair up the obtained scr and scl means with the right x values and plot their normalised version along with everything else
chosen x values to pair up with: start_times of the raw files from which they were obtained. 
"""

"""
now trying to plot them all but this time with min-max normalised version of the aggregate values
"""

"""
min-max for scl and scr
"""

"""
min-max for temp and acc
"""

"""
now to plot TET data, start and end times, 15 minute aggr scl (obtained from per minute aggr scl) all in one graph and scl,scr from flirt
"""

# Plotting
plt.figure(figsize=(25, 16))  

# Plot the main TET data
plt.plot(x*6, y, color='blue', label='TET data', linewidth=3)  # Thicker line

# Scatter for start and end times
plt.scatter(start_times, ys, color='orange', label='Start times of raw EDA files', s=150)  # Larger dots
plt.scatter(end_times, ye, color='green', label='End times of raw EDA files', s=150)  # Larger dots

# Plotting the additional data with thicker lines and larger markers
plt.plot(x_aggr_temp, y_aggr_temp, marker='o', linestyle='-', color='red', label='Temperature averaged over 15 minutes; scaled down by 10', linewidth=3, markersize=10)
plt.plot(x_aggr_devrec, y_aggr_devrec, marker='o', linestyle='-', color='magenta', label='Device recording status over 15 minutes; scaled down by 100', linewidth=3, markersize=10)

# Setting the limits and ticks for x-axis
plt.xlim(0, 24)
plt.xticks(range(0, 25, 1), fontsize=20)  # Larger x-axis ticks

# Adding title and labels with larger font sizes
plt.title(title, fontsize=30)  # Larger title
plt.xlabel('Hours since start of day', fontsize=25)  # Larger x-axis label
plt.ylabel('Magnitude', fontsize=25)  # Larger y-axis label

# Larger y-ticks
plt.yticks(fontsize=20)

# Show legend with larger font
plt.legend(fontsize=20)

# Save the figure in high resolution (dpi=300)
plt.savefig(os.path.join(fig_folder, 'TET_q8_aggr_temp_devrec.png'), bbox_inches='tight', dpi=300)

# Display the plot
plt.show()


# In[55]:


"""
adding temp and acc values from the aggregated files
"""

x, y, q, title, fig_folder = get_req_ips(parentfolder, folder2, dict_TET_map, question_dict)

"""
now need to pair up the obtained scr and scl means with the right x values and plot their normalised version along with everything else
chosen x values to pair up with: start_times of the raw files from which they were obtained. 
"""

"""
now trying to plot them all but this time with min-max normalised version of the aggregate values
"""

"""
min-max for scl and scr
"""

"""
min-max for temp and acc
"""

"""
now to plot TET data, start and end times, 15 minute aggr scl (obtained from per minute aggr scl) all in one graph and scl,scr from flirt
"""

# Plotting
plt.figure(figsize=(35, 26))  
plt.plot(x*6, y, label='TET data')  # Plot using x_val as x-axis and y_val as y-axis
plt.scatter(start_times, ys, color = 'orange', label='Start times')
plt.scatter(end_times, ye, color = 'green', label='End times')

#plt.plot(x_aggr, normalized_val, marker='o', linestyle='-', color = 'red', label='SCL aggregated per 15 minutes, normalised')
plt.plot(x_aggr_temp, normalized_val_temp, marker='o', linestyle='-', color = 'navy', label='Temperature aggregated per 15 minutes, normalised')
#plt.plot(x_aggr_acc, normalized_val_acc, marker='o', linestyle='-', color = 'magenta', label='Standard deviation of the accelerometer magnitude aggregated per 15 minutes, normalised')


plt.xlim(0, 24)
plt.xticks(range(0, 25, 1))

# Adding title and labels
plt.title(title)
plt.xlabel('hours since start of day')
plt.ylabel('intensity')

# Show legend
plt.legend()

#plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_aggr_scl_temp_acc.png')))
plt.savefig(os.path.join(fig_folder, ('TET_' + q + '_aggr_temp_devrec.png')))
plt.show()


# In[ ]:




