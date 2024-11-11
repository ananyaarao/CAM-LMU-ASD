#!/usr/bin/env python
# coding: utf-8

# In[16]:


"""
from l2s2 (l2s2_TET_Data_Processing.ipynb)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

"""
turning the x and y values obtaining for each day into a function as well
"""
def drop_lead_trail_nans(arr):
    #boolean mask for non-NaN values
    non_nan_mask = ~np.isnan(arr)    
    #finding the indices of the first and last non-NaN values
    first_non_nan = np.argmax(non_nan_mask)  #First True value (non-NaN)
    last_non_nan = len(arr) - np.argmax(non_nan_mask[::-1])  #Last True value (non-NaN)    
    # Slice the array to drop leading and trailing NaNs
    cleaned_arr = arr[first_non_nan:last_non_nan]

    return cleaned_arr

def giv_x_y_vals(mainfolder, qnum, ger):
    #ger = True if subject recruited in Germany. Else False
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
        """    
        if folder != '18_3_24_n10_19_3_24_d': #'15_3_24_n7_16_3_24_d':
            continue
        """    
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
                    if df_tet[pl3].iloc[indices_tet[i]] == question_dict[qnum]:
                        if ind !=0:
                            print("ind has already been assigned, assigning this new index to another variable x_new and y_new: ", indices_tet[i], " this is its position in the indices list: ", i)
                            ind = indices_tet[i]
                            if qnum == 'q7':
                                if df_tet[pl2].iloc[ind+1] != expr3 and df_tet[pl2].iloc[ind+2] != expr3:
                                    print("Warning! Below this index, no x_new and y_new data may have been submitted therefore no data! -> ", ind)
                                else:
                                    x_new = df_tet['x_val'].iloc[indices_tet[i]:i1]
                                    y_new = df_tet['y_val'].iloc[indices_tet[i]:i1]
                                    print("x_new and y_new have been assigned and the correct index for q is ind which is: ", ind)

                                    if np.array_equal(np.array(x_new), np.array(x)) and np.array_equal(np.array(y_new), np.array(y)):
                                        print("both the old data and new data entered by the participant for the same question are identical, assigning new data to the dictionaries")
                                    else:
                                        print("the old data and new data entered by the participant for the same question may not be identical! Checking the data more closely")
                                       
                                        x_arr = drop_lead_trail_nans(np.array(x))
                                        x_new_arr = drop_lead_trail_nans(np.array(x_new))
                                        y_arr = drop_lead_trail_nans(np.array(y))
                                        y_new_arr = drop_lead_trail_nans(np.array(y_new))
        
                                        if len(x_arr) != len(x_new_arr) or len(y_arr) != len(y_new_arr):
                                            print("data not identical even after dropping leading and trailing nan values, assigning the new values to the dictionaries")
                                        else:
                                            if np.sum(x_new_arr-x_arr) == 0 and np.sum(y_new_arr-y_arr) == 0:
                                                print("data identical, but assigning the new values to dictionaries anyway")
                                            else: 
                                                print("data not identical after subtraction check, assigning the new values to the dictionaries")
                
                                    dict_TET_x[folder] = np.array(x_new)
                                    dict_TET_y[folder] = np.array(y_new)
                            else:
                                if df_tet[pl2].iloc[ind+1] != expr3:
                                    print("Warning! Below this index, no x and y data may have been submitted therefore no data! -> ", ind)
                                else:
                                    x_new = df_tet['x_val'].iloc[indices_tet[i]:i1]
                                    y_new = df_tet['y_val'].iloc[indices_tet[i]:i1]
                                    print("x_new and y_new have been assigned and the correct index for q is ind which is: ", ind)
                                    
                                    if np.array_equal(np.array(x_new), np.array(x)) and np.array_equal(np.array(y_new), np.array(y)):
                                        print("both the old data and new data entered by the participant for the same question are identical, assigning new data to the dictionaries")   
                                    else:
                                        print("the old data and new data entered by the participant for the same question may not be identical! Checking the data more closely")
                                       
                                        x_arr = drop_lead_trail_nans(np.array(x))
                                        x_new_arr = drop_lead_trail_nans(np.array(x_new))
                                        y_arr = drop_lead_trail_nans(np.array(y))
                                        y_new_arr = drop_lead_trail_nans(np.array(y_new))
        
                                        if len(x_arr) != len(x_new_arr) or len(y_arr) != len(y_new_arr):
                                            print("data not identical even after dropping leading and trailing nan values, assigning the new values to the dictionaries")
                                        else:
                                            if np.sum(x_new_arr-x_arr) == 0 and np.sum(y_new_arr-y_arr) == 0:
                                                print("data identical, but assigning the new values to dictionaries anyway")
                                            else: 
                                                print("data not identical after subtraction check, assigning the new values to the dictionaries")
                
                                    dict_TET_x[folder] = np.array(x_new)
                                    dict_TET_y[folder] = np.array(y_new)
                        else:
                            ind = indices_tet[i]
                            x_new = 0
                            y_new = 0
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
    return dict_TET_x, dict_TET_y, x_new, y_new, x, y
    
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
       
    
    


# In[3]:


"""
from l2s3 (l2s3_EDA_Data_Preprocessing.ipynb)
"""
import pandas as pd
import numpy as np
import os
import warnings
import csv

#pip install flirt
"""
flirt imports: optional

import flirt.with_
flirt.with_.me()
import flirt.reader.empatica


#now need function to one by one collect each raw avro file for a particular day, calculate eda fearures, get the mean of the columns tonic mean and phasic mean. These two measures for each day are to be collected in a dictionary. 


def give_means(file):
    for file in os.listdir(ipfileDir):
        if file.endswith('eda_cet.csv'):
            df_eda_eplus = pd.read_csv(os.path.join(ipfileDir, file)) #filepath joining functionality 
            new_cet_index = pd.to_datetime(df_eda_eplus['CET_timestamp'])
            df_eda_eplus.index = new_cet_index #indexing by date-time object to avoid error with flirt package
            eda_feat_eplus = flirt.get_eda_features(df_eda_eplus['eda'],
                                          window_length = 60, 
                                          window_step_size = 1,
                                          data_frequency = 4)
    return(eda_feat_eplus['tonic_mean'].mean(skipna=True), eda_feat_eplus['phasic_mean'].mean(skipna=True)) 
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as scisig
import pywt
import os
import datetime

from sklearn.metrics.pairwise import rbf_kernel

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

def predict_binary_classifier(X):
    ''''
    X: num test data by 13 features
    '''

    # Get params
    params = binary_classifier()

    # compute kernel for all data points
    K = rbf_kernel(params['support_vec'], X, gamma=params['gamma'])

    # Prediction = sign((sum_{i=1}^n y_i*alpha*K(x_i,x)) + rho)
    predictions = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        # prediction is calculated and then divided into positive (no artefact) or negative (artefact)
        predictions[i] = np.sign(np.sum(params['dual_coef']*K[:, i]) + params['intercept'])

    return predictions


def binary_classifier():
    gamma = 0.1

    # dual coef = y_i*alpha_i
    dual_coef = np.array([[-1.12775599e+02,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -4.65947457e+02,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.17935400e+02,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -2.92534132e+02,  -1.00000000e+03,
                           -1.00000000e+03,  -3.69965631e+01,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,  -1.00000000e+03,
                           -1.00000000e+03,  -1.00000000e+03,   1.00000000e+03,
                           1.00000000e+03,   1.00000000e+03,   1.00000000e+03,
                           7.92366387e+02,   3.00553142e+02,   2.22950860e-01,
                           1.00000000e+03,   1.00000000e+03,   5.58636056e+02,
                           1.21751544e+02,   1.00000000e+03,   1.00000000e+03,
                           2.61920652e+00,   9.96570403e+02,   1.00000000e+03,
                           1.00000000e+03,   1.00000000e+03,   1.00000000e+03,
                           1.00000000e+03,   1.00000000e+03,   1.02270060e+02,
                           5.41288840e+01,   1.91650287e+02,   1.00000000e+03,
                           1.00000000e+03,   1.00000000e+03,   1.00000000e+03,
                           1.00000000e+03,   2.45152637e+02,   7.53766346e+02,
                           1.00000000e+03,   1.00000000e+03,   3.63211198e+00,
                           1.00000000e+03,   3.31675798e+01,   5.64620367e+02,
                           1.00000000e+03,   1.00000000e+03,   1.00000000e+03,
                           2.66900636e+02,   1.00000000e+03,   6.54763900e+02,
                           3.38216549e+02,   6.86434772e+01,   2.78998678e+02,
                           6.97557950e+02,   1.00000000e+03]])

    # intercept = rho
    intercept = np.array([-2.63232929])

    # support vectors = x_i
    support_vec = np.array([[0.02809756, 0.0455, 0.025, 0.00866667, 0.03799132, -0.00799413, 0.01061208, 0.016263, 0.00671743, 0.00572262, 0.00578504, 0.00542415, 0.00318195],
                            [0.00060976, 0.0035, 0.007, 0.00087179, 0.00024191, -0.0005069, 0.0005069, 0.0070711, 0.00306413, 0.0031833, 0.0107827, 0.0066959, 0.0022981],
                            [3.49731707, 0.092, 0.054, 0.01923077, 3.53815367, -0.02236652, 0.02659884, 0.062225, 0.0316782, 0.01818914, 0.06607571, 0.03342241, 0.099702],
                            [2.52643902, 0.058, 0.055, 0.0114359, 2.54031008, -0.01070662, 0.01296803, 0.043134, 0.01649923, 0.01579683, 0.03326171, 0.05004163, 0.013965],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.74622599e-18, -2.42947453e-17, 3.36047450e-17, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [3.89758537, 0.167, 0.27, 0.06717949, 3.87923565, -0.04130143, 0.05403825, 0.047376, 0.0328098, 0.01255584, 0.03676955, 0.14237773, 0.11031],
                            [0.93326829, 0.0855, 0.106, 0.01169231, 0.92669874, -0.02740927, 0.02740927, 0.043841, 0.01131377, 0.01595008, 0.0231871, 0.02414775, 0.0139655],
                            [4.64253659, 0.106, 0.13, 0.03661538, 4.63806066, -0.03168223, 0.03168223, 0.10182, 0.0559785, 0.03369301, 0.06341563, 0.08583294, 0.0251025],
                            [0.29312195, 0.028, 0.039, 0.00682051, 0.28575076, -0.00648365, 0.00648365, 0.0056569, 0.00367694, 0.00126494, 0.00364005, 0.01814984, 0.006364],
                            [3.08187805, 0.0615, 0.123, 0.03435897, 3.11862292, -0.02260403, 0.02260403, 0.053033, 0.0397394, 0.01570345, 0.0338851, 0.10069204, 0.16652],
                            [2.43902439e-05, 5.00000000e-04, 1.00000000e-03, 1.02564103e-04, 2.43769719e-05, -7.19856842e-05, 7.19856842e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -4.05052739e-10, -2.77557303e-09, 5.77955577e-09, 7.07110000e-04, 1.17851667e-04, 2.88676449e-04, 2.04124145e-04, 1.44336183e-04, 0.00000000e+00],
                            [0.83290244, 0.099, 0.172, 0.02610256, 0.82408369, -0.0168393, 0.0168393, 0.13011, 0.02875613, 0.04987211, 0.03786379, 0.02684837, 0.0155565],
                            [0.92597561, 0.017, 0.009, 0.00369231, 0.92583814, -0.00670974, 0.00670974, 0.012021, 0.00506763, 0.00420523, 0.01259266, 0.0115391, 0.00265165],
                            [2.43902439e-05, 5.00000000e-04, 1.00000000e-03, 2.56410256e-05, 2.18000765e-04, -5.56411248e-04, 5.56411248e-04, 9.19240000e-03, 2.71058333e-03, 4.25246049e-03, 2.49833278e-03, 7.64311464e-03, 0.00000000e+00],
                            [0.88760976, 0.0205, 0.022, 0.00489744, 0.88799505, -0.00346772, 0.00461828, 0.011314, 0.00447838, 0.00394135, 0.01327278, 0.01434142, 0.00406585],
                            [9.21263415, 0.118, 0.472, 0.0695641, 9.19153391, -0.02181738, 0.02181738, 0.16688, 0.07130037, 0.06135461, 0.04328934, 0.04277416, 0.0829085],
                            [0.48378049, 0.017, 0.026, 0.00794872, 0.48333175, -0.00337375, 0.00350864, 0.016971, 0.0089568, 0.00472601, 0.01168189, 0.01629524, 0.0226275],
                            [0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 9.65026603e-122, -2.00921455e-120, 4.22507597e-120, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000, 0.00000000e+000],
                            [0.10897561, 0.03, 0.033, 0.00553846, 0.12761266, -0.00442938, 0.00556735, 0.025456, 0.00872107, 0.00870258, 0.01130487, 0.01554551, 0.0123745],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -1.38812548e-09, -2.34438020e-08, 2.34438020e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0.66663415, 0.052, 0.05, 0.00510256, 0.66182973, -0.01361869, 0.01361869, 0.0049497, 0.00296982, 0.00208565, 0.00424264, 0.00961131, 0.012374],
                            [3.74146341e+00, 6.60000000e-02, 7.00000000e-02, 2.41025641e-02, 3.72790310e+00, -1.65194036e-02, 1.65194036e-02, 2.33350000e-02, 2.29102000e-02, 3.87787571e-04, 7.25086202e-03, 8.04828002e-03, 2.26270000e-02],
                            [2.43902439e-05, 5.00000000e-04, 1.00000000e-03, 1.02564103e-04, 2.44149661e-05, -7.19856850e-05, 7.19856850e-05, 7.07110000e-04, 1.17851667e-04, 2.88676449e-04, 2.04124145e-04, 1.44336183e-04, 0.00000000e+00],
                            [1.14713659e+01, 1.68000000e-01, 3.24000000e-01, 8.83589744e-02, 1.13977278e+01, -4.35202063e-02, 4.35202063e-02, 1.20920000e-01, 1.15826000e-01, 5.32593935e-03, 4.29825546e-02, 1.11681949e-01, 1.82080000e-01],
                            [1.63631707, 0.0825, 0.138, 0.02410256, 1.65473267, -0.02914746, 0.02927458, 0.074953, 0.02899134, 0.03271076, 0.02718317, 0.09610564, 0.012728],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.01460518e-42, -2.71490067e-40, 2.71490067e-40, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0.52358537, 0.038, 0.03, 0.00769231, 0.52319376, -0.01066405, 0.01066405, 0.026163, 0.01025307, 0.00912966, 0.02678697, 0.04011893, 0.00866185],
                            [0.10931707, 0.103, 0.407, 0.04461538, 0.13188551, -0.01686662, 0.02506229, 0.1492, 0.0384195, 0.06327203, 0.06411448, 0.05508901, 0],
                            [0.0444878, 0.0245, 0.04, 0.00984615, 0.03577326, -0.00573919, 0.00573919, 0.013435, 0.0078961, 0.00418135, 0.01136515, 0.01291603, 0.0134352],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.03127202e-08, -2.56175141e-07, 5.37317466e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.27917545e-05, -7.79437718e-04, 7.79437718e-04, 3.04060000e-02, 5.06766667e-03, 1.24131975e-02, 1.34721936e-02, 5.34029589e-02, 0.00000000e+00],
                            [2.43902439e-05, 5.00000000e-04, 1.00000000e-03, 1.02564103e-04, 2.60691650e-05, -7.19856850e-05, 7.19856850e-05, 7.07110000e-04, 1.17851667e-04, 2.88676449e-04, 2.04124145e-04, 1.44336183e-04, 0.00000000e+00],
                            [0.46446341, 0.033, 0.03, 0.00933333, 0.46299034, -0.00866364, 0.00866364, 0.033941, 0.01357644, 0.01214903, 0.02164486, 0.02701617, 0.012374],
                            [5.89978049, 0.117, 0.112, 0.04453846, 5.88525247, -0.02253416, 0.02253416, 0.084146, 0.0492146, 0.01985341, 0.06802812, 0.09041259, 0.045255],
                            [0.01317073, 0.0195, 0.015, 0.00538462, 0.00829287, -0.00622806, 0.00622806, 0.026163, 0.01145514, 0.00926554, 0.00690652, 0.02540613, 0.018031],
                            [1.16509756, 0.028, 0.02, 0.01051282, 1.16338281, -0.01379371, 0.01379371, 0.020506, 0.01461345, 0.00563317, 0.01416569, 0.01971055, 0.0281075],
                            [3.67914634, 0.1235, 0.126, 0.02676923, 3.67052968, -0.04266586, 0.04266586, 0.041719, 0.0233342, 0.0106888, 0.03232337, 0.07260248, 0.050912],
                            [0.11331707, 0.0015, 0.004, 0.0014359, 0.11329803, -0.00042144, 0.00042144, 0.0021213, 0.0014142, 0.00109543, 0.00124164, 0.00053231, 0.00070713],
                            [1.11256098, 0.026, 0.016, 0.00561538, 1.09093248, -0.00174647, 0.00490015, 0.02192, 0.01272782, 0.00816993, 0.02111102, 0.04921207, 0.012021],
                            [0.06846341, 0.007, 0.01, 0.00307692, 0.06774886, -0.00179795, 0.00190969, 0.0056569, 0.00311126, 0.00162791, 0.00195576, 0.00721732, 0.01096],
                            [1.16454634e+01, 1.78500000e-01, 3.20000000e-01, 8.94615385e-02, 1.15869935e+01, -1.15451745e-02, 1.59897956e-02, 1.37890000e-01, 1.23393333e-01, 1.01170444e-02, 3.66151153e-02, 1.46607419e-01, 1.94455000e-01],
                            [3.45158537, 0.1375, 0.052, 0.01676923, 3.44594643, -0.03141983, 0.03141983, 0.038184, 0.0272946, 0.00958649, 0.01698014, 0.06290749, 0.1393],
                            [3.12563415, 0.0535, 0.111, 0.02897436, 3.17337638, -0.02835417, 0.02835417, 0.054447, 0.0278601, 0.0188188, 0.00755315, 0.03628251, 0.055154],
                            [8.50975610e-02, 1.00000000e-03, 4.00000000e-03, 8.20512821e-04, 8.50491997e-02, -1.84870042e-04, 2.35933619e-04, 1.41420000e-03, 1.41420000e-03, 2.60312573e-11, 4.08248290e-04, 2.88668284e-04, 7.07110000e-04],
                            [0.82373171, 0.048, 0.121, 0.01853846, 0.82149219, -0.0053288, 0.00684639, 0.041012, 0.0208598, 0.01423898, 0.02609294, 0.02676908, 0.01078335],
                            [4.39680488, 0.223, 0.354, 0.09258974, 4.35973108, -0.03206468, 0.03450864, 0.20506, 0.0971572, 0.07235446, 0.13713059, 0.23019854, 0.32138],
                            [5.66058537, 0.0285, 0.093, 0.01282051, 5.66682734, -0.00633008, 0.00633008, 0.040305, 0.01513214, 0.01889847, 0.01503912, 0.03383458, 0],
                            [0.13329268, 0.011, 0.021, 0.00338462, 0.13419267, -0.00262455, 0.00262455, 0.0035355, 0.00226272, 0.00092195, 0.00772172, 0.00411547, 0.0038891],
                            [0.15463415, 0.0325, 0.065, 0.01617949, 0.15422134, -0.00766504, 0.00766504, 0.067882, 0.02286322, 0.02270081, 0.02939288, 0.0224428, 0.017501],
                            [1.47902439e-01, 1.50000000e-03, 2.00000000e-03, 3.84615385e-04, 1.48269290e-01, -1.36058722e-04, 1.36058722e-04, 2.12130000e-03, 8.24950000e-04, 9.39849132e-04, 5.16397779e-04, 5.91603500e-04, 0.00000000e+00],
                            [2.76797561, 0.071, 0.17, 0.03212821, 2.84223399, -0.01692731, 0.01692731, 0.04879, 0.03441267, 0.00934515, 0.03221283, 0.05768286, 0.092806],
                            [1.30939024, 0.044, 0.066, 0.0165641, 1.2967273, -0.01727205, 0.01727205, 0.03182, 0.01456652, 0.01056655, 0.00732632, 0.02987207, 0.038891],
                            [0.0914878, 0.038, 0.028, 0.00364103, 0.08295897, -0.00877545, 0.00877545, 0.032527, 0.00648182, 0.01277828, 0.01289089, 0.01040763, 0.0042426],
                            [0.13621951, 0.0015, 0.006, 0.00174359, 0.13689296, -0.00036169, 0.00040731, 0.0021213, 0.00153205, 0.00082663, 0.00058452, 0.00069522, 0.00088391],
                            [0.05692683, 0.007, 0.006, 0.00189744, 0.05532006, -0.00145672, 0.00145672, 0.0056569, 0.00311126, 0.00184393, 0.00420714, 0.00465287, 0.0070711],
                            [0.07460976, 0.002, 0.006, 0.00097436, 0.07430141, -0.00035004, 0.00038011, 0.0028284, 0.00113136, 0.0011832, 0.00070711, 0.0005916, 0.00070711],
                            [0.04782927, 0.006, 0.011, 0.00353846, 0.04406202, -0.00232859, 0.00232859, 0.012021, 0.00438408, 0.00442728, 0.00363318, 0.00540593, 0.0091924],
                            [4.443, 0.141, 0.076, 0.02310256, 4.40858239, -0.03710778, 0.03710778, 0.03182, 0.0271528, 0.00465324, 0.03506173, 0.07970664, 0.11278],
                            [8.79678049, 0.057, 0.208, 0.04194872, 8.784878, -0.01132933, 0.01132933, 0.08061, 0.04695182, 0.039817, 0.0405623, 0.01937402, 0.033234],
                            [2.58236585, 0.063, 0.128, 0.02112821, 2.5705713, -0.0079298, 0.01979542, 0.062225, 0.0309712, 0.02172778, 0.02949491, 0.02741888, 0.02687],
                            [0.08992683, 0.0015, 0.006, 0.00030769, 0.09000535, -0.00020308, 0.00020308, 0.0021213, 0.00106065, 0.00116188, 0.0007746, 0.00086603, 0.00053035],
                            [0.09085366, 0.0175, 0.037, 0.00694872, 0.09607742, -0.00456388, 0.00456388, 0.0098995, 0.00523258, 0.00310646, 0.01357571, 0.0133944, 0.0056569],
                            [1.34473171, 0.0255, 0.022, 0.00953846, 1.37010789, -0.00558419, 0.00558419, 0.030406, 0.0134351, 0.00877511, 0.00929516, 0.03188089, 0.0265165],
                            [0.14253659, 0.001, 0.004, 0.00097436, 0.14237889, -0.0002998, 0.0002998, 0.0014142, 0.0011785, 0.00057734, 0.0005164, 0.00069521, 0.00106066],
                            [0.07617073, 0.001, 0.004, 0.00179487, 0.07597272, -0.00025949, 0.00025949, 0.0014142, 0.0011785, 0.00057734, 0.0005164, 0.00063245, 0.00070711],
                            [0.28502439, 0.0025, 0.01, 0.00241026, 0.28596915, -0.000355, 0.000355, 0.12869, 0.02333393, 0.05162999, 0.0313152, 0.13233722, 0.0044194],
                            [5.97658537, 0.0645, 0.106, 0.02925641, 5.95365623, -0.01454886, 0.01454886, 0.045962, 0.02913296, 0.02145587, 0.04602717, 0.06410626, 0.053033],
                            [4.19787805, 0.0405, 0.072, 0.02764103, 4.21230508, -0.01456906, 0.01468492, 0.030406, 0.02206174, 0.01003006, 0.02031748, 0.03873656, 0.034295],
                            [0.06904878, 0.0025, 0.005, 0.00117949, 0.06819891, -0.00023428, 0.00033805, 0.0035355, 0.00098994, 0.00154918, 0.001, 0.0007071, 0.00070711],
                            [2.07410488e+01, 1.10000000e-02, 4.40000000e-02, 1.24102564e-02, 2.07288498e+01, -5.11402880e-02, 5.11402880e-02, 1.55560000e-02, 1.55560000e-02, 0.00000000e+00, 5.68037557e-03, 3.17543685e-03, 7.77820000e-03],
                            [0.15141463, 0.0025, 0.008, 0.00161538, 0.15286961, -0.00066236, 0.00066236, 0.0049497, 0.0021213, 0.00180276, 0.00235584, 0.01268589, 0.0021213],
                            [1.07970732, 0.0275, 0.046, 0.00725641, 1.0819483, -0.0025949, 0.00261392, 0.026163, 0.00754248, 0.00945165, 0.01400506, 0.00566908, 0.011137],
                            [1.45278049e+00, 2.50000000e-02, 3.40000000e-02, 8.23076923e-03, 1.46401853e+00, -5.22375992e-03, 7.56803574e-03, 8.48530000e-03, 6.71755000e-03, 1.39641061e-03, 4.14024959e-03, 1.47976972e-02, 2.03295000e-02],
                            [1.18829268e-01, 1.00000000e-03, 4.00000000e-03, 1.17948718e-03, 1.18657803e-01, -3.33958979e-04, 3.55599268e-04, 1.41420000e-03, 1.41420000e-03, 2.60312573e-11, 6.32455532e-04, 5.32284214e-04, 7.07110000e-04],
                            [0.09217073, 0.0085, 0.007, 0.00258974, 0.07952256, -0.00104703, 0.00138337, 0.006364, 0.00466692, 0.00203719, 0.00509166, 0.01307342, 0.021213],
                            [0.06936585, 0.0095, 0.015, 0.00394872, 0.06837444, -0.00205373, 0.00205373, 0.0084853, 0.00296984, 0.0030984, 0.00234521, 0.00419839, 0.0017678],
                            [5.05807317, 0.049, 0.082, 0.02402564, 5.06327737, -0.01120311, 0.01120311, 0.031113, 0.0239, 0.01338272, 0.01117139, 0.04351642, 0.020506],
                            [0.26421951, 0.04, 0.068, 0.00902564, 0.2587529, -0.01040894, 0.01040894, 0.025456, 0.01060666, 0.00890233, 0.01111643, 0.04563416, 0.011314],
                            [3.59336585, 0.0575, 0.054, 0.02094872, 3.58195886, -0.01804095, 0.01838506, 0.043134, 0.0336584, 0.01240579, 0.01683523, 0.04717173, 0.038184],
                            [1.29187805, 0.026, 0.016, 0.00689744, 1.27916244, -0.00322078, 0.00490015, 0.025456, 0.01032378, 0.00861112, 0.01863263, 0.0636921, 0.038537],
                            [6.28670732, 0.1245, 0.127, 0.03102564, 6.35501978, -0.01747513, 0.02813757, 0.084146, 0.04690465, 0.0254467, 0.06541464, 0.18275149, 0.15008],
                            [10.64578049, 0.079, 0.284, 0.04564103, 10.64447668, -0.01946271, 0.01947497, 0.10889, 0.04186, 0.05739752, 0.06891299, 0.05417812, 0.050205],
                            [3.32470732, 0.092, 0.046, 0.01687179, 3.32977984, -0.02794509, 0.02794509, 0.072125, 0.0288498, 0.02428699, 0.06277798, 0.10343739, 0.061518],
                            [0.07358537, 0.001, 0.004, 0.00153846, 0.0735262, -0.00027514, 0.00027514, 0.0014142, 0.0009428, 0.00073029, 0.00075277, 0.00053228, 0.00070711]])

    return {'dual_coef': dual_coef,
            'support_vec': support_vec,
            'intercept': intercept,
            'gamma': gamma}

def interpolateDataTo8Hz(data,sample_rate): #interpolateDataTo8Hz(dict_data['eda'], (1/dict_data['eda']['sampRate'].iloc[0]))
    if sample_rate<8:
        # Upsample by linear interpolation
        data = data.resample("125L").mean()
    else:
        if sample_rate>8:
            # Downsample
            idx_range = list(range(0,len(data))) # TODO: double check this one
            data = data.iloc[idx_range[0::int(int(sample_rate)/8)]]
        # Set the index to be 8Hz
        data.index = pd.timedelta_range(start='0S', periods=len(data), freq='125L')
        #return data - RETURN STATEMENT 3 - ARRANGE THE APPROPRIATE RETURN STATMENT AT EDA_ARTIFACT_DETECTION_NOTAG AND PREPRO

    # Interpolate all empty values
    data = data.interpolate()
    return data #- RETURN STATEMENT 4 - ARRANGE THE APPROPRIATE RETURN STATMENT AT EDA_ARTIFACT_DETECTION_NOTAG AND PREPRO

def butter_lowpass(cutoff, fs, order=5):
    # Filtering Helper functions
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False) #therefore digital filter
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Filtering Helper functions
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y

def getWaveletData(data):
    '''
    This function computes the wavelet coefficients

    INPUT:
        data:           DataFrame, index is a list of timestamps at 8Hz, columns include EDA, filtered_eda

    OUTPUT:
        wave1Second:    DateFrame, index is a list of timestamps at 1Hz, columns include OneSecond_feature1, OneSecond_feature2, OneSecond_feature3 
        waveHalfSecond: DateFrame, index is a list of timestamps at 2Hz, columns include HalfSecond_feature1, HalfSecond_feature2 
    '''
    startTime = data.index[0]
    print('from getWaveletData, starttime is: ', startTime, type(startTime)) #NOT RETURN STATEMENT BUT LOOK OUT

    # Create wavelet dataframes
    oneSecond = pd.date_range(start=startTime, periods=len(data), freq='1s')
    halfSecond = pd.date_range(start=startTime, periods=len(data), freq='500L')
    #oneSecond = pd.timedelta_range(start=startTime, periods=len(data), freq='1s') #OPTIONAL (NOT) RETURN STATEMENTS TO REPRODUCE ERRORS IF REQUIRED
    #halfSecond = pd.timedelta_range(start=startTime, periods=len(data), freq='500L')

    # Compute wavelets
    cA_n, cD_3, cD_2, cD_1 = pywt.wavedec(data['eda'], 'Haar', level=3) #3 = 1Hz, 2 = 2Hz, 1=4Hz
    
    # Wavelet 1 second window
    N = int(len(data)/8)
    coeff1 = np.max(abs(np.reshape(cD_1[0:4*N],(N,4))), axis=1)
    coeff2 = np.max(abs(np.reshape(cD_2[0:2*N],(N,2))), axis=1)
    coeff3 = abs(cD_3[0:N])
    wave1Second = pd.DataFrame({'OneSecond_feature1':coeff1,'OneSecond_feature2':coeff2,'OneSecond_feature3':coeff3})
    wave1Second.index = oneSecond[:len(wave1Second)]
    
    # Wavelet Half second window
    N = int(np.floor((len(data)/8.0)*2))
    coeff1 = np.max(abs(np.reshape(cD_1[0:2*N],(N,2))),axis=1)
    coeff2 = abs(cD_2[0:N])
    waveHalfSecond = pd.DataFrame({'HalfSecond_feature1':coeff1,'HalfSecond_feature2':coeff2})
    waveHalfSecond.index = halfSecond[:len(waveHalfSecond)]

    return wave1Second,waveHalfSecond


def getDerivatives(eda):
    deriv = (eda[1:-1] + eda[2:])/ 2. - (eda[1:-1] + eda[:-2])/ 2.
    second_deriv = eda[2:] - 2*eda[1:-1] + eda[:-2]
    #put print statements under these
    return deriv,second_deriv

def getDerivStats(eda):
    deriv, second_deriv = getDerivatives(eda)
    maxd = max(deriv)
    mind = min(deriv)
    maxabsd = max(abs(deriv))
    avgabsd = np.mean(abs(deriv))
    max2d = max(second_deriv)
    min2d = min(second_deriv)
    maxabs2d = max(abs(second_deriv))
    avgabs2d = np.mean(abs(second_deriv))
    
    return maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d


def getStats(data):
    eda = data['eda'].values
    filt = data['filtered_eda'].values
    maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d = getDerivStats(eda)
    maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f = getDerivStats(filt)
    amp = np.mean(eda)
    amp_f = np.mean(filt)
    return amp, maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f


def computeWaveletFeatures(waveDF):
    maxList = waveDF.max().tolist()
    meanList = waveDF.mean().tolist()
    stdList = waveDF.std().tolist()
    medianList = waveDF.median().tolist()
    aboveZeroList = (waveDF[waveDF>0]).count().tolist()

    return maxList,meanList,stdList,medianList,aboveZeroList


def getWavelet(wave1Second,waveHalfSecond):
    max_1,mean_1,std_1,median_1,aboveZero_1 = computeWaveletFeatures(wave1Second)
    max_H,mean_H,std_H,median_H,aboveZero_H = computeWaveletFeatures(waveHalfSecond)
    return max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H


def getFeatures(data,w1,wH):
    # Get DerivStats
    amp,maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f = getStats(data)
    statFeat = np.hstack([amp,maxd,mind,maxabsd,avgabsd,max2d,min2d,maxabs2d,avgabs2d,amp_f,maxd_f,mind_f,maxabsd_f,avgabsd_f,max2d_f,min2d_f,maxabs2d_f,avgabs2d_f])

    # Get Wavelet Features
    max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H = getWavelet(w1,wH)
    waveletFeat = np.hstack([max_1,mean_1,std_1,median_1,aboveZero_1,max_H,mean_H,std_H,median_H,aboveZero_H])

    all_feat = np.hstack([statFeat,waveletFeat])
    
    if np.Inf in all_feat:
        print("Inf")
    
    if np.NaN in all_feat:
        print("NaN")

    return list(all_feat)


def createFeatureDF(data):
    '''
    INPUTS:
        filepath:           string, path to input file  
    OUTPUTS:
        features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
        data:               DataFrame, index is a list of timestamps at 8Hz, columns include eda, filtered_eda
    '''
    # Load data from q sensor
    wave1sec,waveHalf = getWaveletData(data)
    
    # Create 5 second timestamp list
    timestampList = data.index.tolist()[0::40]
    
    # feature names for DataFrame columns
    allFeatureNames = ['raw_amp','raw_maxd','raw_mind','raw_maxabsd','raw_avgabsd','raw_max2d','raw_min2d','raw_maxabs2d','raw_avgabs2d','filt_amp','filt_maxd','filt_mind',
        'filt_maxabsd','filt_avgabsd','filt_max2d','filt_min2d','filt_maxabs2d','filt_avgabs2d','max_1s_1','max_1s_2','max_1s_3','mean_1s_1','mean_1s_2','mean_1s_3',
        'std_1s_1','std_1s_2','std_1s_3','median_1s_1','median_1s_2','median_1s_3','aboveZero_1s_1','aboveZero_1s_2','aboveZero_1s_3','max_Hs_1','max_Hs_2','mean_Hs_1',
        'mean_Hs_2','std_Hs_1','std_Hs_2','median_Hs_1','median_Hs_2','aboveZero_Hs_1','aboveZero_Hs_2']

    # Initialize Feature Data Frame
    features = pd.DataFrame(np.zeros((len(timestampList),len(allFeatureNames))),columns=allFeatureNames,index=timestampList)
    
    # Compute features for each 5 second epoch
    for i in range(len(features)-1):
        start = features.index[i]
        end = features.index[i+1]
        this_data = data[start:end]
        this_w1 = wave1sec[start:end]
        this_w2 = waveHalf[start:end]
        features.iloc[i] = getFeatures(this_data,this_w1,this_w2)
    return features


def classifyEpochs(features,featureNames,classifierName):
    '''
    This function takes the full features DataFrame and classifies each 5 second epoch into artifact, questionable, or clean

    INPUTS:
        features:           DataFrame, index is a list of timestamps for each 5 seconds, contains all the features
        featureNames:       list of Strings, subset of feature names needed for classification
        classifierName:     string, type of SVM (binary or multiclass)

    OUTPUTS:
        labels:             Series, index is a list of timestamps for each 5 seconds, values of -1, 0, or 1 for artifact, questionable, or clean
    '''
    # Only get relevant features
    features = features[featureNames]
    X = features[featureNames].values

    # Classify each 5 second epoch and put into DataFrame
    featuresLabels = predict_binary_classifier(X)
    
    return featuresLabels


def getSVMFeatures(key):
    '''
    This returns the list of relevant features

    INPUT:
        key:                string, either "Binary" or "Multiclass"

    OUTPUT:
        featureList:        list of Strings, subset of feature names needed for classification
    '''
    if key == "Binary":
        return ['raw_amp','raw_maxabsd','raw_max2d','raw_avgabs2d','filt_amp','filt_min2d','filt_maxabs2d','max_1s_1',
                                'mean_1s_1','std_1s_1','std_1s_2','std_1s_3','median_1s_3']
    elif key == "Multiclass":
        return ['filt_maxabs2d','filt_min2d','std_1s_1','raw_max2d','raw_amp','max_1s_1','raw_maxabs2d','raw_avgabs2d',
                                    'filt_max2d','filt_amp']
    else:
        print('Error!! Invalid key, choose "Binary" or "Multiclass"\n\n')
        return


def classify(data):
    '''
    This function wraps other functions in order to load, classify, and return the label for each 5 second epoch of Q sensor data.

    INPUT:
        data
    OUTPUT:
        featureLabels:          Series, index is a list of timestamps for each 5 seconds, values of -1, 0, or 1 for artifact, questionable, or clean
        data:                   DataFrame, only output if fullFeatureOutput=1, index is a list of timestamps at 8Hz, columns include eda, filtered_eda
    '''
    
    # Get correct feature names for classifier
    classifierName = 'Binary'

    # Get pickle List and featureNames list
    featureNames = getSVMFeatures(classifierName)

    # Create the feature array and then apply the classifier    
    features = createFeatureDF(data)
    labels   = classifyEpochs(features, featureNames, classifierName)

    return labels, data

def plotData_notag(data, labels, filepath, subject, tag, filteredPlot=0, secondsPlot=0): #maybe change filteredPlot to 1, plotData_notag(data, labels, dir_out, subject, tag, 1, 0)
    '''
    This function plots the Q sensor EDA data with shading for artifact (red) and questionable data (grey). 
        Note that questionable data will only appear if you choose a multiclass classifier

    INPUT:
        data:                   DataFrame, indexed by timestamps at 8Hz, columns include EDA and filtered_eda
        labels:                 array, each row is a 5 second period and each column is a different classifier
        filteredPlot:           binary, 1 for including filtered EDA in plot, 0 for only raw EDA on the plot, defaults to 0
        secondsPlot:            binary, 1 for x-axis in seconds, 0 for x-axis in minutes, defaults to 0

    OUTPUT:
        [plot]                  the resulting plot has N subplots (where N is the length of classifierList) that have linked x and y axes 
                                    and have shading for artifact (red) and questionable data (grey)

    '''
    
    # Initialize x axis
    if secondsPlot:
        scale = 1.0
    else:
        scale = 60.0
    time_m = np.arange(0,len(data))/(8.0*scale)
    
    # Initialize Figure
    plt.figure(figsize=(10,5))

    # For each classifier, label each epoch and plot
    key = 'Binary'
        
    # Initialize Subplots
    ax = plt.subplot(1,1,1)

    # Plot EDA
    ax.plot(time_m,data['eda'], c='b', label ='Raw SC')

    # For each epoch, shade if necessary
    for i in range(0,len(labels)-1):
        if labels[i]==-1:
            # artifact
            start = i*40/(8.0*scale)
            end = start+5.0/scale
            ax.axvspan(start, end, facecolor='red', alpha=0.7, edgecolor ='none', label = 'artifact') #better to give this a label so that in legend it doesn't keep getting incorrectly labelled as filtered data
        elif labels[i]==0:
            # Questionable
            start = i*40/(8.0*scale)
            end = start+5.0/scale
            ax.axvspan(start, end, facecolor='.5', alpha=0.5,edgecolor ='none', label = 'questionable')

    # Plot filtered data if requested
    if filteredPlot:
        ax.plot(time_m-.625/scale,data['filtered_eda'], c='g', linestyle='--', label = 'Filtered SC') #maybe make this dashed line so that the underlying raw data in blue can still be seen because now the raw data is being completely covered.
        plt.legend(['Raw SC','Filtered SC', 'Artifact'],loc=0)
    else:
        plt.legend(['Raw SC', 'Artifact'],loc=0)

    # Label and Title each subplot
    plt.ylabel('$\mu$S')
    plt.title(key)
    
    # Only include x axis label on final subplot
    if secondsPlot:
        plt.xlabel('Time (s)')    
    else:
        plt.xlabel('Time (min)')

    # Display the plot
    plt.subplots_adjust(hspace=.3)
    plt.show()
    plt.savefig(os.path.join(filepath, subject + '_' + tag + '_artefacts.png'), dpi = 300) #take out the exp after experimenting is done
    
    return


#if __name__ == "__main__":

def EDA_artifact_detection_notag(dict_data, dir_out, subject, tag):
    
    # make sure data has 8Hz
    data = interpolateDataTo8Hz(dict_data['eda'], (1/dict_data['eda']['sampRate'].iloc[0])) #interpolateDataTo8Hz(dict_df['eda'], (1/dict_df['sampRate'].iloc[0]))#data  = interpolateDataTo8Hz(dict_df['eda'], (1/dict_df['eda']['sampRate'].iloc[0]))
    
    # forward propagate data to fill NAs after merging
    data = data.ffill()
    #return data
    
    
    # get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
    data['filtered_eda'] =  butter_lowpass_filter(data['eda'], 1.0, 8, 6) #butter_lowpass_filter(data, 1.0, 8, 6) #butter_lowpass_filter(data['eda'], 1.0, 8, 6)
    #return data - RETURN STATEMENT 5 (TO SEE WHAT THE FILTERED EDA COLUMN LOOKS LIKE) - ARRANGE RETURNS IN OUTER FUNCTIONS ACCORDINGLY    
    
    
    # classify the data
    labels, data = classify(data)

    # plot data
    plotData_notag(data, labels, dir_out, subject, tag, 1, 0)   #change the second last entry back to 1 after experimenting is done - changed back

    # save labels
    fullOutputPath = os.path.join(dir_out, subject + '_' + tag + '_artefacts.csv')

    #featureLabels = pd.DataFrame(labels, index=pd.timedelta_range(start=data.index[0], periods=len(labels), freq='5s'),
                                 #columns=['Binary'])
    featureLabels = pd.DataFrame(labels, index=pd.date_range(start=data.index[0], periods=len(labels), freq='5s'),
                                 columns=['Binary'])

    featureLabels.reset_index(inplace=True)
    featureLabels.rename(columns={'index':'StartTime'}, inplace=True)
    #featureLabels['EndTime'] = featureLabels['StartTime']+datetime.timedelta(seconds=5)
    featureLabels['EndTime'] = featureLabels['StartTime']+timedelta(seconds=5)
    featureLabels.index.name = 'EpochNum'

    featureLabels.to_csv(fullOutputPath)
    
    return featureLabels


"""

This is a preprocessing pipeline for psychophysiological data collected with 
Empatica wristbands, either E4 or E+. It uses EDA Explorer to detect 
artefacts in the data based on EDA, temperature and acceleration data using
a binary classifier (noise versus okay). For the preprocessing, it uses 
functions from the NeuroKit2 package. Specifically for the EDA preprocessing, 
it also draws inspiration from Ledalab. 

Arguments: 
    empatica   : either 'e4' or 'e+' or 'cut' (4Hz EDA and 64Hz BVP only)
    winwidth   : width of the window for smoothing of EDA with Gaussian kernel (int)
    lowpass    : lowpass filter frequency for EDA - has to be no larger than half the sample rate
    dir_out    : output directory for all the results
    dir_path   : input directory
    exclude    : list of patterns to be excluded from preprocessing

The function creates preprocessed data files for EDA and BVP as well as plots 
to check the data quality. 

(c) Irene Sophia Plank, 10planki@gmail.com

"""

# load all modules
import neurokit2 as nk
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import simple_colors
import math
import glob
import os
import warnings

#warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')

from avro.datafile import DataFileReader
from avro.io import DatumReader
from datetime import datetime

#added by me:
from datetime import timedelta
import pytz
from pathlib import Path
#from EDA_artifactdetection_short import EDA_artifact_detection

###### Helper functions

def gauss_smoothing(data, winwidth):
    # Gaussian smoothing of EDA data
    
    # pad to remove border errors
    fdata = pd.concat([pd.Series(data.iloc[0]), data, pd.Series(data.iloc[-1])])
    
    # ensure an even window width
    winwidth = math.floor(winwidth/2)*2
    
    # extend data to reduce convolution error at beginning and end
    data_ext = pd.concat([pd.Series([fdata.iloc[0]]*int(winwidth/2)), fdata, pd.Series([fdata.iloc[-1]]*int(winwidth/2))])
    
    # apply normpdf (?)
    x = np.array(range(1, winwidth+2))
    mu = winwidth/2+1
    sigma = winwidth/8
    window = np.exp(-0.5 * ((x - mu)/sigma)**2) / (math.sqrt(2*math.pi) * sigma)
    window = window / sum(window)
    
    # perform convolution for smoothing
    sdata_ext = np.convolve(data_ext, window)
    
    # cut to length of data
    return sdata_ext[(1+winwidth):(len(sdata_ext)-winwidth-1)]

def int_missing(df_eda, df_bvp, df_temp, df_acc, f):
    # performing linear interpolation for bvp, temp and acc as well as cubic
    # spline interpolation for eda
    
    # only process temp if it is not empty
    if len(df_temp) > 0:
        sampRate = df_temp['sampRate'].iloc[0]
        df_temp = df_temp.interpolate()
        df_temp = df_temp.bfill()
        df_temp['sampRate'] = sampRate
    
    # only process acc if it is not empty
    if len(df_acc) > 0:
        sampRate = df_acc['sampRate'].iloc[0]
        df_acc = df_acc.interpolate()
        df_acc = df_acc.bfill()
        df_acc['sampRate'] = sampRate
    
    # EDA
    #sampRate = df_eda['sampRate'].iloc[0]  - commented out in this line and added below because the first sampRate value was turning up as NaN
    raw    = df_eda['eda']
    df_eda = df_eda.interpolate(method='spline', order=3)
    df_eda = df_eda.bfill()
    df_eda['raw'] = raw
    sampRate = df_eda['sampRate'].iloc[0]
    df_eda['sampRate'] = sampRate
    
    # BVP
    sampRate = df_bvp['sampRate'].iloc[0]
    raw    = df_bvp['bvp']
    df_bvp = df_bvp.interpolate()
    df_bvp = df_bvp.bfill()
    df_bvp['raw'] = raw
    df_bvp['sampRate'] = sampRate
    
    # print how much was interpolated for bvp and eda
    per_eda = np.mean(raw.isna())
    per_bvp = np.mean(raw.isna())
    if round(per_eda, 2) == round(per_bvp, 2):
        if per_eda >= 0.2:
            print(simple_colors.red(datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_eda*100,2)) + ' percent of data were interpolated', 'bold'))
        elif per_eda >= 0.01:
            print(simple_colors.yellow(datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_eda*100,2)) + ' percent of data were interpolated', 'bold'))
    else:
        if per_eda >= 0.2:
            print(simple_colors.red(datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_eda*100,2)) + ' percent EDA were interpolated', 'bold'))
        elif per_eda >= 0.01:
            print(simple_colors.yellow(datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_eda*100,2)) + ' percent EDA were interpolated', 'bold'))
        if per_bvp >= 0.2:
            print(simple_colors.red(datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_bvp*100,2)) + ' percent BVP were interpolated', 'bold'))
        elif per_bvp >= 0.01:
            print(simple_colors.yellow(datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_bvp*100,2)) + ' percent BVP were interpolated', 'bold'))
    
    # write to log file
    f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - ' + str(round(per_bvp*100,2)) + '% BVP and ' + str(round(per_eda*100,2)) + '% EDA were interpolated')
    
    # return the dataframes
    return df_eda, df_bvp, df_temp, df_acc

def na_missing(df_eda, df_bvp, labels):
    # taking the labels from the artefact detection classifier and create column
    # with NaNs where artefacts were detected
    
    # join label dataframes with the dataframes containing EDA and BVP
    labels.index = labels['StartTime']
    df_eda = df_eda.join(labels, how='outer')
    df_bvp = df_bvp.join(labels, how='outer')
    
    # fill up the NaNs with the preceeding label
    df_eda['Binary'] = df_eda['Binary'].fillna(method='ffill')
    df_bvp['Binary'] = df_bvp['Binary'].fillna(method='ffill')
    
    # calculate the new column and drop the unnecessary ones
    df_eda['eda'] = np.where(df_eda['Binary'] == 1, df_eda['eda'], np.nan)
    df_bvp['bvp'] = np.where(df_bvp['Binary'] == 1, df_bvp['bvp'], np.nan)
    df_eda = df_eda.drop(columns=['StartTime', 'EndTime'])
    df_bvp = df_bvp.drop(columns=['StartTime', 'EndTime'])
    
    return df_eda, df_bvp

###### Cut and convert the data

def read_avro(filepath, timezone = 'Europe/Berlin'):
    # reading in avro data and converting it into data frames with the correct time stamp
    
    # read in the avro data
    reader = DataFileReader(open(filepath, "rb"), DatumReader())
    for user in reader:
        dict_data = user
    reader.close()

    target_timezone = pytz.timezone(timezone) #change to required timezone if needed - done
    # temperature: 
    startTime = datetime.utcfromtimestamp((float(dict_data['rawData']['temperature']['timestampStart'])/(10**(len(str(dict_data['rawData']['temperature']['timestampStart']))-10)))).replace(tzinfo=pytz.utc).astimezone(target_timezone) #1000000
    #cet = pytz.timezone('Europe/Berlin') #me
    #startTime = cet.localize(startTime) #me
    sampRate  = dict_data['rawData']['temperature']['samplingFrequency']
    df_temp   = pd.DataFrame(dict_data['rawData']['temperature']['values'], columns=["temp"])
    if sampRate > 0.0:
        freq      = str(round(1/sampRate)) + 'S'
        time      = pd.date_range(startTime, periods=len(df_temp), freq=freq)
        df_temp.index = time
        df_temp['sampRate'] = round(1/sampRate)
    
    # acceleration
    startTime = datetime.utcfromtimestamp((float(dict_data['rawData']['accelerometer']['timestampStart'])/(10**(len(str(dict_data['rawData']['accelerometer']['timestampStart']))-10)))).replace(tzinfo=pytz.utc).astimezone(target_timezone)
    #cet = pytz.timezone('Europe/Berlin') #me
    #startTime = cet.localize(startTime) #me
    sampRate  = dict_data['rawData']['accelerometer']['samplingFrequency']
    df_acc    = pd.DataFrame({'accx_raw': dict_data['rawData']['accelerometer']['x'],
                              'accy_raw': dict_data['rawData']['accelerometer']['y'],
                              'accz_raw': dict_data['rawData']['accelerometer']['z']})
    if sampRate > 0.0:
        freq      = str(round(1/sampRate, 6)) + 'S'
        time      = pd.date_range(startTime, periods=len(df_acc), freq=freq)
        df_acc.index = time
        df_acc['sampRate'] = round(1/sampRate, 6)
    
    # bvp
    startTime = datetime.utcfromtimestamp((float(dict_data['rawData']['bvp']['timestampStart'])/(10**(len(str(dict_data['rawData']['bvp']['timestampStart']))-10)))).replace(tzinfo=pytz.utc).astimezone(target_timezone)
    #cet = pytz.timezone('Europe/Berlin') #me
    #startTime = cet.localize(startTime) #me
    sampRate  = dict_data['rawData']['bvp']['samplingFrequency']
    df_bvp    = pd.DataFrame({'bvp': dict_data['rawData']['bvp']['values']})
    if sampRate > 0.0:
        freq      = str(round(1/sampRate, 6)) + 'S'
        time      = pd.date_range(startTime, periods=len(df_bvp), freq=freq)
        df_bvp.index = time
        df_bvp['sampRate'] = round(1/sampRate, 6)
        
    # eda
    startTime = datetime.utcfromtimestamp((float(dict_data['rawData']['eda']['timestampStart'])/(10**(len(str(dict_data['rawData']['eda']['timestampStart']))-10)))).replace(tzinfo=pytz.utc).astimezone(target_timezone)
    #print("start time using old utc_cet conversion method ", startTime)
    #cet = pytz.timezone('Europe/Berlin') #me
    #startTime = cet.localize(startTime) #me this stuff doesn't work so commented out
    #print("start time after attempted another attempted cet conversion ", startTime)
    sampRate  = dict_data['rawData']['eda']['samplingFrequency']
    df_eda    = pd.DataFrame({'eda': dict_data['rawData']['eda']['values']})
    if sampRate > 0.0:
        freq      = str(round(1/sampRate, 2)) + 'S'
        time      = pd.date_range(startTime, periods=len(df_eda), freq=freq)
        df_eda.index = time
        df_eda['sampRate'] = round(1/sampRate, 2)
    
    # return all data frames
    return df_temp, df_acc, df_bvp, df_eda


def convert_eplus_notag(dir_path, f, timezone):
    # reading in and converting data collected with Embrace Plus
    
    # get list of all files of this participant
    #fls = glob.glob(os.path.join(dir_path, part + '*.avro')) #glob.glob(os.path.join(dir_path, 'participant_data',
                                 #'*', '*', 'raw_data', 'v*', part + '*.avro'))
    fls = glob.glob(os.path.join(dir_path, '*.avro')) #glob.glob(os.path.join(dir_path, 'participant_data',
                                 #'*', '*', 'raw_data', 'v*', part + '*.avro'))
    
    # check if any data was found
    if len(fls) < 1:
        print(simple_colors.red(datetime.now().strftime("%H:%M:%S") + '- no data was found for participant ' + part, 'bold'))
        f.write('\n' + datetime.now().strftime("%H:%M:%S") + '- no data was found for participant ' + part)
        return {}
    
    # sort by start time in unix
    fls = sorted(fls, key=lambda i: int(os.path.splitext(os.path.basename(i))[0][-9:]))
    
    # initialise empty data frames
    ls_temp = list()
    ls_acc  = list()
    ls_eda  = list()
    ls_bvp  = list()
    
    # loop through files
    for fl in fls:
        # read in the avro data
        df_temp, df_acc, df_bvp, df_eda = read_avro(fl, timezone)
        # check timing difference: 
        if len(ls_temp) > 0 & len(df_temp) > 0: 
            # temperature: 1Hz -> should be about 1 per second
            idx_temp = pd.date_range(ls_temp[-1].index[-1]+pd.Timedelta(seconds=df_temp['sampRate'].iloc[0]), 
                                     df_temp.index[0]-pd.Timedelta(seconds=df_temp['sampRate'].iloc[0]), freq=str(df_temp['sampRate'].iloc[0])+'S')
            """
            start_time_temp = ls_temp[-1].index[-1]+pd.Timedelta(seconds=df_temp['sampRate'].iloc[0])
            end_time_temp = df_temp.index[0]-pd.Timedelta(seconds=df_temp['sampRate'].iloc[0])
            #making them time-zone aware
            start_time_temp = start_time_temp.tz_localize('Europe/Berlin') if start_time_temp.tzinfo is None else start_time_temp
            end_time_temp = end_time_temp.tz_localize('Europe/Berlin') if end_time_temp.tzinfo is None else end_time_temp
            
            idx_temp = pd.date_range(start_time_temp, 
                                     end_time_temp, freq=str(df_temp['sampRate'].iloc[0])+'S', tz = 'Europe/Berlin') #me
            """
            tdf_temp = pd.DataFrame(np.nan, index=idx_temp, columns=['temp'])
            df_temp  = pd.concat([tdf_temp, df_temp])
        
        if len(ls_acc) > 0 & len(df_acc) > 0: 
            # acceleration: 64Hz -> should be about 1 every 15.625ms
            idx_acc  = pd.date_range(ls_acc[-1].index[-1]+pd.Timedelta(seconds=df_acc['sampRate'].iloc[0]), 
                                     df_acc.index[0]-pd.Timedelta(seconds=df_acc['sampRate'].iloc[0]), freq=str(df_acc['sampRate'].iloc[0])+'S')
            """
            start_time_acc = ls_acc[-1].index[-1]+pd.Timedelta(seconds=df_acc['sampRate'].iloc[0])
            end_time_acc = df_acc.index[0]-pd.Timedelta(seconds=df_acc['sampRate'].iloc[0])
            #making them time-zone aware
            start_time_acc = start_time_acc.tz_localize('Europe/Berlin') if start_time_acc.tzinfo is None else start_time_acc
            end_time_acc = end_time_acc.tz_localize('Europe/Berlin') if end_time_acc.tzinfo is None else end_time_acc
            
            idx_acc  = pd.date_range(start_time_acc, 
                                     end_time_acc, freq=str(df_acc['sampRate'].iloc[0])+'S', tz = 'Europe/Berlin') #me
            """
            tdf_acc  = pd.DataFrame(np.nan, index=idx_acc, columns=['accx', 'accy', 'accz'])
            df_acc   = pd.concat([tdf_acc, df_acc])
        
        if len(ls_eda) > 0 & len(df_eda) > 0:     
            # eda: 4Hz -> should be about 1 every 250ms
            idx_eda  = pd.date_range(ls_eda[-1].index[-1]+pd.Timedelta(seconds=df_eda['sampRate'].iloc[0]), 
                                     df_eda.index[0]-pd.Timedelta(seconds=df_eda['sampRate'].iloc[0]), freq=str(df_eda['sampRate'].iloc[0])+'S')
            
            """
            start_time_eda = ls_eda[-1].index[-1]+pd.Timedelta(seconds=df_eda['sampRate'].iloc[0])
            end_time_eda = df_eda.index[0]-pd.Timedelta(seconds=df_eda['sampRate'].iloc[0])
            #making them time-zone aware
            start_time_eda = start_time_eda.tz_localize('Europe/Berlin') if start_time_eda.tzinfo is None else start_time_eda
            end_time_eda = end_time_eda.tz_localize('Europe/Berlin') if end_time_eda.tzinfo is None else end_time_eda
            
            idx_eda  = pd.date_range(start_time_eda, 
                                     end_time_eda, freq=str(df_eda['sampRate'].iloc[0])+'S', tz = 'Europe/Berlin') #me
            """
            tdf_eda  = pd.DataFrame(np.nan, index=idx_eda, columns=['eda'])
            df_eda   = pd.concat([tdf_eda, df_eda])
           
        
        if len(ls_bvp) > 0 & len(df_bvp) > 0:     
            # bvp: 1Hz -> 64Hz -> should be about 1 every 15.625ms
            idx_bvp = pd.date_range(ls_bvp[-1].index[-1]+pd.Timedelta(seconds=df_bvp['sampRate'].iloc[0]), 
                                     df_bvp.index[0]-pd.Timedelta(seconds=df_bvp['sampRate'].iloc[0]), freq=str(df_bvp['sampRate'].iloc[0])+'S')
            """
            start_time_bvp = ls_bvp[-1].index[-1]+pd.Timedelta(seconds=df_bvp['sampRate'].iloc[0])
            end_time_bvp = df_bvp.index[0]-pd.Timedelta(seconds=df_bvp['sampRate'].iloc[0])
            #making them time-zone aware
            start_time_bvp = start_time_bvp.tz_localize('Europe/Berlin') if start_time_bvp.tzinfo is None else start_time_bvp
            end_time_bvp = end_time_bvp.tz_localize('Europe/Berlin') if end_time_bvp.tzinfo is None else end_time_bvp
            
            idx_bvp = pd.date_range(start_time_bvp, 
                                     end_time_bvp, freq=str(df_bvp['sampRate'].iloc[0])+'S', tz = 'Europe/Berlin') #me
            """
            tdf_bvp  = pd.DataFrame(np.nan, index=idx_bvp, columns=['bvp'])
            df_bvp   = pd.concat([tdf_bvp, df_bvp])
            
        # append to lists
        if len(df_bvp) > 0:
            ls_bvp.append(df_bvp)
        if len(df_temp) > 0:
            ls_temp.append(df_temp)
        if len(df_acc) > 0:
            # scale the accelometer to +-2g: "each ADC count will be = 1/2048g" (email from 12.12.2023)
            df_acc["accx"] = df_acc["accx_raw"]/2048
            df_acc["accy"] = df_acc["accy_raw"]/2048
            df_acc["accz"] = df_acc["accz_raw"]/2048
            
            ls_acc.append(df_acc)
        if len(df_eda) > 0:
            ls_eda.append(df_eda)
             #return tdf_eda, df_eda, ls_eda
    #"""
    # concat list of dataframes to dataframes in dictionary
    return {
        'temp' : pd.concat(ls_temp),
        'acc'  : pd.concat(ls_acc),
        'bvp'  : pd.concat(ls_bvp),
        'eda'  : pd.concat(ls_eda)
    }
    #"""


###### EDA

def eda_prepro_notag(dir_out, df_eda, subject, key, winwidth, lowpass, f, timezone): #(dir_out, df_eda, subject, key, winwidth, [], f) 
    # preprocessing EDA data

    # get sampling rate in Hz
    sr = 1/df_eda['sampRate'].iloc[0]
    print("sr = ", sr)
    # if a filter was supposed to be applied
    if isinstance(lowpass, int):
        # check if digital filter critical frequency 0 < lowpass < fs/2
        if (lowpass > 0) & (lowpass < sr/2):
            fil = scipy.signal.butter(1, 5, fs = sr)
            data = scipy.signal.sosfilt(fil, df_eda['eda'])
        else:
            data = df_eda['eda']
            print(simple_colors.red('No filtering applied.', 'bold'), 'Critical frequency must be at least half the sampling rate for smoothing to be performed.')
            f.write('\n' + 'No filtering applied. Critical frequency must be at least half the sampling rate for smoothing to be performed.')
    else:
        print("no filtering applied") #added by me
        data = df_eda['eda']
    
    # smooth the data - same as GAUSS option in ledalab
    df_eda['eda_smooth'] = gauss_smoothing(data, winwidth)

    #return df_eda
    
    # process the data, including TTG to detect peaks
    signals, info = nk.eda_process(df_eda['eda_smooth'], sampling_rate = sr)
    #return df_eda, signals, info

    
    # combine the data frames
    signals.index = df_eda.index
    df_eda = signals.join(df_eda)
    
    
    df_eda = df_eda.drop(['EDA_Raw', 'eda_smooth'], axis=1).rename(columns={"eda": "EDA_Raw"}, errors="raise")
    signals = df_eda.reset_index()
    
    # visualise the signals
    matplotlib.rcParams['figure.figsize'] = (100, 10)
    nk.eda_plot(signals, info)
    plt.savefig(os.path.join(dir_out, subject + '_' + key + '_eda_signals.png'), dpi = 300)
    
    # close all figures
    plt.close("all") 

    #return df_eda, signals, info
    
    #filepath = the empatica folder for that day
    
    filepath = Path(dir_path).parent
    print(filepath)
    df_eda_filtered = additional_filters(df_eda, filepath, timezone) 
    # save data to csv
    df_eda.rename(columns={"raw": "EDA_Raw_noint"}).to_csv(os.path.join(dir_out, subject + '_' + key + '_eda_signals.csv'), index = True)
    info.pop('sampling_rate')
    info_df = pd.DataFrame.from_dict(info)
    info_df.to_csv(os.path.join(dir_out, subject + '_' + key + '_eda_scr.csv'), index = True)
    df_eda_filtered.rename(columns={"raw": "EDA_Raw_noint"}).to_csv(os.path.join(dir_out, subject + '_' + key + '_eda_signals_filtered.csv'), index = True)
    #call back the previously saved fig or re-visualise the eda figure and whereever the data has been turned NA, put red lines just like the artifact plot function
    
    return df_eda, df_eda_filtered, signals, info
    #return
    
"""
def additional_filters(df_eda, filepath):
    #read in temp.csv as a dataframe from preprocessed files folder
    #read agg eda.csv as a dataframe from aggr_p_min folder
    #add a new column to df_eda called "temp_data_validity" -> +1 if within required temperature range and -1 if not; match by timestamp
    #add a new column to df_eda called "device_recording_validity" -> -1 if "device_not_recording" or "device_not_worn_correctly" and +1 if not; for every
    #now for all the rows of df_eda, if the "temp_data_validity" or "device_recording_validity" columns are -1, all eda columns are made NA
    return df_eda
"""



def additional_filters(df_eda, filepath, timezone):
    folder11 = 'aggr_p_min'
    folder12 = 'avro_files'
    #folder13 = 'avro2csv'
    folder14 = 'preprocessed_files_debug'
    folder141 = 'data_preproc_debug'
    #read in temp.csv as a dataframe from aggr_p_min folder (not preprocessed files folder because it is data for every 1 second and the timestamps do not match. Therefore there is no coding-based advantage to using it. Moreover, a second by second data elimination is not needed. The per minute aggregated data is a better and stricter way of filtering out untrustworthy data because the entire minute's data, if it is not valid, can be eliminated.
    for file in os.listdir(os.path.join(filepath, folder11)):
        if file.endswith('temperature.csv'):
            temp_file = pd.read_csv(os.path.join(filepath, folder11, file))
    # Define the time zones
    utc_zone = pytz.utc
    req_zone = pytz.timezone(timezone)

    # Function to convert UTC to Europe/Berlin time
    def from_isoutc_to_req(iso_timestamp):
        # Parse the ISO 8601 timestamp into a datetime object
        utc_time = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    
        # Localize the datetime object to UTC
        #utc_time = utc_zone.localize(utc_time) check this line and see if it is still required
    
        # Convert to Berlin time
        req_time = utc_time.astimezone(req_zone)
        #print(req_time, type(req_time))
    
        return req_time #.isoformat()

    # Apply the conversion function to the 'utc_timestamps' column and create a new column 'converted_timestamps'
    temp_file['converted_timestamps'] = temp_file['timestamp_iso'].apply(from_isoutc_to_req)
    
    #read agg eda.csv as a dataframe from aggr_p_min folder
    for file in os.listdir(os.path.join(filepath, folder11)):
        if file.endswith('eda.csv'):
            eda_aggr_file = pd.read_csv(os.path.join(filepath, folder11, file))

    # Apply the conversion function to the 'utc_timestamps' column and create a new column 'converted_timestamps'
    eda_aggr_file['converted_timestamps'] = eda_aggr_file['timestamp_iso'].apply(from_isoutc_to_req)

    
    #add a new column to df_eda called "temp_data_validity" -> +1 if within required temperature range and -1 if not; match by timestamp
    #tenative min and max values => modify to verified values
    temp_min= 30
    temp_max = 40
    # Create a new column in df_eda initialized with NaN
    df_eda['temp_data_validity'] = np.nan

    # Create a boolean mask for valid temperatures in temp_file
    valid_temp_mask = (temp_file['temperature_celsius'] > temp_min) & (temp_file['temperature_celsius'] < temp_max)

    # Convert the mask to 1 and -1
    temp_validity = np.where(valid_temp_mask, 1, -1)

    # Use pd.cut to bin df_eda.index based on temp_file['converted_timestamps']
    bins = temp_file['converted_timestamps']
    df_eda['temp_interval'] = pd.cut(df_eda.index, bins=bins, labels=False, include_lowest=True, right=False)

    # Map the interval indices to their corresponding validity values
    interval_to_validity = dict(enumerate(temp_validity[:-1]))
    df_eda['temp_data_validity'] = df_eda['temp_interval'].map(interval_to_validity)

    # Handle values before the first timestamp and after the last timestamp
    df_eda.loc[df_eda.index < bins.min(), 'temp_data_validity'] = temp_validity[0]
    df_eda.loc[df_eda.index >= bins.max(), 'temp_data_validity'] = temp_validity[-1]

    # Drop the temporary 'temp_interval' column
    df_eda.drop('temp_interval', axis=1, inplace=True)

    #add a new column to df_eda called "device_recording_validity" -> -1 if "device_not_recording" or "device_not_worn_correctly" and +1 if not; for every
    # Create a new column in df_eda initialized with NaN
    df_eda['device_recording_validity'] = np.nan

    # Create a boolean mask for valid recording entries in eda_aggr_file
    valid_eda_rec_mask = valid_eda_rec_mask = ~((eda_aggr_file['missing_value_reason'] == 'device_not_recording') | 
                           (eda_aggr_file['missing_value_reason'] == 'device_not_worn_correctly'))#eda_aggr_file['eda_scl_usiemens'] != np.nan #(eda_aggr_file['missing_value_reason'] != 'device_not_recording') & (eda_aggr_file['missing_value_reason'] != 'device_not_worn_correctly')

    # Convert the mask to 1 and -1
    dev_rec_validity = np.where(valid_eda_rec_mask, 1, -1)

    # Use pd.cut to bin df_eda.index based on eda_aggr_file['converted_timestamps']
    bins = eda_aggr_file['converted_timestamps']
    df_eda['eda_interval'] = pd.cut(df_eda.index, bins=bins, labels=False, include_lowest=True, right=False)

    # Map the interval indices to their corresponding validity values
    interval_to_validity = dict(enumerate(dev_rec_validity[:-1]))
    df_eda['device_recording_validity'] = df_eda['eda_interval'].map(interval_to_validity)

    # Handle values before the first timestamp and after the last timestamp
    df_eda.loc[df_eda.index < bins.min(), 'device_recording_validity'] = dev_rec_validity[0]
    df_eda.loc[df_eda.index >= bins.max(), 'device_recording_validity'] = dev_rec_validity[-1]

    # Drop the temporary 'temp_interval' column
    df_eda.drop('eda_interval', axis=1, inplace=True)

    """    
    # Print some diagnostics
    print(f"Number of -1 entries: {(df_eda['device_recording_validity'] == -1).sum()}")
    print(f"Number of 1 entries: {(df_eda['device_recording_validity'] == 1).sum()}")
    print(f"Total entries: {len(df_eda)}")

    # Check the distribution of missing_value_reason in eda_aggr_file
    print(eda_aggr_file['missing_value_reason'].value_counts())
    """
    #now for all the rows of df_eda, if the "temp_data_validity" or "device_recording_validity" columns are -1, all eda columns are made NA -> maybe not
    return df_eda
    
###### BVP and HR


def bvp_prepro_notag(dir_out, df_bvp, subject, key):
    # preprocessing BVP and HR data

    # get sampling rate in Hz
    sr  = 1/df_bvp['sampRate'].iloc[0]
    
    # process the data following Elgendi et al. (2013)
    # settings: 
    #   peakwindow=0.111,
    #   beatwindow=0.667,
    #   beatoffset=0.02,
    #   mindelay=0.3
    #   minimum peak height of 0
    signals, info = nk.ppg_process(df_bvp['bvp'], sampling_rate = sr)
    
    # plot the data
    matplotlib.rcParams['figure.figsize'] = (20, 10)
    nk.ppg_plot(signals, info)
    plt.savefig(os.path.join(dir_out, subject + '_' + key + '_bvp_signals.png'), dpi = 300)
    signals_split = np.array_split(signals, 10)
    count = 0
    for s in signals_split:
        count = count + 1
        # graphs can only be created when there are more than three peaks
        if sum(s['PPG_Peaks'] == 1) > 3:
            nk.ppg_plot(s, info)
            plt.savefig(os.path.join(dir_out, subject + '_' + key + '_bvp_signals_' + str(count) + '.png'), dpi = 300)
    
    # calculate HRV
    hrv_indices = nk.hrv(signals, sampling_rate = info['sampling_rate'], show = True)
    plt.savefig(os.path.join(dir_out, subject + '_' + key + '_bvp_hrv.png'), dpi = 300)
    
    # close all figures
    plt.close("all") 
    
    # save signals and hrv_indices to csv
    signals.index = df_bvp.index
    df_bvp = signals.join(df_bvp)
    df_bvp.drop(['bvp'], axis=1).rename(columns={"raw": "PPG_Raw_noint"}).to_csv(os.path.join(dir_out, subject + '_' + key + '_bvp_signals.csv'), index = True)
    hrv_indices.to_csv(os.path.join(dir_out, subject + '_' + key + '_bvp_hrv.csv'), index = True)
    
    return 


def preproPSYPHY_notag(mainfolder, dir_path, dir_out, subject_list, empatica, date, timezone, exclude = [], winwidth = 8, lowpass = 5, max_art = 50, art_cor = True): #change max_art to 100/3 - the original value

    # start writing a log file
    log_file_new = os.path.join(mainfolder, 'subject_log.txt')
    f = open(log_file_new, "a")

    # remove excluded participants
    
    for e in exclude:
        for part in subject_list:
            if e in part: 
                subject_list.remove(part)
    
    
    # loop through the participants
    for subject in subject_list:
        print(subject)
       
        # print a message
        print(simple_colors.blue(datetime.now().strftime("%H:%M:%S") + ' - processing participant ' + subject + ' for date ' + date, 'bold'))
        f.write('\n\n' + datetime.now().strftime("%H:%M:%S") + ' - processing participant ' + subject + ' for date ' + date)

        # read in and convert the data
        if empatica == 'e+':

            # convert eplus data
            dict_data = convert_eplus_notag(dir_path, f, timezone) #RETURN STATMENT 2 - COMMENT OUT THE REST OF THE FUNCTION AND RETURN 
            #tdf_eda, df_eda, ls_eda = convert_eplus_notag(dir_path, f, timezone) #RETURN STATEMENT 1 - comment out the rest of the this function
        
        
        # if no data was found for this participant, continue with the next one
        if len(dict_data) < 1:
            continue

        print(simple_colors.green(datetime.now().strftime("%H:%M:%S") + ' - conversion done for subject ' + subject + ' for date ' + date, 'bold'))
        f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - conversion done')

        #return dict_data
        
        
        # loop through the blocks and preprocess the data
        for key, dict_df in dict_data.items():
            print(key)
            
            # detect artifacts using the EDA Explorer classifier
            if key == 'eda':
                    #labels  = EDA_artifact_detection(dict_df, dir_out, part, key)
                    #data = EDA_artifact_detection(dict_df, dir_out, part, key)
                    #data = EDA_artifact_detection(dict_df, dict_data, dir_out, part, key)
                    labels = EDA_artifact_detection_notag(dict_data, dir_out, subject, key)
                    #print(labels)
            elif key == 'temp' and len(dict_df) > 0:# simply save temp and acc, if they exist
                try:
                    dict_df['temp'].to_csv(os.path.join(dir_out, subject + '_' + key + '_temp.csv'))
                    continue
                except Exception as e:
                    print(f"An error occurred while saving the 'temp' data: {e}")
                    f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - error saving temp data '+ e)
                    continue
            elif key == 'acc' and len(dict_df) > 0:
                try:
                    dict_data['acc'].to_csv(os.path.join(dir_out, subject + '_' + key + '_acc.csv'))
                    continue
                except Exception as e:
                    print(f"An error occurred while saving the 'acc' data: {e}")
                    f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - error saving acc data '+ e)
                    continue
            else:
                    continue
            #return data
            
            per_art = sum(labels['Binary'] == -1)*100/len(labels)
            print("per_art is: ", per_art)
            
            # add the percent to the tags object            
            #tags['artefact%'].loc[(tags['part'] == part) & (tags['tag'] == key)] = per_art
            
            # only preprocess if less than 20% artefacts
            if per_art < max_art:

                print(simple_colors.green(datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': artifact detection done , for date ' + date, 'bold'))
                f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': artifact detection done')

                # replacing artefacts with NaNs and then interpolating them
                if art_cor:
                    #return dict_data['eda'], labels
                   
                    #df_eda, df_bvp = na_missing(dict_df['eda'], dict_df['bvp'], labels)
                    df_eda, df_bvp = na_missing(dict_data['eda'], dict_data['bvp'], labels) #me
                    #return df_eda, labels
                    
                    df_eda, df_bvp, [], [] = int_missing(df_eda, df_bvp, [], [], f)
                    #return df_eda
                    
                    print(simple_colors.green(datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': artifact correction done, for date ' + date, 'bold'))
                    f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': artifact correction done')
                else:
                    df_eda = dict_df['eda']
                    df_bvp = dict_df['bvp']

                # preprocess EDA and BVP data with neurokit
                df_eda, df_eda_filtered, signals, info = eda_prepro_notag(dir_out, df_eda, subject, key, winwidth, [], f, timezone) 
                #return df_eda
                
                #bvp_prepro_notag(dir_out, df_bvp, subject, key)

                
                
                print(simple_colors.green(datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': preprocessing done' + ' for date ' + date, 'bold'))
                f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': preprocessing done')

                

            else: 

                print(simple_colors.red(datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': STOPPED due to ' + str(round(per_art,2)) + '% artefacts', 'bold'))
                f.write('\n' + datetime.now().strftime("%H:%M:%S") + ' - block ' + key + ': STOPPED due to ' + str(round(per_art,2)) + '% artefacts')
                f.close()
                return None, None, None, None
                
    #tags.to_csv(tag_file[:-4] + '_prepro.csv')
    f.close()
    return df_eda, df_eda_filtered, signals, info
    


# In[17]:


"""
convert to python script to use in other scripts as needed
"""
#get_ipython().system('jupyter nbconvert --to script l2script_functions.ipynb')


# In[ ]:




