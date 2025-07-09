#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:40:43 2024

@author: justo
"""

# %% Libraries 
import ast
from os import listdir
import sys
import math
import time
import joblib
import pyedflib
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from connectivity_methods import connectivity_analysis
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from timescoring.annotations import Annotation
from timescoring import scoring, visualization

#%% Variables 

# Epoch definition
epoch_length = 60      # Epoch duration in seconds
epoch_overlap = 30   # Epoch overlap in seconds
extension_length = 10  # Duration of the signal in seconds before and after the epoch that is selected to avoid artifacts when filtering (padding, lenght basically)

# Buffer 
buffer_length = 5*60      # Buffer duration in seconds
buffer_threshold = 0.5 # Ratio of the buffer that has to be preictal in order to raise a warning

# Paths
rawEDFs_directory = '/Volumes/MyBook/justo/RAW_EDFs/'
rawEDF_files = listdir(rawEDFs_directory) # List of all the EDF files in the directory

mainDirectory = '/Volumes/MyBook/justo/oneMINUTE/'
metadata_directory = mainDirectory + 'metadata/'

results_directory = mainDirectory + 'results_online/'

# Load list of best classifiers
subject_results_file = metadata_directory + 'best_classifiers/subject_optimization_results.csv'
bestClassifiers = pd.read_csv(subject_results_file, index_col=0).to_dict()['best_combination']

# %% RAW EDF selection

slurm_array_id = 25

rawEDF_fileName = rawEDF_files[slurm_array_id]
subject = rawEDF_fileName.split("-")[0] # Patient code is the first part of the file name
date = rawEDF_fileName.split("-")[1] # Date is the second part of the file name

bestClassifier = bestClassifiers[subject] # Get the best classifier for that patient
print("Best classifier for subject {}: {}".format(subject, bestClassifier))
re_reference_method = bestClassifier.split('-')[0] # The re-referencing method is the first part of the classifier name
band = bestClassifier.split('-')[1] # The band of the classifier is the second part of the classifier name
connectivity_method = bestClassifier.split('-')[2] # The connectivity method is the third part of the classifier name
# Adapt the names of the connectivity methods to the ones used in the train
if connectivity_method == 'PLockV':
    connectivity_method = 'phase_lock'
elif connectivity_method == 'PLagI':
    connectivity_method = 'phase_lag'
elif connectivity_method == 'CrossCorrelation':
    connectivity_method = 'cross_correlation'
feature_selection = bestClassifier.split('-')[3] # The feature selection method is the fourth part of the classifier name
if feature_selection == 'EigVals':
    feature_selection = 'Eig'
classifier_type = bestClassifier.split('-')[4] # The classifier type is the fifth part of the classifier name
network_type = bestClassifier.split('-')[5] # The network type is the sixth part of the classifier name

metadata_file = metadata_directory + 'unified/' + subject + '.txt'
with open(metadata_file, 'r') as f:
    metadataDict = eval(f.read())
    # if there is more than one date 
    if np.size(metadataDict['files']) > 1:
        electrodes = metadataDict['electrodes'][0] # The electrodes are the first element of the metadata dictionary regarless of the date
    else:
        electrodes = metadataDict['electrodes']

# Remove 'nt' from any element in electrodes that contains 'HAnt'
electrodes = [e.replace('HAnt', 'HA') if 'HAnt' in e else e for e in electrodes]


# if network_type == 'Full':
#     classifiers = [mainDirectory + 'classifiers/' + re_reference_method + '/' + '-'.join([subject, connectivity_method, band, feature_selection+'_'+classifier_type, str(i)]) + '.sav' for i in range(1,11)]
#     # IF USING THE FULL NETWORK, WE NEED TO LOAD THE ELECTRODES FROM THE METADATA FILE
#     epileptogenic_network_patient = electrodes # The electrodes are the same as the ones in the metadata file, so we can use them directly
# elif network_type == 'JustEN':
#     classifiers = [mainDirectory + 'classifiers_justEN/' + re_reference_method + '/' + '-'.join([subject, connectivity_method, band, feature_selection+'_'+classifier_type, str(i)]) + '.sav' for i in range(1,11)]
#     EN_method = 'SC_R(4,8)'
#     ENs_file = metadata_directory + 'ENs/' + 'Neuroimage_results.xlsx' 
#     # IF USING THE EPILEPTOGENIC NETWORKS, WE NEED TO LOAD THE ELECTRODES FROM THE ENs file
#     epileptogenic_networks = pd.read_excel(ENs_file, index_col=0, dtype='object', engine='openpyxl') 
#     epileptogenic_networks = epileptogenic_networks[(epileptogenic_networks.subject == subject) & (epileptogenic_networks.time_frame == 'NS') & (epileptogenic_networks.method == EN_method)].reset_index(drop=True)
#     epileptogenic_network_patient = ast.literal_eval(epileptogenic_networks[epileptogenic_networks.subject == subject]['EN'].values[0])
#     epileptogenic_network_patient = [item.strip('\'"') for item in epileptogenic_network_patient]

# matrix_indices_ofENs = [electrodes.index(node.split('-')[0]) for node in epileptogenic_network_patient if node.split('-')[0] in electrodes ]


if network_type == 'Full':
    classifiers = [mainDirectory + 'classifiers/' + re_reference_method + '/' + '-'.join([subject, connectivity_method, band, feature_selection+'_'+classifier_type, str(i)]) + '.sav' for i in range(1,11)]
    # IF USING THE FULL NETWORK, WE NEED TO LOAD THE ELECTRODES FROM THE METADATA FILE
elif network_type == 'JustEN':
    classifiers = [mainDirectory + 'classifiers_justEN/' + re_reference_method + '/' + '-'.join([subject, connectivity_method, band, feature_selection+'_'+classifier_type, str(i)]) + '.sav' for i in range(1,11)]
    EN_method = 'SC_R(4,8)'
    ENs_file = metadata_directory + 'ENs/' + 'Neuroimage_results.xlsx' 
    # IF USING THE EPILEPTOGENIC NETWORKS, WE NEED TO LOAD THE ELECTRODES FROM THE ENs file
    epileptogenic_networks = pd.read_excel(ENs_file, index_col=0, dtype='object', engine='openpyxl') 
    epileptogenic_networks = epileptogenic_networks[(epileptogenic_networks.subject == subject) & (epileptogenic_networks.time_frame == 'NS') & (epileptogenic_networks.method == EN_method)].reset_index(drop=True)
    epileptogenic_network_patient = ast.literal_eval(epileptogenic_networks[epileptogenic_networks.subject == subject]['EN'].values[0])
    epileptogenic_network_patient = [item.strip('\'"') for item in epileptogenic_network_patient]


#%% Functions

def calculate_epochs(sample_interval, epoch_length_samples, epoch_overlap_samples):
    """
    This function computes the number of epochs of specific duration and overlap that can fit within a signal.
    
    Inputs:
        sample_interval: 
            Total number of samples of the signal that wants to be epoched.
        epoch_length_samples: 
            Number of samples that should be contained within one epoch.
        epoch_overlap_samples:
            Number of samples of the overlap between epochs.
            
    Outputs:
        number_of_epochs: 
            Number of epochs of specified duration and overlap that fit in the signal.
        number_of_samples: 
            Resulting TOTAL number of samples adding the samples of all of the epochs. Given the overlap, the resulting epochs will have more samples (adding them up) than the original signal.
    """
    # number of epochs that fit in the sample interval 
    number_of_epochs = 1 + math.ceil( (sample_interval - epoch_length_samples) / (epoch_length_samples - epoch_overlap_samples) )
    # total number of samples 
    number_of_samples = epoch_length_samples + (number_of_epochs-1)*(epoch_length_samples - epoch_overlap_samples)
    return [number_of_epochs, number_of_samples]

def time_from_epochs(number_of_epochs, epoch_length, epoch_overlap):
    time_covered = math.floor((number_of_epochs-1)*(epoch_length - epoch_overlap)+ epoch_length)
    return time_covered

def notch_filter(input_signal, sampling_frequency):
    # Notch filtering
    electrical_noise_og = 50.0  # Frequency to be removed from signal (Hz)
    electrical_noise = electrical_noise_og
    nyq = 0.5 * sampling_frequency
    harmonics = 2 # we start with 2 cause if not we are filtering at the first harmonic twice
    filtered_signal = input_signal
    while electrical_noise < nyq:
        Q = electrical_noise / 5 # Quality factor is defined as cut-off frequency / fH - fL, where fH and fL are the high and low bands of the stopband filter. Therefore, the value in the denominator indicates the range around the cut off freq that will be removed too 
        b_notch, a_notch = signal.iirnotch(electrical_noise, Q, sampling_frequency)
        filtered_signal = signal.filtfilt(b_notch, a_notch, filtered_signal)
        electrical_noise = electrical_noise_og * harmonics
        harmonics += 1 
    return filtered_signal

def slow_drift_filter(input_signal, sampling_frequency, order=4):
    cutoff = 1
    nyq = 0.5 * sampling_frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    filtered_signal = signal.filtfilt(b, a, input_signal)
    return filtered_signal

def rereference_Laplacian(input_signal,electrodes_channels_indices):
    re_ref_signal = np.zeros(input_signal.shape)
    for electrode in electrodes_channels_indices:
        if len(electrode)>1: # Only re-reference those electrodes with at least two channels
            for channel_position_in_electrode, channel in enumerate(electrode): # Here the electrode is a list with all the indices of the channels of that electrode                   
                # for the first channel
                if channel_position_in_electrode == 0: 
                    re_ref_signal[channel,:] = input_signal[channel,:] - input_signal[channel+1,:]
                # for any channel except 1st and last 
                if channel_position_in_electrode > 0 and channel_position_in_electrode < len(electrode)-1: # the -1 is bc python starts counting from 0
                    re_ref_signal[channel,:] = input_signal[channel,:] - ( input_signal[channel-1,:] + input_signal[channel+1,:] )/2
                # for the last channel
                if channel_position_in_electrode == len(electrode)-1:
                    re_ref_signal[channel,:] = input_signal[channel,:] - input_signal[channel-1,:]
        else: 
            pass
    return re_ref_signal

def rereference_bipolar(input_signal,electrodes_channels_indices):
    re_ref_signal = np.zeros(input_signal.shape)
    for electrode in electrodes_channels_indices:
        if len(electrode)>1: # Only re-reference those electrodes with at least two channels
            for channel_position_in_electrode, channel  in enumerate(electrode): # Here the electrode is a list with all the indices of the channels of that electrode                
                # for any channel except last 
                if channel_position_in_electrode < len(electrode)-1: 
                    re_ref_signal[channel,:] = input_signal[channel,:] - input_signal[channel+1,:]
                # for the last channel
                if channel_position_in_electrode == len(electrode)-1:
                    re_ref_signal[channel,:] = input_signal[channel,:]*0
        else: 
            pass
    return re_ref_signal

def rereference_signal(input_signal,electrodes_channels_indices,re_reference_method):
    if re_reference_method == "laplacian":
        re_ref_signal = rereference_Laplacian(input_signal, electrodes_channels_indices)
    if re_reference_method == "bipolar":
        re_ref_signal = rereference_bipolar(input_signal, electrodes_channels_indices)
    if re_reference_method == "monopolar":
        re_ref_signal = input_signal
    return re_ref_signal

def band_pass_filter(input_signal, sampling_frequency, frequency_band, order=2):
    filtered_signal = np.zeros(input_signal.shape)
    # Initialize frequencies
    frequencies = {"delta": [1,4],
                   "theta": [4,8],
                   "alpha": [8,13],
                   "beta": [13,35],
                   "low_gamma": [35,60],
                   "high_gamma": [60,140]}
    # Associate the name of the selected frequency to the range it covers and that will be the low and upper margings of the filter taking into account nyquist
    nyq = 0.5 * sampling_frequency
    low = frequencies[frequency_band][0] / nyq
    high = frequencies[frequency_band][1] / nyq
    if high > 1: high = 0.9999
    # Design the filter
    b, a = signal.butter(order, [low,high], btype='band', analog=False)
    # # This section tests that the filter has been designed properly
    # w, h = signal.freqz(b, a)
    # plt.plot((sampling_frequency * 0.5 / np.pi) * w, abs(h))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain')
    # plt.grid(True)
    # plt.show()
    #Apply the filter to the signal
    filtered_signal = signal.lfilter(b, a, input_signal, axis=-1)
    return filtered_signal

def are_multiples(a, b):
    if a == 0 or b == 0:
        return a == 0 and b == 0
    return a % b == 0 or b % a == 0

def is_within_ranges(ranges, number):
    """
    Check if a specific number is within any of the given ranges.

    Parameters:
    ranges (np.ndarray): A 2D numpy array where each row represents a range [start, end].
    number (float or int): The number to check.

    Returns:
    bool: True if the number is within any range, False otherwise.
    """
    for start, end in ranges:
        if start <= number <= end:
            return True
    return False

def is_within_any_ranges(ranges1, ranges2, number):
    """
    Check if a specific number is within any of the given ranges in two arrays.

    Parameters:
    ranges1 (np.ndarray): A 2D numpy array where each row represents a range [start, end].
    ranges2 (np.ndarray): Another 2D numpy array where each row represents a range [start, end].
    number (float or int): The number to check.

    Returns:
    bool: True if the number is within any range in either array, False otherwise.
    """
    def is_within_ranges(ranges, number):
        for start, end in ranges:
            if start <= number <= end:
                return True
        return False
    
    return is_within_ranges(ranges1, number) or is_within_ranges(ranges2, number)

def expand_coarse_to_fine(coarse_labels, coarse_epoch_length, overlap):
    """
    Expand coarse labels to fine-grained labels at a resolution of 1 sample.

    Parameters:
    coarse_labels (list of int): List of coarse labels (0 or 1).
    coarse_epoch_length (float): Duration of each coarse epoch in seconds.
    overlap (float): Overlap between consecutive coarse epochs in seconds.

    Returns:
    list of int: Fine-grained labels at a resolution of 1 sample.
    """
    # The step between the start times of consecutive coarse windows:
    step = coarse_epoch_length - overlap
    
    # Calculate total duration.
    # For N windows, total_duration = (N-1)*step + coarse_epoch_length
    N = len(coarse_labels)
    total_duration = (N - 1) * step + coarse_epoch_length
    
    # Start with all 0's at a resolution of 1 sample.
    fine_labels = [0] * int(total_duration * 10)  # Multiply by 10 for 0.1s resolution
    
    # "Paint" each coarse window onto the timeline.
    # If the coarse label is 1, mark the corresponding seconds as 1.
    for i, label in enumerate(coarse_labels):
        start = int(i * step * 10)
        end = int((i * step + coarse_epoch_length) * 10)  # Convert to 0.1s resolution
        if label == 1:
            # Set all samples in this window to 1.
            for t in range(start, end):
                fine_labels[t] = 1
    
    # Convert back to 1-second resolution by averaging every 10 samples.
    fine_labels = [1 if sum(fine_labels[i:i+10]) > 0 else 0 for i in range(0, len(fine_labels), 10)]
    
    return fine_labels

def update_buffer(buffer, new_value):
    """
    Update the buffer by removing the first row and appending new_value at the end.
    
    Parameters:
    buffer (np.ndarray): Array of shape (N, 3).
    new_value (np.ndarray): Array of shape (1, 3) or (3,).
    
    Returns:
    np.ndarray: Updated buffer of shape (N, 3).
    """
    buffer_chopped = buffer[1:]
    # Ensure new_value is of shape (1, 3)
    new_value = np.asarray(new_value).reshape(1, 3)
    buffer_new = np.vstack([buffer_chopped, new_value])
    return buffer_new

# Function to replace NaN with 0
def replace_nan_with_zero(x):
    return 0.0 if np.isnan(x) else x
vectorized_replace = np.vectorize(replace_nan_with_zero, otypes=[float])

def collapse_overlapping_epochs(predictions, conservation='preictal'):
    """
    Given an array of N values, create an array of size (N+1)//2.
    For element 0 of the new array, check elements 0 and 1 of the input array.
    If they are the same, assign that value. If different, assign 1.
    For element i > 0, check elements 2*i-1, 2*i, 2*i+1 of the input array.
    If all are the same, assign that value. If not, assign the majority value.
    If tie, assign 1 if conservation='preictal', else 0.
    """
    N = len(predictions)
    out_len = (N + 1) // 2
    out = np.zeros(out_len, dtype=int)
    for i in range(out_len):
        if i == 0:
            # Check elements 0 and 1
            vals = predictions[0:2]
            if len(vals) < 2:
                out[i] = vals[0]
            elif vals[0] == vals[1]:
                out[i] = vals[0]
            else:
                out[i] = 1
        else:
            idxs = [2*i-1, 2*i, 2*i+1]
            # Only keep indices within bounds
            idxs = [idx for idx in idxs if idx < N]
            vals = predictions[idxs]
            unique, counts = np.unique(vals, return_counts=True)
            if len(unique) == 1:
                out[i] = unique[0]
            else:
                max_count = np.max(counts)
                max_vals = unique[counts == max_count]
                if len(max_vals) == 1:
                    out[i] = max_vals[0]
                else:
                    # Tie: assign 1 if conservation='preictal', else 0
                    out[i] = 1 if conservation == 'preictal' else 0
    return out

# %% Initialization 
print("Loading...")

# Load 3 classifiers
trained_models = [joblib.load(classifier) for classifier in classifiers][0:3]
   
# Open EDF file
file = pyedflib.EdfReader(rawEDFs_directory + rawEDF_fileName) # open file

# Associate all annotations of the EDF with its time points
annotations = file.readAnnotations()
annotations_text = annotations[-1]
annotations_time = annotations[0]

 # Select the annotations that refer to a seizure and the time point. Create two lists, one for starting points and one for ending points of the seizures        Then combine them in a pandas DataFrame
seizures_start, seizures_end = [],[] # two lists containing the start and end, respectively, of each seizure
did_seizure_end = True # This switch accounts for those files in which the Seizure start/end are labelled in multiple languages. So, if a seizure flag is raised in any language, it can be closed in any language.
for time_mark, annotation in zip(annotations_time, annotations_text):
    if ((annotation == 'EEG start') or ('EEG inicio'  in annotation) or ('EEG Inicio'  in annotation)) and did_seizure_end:
        seizures_start.append(time_mark)
        did_seizure_end = False
    if ((annotation == 'EEG end') or ('EEG fin' in annotation) or ('fin' in annotation) or ('EEG Fin'  in annotation)) and not did_seizure_end:
        did_seizure_end = True
        seizures_end.append(time_mark)

# Dataframe containing all the relevant time info of the seizures
seizures = pd.DataFrame({"start":seizures_start, "end":seizures_end}) # Seizure start and end (in seconds)
seizures["duration"] = seizures.end - seizures.start                  # Seizure duration (in seconds)
number_of_seizures = len(seizures)                                    # Total number of seizures
print("seizures: "+str(number_of_seizures))
seizure_time_ratio = seizures.duration.sum()/file.getFileDuration()   # Seizure ratio: seizure time / all signal time

signal_duration = file.getFileDuration() # Duration of the signal in seconds
print("Signal duration: "+str(signal_duration)+" seconds")


# Only run the analysis if there are seizures
if number_of_seizures != 0: 
    
    # Select the channels
    """ Select channels """
    number_of_channels = file.signals_in_file # Number of channels in the recording
    labels = file.getSignalLabels() # ALL channels names (including ECG and unused channels), therefore labels=channel names
    
    # Remove channels that are not necessary
    non_necessary_channels = ['Pleth','PR','OSAT','TRIG','ECG1','ECG2','TTL']
    clean_labels = []
    non_used_labels, non_used_labels_index = [],[]
    for index, label in enumerate(labels): # iterate through all the labels by their index in the list
        included = False
        if (label not in non_necessary_channels): # if the channel is NOT one of the channels that we don't need, we keep going to save it in the "clean" list of labels
            if not ((''.join((x for x in label if not x.isdigit())) == 'C') and (int(''.join((x for x in label if x.isdigit()))) > 25)): # we ignore those channels that are named as "C" with a number higher than 18 bc they are empty channels
                if ''.join((x for x in label if not x.isdigit())) != 'DC': # also ignore the channels that are named DC1, DC2, DC3...
                    if ''.join((x for x in label if not x.isdigit())) != 'Fp': # also ignore the channels that are named Fp1, Fp2, Fp3...
                        clean_labels.append(label) # if all the criteria is met, include that channel in the "clean" list
                        included = True # we specify that that channel has been included
        if not included: # if all channel has not been included, then it means that it is not useful for our study, therefore we include it the list of channels that are not used
            non_used_labels.append(label)
            non_used_labels_index.append(index) # also save the position in the list of labels of those non used channels
    
    # Number of channels after removing the non necesary ones
    clean_labels_indices = np.delete(np.arange(number_of_channels),non_used_labels_index)
    new_number_of_channels = len(clean_labels_indices)
    
    electrodes_names = np.unique([ ''.join((x for x in name if not x.isdigit())) for name in clean_labels]).tolist() # Unique electrode names (i.e. electrode "B")
    electrodes_channels = [[name for name in clean_labels if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in electrodes_names] # Channels grouped by electrodes. List of lists 
    electrodes_channels_indices = [[clean_labels.index(name) for name in clean_labels if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in electrodes_names] # Channels grouped by electrodes. List of lists 
    
    # Redefine the epoch variables in terms of samples 
    sampling_frequency = round(file.getNSamples()[clean_labels_indices[0]]/file.getFileDuration()) # Sampling freq [Hz] of the recording (number of total samples/duration of the recording in seconds), we select the first channel of the clean channels as reference, cause if we selected the first channel of the recording we could be selecting an ECG channel f.e.
    original_sampling_frequency = sampling_frequency
    epoch_length_samples = round(epoch_length * sampling_frequency) # How long, in term of samples, epochs should be according to their duration. I.e. if we use 2s epochs at a 500Hz sampling rate, we would need to select 1000 samples per epoch 
    epoch_overlap_samples = round(epoch_overlap * sampling_frequency) # Epoch overlap in terms of samples
    extension_length_samples = round(extension_length * sampling_frequency)
    
    # Calculate number of epochs that fit into the signal
    seizures['starting_sample'] = round(seizures.start*sampling_frequency) # Exact sample when the seizures start
    seizures['ending_sample'] = round(seizures.end*sampling_frequency) # Exact sample when the seizures end
    number_of_samples_in_signal = np.unique(file.getNSamples())[-1]
    number_of_epochs, _ =  calculate_epochs(number_of_samples_in_signal, epoch_length_samples, epoch_overlap_samples) # Here we obtain the number of epochs that we need for that seizure 
    
    # Ranges of epochs that contain ictal activity
    ictal_ranges = []
    for index, seizure in seizures.iterrows():
        start, _ =  calculate_epochs(seizure.starting_sample, epoch_length_samples, epoch_overlap_samples)
        end, _ =  calculate_epochs(seizure.ending_sample, epoch_length_samples, epoch_overlap_samples)
        ictal_ranges.append((int(start) , int(end)))
    ictal_ranges = np.array(ictal_ranges)
    
    # Array to store all the predictions of the state of the epochs by the classifiers
    epoch_predictions = np.zeros((number_of_epochs, 3))    
    epoch_predictions_proba = np.zeros((number_of_epochs, 3)) # Array to store the probabilities of the state of the epochs by the classifiers
    
    epoch_predictions_duration = np.zeros((number_of_epochs, 1)) # Array to store the duration of the epochs in seconds

    # List to store the warnings
    raised_warnings = []
    
    # Start timing the prediction
    prediction_start_time = time.time()
    
    print("Starting prediction")
    
    # Create a buffer to store the last predicted epochs
    buffer_length_adjusted, _ = calculate_epochs(buffer_length, epoch_length, epoch_overlap)
    buffer = np.zeros((buffer_length_adjusted, 3), dtype=int)
    buffer_proba = np.zeros((buffer_length_adjusted, 3), dtype=float) # Buffer to store the probabilities of the epochs

    # %% Predict classification of epochs (and probabilities) using the trained models
    
    # Explore the signal per epochs
    for epoch in range(number_of_epochs-1):
        
        epoch_time_start = time.time() # Start time of the epoch

        # Initialize array where the signal is stored
        data_bufs = np.zeros((new_number_of_channels, epoch_length_samples+extension_length_samples*2))
        
        # Start and end in sample terms
        starting_sample = epoch*(epoch_length_samples-epoch_overlap_samples)
        ending_sample = starting_sample + epoch_length_samples
        
        # Array where bad channels are stored
        bad_channels = []
        
        # The conditial account for first al last epochs that cannot be padded with real signal but with zeros
        if (starting_sample > extension_length_samples) and (starting_sample < number_of_samples_in_signal - epoch_length_samples - extension_length_samples*2):
            for channel_position_in_buffer, channel_index_in_EDF_file in zip(np.arange(new_number_of_channels), clean_labels_indices): # Since some channels are removed as they are non-necessary (i.e. ECG channels), the indexing of the resulting array does not match the indexing of the EDF file, that is, if we for example don't want channel 1 from the EDF but we want channel 2, in our array channel 1 is in fact channel 2 of the EDF, as we removed the 1st one      
                # Retreive the signal
                data_bufs[channel_position_in_buffer, :] = file.readSignal(channel_index_in_EDF_file, start=starting_sample-extension_length_samples, n=epoch_length_samples+extension_length_samples*2) # save the signal in the buffer
                # Detect bad channels
                if len(np.unique(data_bufs[channel_position_in_buffer, :])) == 1: # if all the elements in a channel are the same, that channel is useless, so we add it to a list o channels to remove from the array
                    bad_channels.append(channel_position_in_buffer) 
        else:
            for channel_position_in_buffer, channel_index_in_EDF_file in zip(np.arange(new_number_of_channels), clean_labels_indices): # Since some channels are removed as they are non-necessary (i.e. ECG channels), the indexing of the resulting array does not match the indexing of the EDF file, that is, if we for example don't want channel 1 from the EDF but we want channel 2, in our array channel 1 is in fact channel 2 of the EDF, as we removed the 1st one      
                data_bufs[channel_position_in_buffer, extension_length_samples:extension_length_samples+epoch_length_samples] = file.readSignal(channel_index_in_EDF_file, start=starting_sample, n=epoch_length_samples) # save the signal in the buffer
                if len(np.unique(data_bufs[channel_position_in_buffer, :])) == 1: # if all the elements in a channel are the same, that channel is useless, so we add it to a list o channels to remove from the array
                    bad_channels.append(channel_position_in_buffer)   

        # Remove bad channels
        # bad_channels = np.unique(bad_channels)
        # if bad_channels.size != 0:
        #     data_bufs = np.delete(data_bufs, bad_channels, axis=0) 
        #     # As we now have an array that is different, as we have removed bad channels, we need to reorganize the electrodes to perform re-referencing later
        #     new_clean_labels = np.delete(clean_labels, bad_channels).tolist()
        #     electrodes_names = np.unique([ ''.join((x for x in name if not x.isdigit())) for name in new_clean_labels]).tolist() # Unique electrode names (i.e. electrode "B")
        #     electrodes_channels = [[name for name in new_clean_labels if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in electrodes_names] # Channels grouped by electrodes. List of lists 
        #     electrodes_channels_indices = [[new_clean_labels.index(name) for name in new_clean_labels if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in electrodes_names] # Channels grouped by electrodes. List of lists 


        print("Epoch: {}/{}".format(epoch+1, number_of_epochs-1))
        data_rereferenced = rereference_signal(data_bufs, electrodes_channels_indices, re_reference_method)
        del data_bufs 
                         
         # Apply notch filter (low pass) to the noisy signal using signal.filtfilt
        data_notch_filtered = notch_filter(data_rereferenced, sampling_frequency)
        del data_rereferenced
        
        # High pass filter at 1Hz to remove slow drifts
        data_high_filtered = slow_drift_filter(data_notch_filtered, sampling_frequency)
        del data_notch_filtered
        
        # Depadd the signal
        data_depadded = data_high_filtered[:,extension_length_samples:epoch_length_samples+extension_length_samples]
        del data_high_filtered
    
        # Resample the signal if needed
        if (sampling_frequency != 512) and (sampling_frequency != 500): # If the sampling rate is different than 512Hz or 500Hz, resample 
            if are_multiples(sampling_frequency,512): # If the sampling freq is a multiple of 512Hz, resample to 512Hz.
                new_sampling_frequency = 512
            if are_multiples(sampling_frequency,500): # If the sampling freq is a multiple of 500, resample to 500Hz.
                new_sampling_frequency = 500
            new_epoch_length_samples = round(epoch_length * new_sampling_frequency) # Epoch duration in terms of samples of the resampled signals
            data_resampled = signal.resample(data_depadded, new_epoch_length_samples, axis=-1)
        else: # if the sampling was already 512 or 500, keep that signals as they are, only rename the variables 
            if sampling_frequency == 512:
                new_sampling_frequency = 512
                data_resampled = data_depadded
            if sampling_frequency == 500:
                new_sampling_frequency = 500
                data_resampled = data_depadded
        del data_depadded 
        
        # Filter the signal if needed
        if band == 'all_bands':
            epoch_data = data_resampled
        else:
            epoch_data = band_pass_filter(data_resampled, new_sampling_frequency, band)
            
        # Select the method to apply
        if connectivity_method == 'PAC': 
            from connectivity_methods import PAC
            method = PAC
            
        if connectivity_method == 'spectral_coherence': 
            from connectivity_methods import spectral_coherence
            method = spectral_coherence
            
        if connectivity_method == 'correlation': 
            from connectivity_methods import correlation
            method = correlation
    
        if connectivity_method == 'cross_correlation': 
            from connectivity_methods import cross_correlation
            method = cross_correlation
            
        if connectivity_method == 'phase_lag': 
            from connectivity_methods import phase_lag
            method = phase_lag
            
        if connectivity_method == 'phase_lock': 
            from connectivity_methods import phase_lock
            method = phase_lock
            
        if network_type == 'JustEN':
            # If we are using the epileptogenic networks, we only compute the connectivity between the electrodes that are part of the EN
            clean_labels = [e.replace('HAnt', 'HA') if 'HAnt' in e else e for e in clean_labels]

            matrix_indices_ofEN_inthecleanlabels = [clean_labels.index(node.split('-')[0]) for node in epileptogenic_network_patient if node.split('-')[0] in clean_labels ]
            matrix_indices_ofENs = clean_labels_indices[matrix_indices_ofEN_inthecleanlabels]
            epoch_data = epoch_data[matrix_indices_ofENs, :]  # Select only the channels in the epileptogenic network

        # Compute connectivity matrices
        connectivity_matrix = connectivity_analysis(epoch_data, method, fs=new_sampling_frequency);
        connectivity_matrix_nonNaN = vectorized_replace(connectivity_matrix)
        # Flatten the CM to feed it to the classifier
        connectivity_matrix_flat = connectivity_matrix_nonNaN.reshape(1,-1) 

        if feature_selection == 'Eig':
            eigvalues, _ = np.linalg.eig(connectivity_matrix_nonNaN)
            eigvalues = np.real(eigvalues)  # Ensure we are working with real values
            connectivity_matrix_flat = eigvalues.reshape(1, -1)  # Reshape to 2D array for classifier input
        
        for i in range(0, len(trained_models)):
            # Predict the state of the epoch and store it
            try:
                prediction = trained_models[i].predict(connectivity_matrix_flat) # Class prediction (0: interictal, 1: preictal)
                prediction_proba = trained_models[0].predict_proba(connectivity_matrix_flat)[0][1] # Probability of the epoch being preictal
            # In some cases artifacts make an epoch impossible to classify, we label it as interictal then
            except Exception as e: 
                print(f"Error during prediction for epoch {epoch}, model {i}: {e}")
                prediction = np.nan
                prediction_proba = np.nan
                                
            # Store the prediction in the array
            epoch_predictions[epoch, i] = prediction.item() if hasattr(prediction, "item") else prediction
            epoch_predictions_proba[epoch, i] = prediction_proba.item() if hasattr(prediction_proba, "item") else prediction_proba

        epoch_time_end = time.time() # End time of the epoch
        epoch_predictions_duration[epoch, 0] = (epoch_time_end - epoch_time_start)   

# %% Post-process the predictions
epoch_predictions_postprocessed = np.zeros((number_of_epochs, 3)) # Array to store the post-processed predictions
epoch_predictions_proba_postprocessed = np.zeros((number_of_epochs, 3)) # Array to store the post-processed probabilities
raised_warnings = np.zeros((number_of_epochs, 1), dtype=bool) # Array to store the warnings raised by the classifier
raised_proba = np.zeros((number_of_epochs, 1), dtype=float) # Array to store the warnings raised by the classifier based on probabilities

for epoch in range(number_of_epochs-1):
        
        print("Post-processing epoch: {}/{}".format(epoch+1, number_of_epochs-1))
        
        # Update the buffer with the predictions and probabilities
        buffer = update_buffer(buffer, epoch_predictions[epoch])
        buffer_proba = update_buffer(buffer_proba, epoch_predictions_proba[epoch])

        # Percentage of the buffer predicted as preictal
        percent_of_preictal = buffer.sum()/buffer.size # Percentage of the buffer predicted as preictal
        percent_of_preictal_proba = buffer_proba.sum()/buffer_proba.size # Percentage of the buffer predicted as preictal based on probabilities

        if percent_of_preictal >= buffer_threshold: # If the percentage of preictal epochs in the buffer is higher than the threshold, we raise a warning
            raised_warnings[epoch] = True
        else:
            raised_warnings[epoch] = False
        
        raised_proba[epoch] = percent_of_preictal_proba # Store the percentage of preictal epochs in the buffer based on probabilities

# Plot the probabilities of the epochs
plt.figure(figsize=(20, 4), dpi=300)
# Plot vertical orange lines for each epoch where raised_warnings is True
for idx in np.where(raised_warnings[:number_of_epochs-1].flatten())[0]:
    plt.axvline(x=idx, color='orange', linestyle='-', linewidth=1, label='Warning' if idx == np.where(raised_warnings[:number_of_epochs-1].flatten())[0][0] else "")
plt.plot(np.arange(0, number_of_epochs-1), raised_proba[:number_of_epochs-1].flatten(), color='blue', label='Probability of preictal')
plt.axhline(y=buffer_threshold, color='red', linestyle='--', label='Threshold')
plt.xlim(0, len(raised_proba[:number_of_epochs-1].flatten()))
plt.xlabel('Epoch number')
plt.ylabel('Probability of preictal')
plt.title(f'Probability of preictal epochs - Subject:{subject} - Date:{date}')
plt.legend()
plt.savefig(results_directory+'figures/'+subject+'-'+date+'-Probability_of_preictal_epochs.svg')
plt.show()
    
# %% Represent model classification of epochs
    
# Ranges of epochs that contain preictal activity
preictal_ranges = []
for ictal_range in ictal_ranges:
    duration_preictal_range = (5*2)-1
    start = ictal_range[0] - duration_preictal_range
    end = ictal_range[0] -1
    preictal_ranges.append((int(start) , int(end)))
preictal_ranges = np.array(preictal_ranges)

# Create "ground-truth" of the state of the signal
preictal_ground_truth = np.array([int(is_within_ranges(preictal_ranges,epoch)) for epoch in range(number_of_epochs)])
 
# Plot real states of the epochs
plt.figure(figsize=(20, 4), dpi=300)
# Plot vertical orange lines for each epoch where raised_warnings is True
for idx in np.where(raised_warnings[:number_of_epochs-1].flatten())[0]:
    plt.axvline(x=idx, color='orange', linestyle='-', linewidth=1, label='Warning' if idx == np.where(raised_warnings[:number_of_epochs-1].flatten())[0][0] else "")
plt.plot(np.arange(0, number_of_epochs-1), raised_proba[:number_of_epochs-1].flatten(), color='blue', label='Probability of preictal')
plt.axhline(y=buffer_threshold, color='red', linestyle='--', label='Threshold')
for index, value in enumerate(preictal_ground_truth):
    if value == 1:
        plt.axvline(x=index, color='green', linestyle='-', linewidth=2, label='Preictal' if index == 0 else "")
for ranges in ictal_ranges:
    for ictal_act in range(ranges[0], ranges[1] + 1):
        plt.axvline(x=ictal_act, color='red', linestyle='-', linewidth=2, label='Ictal' if ictal_act == ictal_ranges[0][0] else "")
# Set the limits of the x-axis
plt.xlim(0, len(preictal_ground_truth))
# Set labels and title
plt.xlabel('Epoch number')
plt.ylabel('State')
plt.title(f'Classification of epochs (Green: preictal; Red: real seizure) - Subject:{subject} - Date:{date}')
plt.legend()
plt.savefig(results_directory+'figures/'+subject+'-'+date+'-Classification_of_epochs.svg')
plt.show()

# Create "ground-truth" and predcitions of the state of the signal but time-wise (each epoch is a minute without overlap)
collapsed_grountruth = collapse_overlapping_epochs(preictal_ground_truth[:number_of_epochs].flatten(), conservation='preictal')
collapsed_warnings = collapse_overlapping_epochs(raised_warnings[:number_of_epochs].flatten(), conservation='preictal')

# %% Event-Based Metrics: TimeScoring

# Scoring with SzScore
ref = Annotation(collapsed_grountruth, 1/60)
hyp = Annotation(collapsed_warnings, 1/60)
    
# by samples
scores = scoring.SampleScoring(ref, hyp)
figSamples = visualization.plotSampleScoring(ref, hyp)
figSamples.suptitle(f'Subject: {subject} - Date: {date}', y=1.05)
figSamples.savefig(results_directory+'figures/'+subject+'-'+date+'-SzScore_SAMPLES.svg')

# by events
param = scoring.EventScoring.Parameters(
    toleranceStart=5*60,
    toleranceEnd=10*60,
    minOverlap=0,
    maxEventDuration=10*60,
    minDurationBetweenEvents=1*60)

scores = scoring.EventScoring(ref, hyp, param)
sensitivity = scores.sensitivity
precision = scores.precision
f1 = scores.f1
fpRate = scores.fpRate

figEvents = visualization.plotEventScoring(ref, hyp, param)
figEvents.savefig(results_directory+'figures/'+subject+'-'+date+'-SzScore_EVENTS.svg')

# Performance of the classifier based on metrics
model_accuracy = balanced_accuracy_score(collapsed_grountruth, collapsed_warnings)
model_recall = recall_score(collapsed_grountruth, collapsed_warnings,average=None);
model_precision = precision_score(collapsed_grountruth, collapsed_warnings,average=None);
print("\nAccuracy: {} % \nRecall preictal: {} % \nPrecision interictal: {} %".format( round(model_accuracy*100,4), round(model_recall[1]*100,4), round(model_precision[0]*100,4) ))
    
# Confusion Matrix
confusionMatrix = confusion_matrix(collapsed_grountruth, collapsed_warnings)
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels = ['Interictal', 'Preictal'])
cm_display.plot()
plt.title('Confusion Matrix '+ '-'.join([subject, date]))
plt.xlabel('Predicted label (Number of epochs)')
plt.ylabel('True label (Number of epochs)')
plt.savefig(results_directory+'figures/'+subject+'-'+date+'-Confusion_Matrix.svg')
plt.show()
    
# Percentage of the signal labelled as preictal
print("\nPercent of signal predicted as preictal: {} %".format(round(sum(collapsed_warnings)/len(collapsed_warnings)*100, 3)))


onlinePredictionResults = pd.DataFrame({'patient': [subject],
                                        'date': [date],
                                        'balanced_accuracy': [model_accuracy],
                                        'recall_interictal': [model_recall[0]],
                                        'recall_preictal': [model_recall[1]],
                                        'precision_interictal':[ model_precision[0]],
                                        'precision_preictal':[ model_precision[1]],
                                        'SzScore_Events_sensitivity': [sensitivity], 
                                        'SzScore_Events_precision': [precision], 
                                        'SzScore_Events_f1': [f1], 
                                        'SzScore_Events_fpRate': [fpRate], 
                                        'percent_preictal': [round(sum(collapsed_warnings)/len(collapsed_warnings)*100, 3)],
                                        'time_to_predict_one_epoch_mean': [epoch_predictions_duration.mean()], #in seconds
                                        'time_to_predict_one_epoch_std': [epoch_predictions_duration.std()]})

import pickle 
with open(results_directory + subject+'-'+date+'-'+'onlinePredictionResult.pkl', 'wb') as f:
    pickle.dump(onlinePredictionResults, f)
