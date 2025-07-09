#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:12:25 2023
@author: Justo Montoya Gálvez
"""

# %% Libraries 
import sys
import math
import h5py
import pyedflib
import numpy as np
import pandas as pd
from json import dump
from os import listdir
from scipy import signal
from random import sample

#%% Variable initilization 
slurm_array_id = int(sys.argv[1])

# %% Variables
""" Type of re-reference method """
# re_reference_method = "monopolar"
re_reference_method = "bipolar"
# re_reference_method = "laplacian"

""" Epochs """
interictal_ratio = 2  # Ratio of the seizure duration (interictal/preictal) employed as interictal 
preictal_duration = 5*60 # Duration of the preictal state in seconds
epoch_length = 60      # Epoch duration in seconds
epoch_overlap = 30   # Epoch overlap in seconds
extension_length = 10  # Duration of the signal in seconds before and after the epoch that is selected to avoid artifacts when filtering

""" Sampling limitations """
around_seizure_excluded = 30 # Time in minutes before a seizure excluded from interictal selection

""" Paths """
# Local
# directory = '/Users/justo/Desktop/cohort/'
# output_directory = '/Users/justo/Desktop/cohort/'

# %% Cluster

# CLUSTER: 
directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/Raw_EDFs_cohort21/'
output_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/epochs/'
metadata_directory='/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/single/'
wo_seizure_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/WOseizure/'


# %%
current_subs = listdir(directory)
# current_subs = ['VBM-2013MAR12-seeg.EDF', 'SDA-2016APR28-seeg.EDF']

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
    cutoff = 0.5
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
                    re_ref_signal[:,channel,:] = input_signal[:,channel,:] - input_signal[:,channel+1,:]
                # for any channel except 1st and last 
                if channel_position_in_electrode > 0 and channel_position_in_electrode < len(electrode)-1: # the -1 is bc python starts counting from 0
                    re_ref_signal[:,channel,:] = input_signal[:,channel,:] - ( input_signal[:,channel-1,:] + input_signal[:,channel+1,:] )/2
                # for the last channel
                if channel_position_in_electrode == len(electrode)-1:
                    re_ref_signal[:,channel,:] = input_signal[:,channel,:] - input_signal[:,channel-1,:]
        else: 
            pass
    return re_ref_signal

def rereference_bipolar(input_signal,electrodes_channels_indices):
    re_ref_signal = np.zeros(input_signal.shape)
    for electrode in electrodes_channels_indices:
        if len(electrode)>1: # Only re-reference those electrodes with at least two channels
            for channel_position_in_electrode, channel in enumerate(electrode): # Here the electrode is a list with all the indices of the channels of that electrode                
                # for any channel except last 
                if channel_position_in_electrode < len(electrode)-1: 
                    re_ref_signal[:,channel,:] = input_signal[:,channel,:] - input_signal[:,channel+1,:]
                # for the last channel
                if channel_position_in_electrode == len(electrode)-1:
                    re_ref_signal[:,channel,:] = input_signal[:,channel,:]*0
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

#%% Data loading and early stage preprocessing

"""" Retreive file from folder..."""

file_name = current_subs[slurm_array_id]

# %% File opening
file = pyedflib.EdfReader(directory+file_name) # open file

patient_code = file_name.split("-")[0]
date =  file_name.split("-")[1]
print(patient_code, date)
print(re_reference_method)

"""" Associate all annotations of the EDF with its time points """
annotations = file.readAnnotations()
annotations_text = annotations[-1]
annotations_time = annotations[0]

""" Select the annotations that refer to a seizure and the time point
    Create two lists, one for starting points and one for ending points of the seizures
    Then combine them in a pandas DataFrame
"""
seizures_start, seizures_end = [],[] # two lists containing the start and end, respectively, of each seizure
subC_seizures_start = [] #start of subclinical seizures
did_seizure_end = True # This switch accounts for those files in which the Seizure start/end are labelled in multiple languages. So, if a seizure flag is raised in any language, it can be closed in any language.
for time_mark, annotation in zip(annotations_time, annotations_text):
    if ((annotation == 'EEG start') or ('eeg inicio' in annotation.lower()) or ('inicio seeg' in annotation.lower())) and did_seizure_end:
        seizures_start.append(time_mark)
        did_seizure_end = False
    if ((annotation == 'EEG end') or ('eeg fin' in annotation.lower()) or ('fin' in annotation) or ('fin seeg' in annotation.lower())) and not did_seizure_end:
        did_seizure_end = True
        seizures_end.append(time_mark)
    if annotation == 'SC':
        subC_seizures_start.append(time_mark)

# Dataframe containing all the relevant time info of the seizures
seizures = pd.DataFrame({"start":seizures_start, "end":seizures_end}) # Seizure start and end (in seconds)
seizures["duration"] = seizures.end - seizures.start                  # Seizure duration (in seconds)
number_of_seizures = len(seizures)                                    # Total number of seizures
print("seizures: "+str(number_of_seizures))
seizure_time_ratio = seizures.duration.sum()/file.getFileDuration()   # Seizure ratio: seizure time / all signal time

scSeizures = pd.DataFrame({"start":subC_seizures_start}) # Subclinical seizure start (in seconds)

# %% Retreive signal
if number_of_seizures == 0:
    with open(wo_seizure_directory+file_name+'.txt', 'w') as f:
        f.write(file_name)
else:
     
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
    
    """ Epoch definition """
    sampling_frequency = round(file.getNSamples()[clean_labels_indices[0]]/file.getFileDuration()) # Sampling freq [Hz] of the recording (number of total samples/duration of the recording in seconds), we select the first channel of the clean channels as reference, cause if we selected the first channel of the recording we could be selecting an ECG channel f.e.
    original_sampling_frequency = sampling_frequency
    epoch_length_samples = round(epoch_length * sampling_frequency) # How long, in term of samples, epochs should be according to their duration. I.e. if we use 2s epochs at a 500Hz sampling rate, we would need to select 1000 samples per epoch 
    epoch_overlap_samples = round(epoch_overlap * sampling_frequency) # Epoch overlap in terms of samples
    extension_length_samples = round(extension_length * sampling_frequency)
    
    """ Retreive signals """
    print("   Obtaining signal data...")
    # Initialize important variables that are commonly used trhough the code
    seizures['starting_sample'] = round(seizures.start*sampling_frequency) # Exact sample when the seizures start
    seizures['ending_sample'] = round(seizures.end*sampling_frequency) # Exact sample when the seizures end
    seizures["sample_interval_preictal"] = round(preictal_duration*sampling_frequency) # number of samples that should be selected as preictal for each seizure

    # If the seizure start is close to the recording start, and the distance to start it is smaller than the defined preictal duration, we need to adapt so that we select from the start of the recording
    if seizures.starting_sample[0] < seizures.sample_interval_preictal[0]:
        seizures.loc[0,'sample_interval_preictal'] = seizures.starting_sample[0]

    seizures["sample_interval_interictal"] = round(seizures.sample_interval_preictal*interictal_ratio) # number of samples that should be selected as interictal for each seizure
    scSeizures['starting_sample'] = round(scSeizures.start*sampling_frequency) # Exact sample when the subclinical seizures start

    # All samples that contain preictal, ictal, around seizure and around subclinical seizure and cannot be selected as interictal
    excluded = []
    samples_around_seizure_excluded = (around_seizure_excluded*60/epoch_length)*epoch_length_samples
    for index,seizure in seizures.iterrows():
        # We create this list of excluded samples that contains all the samples from the preictal and ictal states, for that we remove from the starting sample of the seizure (- the duration in samples of the preictal) and also (- 30 min before the preictal) til the ending sample of the seizure + 30 mins more 
        excluded.append(np.arange(seizure.starting_sample-seizure.sample_interval_preictal-samples_around_seizure_excluded, seizure.ending_sample+samples_around_seizure_excluded))

    for index, scSeizure in scSeizures.iterrows():
        excluded.append(np.arange(scSeizure.starting_sample-samples_around_seizure_excluded, scSeizure.starting_sample+samples_around_seizure_excluded)) # We add the subclinical seizures to the excluded samples

    # Here we also exclude the same number of samples as the epoch duration from the end of the recording, in case that the interictal epoch falls at the end.
    excluded.append(np.arange(file.getNSamples()[clean_labels_indices[0]]-epoch_length_samples, file.getNSamples()[clean_labels_indices[0]]) ) 
    excluded.append(np.arange(0,extension_length_samples)) # Remove the padding from the beginning
    excluded_samples_from_interictal = np.hstack(excluded) # Flatten the list
    available_samples = np.arange(file.getNSamples()[clean_labels_indices[0]]) # Create an array with all samples in the recording
    available_samples = np.setdiff1d(available_samples.astype(int), excluded_samples_from_interictal.astype(int)) # Remove samples that are not available for interictal selection
    
    # Here we extract the signal from the EDF file
    all_selected = []
    for index,seizure in seizures.iterrows():
        
        # PREICTAL: Create buffer to store the preictal signal. Rows are the channels, columns the number of samples and each stack of the matrix is an epoch
        number_of_epochs_preictal, _ =  calculate_epochs(seizure.sample_interval_preictal, epoch_length_samples, epoch_overlap_samples) # Here we obtain the number of epochs that we need for that seizure 
        preictal_bufs = np.zeros((number_of_epochs_preictal, new_number_of_channels, epoch_length_samples+extension_length_samples*2))
        # Fill the buffer with the actual signal
        bad_channels = []
        for channel_position_in_buffer, channel_index_in_EDF_file in zip(np.arange(new_number_of_channels), clean_labels_indices): # Since some channels are removed as they are non-necessary (i.e. ECG channels), the indexing of the resulting array does not match the indexing of the EDF file, that is, if we for example don't want channel 1 from the EDF but we want channel 2, in our array channel 1 is in fact channel 2 of the EDF, as we removed the 1st one
            for epoch in np.arange(number_of_epochs_preictal): # we fill the buffer epoch per epoch to apply the overlap
                if epoch == 0: # The 1st epoch goes from the starting of the preictal state to + the number of samples in an epoch
                    start = seizure.starting_sample-seizure.sample_interval_preictal-extension_length_samples
                    if start < 0: # if the starting sample is negative, we need to start from 0
                        preictal_bufs[epoch, channel_position_in_buffer, extension_length_samples:] = file.readSignal(channel_index_in_EDF_file, start=0, n=epoch_length_samples+extension_length_samples) # save the signal in the buffer
                        last_ending_sample = int(seizure.starting_sample-seizure.sample_interval_preictal + epoch_length_samples)
                    else:
                        preictal_bufs[epoch, channel_position_in_buffer, :] = file.readSignal(channel_index_in_EDF_file, start=seizure.starting_sample-seizure.sample_interval_preictal-extension_length_samples, n=epoch_length_samples+extension_length_samples*2) # save the signal in the buffer
                        last_ending_sample = int(seizure.starting_sample-seizure.sample_interval_preictal + epoch_length_samples) # we store the position of the last sample of the epoch
                else:
                    # Now we sample from the postion that was the last sample of the previous epoch, minus the overlap, till the duration of an epoch
                    preictal_bufs[epoch, channel_position_in_buffer, :] = file.readSignal(channel_index_in_EDF_file, start=last_ending_sample-epoch_overlap_samples-extension_length_samples, n=epoch_length_samples+extension_length_samples*2)
                    last_ending_sample = int(last_ending_sample-epoch_overlap_samples + epoch_length_samples) # update the value of the last sample of the epoch
                    if len(np.unique(preictal_bufs[epoch, channel_position_in_buffer, :])) == 1: # if all the elements in a channel are the same, that channel is useless, so we add it to a list o channels to remove from the array
                        bad_channels.append(channel_position_in_buffer)
        
        # INTERICTAL
        number_of_epochs_interictal, _ = calculate_epochs(seizure.sample_interval_interictal, epoch_length_samples, 0) # since the epochs of interictal are selected randomly from the signal that is not labeled as preictal or ictal, there is no overlap (overlap=0) between epochs, we just select random epochs
        interictal_bufs = np.zeros((number_of_epochs_interictal, new_number_of_channels, epoch_length_samples+extension_length_samples*2))
        # Select the random samples that will be the starting point of the epochs of interictal
        interictal_random_samples = sample(list(available_samples), number_of_epochs_interictal)
        # identify all the samples that will be part of the epochs and remove them from the available samples so that we don't select twice the same signal
        selected_interictal_samples = np.unique(np.hstack([np.arange(start_sample, start_sample + epoch_length_samples) for start_sample in interictal_random_samples]))
        all_selected.append(selected_interictal_samples)
        available_samples = np.setdiff1d(available_samples, selected_interictal_samples)
        # Fill the buffer with the actual signal
        for channel_position_in_buffer, channel_index_in_EDF_file in zip(np.arange(new_number_of_channels), clean_labels_indices):
            for epoch, random_interictal_starting_sample in enumerate(interictal_random_samples):
                interictal_bufs[epoch, channel_position_in_buffer, :] = file.readSignal(channel_index_in_EDF_file, start=random_interictal_starting_sample-extension_length_samples, n=epoch_length_samples+extension_length_samples*2) # save the signal in the buffer        
        
        bad_channels = np.array(bad_channels).astype(int) # Array has to be dtype=int to then remove the bad channels
        if bad_channels.size != 0:
            # Remove bad channels
            preictal_bufs = np.delete(preictal_bufs, bad_channels, axis=1) 
            interictal_bufs = np.delete(interictal_bufs, bad_channels, axis=1)
            # As we now have an array that is different, as we have removed bad channels, we need to reorganize the electrodes to perform re-referencing later
            new_clean_labels = np.delete(clean_labels, bad_channels).tolist()
            electrodes_names = np.unique([ ''.join((x for x in name if not x.isdigit())) for name in new_clean_labels]).tolist() # Unique electrode names (i.e. electrode "B")
            electrodes_channels = [[name for name in new_clean_labels if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in electrodes_names] # Channels grouped by electrodes. List of lists 
            electrodes_channels_indices = [[new_clean_labels.index(name) for name in new_clean_labels if electrode == ''.join((x for x in name if not x.isdigit()))] for electrode in electrodes_names] # Channels grouped by electrodes. List of lists 
        
                
        print("   Seizure {}: Signal data obtained!".format(index+1))
        
        """ Early stage preprocessing:
            1. Re-reference
            2. Notch filter at 50Hz to remove electrical noise (and harmonics)
            3. High-pass at 1Hz to remove slow drifts
            4. Remove padding from epochs
            5. Resample
        """
        print("               Early-stage pre-processing...")
    
    # test = np.hstack(test)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(100,10),dpi=200)
    # plt.vlines(seizures['starting_sample'], ymin=0, ymax=1,colors='red')
    # plt.vlines(excluded_samples_from_interictal, ymin=0, ymax=.5, colors='orange')
    # plt.vlines(test2, ymin=0, ymax=.5, colors='green')
    # plt.xlim(0,7200)
    # plt.show()
    
# %% Re-reference

        """ Re-reference the channels """
        preictal_rereferenced = rereference_signal(preictal_bufs, electrodes_channels_indices, re_reference_method)
        interictal_rereferenced = rereference_signal(interictal_bufs, electrodes_channels_indices, re_reference_method)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Uncomment this section to check that the re-reference has been properly applied #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # import matplotlib.pyplot as plt
        # # Check with synthetic signals
        # # Set the parameters of the sine wave
        # amplitude = 1 # Amplitude of the wave
        # frequency = 50 # Frequency of the wave in Hertz
        # phase = 0 # Phase shift of the wave in radians
        # sampling_rate = 500 # Number of samples per second
        # duration = 2 # Duration of the wave in seconds
        # # Create a numpy array of time values from 0 to duration
        # time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        # # Calculate the sine wave values for each time value
        # noise = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # # Another signal at other frequency
        # frequency = 5
        # sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # # Slow drift at 0.5Hz
        # frequency = .5
        # slow_drift = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # # Create signals from 3 different electrodes, they share in common a sine at 50Hz
        # signal_el1 = sine_wave + noise
        # signal_el2 = slow_drift + noise
        # signal_el3 = noise
        # plt.plot(time, signal_el1)
        # plt.title('electrode 1')
        # plt.show()
        # plt.plot(time, signal_el2)
        # plt.title('electrode 2')
        # plt.show()
        # plt.plot(time, signal_el3)
        # plt.title('electrode 3')
        # plt.show()
        # # Display as arrays and filter
        # array_of_synthetic_signals = np.array([[signal_el1, signal_el2, signal_el3],[signal_el1, signal_el2, signal_el3]])
        # signal_rereferenced = rereference_signal(array_of_synthetic_signals,[[0,1,2]], re_reference_method)
        # plt.plot(time, signal_rereferenced[0][0])
        # plt.title('Re-ref signal el1')
        # plt.show()
        # plt.plot(time, signal_rereferenced[0][1])
        # plt.title('Re-ref signal el2')
        # plt.show()
        # plt.plot(time, signal_rereferenced[0][2])
        # plt.title('Re-ref signal el3')
        # plt.show()
        
        del preictal_bufs, interictal_bufs    
        
# %% Notch
 
        """ Apply notch filter (low pass) to the noisy signal using signal.filtfilt """
        preictal_notch_filtered = notch_filter(preictal_rereferenced, sampling_frequency)
        interictal_notch_filtered = notch_filter(interictal_rereferenced, sampling_frequency)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Uncomment this section to check that the notch filter has been properly applied #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
        # import matplotlib.pyplot as plt
        # plt.psd(preictal_resampled[0][0], Fs=new_sampling_frequency)
        # plt.title('PSD unfiltered signal')
        # plt.show()
        # plt.psd(preictal_notch_filtered[0][0], Fs=new_sampling_frequency)
        # plt.title('PSD notch filtered signal @ 50Hz + harmonics')
        # plt.show()
        # # Check with synthetic signals
        # # Set the parameters of the sine wave
        # amplitude = 1 # Amplitude of the wave
        # frequency = 50 # Frequency of the wave in Hertz
        # phase = 0 # Phase shift of the wave in radians
        # sampling_rate = 500 # Number of samples per second
        # duration = 2 # Duration of the wave in seconds
        # # Create a numpy array of time values from 0 to duration
        # time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        # # Calculate the sine wave values for each time value
        # noise = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # # Plot it
        # plt.plot(time, noise)
        # plt.title('Noise at 50Hz')
        # plt.show()
        # # Another signal at other frequency
        # frequency = 5
        # sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # plt.plot(time, sine_wave)
        # plt.title('Sine at 5 Hz')
        # plt.show()
        # # Sum them 
        # synthetic_signal = noise + sine_wave
        # plt.plot(time, synthetic_signal)
        # plt.title('Sine @ 5Hz + Noise @ 50Hz')
        # plt.show()
        # # Filter
        # signal_filtered = notch_filter(synthetic_signal, sampling_rate)
        # plt.plot(time, signal_filtered)
        # plt.show()
        # # Display as arrays
        # array_of_synthetic_signals = np.array([[synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal],[synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal]])
        # signal_filtered = notch_filter(array_of_synthetic_signals, sampling_rate)
        # plt.plot(time, signal_filtered[0][0])
        # plt.title('Notch filtered signal')
        # plt.show()

        del preictal_rereferenced, interictal_rereferenced
        
# %% High pass (detrend)        

        """ High pass filter at 1Hz to remove slow drifts """
        preictal_high_filtered = slow_drift_filter(preictal_notch_filtered, sampling_frequency)
        interictal_high_filtered = slow_drift_filter(interictal_notch_filtered, sampling_frequency)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # # # # 
        # Uncomment this section to check that the slow drift (high-pass) filter has been properly applied #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  # # # # # # # #
        
        # import matplotlib.pyplot as plt
        # plt.psd(preictal_high_filtered[0][0], Fs=new_sampling_frequency)
        # plt.title('High pass filtered @ 1Hz')
        # plt.show()
        # # Check with synthetic signals
        # # Set the parameters of the sine wave
        # amplitude = 1 # Amplitude of the wave
        # frequency = 50 # Frequency of the wave in Hertz
        # phase = 0 # Phase shift of the wave in radians
        # sampling_rate = 500 # Number of samples per second
        # duration = 2 # Duration of the wave in seconds
        # # Create a numpy array of time values from 0 to duration
        # time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        # # Calculate the sine wave values for each time value
        # noise = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # # Plot it
        # plt.plot(time, noise)
        # plt.title('Noise at 50Hz')
        # plt.show()
        # # Another signal at other frequency
        # frequency = 5
        # sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # plt.plot(time, sine_wave)
        # plt.title('Sine at 5 Hz')
        # plt.show()
        # # Slow drift at 0.5Hz
        # frequency = .5
        # slow_drift = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # plt.plot(time, slow_drift)
        # plt.title('Sine at 1 Hz')
        # plt.show()
        # # Sum them 
        # synthetic_signal = noise + sine_wave + slow_drift
        # plt.plot(time, synthetic_signal)
        # plt.title('Sine @ 5Hz + Noise @ 50Hz + Slow drift @ 0.5Hz')
        # plt.show()
        # # Display as arrays and filter
        # array_of_synthetic_signals = np.array([[synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal],[synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal]])
        # signal_filtered = slow_drift_filter(array_of_synthetic_signals, sampling_rate)
        # plt.plot(time, signal_filtered[0][0])
        # plt.title('Slow drift filtered signal')
        # plt.show()
        
        del preictal_notch_filtered, interictal_notch_filtered
        
# %% Remove the padding 
        
        preictal_depadded = preictal_high_filtered[:,:,extension_length_samples:epoch_length_samples+extension_length_samples]
        interictal_depadded = interictal_high_filtered[:,:,extension_length_samples:epoch_length_samples+extension_length_samples]
        
        del preictal_high_filtered, interictal_high_filtered

# %% Resample

        # Resample the signal if needed
        if (sampling_frequency != 512) and (sampling_frequency != 500): # If the sampling rate is different than 512Hz or 500Hz, resample 
            if are_multiples(sampling_frequency,512): # If the sampling freq is a multiple of 512Hz, resample to 512Hz.
                new_sampling_frequency = 512
            if are_multiples(sampling_frequency,500): # If the sampling freq is a multiple of 500, resample to 500Hz.
                new_sampling_frequency = 500
            new_epoch_length_samples = round(epoch_length * new_sampling_frequency) # Epoch duration in terms of samples of the resampled signals
            preictal_resampled = signal.resample(preictal_depadded, new_epoch_length_samples, axis=-1)
            interictal_resampled = signal.resample(interictal_depadded, new_epoch_length_samples, axis=-1)
        else: # if the sampling was already 512 or 500, keep that signals as they are, only rename the variables 
            if sampling_frequency == 512:
                new_sampling_frequency = 512
                preictal_resampled = preictal_depadded
                interictal_resampled = interictal_depadded
            if sampling_frequency == 500:
                new_sampling_frequency = 500
                preictal_resampled = preictal_depadded
                interictal_resampled = interictal_depadded 
                
        del preictal_depadded, interictal_depadded # Remove variables that are no longer needed to lower computing expenses
      
# %%
        # Save the signal for all bands (unfiltered; all_bands)
        with h5py.File(output_directory+re_reference_method+'/all_bands/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_resampled)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/all_bands/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_resampled)
            data_file.close()
        del data_file
        
        """ Band decomposition """
        # DELTA
        preictal_delta_filtered = band_pass_filter(preictal_resampled, new_sampling_frequency, 'delta')
        interictal_delta_filtered = band_pass_filter(interictal_resampled, new_sampling_frequency, 'delta')
        
        with h5py.File(output_directory+re_reference_method+'/delta/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_delta_filtered)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/delta/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_delta_filtered)
            data_file.close()
        del data_file
            
        del preictal_delta_filtered, interictal_delta_filtered
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # Uncomment this section to check that the sband filter has been properly applied #
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
        # import matplotlib.pyplot as plt
        # # Check with synthetic signals
        # # Set the parameters of the sine wave
        # amplitude = 1 # Amplitude of the wave
        # frequency = 70 # Frequency of the wave in Hertz
        # phase = 0 # Phase shift of the wave in radians
        # sampling_rate = 500 # Number of samples per second
        # duration = 2 # Duration of the wave in seconds
        # # Create a numpy array of time values from 0 to duration
        # time = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        # # Calculate the sine wave values for each time value
        # noise = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # # Plot it
        # plt.plot(time, noise)
        # plt.title('Noise at 50Hz')
        # plt.show()
        # # Another signal at other frequency
        # frequency = 36
        # sine_wave = amplitude * np.sin(2 * np.pi * frequency * time + phase)
        # plt.plot(time, sine_wave)
        # plt.title('Sine at 9 Hz')
        # plt.show()
        # # Sum them 
        # synthetic_signal = noise + sine_wave
        # plt.plot(time, synthetic_signal)
        # plt.title('Sine @ 9Hz + Noise @ 50Hz')
        # plt.show()
        # # Filter
        # signal_filtered = band_pass_filter(synthetic_signal, sampling_rate,'high_gamma')
        # plt.plot(time, signal_filtered)
        # plt.show()
        # # Display as arrays
        # array_of_synthetic_signals = np.array([[synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal],[synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal,synthetic_signal]])
        # signal_filtered = band_pass_filter(array_of_synthetic_signals, sampling_rate,'high_gamma')
        # plt.plot(time, signal_filtered[0][0])
        # plt.title('High gamma filtered signal')
        # plt.show()
        # plt.psd(array_of_synthetic_signals[0][0],Fs=sampling_rate)
        # plt.show()
        # plt.psd(signal_filtered[0][0],Fs=sampling_rate)
        # plt.show()
        
        # THETA
        preictal_theta_filtered = band_pass_filter(preictal_resampled, new_sampling_frequency, 'theta')
        interictal_theta_filtered = band_pass_filter(interictal_resampled, new_sampling_frequency, 'theta')
        
        with h5py.File(output_directory+re_reference_method+'/theta/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_theta_filtered)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/theta/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_theta_filtered)
            data_file.close()
        del data_file
            
        del preictal_theta_filtered, interictal_theta_filtered
        
        # ALPHA
        preictal_alpha_filtered = band_pass_filter(preictal_resampled, new_sampling_frequency, 'alpha')
        interictal_alpha_filtered = band_pass_filter(interictal_resampled, new_sampling_frequency, 'alpha')
        
        with h5py.File(output_directory+re_reference_method+'/alpha/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_alpha_filtered)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/alpha/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_alpha_filtered)
            data_file.close()
        del data_file
            
        del preictal_alpha_filtered, interictal_alpha_filtered
        
        # BETA
        preictal_beta_filtered = band_pass_filter(preictal_resampled, new_sampling_frequency, 'beta')
        interictal_beta_filtered = band_pass_filter(interictal_resampled, new_sampling_frequency, 'beta')
        
        with h5py.File(output_directory+re_reference_method+'/beta/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_beta_filtered)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/beta/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_beta_filtered)
            data_file.close()
        del data_file
            
        del preictal_beta_filtered, interictal_beta_filtered
    
        # LOW GAMMA
        preictal_low_gamma_filtered = band_pass_filter(preictal_resampled, new_sampling_frequency, 'low_gamma')
        interictal_low_gamma_filtered = band_pass_filter(interictal_resampled, new_sampling_frequency, 'low_gamma')
        
        with h5py.File(output_directory+re_reference_method+'/low_gamma/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_low_gamma_filtered)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/low_gamma/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_low_gamma_filtered)
            data_file.close()
        del data_file
            
        del preictal_low_gamma_filtered, interictal_low_gamma_filtered
    
        # HIGH GAMMA
        preictal_high_gamma_filtered = band_pass_filter(preictal_resampled, new_sampling_frequency, 'high_gamma')
        interictal_high_gamma_filtered = band_pass_filter(interictal_resampled, new_sampling_frequency, 'high_gamma')
        
        with h5py.File(output_directory+re_reference_method+'/high_gamma/preictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=preictal_high_gamma_filtered)
            data_file.close()
        del data_file
        with h5py.File(output_directory+re_reference_method+'/high_gamma/interictal_'+patient_code+"-"+date+'-seizure-'+str(index+1)+'.hdf5', 'w') as data_file:
            data_file.create_dataset("dataset", data=interictal_high_gamma_filtered)
            data_file.close()
        del data_file
            
        del preictal_high_gamma_filtered, interictal_high_gamma_filtered

# %%
    
        print("               Pre-processing completed!")
    
    # Remove bad channels from the list of labels
    clean_labels = np.delete(clean_labels, bad_channels).tolist()
    
    """ Save metadata """
    metadata = {"patient": patient_code,
                "date": date,
                "number_of_seizures": number_of_seizures,
                "seizure_duration_ratio": seizure_time_ratio,
                "seizure_duration_mean": seizures.duration.mean(),
                "seizure_duration_std": seizures.duration.std(),
                "seizure_duration_min": seizures.duration.min(),
                "seizure_duration_max": seizures.duration.max(),
                "number_of_subclinical_seizures": len(scSeizures),
                "original_sampling_frequency": original_sampling_frequency,
                "sampling_frequency": new_sampling_frequency,
                "recording_duration": file.getFileDuration(),
                "sex": file.getSex(),
                "birth_date": file.getBirthdate(),
                "number_of_electrodes": len(clean_labels),
                "electrodes": clean_labels}
    with open(metadata_directory+patient_code+'-'+date+'.txt', "w") as fp:
        dump(metadata, fp)  # encode dict into JSON
    
    file.close()
    
