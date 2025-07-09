#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from json import loads, dump
from os import listdir

# All subs
# directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/metadata/single/'
# output_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/metadata/unified/'
# Known outcome
directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/single/'
output_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/unified/'

files = listdir(directory)
unique_patients = []
for file_name in listdir(directory): 
    
    patient_code = file_name.split("-")[0]
    if patient_code not in unique_patients:
        unique_patients.append(patient_code)
    
for patient in unique_patients:
    
    patient_file_names = [file_name for file_name in files if patient in file_name]
    
    # Is there's only one file, just rename it
    if len(patient_file_names)==1:
        with open(directory+patient_file_names[0]) as f: 
            data = f.read() 
            metadata = loads(data) # reconstructing the data as a dictionary 
        metadata['files'] = patient_file_names[0]
        with open(output_directory+patient+'.txt', "w") as fp:
                dump(metadata, fp)  # encode dict into JSON
    
    # If there is more than a file per patientm combine it 
    else:
        dates = []
        n_seiz = []
        original_sampling_frequency = []
        sampling_frequency = []
        recording_duration = []
        electrodes = []
        
        for patient_file_name in patient_file_names:
            # Reading the metadata .txt files
            with open(directory+'/'+patient_file_name) as f: 
                data = f.read() 
                metadata = loads(data) # reconstructing the data as a dictionary 
            
            dates.append(metadata['date'])
            n_seiz.append(metadata['number_of_seizures'])
            original_sampling_frequency.append(metadata['original_sampling_frequency'])
            sampling_frequency.append(metadata['sampling_frequency'])
            recording_duration.append(metadata['recording_duration'])
            electrodes.append(metadata['electrodes'])
        
        unified_metadata = {"patient": patient,
                    "date": dates,
                    "number_of_seizures": sum(n_seiz),
                    "original_sampling_frequency": original_sampling_frequency,
                    "sampling_frequency": sampling_frequency,
                    "recording_duration": sum(recording_duration),
                    "sex": metadata['sex'],
                    "birth_date": metadata['birth_date'],
                    "files": patient_file_names,
                    "electrodes":electrodes}
        
        with open(output_directory+patient+'.txt', "w") as fp:
                dump(unified_metadata, fp)  # encode dict into JSON