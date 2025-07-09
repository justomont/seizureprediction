#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:38:51 2023

@author: justo
"""
import os
import sys
import h5py
from os import listdir
from json import loads
from connectivity_methods import connectivity_analysis

# To run it through the cluster
# sbatch --array=0-9 connectivity_analysis.sh    but change the number for the number of patients in metadata unified

slurm_array_id = int(sys.argv[1])

""" Variables for the initialization of the loops """
re_reference_method = "bipolar"
# re_reference_method = "laplacian"
# re_reference_method = "monopolar"


# connectivity_method ='correlation'
# connectivity_method ='cross_correlation'
# connectivity_method ='phase_lag'
# connectivity_method ='phase_lock'
connectivity_method ='PAC'
# connectivity_method ='spectral_coherence_real'
# connectivity_method ='spectral_coherence_imag'
# connectivity_method = 'granger_causality'


bands = ['all_bands', 'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']

if connectivity_method == 'PAC': 
    from connectivity_methods import PAC
    method = PAC
    bands = ['all_bands']
    
if connectivity_method == 'spectral_coherence_real': 
    from connectivity_methods import spectral_coherence_real
    method = spectral_coherence_real

if connectivity_method == 'spectral_coherence_imag':
    from connectivity_methods import spectral_coherence_imag
    method = spectral_coherence_imag
    
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

if connectivity_method == 'granger_causality': 
    from connectivity_methods import granger_causality
    method = granger_causality


""" File paths """
# Local
# metadata_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/metadata/unified/'
# single_metadata_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/metadata/single/'
# epochs_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/epochs/'
# output_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/test/'

# Cluster: 
metadata_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/unified/'
single_metadata_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/single/'
epochs_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/epochs/'
output_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/connectivity_matrices/'

# file_metadata = listdir(metadata_directory)[slurm_array_id] # Parallelize per patient     
test = ['SDA.txt', 'VBM.txt']
file_metadata = test[slurm_array_id] # Parallelize per patient    

# Open the metadata file to check if there is more than one date recorded
with open(metadata_directory+file_metadata) as f: 
    data = f.read() 
    patient_metadata = loads(data) # reconstructing the data as a dictionary 

print(patient_metadata['patient'])

for band in bands:

    print(re_reference_method, connectivity_method, band)
    
    # If it has more than one date:
    if type(patient_metadata['date']) == list: 
        
        previous_file_seizure_number = 0
        
        for patient_date in patient_metadata['date']: # iterate through the dates
    
            # Since there is a metadata file for each one of the days, we need to open that specific date.
            with open(single_metadata_directory+patient_metadata['patient']+'-'+patient_date+'.txt') as f: 
                data = f.read() 
                single_patient_metadata = loads(data) # reconstructing the data as a dictionary 
                
            for seizure_number in range(single_patient_metadata['number_of_seizures']): # iterate through seizures
    
                # Open the PREICTAL data associated with that day and seizure
                h5f = h5py.File(epochs_directory+re_reference_method+'/'+band+'/preictal_'+patient_metadata['patient']+'-'+patient_date+'-seizure-'+str(seizure_number+1)+'.hdf5','r')
                preictal_data_single = h5f['dataset'][:] # data associated with only one seizure
                h5f.close()
                del h5f
                
                # Compute connectivity matrices
                preictal_CM = connectivity_analysis(preictal_data_single, method, fs=single_patient_metadata['sampling_frequency']);
                
                del preictal_data_single # delete variable
                
                # Save preictal connectivity matrices
                isExist = os.path.exists(output_directory+connectivity_method+'/'+re_reference_method+'/'+band)
                if not isExist:
                    os.makedirs(output_directory+connectivity_method+'/'+re_reference_method+'/'+band)
                    with h5py.File(output_directory+connectivity_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_preictal-seizure-'+str(seizure_number+previous_file_seizure_number+1)+'.hdf5', 'w') as data_file:
                        data_file.create_dataset("dataset", data=preictal_CM)
                        data_file.close()
                else:
                    with h5py.File(output_directory+connectivity_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_preictal-seizure-'+str(seizure_number+previous_file_seizure_number+1)+'.hdf5', 'w') as data_file:
                        data_file.create_dataset("dataset", data=preictal_CM)
                        data_file.close()
                
                del data_file, preictal_CM # delete used variables
                
                # INTERICTAL
                h5f = h5py.File(epochs_directory+re_reference_method+'/'+band+'/interictal_'+patient_metadata['patient']+'-'+patient_date+'-seizure-'+str(seizure_number+1)+'.hdf5','r')
                interictal_data_single = h5f['dataset'][:]
                h5f.close()
                del h5f
                
                # Compute connectivity matrices
                interictal_CM = connectivity_analysis(interictal_data_single, method, fs=single_patient_metadata['sampling_frequency']);
            
                del interictal_data_single # delete used variable
                
                # Save interictal connectivity matrices
                with h5py.File(output_directory+connectivity_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_interictal-seizure-'+str(seizure_number+previous_file_seizure_number+1)+'.hdf5', 'w') as data_file:
                    data_file.create_dataset("dataset", data=interictal_CM)
                    data_file.close()
                
                del data_file, interictal_CM
            
            previous_file_seizure_number += single_patient_metadata['number_of_seizures']
        
                
    # in the case that there is only one date per patient, open just that date          
    else: 
        patient_date = patient_metadata['date']
        for seizure_number in range(patient_metadata['number_of_seizures']):
            
            # Open the PREICTAL data associated with that day and seizure
            h5f = h5py.File(epochs_directory+re_reference_method+'/'+band+'/preictal_'+patient_metadata['patient']+'-'+patient_date+'-seizure-'+str(seizure_number+1)+'.hdf5','r')
            preictal_data_single = h5f['dataset'][:] # data associated with only one seizure
            h5f.close()
            del h5f
            
            # Compute connectivity matrices
            preictal_CM = connectivity_analysis(preictal_data_single, method, fs=patient_metadata['sampling_frequency']);
            
            del preictal_data_single # delete variable
            
            # Save preictal connectivity matrices
            isExist = os.path.exists(output_directory+connectivity_method+'/'+re_reference_method+'/'+band)
            if not isExist:
                os.makedirs(output_directory+connectivity_method+'/'+re_reference_method+'/'+band)
                with h5py.File(output_directory+connectivity_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_preictal-seizure-'+str(seizure_number+1)+'.hdf5', 'w') as data_file:
                    data_file.create_dataset("dataset", data=preictal_CM)
                    data_file.close()
            else:
                with h5py.File(output_directory+connectivity_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_preictal-seizure-'+str(seizure_number+1)+'.hdf5', 'w') as data_file:
                    data_file.create_dataset("dataset", data=preictal_CM)
                    data_file.close()
            
            del data_file, preictal_CM # delete used variables
            
            # INTERICTAL
            h5f = h5py.File(epochs_directory+re_reference_method+'/'+band+'/interictal_'+patient_metadata['patient']+'-'+patient_date+'-seizure-'+str(seizure_number+1)+'.hdf5','r')
            interictal_data_single = h5f['dataset'][:]
            h5f.close()
            del h5f
            
            # Compute connectivity matrices
            interictal_CM = connectivity_analysis(interictal_data_single, method, fs=patient_metadata['sampling_frequency']);
            
            del interictal_data_single # delete used variable
            
            # Save interictal connectivity matrices
            with h5py.File(output_directory+connectivity_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_interictal-seizure-'+str(seizure_number+1)+'.hdf5', 'w') as data_file:
                data_file.create_dataset("dataset", data=interictal_CM)
                data_file.close()
                
            del data_file, interictal_CM

print(f"Connectivity analysis for {patient_metadata['patient']} with method {connectivity_method} completed.")
