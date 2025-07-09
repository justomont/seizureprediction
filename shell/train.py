#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 13:43:11 2024

@author: justo
"""
# %% Import Libraries

import h5py
import sys
import time
import joblib
import numpy as np
import pandas as pd
from os import listdir
from json import loads
from sklearn import svm, dummy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

# Function to replace NaN with 0
def replace_nan_with_zero(x):
    return 0 if isinstance(x, float) and np.isnan(x) else x
vectorized_replace = np.vectorize(replace_nan_with_zero)

# %% Paths 

time_to_run_start = time.time()

# PC 
# metadata_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/metadata/unified/'
# single_metadata_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/metadata/single/'
# matrices_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/connectivity_matrices/'
# classifiers_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/classifiers/'
# results_directory = '/Users/justo/Library/CloudStorage/GoogleDrive-justo.montoya@upf.edu/Mi unidad/SPred/Code/cluster_ready/cohort21/results/'

# Cluster
metadata_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/unified/'
single_metadata_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/metadata/single/'
matrices_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/connectivity_matrices/'
classifiers_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/classifiers/'
results_directory = '/gpfs42/projects/lab_rrocamora/shared_data/jmontoya/seizure_pred/oneMIN/results/'

# %% Variables 

# Select the connectivity method, re-reference and band to be studied
methods = ['correlation', 'phase_lock', 'cross_correlation', 'PAC', 'phase_lag']
# Script is parallelized per connectivity method
slurm_array_id = int(sys.argv[1])
written_method = methods[slurm_array_id]
print(written_method)

# Select reference method
re_reference_method = 'bipolar'
print(re_reference_method)

# Bands to be considered
bands = ['all_bands', 'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']

# Some methods do not use all bands, for them remove those unused 
if written_method == 'PAC':
    bands = ['all_bands']
if written_method == 'phase_lock':
    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma']

# Initializ dataframe storing the results 
results = pd.DataFrame(columns=['patient','classification_method', 'value', 'metric','FC_method', 'fold', 'train_time'])

# Retrieve list of patients
list_file_metadata = listdir(metadata_directory)

# %% Loops 

#Iterate through all the patients
for file_metadata in list_file_metadata:
    
    for band in bands:
        print('\n')
        print(band)
        
# %% Load files
        
        # Open metadata of the patient
        with open(metadata_directory+file_metadata) as f: 
            data = f.read() 
            patient_metadata = loads(data) # reconstructing the data as a dictionary 
        patient_code = patient_metadata['patient']
        number_of_seizures = patient_metadata['number_of_seizures']
        print(patient_code, '#seizures:',number_of_seizures)
    
        """ Loading the connectivity matrices """
        preictal_data, interictal_data = [],[]
        # Since some patients may have only one day, we need to check is the date is a list or just a single date
        if type(patient_metadata['date']) == list:
            for patient_date in patient_metadata['date']:
                # Since there is a metadata file for each one of the days, we need to open that specific date.
                with open(single_metadata_directory+patient_metadata['patient']+'-'+patient_date+'.txt') as f: 
                    data = f.read() 
                    single_patient_metadata = loads(data) # reconstructing the data as a dictionary 
                for seizure_number in range(single_patient_metadata['number_of_seizures']):
                    # preictal
                    h5f = h5py.File(matrices_directory+written_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_preictal-seizure-'+str(seizure_number+1)+'.hdf5','r')
                    preictal_data_single = h5f['dataset'][:]
                    h5f.close()
                    preictal_data.append(preictal_data_single)
                    # interictal
                    h5f = h5py.File(matrices_directory+written_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_interictal-seizure-'+str(seizure_number+1)+'.hdf5','r')
                    interictal_data_single = h5f['dataset'][:]
                    h5f.close()
                    interictal_data.append(interictal_data_single)
        else: 
            patient_date = patient_metadata['date']
            for seizure_number in range(patient_metadata['number_of_seizures']):
                # preictal
                h5f = h5py.File(matrices_directory+written_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_preictal-seizure-'+str(seizure_number+1)+'.hdf5','r')
                preictal_data_single = h5f['dataset'][:]
                h5f.close()
                preictal_data.append(preictal_data_single)
                # interictal
                h5f = h5py.File(matrices_directory+written_method+'/'+re_reference_method+'/'+band+'/'+patient_metadata['patient']+'_interictal-seizure-'+str(seizure_number+1)+'.hdf5','r')
                interictal_data_single = h5f['dataset'][:]
                h5f.close()
                interictal_data.append(interictal_data_single)
        
        # These variables are lists of preictal and interictal data. Each item of the list is the data associated with each seizure. 
        preictal_data = np.array(preictal_data, dtype=object)
        interictal_data = np.array(interictal_data, dtype=object)
        
        # Same as for previous variables, but stacked
        preictal_data_all = np.vstack(preictal_data)
        interictal_data_all = np.vstack(interictal_data)
        
        # All the connectivity matrices together
        all_data = vectorized_replace(np.vstack([interictal_data_all,preictal_data_all]))
        
        # Labels of the states (0 interictal; 1 preictal)
        all_data_labels = np.array([0]*len(interictal_data_all) + [1]*len(preictal_data_all))
        
        del interictal_data_single, preictal_data_single, preictal_data, preictal_data_all, interictal_data_all, interictal_data, data, f, h5f, number_of_seizures, seizure_number, patient_metadata,  patient_date

        
        """ Uncomment this section to plot a SINGLE connectivity matrix of each state """
        # sns.set_style("ticks")
        # sns.color_palette("mako", as_cmap=True)
        # plt.imshow(preictal_data_single[int(len(preictal_data_single)/2)], vmax=1, vmin=0)
        # plt.title('preictal connectivity matrix')
        # plt.show()
        # plt.imshow(interictal_data_single[int(len(interictal_data_single)/2)], vmax=1, vmin=0)
        # plt.title('interictal connectivity matrix')
        # plt.show()
        
    
    # %% K-Fold of the Functional Connectivity Matrices 
    
        print('\nTrainning using Functional Connectivity Matrices:')    
    
        # Flatten the matrices so that they can be used with sklearn
        all_data_flat = all_data.reshape(all_data.shape[0],-1) 
    
        # Split data in folds
        number_of_folds = 10
        skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(skf.split(all_data_flat, all_data_labels)):
            
            # print(f"Fold {i}/{number_of_folds}")
            print("Fold "+str(i+1)+"/"+str(number_of_folds))
            
            # Split train and test for each fold
            X_train, X_test, y_train, y_test = all_data_flat[train_index], all_data_flat[test_index], all_data_labels[train_index], all_data_labels[test_index]
            
            
            """ Support Vector Machine """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted SVM classifier fit
            SVMclassifier = svm.SVC(kernel='rbf', gamma='auto', probability=True)
            SVMclassifier.fit(X_train, y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = SVMclassifier.predict(X_test)
            y_test_prob = SVMclassifier.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class

            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)
            
            # Store the classification metrics, fold and times of training of the classifier
            results.loc[len(results.index)] = [patient_code,'FC_SVM',accuracy,'accuracy', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',recall[0],'recall_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',recall[1],'recall_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',precision[0],'precision_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',precision[1],'precision_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',auroc,'AUROC', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',fpr,'fpr', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_SVM',tpr,'tpr', written_method+'_'+band, i+1, time_to_train]
            
            # Save the model to use it later
            joblib.dump(SVMclassifier, classifiers_directory + re_reference_method +'/' + '-'.join([patient_code, written_method, band, 'FC_SVM', str(i+1)]) + '.sav')
            
            # Delete model to save space
            del SVMclassifier
            print(' - SVM')
            
            
            """ Random forest """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted Random Forest classifier
            RFclassifier = RandomForestClassifier(class_weight="balanced")
            RFclassifier.fit(X_train, y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = RFclassifier.predict(X_test)
            y_test_prob = RFclassifier.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class

            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)
                  
            # Store the classification metrics, fold and times of training of the classifier
            results.loc[len(results.index)] = [patient_code,'FC_RF',accuracy,'accuracy', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',recall[0],'recall_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',recall[1],'recall_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',precision[0],'precision_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',precision[1],'precision_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',auroc,'AUROC', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',fpr,'fpr', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'FC_RF',tpr,'tpr', written_method+'_'+band, i+1, time_to_train]
            
            # Save the model to use it later
            joblib.dump(RFclassifier, classifiers_directory + re_reference_method +'/'+ '-'.join([patient_code, written_method, band, 'FC_RF', str(i+1)]) + '.sav')
            
            # Delete model to save space
            del RFclassifier
            print(' - RandF')
            
            
            """ Dummy classifier: Random stratified """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted SVM classifier
            Dummyclassifier = dummy.DummyClassifier(strategy='stratified')
            Dummyclassifier.fit(X_train, y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = Dummyclassifier.predict(X_test)
            y_test_prob = Dummyclassifier.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
            
            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)
            
            # Store the classification metrics, fold and times of training of the classifier
            results.loc[len(results.index)] = [patient_code,'stratified',accuracy,'accuracy', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'stratified',recall[0],'recall_interictal','Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'stratified',recall[1],'recall_preictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'stratified',precision[0],'precision_interictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'stratified',precision[1],'precision_preictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'stratified', auroc, 'AUROC', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'stratified', fpr, 'fpr', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'stratified', tpr, 'tpr', 'Dummy', i+1, time_to_train]
            
            
            # Save the model to use it later
            joblib.dump(Dummyclassifier, classifiers_directory + re_reference_method +'/'+ '-'.join([patient_code, 'DummyStratified', str(i+1)]) + '.sav')
            
            print(' - Dummy')
            
            """ Dummy: All interictal """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted SVM classifier
            Dummyclassifier = dummy.DummyClassifier(strategy='constant', constant=0)
            Dummyclassifier.fit(X_train, y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = Dummyclassifier.predict(X_test)
            y_test_prob = Dummyclassifier.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
            
            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)
            
            # Store the classification metrics, fold and times of training of the classifier
            results.loc[len(results.index)] = [patient_code,'all_interictal',accuracy,'accuracy', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_interictal',recall[0],'recall_interictal','Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_interictal',recall[1],'recall_preictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_interictal',precision[0],'precision_interictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_interictal',precision[1],'precision_preictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'all_interictal', auroc, 'AUROC', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'all_interictal', fpr, 'fpr', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'all_interictal', tpr, 'tpr', 'Dummy', i+1, time_to_train]
            
            # Save the model to use it later
            joblib.dump(Dummyclassifier, classifiers_directory+ re_reference_method +'/' + '-'.join([patient_code, 'DummyAllInterictal', str(i+1)]) + '.sav')
            
            print(' - Dummy Inter')
            
            """ Dummy: All preictal """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted SVM classifier
            Dummyclassifier = dummy.DummyClassifier(strategy='constant', constant=1)
            Dummyclassifier.fit(X_train, y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = Dummyclassifier.predict(X_test)
            y_test_prob = Dummyclassifier.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
            
            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)

            # Store the classification metrics, fold and times of training of the classifier
            results.loc[len(results.index)] = [patient_code,'all_preictal',accuracy,'accuracy', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_preictal',recall[0],'recall_interictal','Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_preictal',recall[1],'recall_preictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_preictal',precision[0],'precision_interictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'all_preictal',precision[1],'precision_preictal', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'all_preictal', auroc, 'AUROC', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'all_preictal', fpr, 'fpr', 'Dummy', i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code, 'all_preictal', tpr, 'tpr', 'Dummy', i+1, time_to_train]
            
            # Save the model to use it later
            joblib.dump(Dummyclassifier, classifiers_directory+ re_reference_method +'/' + '-'.join([patient_code, 'DummyAllPreictal', str(i+1)]) + '.sav')
        
            print(' - Dummy Pre')
            
        del skf, i, train_index, test_index, X_train, X_test, y_test, y_train, y_test_prediction, accuracy, recall, precision            

    # %% EIGENVALUE DECOMPOSITION
        
        # Eigenvalue decompose the Connectivity Matrices
        all_data_eigvalues, all_data_eigvects = np.linalg.eig(all_data)
        all_data_leading_eigenvector = all_data_eigvects[:,:,0] # (linalg.eig already returns the eigvals ordered from max to min, same for eigvects, therefore the element 0 is the eigenvector)
        
        del all_data, all_data_eigvects
        
    #%% EigenValues
    
        print('\nTrainning using Eigenvalue Decomposition:')    
    
        # Split data in folds
        skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True)
        
        for i, (train_index, test_index) in enumerate(skf.split(all_data_eigvalues, all_data_labels)):
            
            # print(f"Fold {i}/{number_of_folds}")
            print("Fold "+str(i+1)+"/"+str(number_of_folds))
            
            # Split train and test for each fold
            X_train, X_test, y_train, y_test = all_data_eigvalues[train_index], all_data_eigvalues[test_index], all_data_labels[train_index], all_data_labels[test_index]
            
            
            """ Support Vector Machine """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted SVM classifier
            SVMclassifier = svm.SVC(kernel='rbf', gamma='auto', probability=True)
            SVMclassifier.fit(np.real(X_train), y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = SVMclassifier.predict(np.real(X_test))
            y_test_prob = SVMclassifier.predict_proba(np.real(X_test))[:, 1]  # Get the probabilities for the positive class

            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)
                
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',accuracy,'accuracy', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',recall[0],'recall_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',recall[1],'recall_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',precision[0],'precision_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',precision[1],'precision_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',auroc,'AUROC', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',fpr,'fpr', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_SVM',tpr,'tpr', written_method+'_'+band, i+1, time_to_train]
            
            # Save the model to use it later
            joblib.dump(SVMclassifier, classifiers_directory+ re_reference_method +'/' + '-'.join([patient_code, written_method, band, 'Eig_SVM', str(i+1)]) + '.sav')
            
            del SVMclassifier
            print(' - SVM')
            
            """ Random Forest """
            
            # Measure time (start) it takes to train a classifier
            classifier_trainning_start = time.time()
            
            # Weighted RF classifier
            RFclassifier = RandomForestClassifier(class_weight="balanced")
            RFclassifier.fit(np.real(X_train), y_train)
            
            # Measure time (end) it takes to train a classifier
            classifier_trainning_end = time.time()
            time_to_train = classifier_trainning_end - classifier_trainning_start
            
            # Test the classifier
            y_test_prediction = RFclassifier.predict(np.real(X_test))
            y_test_prob = RFclassifier.predict_proba(np.real(X_test))[:, 1]

            # Calculate the accuracy, recall and precision
            accuracy = balanced_accuracy_score(y_test, y_test_prediction)
            recall = recall_score(y_test, y_test_prediction,average=None);
            precision = precision_score(y_test, y_test_prediction,average=None);

            # Calculate the ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            auroc = roc_auc_score(y_test, y_test_prob)
            
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',accuracy,'accuracy', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',recall[0],'recall_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',recall[1],'recall_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',precision[0],'precision_interictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',precision[1],'precision_preictal', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',auroc,'AUROC', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',fpr,'fpr', written_method+'_'+band, i+1, time_to_train]
            results.loc[len(results.index)] = [patient_code,'EigVals_RF',tpr,'tpr', written_method+'_'+band, i+1, time_to_train]
            
            # Save the model to use it later
            joblib.dump(RFclassifier, classifiers_directory+ re_reference_method +'/' + '-'.join([patient_code, written_method, band, 'Eig_RF', str(i+1)]) + '.sav')
            
            del RFclassifier
            print(' - RandF')
    
        del skf, i, train_index, test_index,  X_train, X_test, y_test, y_train, y_test_prediction, accuracy, recall, precision, number_of_folds

    #%% LEIDA    
      
        # print('LEIDA:')     
      
        # """ LEIDA """
        # # K-means clustering of the leading eigenvectors
        # number_of_states = 2
        
        # for i, (train_index, test_index) in enumerate(skf.split(all_data_leading_eigenvector, all_data_labels)):
        #     print(f"Fold {i}/{number_of_folds}")
            
        #     kmeans = KMeans(n_clusters=number_of_states, init='random').fit(np.real(all_data_leading_eigenvector[train_index]))
            
        #     # Predict the state based on the clusters
        #     kmeans_predictions = kmeans.labels_
        #     accuracy = balanced_accuracy_score(all_data_labels[train_index], kmeans_predictions)
        #     recall = recall_score(all_data_labels[train_index], kmeans_predictions, average=None)
        #     precision = precision_score(all_data_labels[train_index], kmeans_predictions, average=None)
            
        #     # Fractional occupancies
        #     fractional_occupancies = [(kmeans_predictions == state_number).sum()/len(kmeans_predictions) for state_number in range(number_of_states)]
            
        #     results.loc[len(results.index)] = [patient_code,'LEIDA',accuracy,'accuracy', written_method+'_'+band]
        #     results.loc[len(results.index)] = [patient_code,'LEIDA',recall[0],'recall_interictal', written_method+'_'+band]
        #     results.loc[len(results.index)] = [patient_code,'LEIDA',recall[1],'recall_preictal', written_method+'_'+band]
        #     results.loc[len(results.index)] = [patient_code,'LEIDA',precision[0],'precision_interictal', written_method+'_'+band]
        #     results.loc[len(results.index)] = [patient_code,'LEIDA',precision[1],'precision_preictal', written_method+'_'+band]

#%% RESULTS

# plt.figure(figsize=(9, 6),dpi=200)
# sns.set_style('ticks')
# # plt.hlines(0.5, xmin=-2, xmax=6, linestyles='dashed', colors='black')
# sns.barplot(data=results[results.metric=='accuracy'], x="FC_method", y="value", hue='classification_method')
# plt.xticks(rotation=70)
# plt.title('Balanced ccuracy ' + written_method )
# plt.show()

#%% SAVE RESULTS
import pickle 
with open(results_directory + re_reference_method+'/' + written_method+'_'+'results.pkl', 'wb') as f:
    pickle.dump(results, f)
    
time_to_run_end = time.time()
time_to_run = time_to_run_end - time_to_run_start
print('\nTotal time to run:', time_to_run)