import os
import random

def stratified_random_sampling(data_folder):
    
    # exclude empty files
    exclude_files = ['U1Ex1Rep3.csv', 'U3Ex2Rep3.csv']
    
    # Generate list of all file names based on naming convention
    file_names = [filename for filename in os.listdir(data_folder) if filename not in exclude_files]
    
    
    # Define the strata
    strata = [
        'Ex1Rep1', 'Ex1Rep2', 'Ex1Rep3',
        'Ex2Rep1', 'Ex2Rep2', 'Ex2Rep3',
        'Ex3Rep1', 'Ex3Rep2', 'Ex3Rep3'
    ]
    
    num_files = len(file_names) # number of files
    
    # Number of files for training (80%) and testing (20%)
    num_trai = int(num_files * 0.8)
    
    # Calculate the number of files per stratum in the training
    n_str_trai = num_trai // len(strata)
    
    
    # Randomly select files for training from each stratum
    training_files = [random.sample([file_name for file_name in file_names if stratum in file_name], n_str_trai) 
                      for stratum in strata]
    
    # Flatten the list of lists
    training_files = [file_name for sublist in training_files for file_name in sublist]
    
    # Determine testing files as the remaining files
    testing_files = list(set(file_names) - set(training_files))
    
    return training_files, testing_files
    