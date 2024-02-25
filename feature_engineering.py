import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler

# Extraxt and select 5 features and generate a feature matrix

# Time Domain:
# 1. Mean Absolute Values
# 2. Variance
# 3. Root Mean Square

# Frequency Domain:
# 4. Mean Power Frequency (MNF) 

# Time-Frequency Domain:
# 5. Spectral Centroid


def extract_features_emg(all_fat_epoch, all_nonf_epoch):

    # Time Domain Features
    #---------------------------------------------------------------------
    
    # 1. Compute mean absolute value
    Mav_fat = np.mean(np.abs(all_fat_epoch), axis=1)
    Mav_nonf = np.mean(np.abs(all_nonf_epoch), axis=1)
    
    # 2. Compute Variance
    Var_fat = np.var(all_fat_epoch, axis=1)
    Var_nonf = np.var(all_nonf_epoch, axis=1)
    
    # 3. Compute Root Mean Square (RMS)
    Rms_fat = np.sqrt(np.mean(all_fat_epoch**2, axis=1))
    Rms_nonf = np.sqrt(np.mean(all_nonf_epoch**2, axis=1))
    
    
    # Frequency Domain Features
    #------------------------------------------------------------------------
    
    Fs = 1926  # sampling frequency
    
    # Compute the power spectral density (PSD) first
    frequencies_f, psd_f = welch(all_fat_epoch, fs=Fs, nperseg=Fs, axis=1) # fatigue 
    frequencies_n, psd_n = welch(all_nonf_epoch, fs=Fs, nperseg=Fs, axis=1) # non-fatigue
    
    # 4. Compute Mean Power Frequency (MNF) 
    Mnf_fat  = np.sum(frequencies_f * psd_f, axis = 1) / np.sum(psd_f, axis= 1)
    Mnf_nonf = np.sum(frequencies_n * psd_n, axis = 1) / np.sum(psd_n, axis = 1)
    

    # Time-Frequency Domain
    #--------------------------------------------------------------------
    
    # 5. Calculate the Spectral Centroid
    Sc_fat  = np.sum(frequencies_f * psd_f, axis = 1) / np.sum(psd_f, axis = 1)
    Sc_nonf = np.sum(frequencies_n * psd_n, axis = 1) / np.sum(psd_n, axis = 1)
    
    
    # Store and organize all the features in a matrix for fatigue and non-fatigue classes
    feature_matrix_fat = np.array([Mav_fat, Var_fat, Rms_fat, Mnf_fat, Sc_fat]).T  
    feature_matrix_nonf = np.array([Mav_nonf, Var_nonf, Rms_nonf, Mnf_nonf, Sc_nonf]).T  
    
    # Concatenate feature matrices for fatigue and non-fatigue classes
    feature_matrix = np.concatenate((feature_matrix_fat, feature_matrix_nonf), axis=0)
    
    # Create class labels array (0 for non-fatigue, 1 for fatigue)
    class_labels = np.concatenate((np.zeros(feature_matrix_fat.shape[0]), np.ones(feature_matrix_nonf.shape[0])))
    
    
    # Shuffle the order of the classes to prevent bias
    num_samples = feature_matrix.shape[0]  # Number of total samples/epochs
    random_indices = np.random.permutation(num_samples)  # Randomize the order of the indices
    feature_matrix = feature_matrix[random_indices]
    class_labels = class_labels[random_indices]
    
    standard_scaler = StandardScaler()
    feature_matrix = standard_scaler.fit_transform(feature_matrix)
    
    return feature_matrix, class_labels
    
    