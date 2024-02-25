from scipy import signal
import numpy as np
import os
import pandas as pd


def clean_emg_file(files, data_folder):

    all_fat_epoch  = np.array([])  # empty list to store all fatigue epochs from all files
    all_nonf_epoch = np.array([])  # empty list to store all non-fatigue epochs from all files
    
    # iterate over all training files 
    for filename in files:  
        
        # find the file path
        file_path = os.path.join(data_folder, filename)
        
        # load the raw EMG file and assign variable names to the columns
        EMG = pd.read_csv(file_path, names=['time', 'data', 'label'])

        raw_data = EMG['data']
        
        # 1. Apply bandpass filter: 20-500 Hz
        #------------------------------------
        
        # Parameters
        Fs = 1926 # sampling frequency 
        Fnyquist = Fs/2 # Nyquist frequency
        lowCut = 20/Fnyquist # normalized low cutoff frequency
        highCut = 500/Fnyquist # normalized high cutoff frequency
        
        # design the Butterworth bandpass filter
        b, a = signal.butter(4, [lowCut, highCut], btype='band')
        
        # apply the filter
        band_data = signal.filtfilt(b, a, raw_data) # bandpassed data
        
        
        # 2. Apply notch filter: 50 Hz
        #--------------------------------
    
        # Parameters
        notch_freq = 50 # frequency 
        f_normalized = notch_freq/Fnyquist # normalized frequency
        bw = 0.5 # bandwidth
        Q = notch_freq / bw # quality foctor
    
        # desing the notch filter
        b, a = signal.iirnotch(f_normalized, Q, fs=Fs)
    
        # apply the filter
        filtered_data = signal.filtfilt(a, b, band_data) # filtered_data: bandpass + notch
        
        
        # 3. Capture the EMG envelope: RMS
        #-----------------------------------
        # compute the root mean square (RMS) value of the signal within a window which "slides across” the signal.
        
        # calculate rectified signal, take the absolute value
        rec_signal = np.abs(filtered_data - np.mean(filtered_data))
        
        f_cutoff = 20  # final cutoff frequency 
        
        # design 2nd order Butterworth low-pass filter
        b, a = signal.butter(2, f_cutoff * 1.25 / Fnyquist)
        
        # filter the rectified data
        filtered_rec = signal.filtfilt(b, a, rec_signal)
        
        
        # 4. Epoching
        #-----------------------------------
        # Segment the data into 1 sec epochs according to labels
        
        EMG['data'] = filtered_rec # update the EMG table
    
        # Remove the last element if it is a "0"
        if EMG['label'].iloc[-1] == 0:
            EMG.drop(index=len(EMG) - 1, inplace=True)
        
        # First separete the fatigue and non-faigue segments 
        
        # find indices corresponding to fatigue and non-fatigue segments
        fatigue_indices = np.where(EMG['label'] == 1)[0]
        nonFatigue_indices = np.where(EMG['label'] == 0)[0]
        
        # extract fatigue and non-fatigue segments from the data
        fatigueData = EMG['data'][fatigue_indices].values
        nonFatigueData = EMG['data'][nonFatigue_indices].values
    
        # determine the number of epochs to ensure each epoch will be 1 sec 
        n_fatEpochs = len(fatigueData) // Fs
        n_nonfEpochs = len(nonFatigueData) // Fs
    
        # Segment the fatigue data into epochs
        fatEpoch_EMG = np.empty((n_fatEpochs, Fs))
        for ep in range(n_fatEpochs):
            startIdx = ep * Fs
            endIdx = (ep + 1) * Fs
            fatEpoch_EMG[ep] = fatigueData[startIdx:endIdx]
    
        # Segment the non-fatigue data into epochs
        nonfEpoch_EMG = np.empty((n_nonfEpochs, Fs))
        for ep in range(n_nonfEpochs):
            startIdx = ep * Fs
            endIdx = (ep + 1) * Fs
            nonfEpoch_EMG[ep] = nonFatigueData[startIdx:endIdx]
        
        # store the data in a matrix for all epochs from all files
        if all_fat_epoch.size == 0:
            all_fat_epoch = fatEpoch_EMG
        else:    
            all_fat_epoch  = np.vstack((all_fat_epoch, fatEpoch_EMG))
        
        if all_nonf_epoch.size == 0:
            all_nonf_epoch = nonfEpoch_EMG
        else:    
            all_nonf_epoch = np.vstack((all_nonf_epoch, nonfEpoch_EMG))
        
    
    return all_fat_epoch, all_nonf_epoch