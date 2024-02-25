import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# SETUP

# Add the parent directory of the current script to the Python path
parent_dir = os.path.dirname(os.path.abspath(r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-2\Python\main.py'))
sys.path.append(parent_dir)

# Define the data folder
data_folder = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-2\EMG-data'


from utils import stratified_random_sampling
from preprocessing import clean_emg_file
from feature_engineering import extract_features_emg
from RNN import rnn_training, rnn_testing

 
# Divide the dataset into training and testing
training_files, testing_files = stratified_random_sampling(data_folder)


def EMG_linear_model():
    
    # TRAINING
    #--------------------------------------------------------------------------
    
    # preprocess the raw EMG data
    all_fat_epoch, all_nonf_epoch = clean_emg_file(training_files, data_folder)
    
    # extract and select features
    feature_matrix_tra, class_labels_tra = extract_features_emg(all_fat_epoch, all_nonf_epoch)
    
    # Create logistic regression model and balance to weights proportionally to classes for imbalanced dataset
    LR_model = LogisticRegression(class_weight='balanced')
    
    # Fit the model to the training data
    LR_model.fit(feature_matrix_tra, class_labels_tra)
    
    
    # TESTING
    #--------------------------------------------------------------------------
    
    # preprocess the raw EMG data
    all_fat_epoch, all_nonf_epoch = clean_emg_file(testing_files, data_folder)
    
    # extract and select features
    feature_matrix_test, class_labels_test = extract_features_emg(all_fat_epoch, all_nonf_epoch)
    
    # Conduct classification on the testing data
    predictions = LR_model.predict(feature_matrix_test)
    
    # Compute accuracy
    accuracy_lr = accuracy_score(class_labels_test, predictions)
    
    
    ## ROC Curve
    #----------------------   
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(class_labels_test, LR_model.predict_proba(feature_matrix_test)[:,1])
    
    # Plot ROC curve
    plt.figure(figsize=(16, 12))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Linear Model')
    plt.legend(loc='lower right')
    plt.show()
    
    ## Confusion Matrix
    #---------------------- 
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(class_labels_test, predictions)
    
    # Define label names
    label_names = ['Non-Fatigue', 'Fatigue']
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix for Linear Model')
    plt.show()

    return accuracy_lr


def EMG_deep_learning_model():
    
    # TRAINING
    #--------------------------------------------------------------------------
    
    # preprocess the raw EMG data
    all_fat_epoch, all_nonf_epoch = clean_emg_file(training_files, data_folder)
    
    # combine fatigue and non-fatigue data
    feature_matrix_dl = np.concatenate((all_fat_epoch, all_nonf_epoch), axis=0)

    # Create class labels array (0 for non-fatigue, 1 for fatigue)
    class_labels = np.concatenate((np.zeros(all_fat_epoch.shape[0]), np.ones(all_nonf_epoch.shape[0])))                                   

    # Shuffle the order of the classes to prevent bias
    num_samples = feature_matrix_dl.shape[0]  # Number of total samples/epochs
    random_indices = np.random.permutation(num_samples)  # Randomize the order of the indices
    dl_feature_matrix_tra = feature_matrix_dl[random_indices]
    dl_class_vector_tra = class_labels[random_indices]
    
    model = rnn_training(dl_feature_matrix_tra, dl_class_vector_tra)
    
    
    # TESTING
    #--------------------------------------------------------------------------
    
    # preprocess the raw EMG data
    all_fat_epoch, all_nonf_epoch = clean_emg_file(testing_files, data_folder)
    
    # combine fatigue and non-fatigue data
    feature_matrix_dl = np.concatenate((all_fat_epoch, all_nonf_epoch), axis=0)
    
    # Create class labels array (0 for non-fatigue, 1 for fatigue)
    class_labels = np.concatenate((np.zeros(all_fat_epoch.shape[0]), np.ones(all_nonf_epoch.shape[0])))

    # Shuffle the order of the classes to prevent bias
    num_samples = feature_matrix_dl.shape[0]  # Number of total samples/epochs
    random_indices = np.random.permutation(num_samples)  # Randomize the order of the indices
    dl_feature_matrix_test = feature_matrix_dl[random_indices]
    dl_class_vector_test = class_labels[random_indices]
    
    test_accuracy = rnn_testing(dl_feature_matrix_test, dl_class_vector_test, model)
    
    return test_accuracy

if __name__ == "__main__":
    
    accuracy_lr = EMG_linear_model() 
    test_accuracy = EMG_deep_learning_model()
    print('Accuracy of the linear model is:', accuracy_lr)
    print('Accuracy of the deep learning model is:', test_accuracy)