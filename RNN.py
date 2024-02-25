from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

def rnn_training(dl_feature_matrix_tra, dl_class_vector_tra):
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(dl_feature_matrix_tra, dl_class_vector_tra, test_size=0.2, random_state=42)
    
    # Reshape the input data for RNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val   = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Define the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=32, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping])
    
    # Evaluate the model
    val_accuracy = history.history['val_accuracy'][-1]
    print("Validation accuracy of the model:", val_accuracy)
    
    return model

def rnn_testing(dl_feature_matrix_test, dl_class_vector_test, model):
    
    # Reshape the input data for RNN
    dl_feature_matrix_test = dl_feature_matrix_test.reshape(dl_feature_matrix_test.shape[0], dl_feature_matrix_test.shape[1], 1)
    
    test_loss, test_accuracy = model.evaluate(dl_feature_matrix_test, dl_class_vector_test)
    print("Testing accuracy of RNN Model:", test_accuracy)
    
    ## ROC Curve
    
    # Predict probabilities for test data
    y_probs = model.predict(dl_feature_matrix_test)
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(dl_class_vector_test, y_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(16, 12))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - RNN')
    plt.legend(loc="lower right")
    plt.show()
    
    # Confusion Matrix
    
    # Predict probabilities for the test data
    y_pred_prob = model.predict(dl_feature_matrix_test)
    
    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(dl_class_vector_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - RNN')
    plt.show()
    
    return test_accuracy