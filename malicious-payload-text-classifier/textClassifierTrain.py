#%%

import numpy as np
import pandas as pd
import csv
import tensorflow as tf
from data_utils import Data
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Convolution1D, MaxPooling1D, Embedding, ThresholdedReLU, Dropout
from keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix

#%%

# Import the dataset (https://github.com/Morzeux/HttpParamsDataset)
df = pd.read_csv('./Morzeux_HttpParamsDataset_full.csv')

# Keep only the necessary columns
df = df[['payload','attack_type']]

# Assign classes to numbers
labels = {'norm':0, 'sqli':1, 'xss':2, 'path-traversal':3, 'cmdi':4}

# Convert string labels to integers and then stringify them
df['label'] = df['attack_type'].apply(lambda row: labels[row])
df['label'].apply(str)

# Split to train (60%), validation (20%), test (20%) sets
train, valid, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

# Export sets to .csv files
train.to_csv('./train_set.csv',header=False,index=False,quoting=csv.QUOTE_ALL,columns=['label','payload'])
valid.to_csv('./validation_set.csv',header=False,index=False,quoting=csv.QUOTE_ALL,columns=['label','payload'])
test.to_csv('./test_set.csv',header=False,index=False,quoting=csv.QUOTE_ALL,columns=['label','payload'])

#%%

# Set Variables
alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
input_size = 400
num_of_classes = len(labels)

# Set paths of the sets
train_data_path = "./train_set.csv"
valid_data_path = "./validation_set.csv"
test_data_path = "./test_set.csv"

# Model's name to be saved
save_model_file = "textModel.h5"

# Load training data
training_data = Data(data_source=train_data_path, alphabet=alphabet, input_size=input_size ,num_of_classes=num_of_classes)
training_data.load_data()
training_inputs, training_labels, _ = training_data.get_all_data()

# Load validation data
validation_data = Data(data_source=valid_data_path, alphabet=alphabet, input_size=input_size, num_of_classes=num_of_classes)
validation_data.load_data()
validation_inputs, validation_labels, _ = validation_data.get_all_data()

# Load test data
test_data = Data(data_source=test_data_path, alphabet=alphabet, input_size=input_size, num_of_classes=num_of_classes)
test_data.load_data()
test_inputs, test_labels, _ = test_data.get_all_data()

#%%
def create_model(input_size, alphabet_size, conv_layers, fc_layers, num_of_classes):
    # Input layer
    inputs = Input(shape=(input_size,), name='sent_input', dtype='int64')
            
    # Embedding layers
    x = Embedding(len(alphabet) + 1, 128, input_length=input_size)(inputs)

    # 1D Convolutional layers
    for cl in conv_layers:
        x = Convolution1D(cl[0], cl[1])(x)
        x = ThresholdedReLU(1e-6)(x)
        if cl[2] != -1:
            x = MaxPooling1D(cl[2])(x)

    x = Flatten()(x)

    # Fully Connected layers
    for fl in fc_layers:
        x = Dense(fl)(x)
        x = ThresholdedReLU(1e-6)(x)
        x = Dropout(0.5)(x)
            
    # Output layer
    predictions = Dense(num_of_classes, activation='softmax')(x)
    return inputs, predictions

#%%

# Model Architecture
conv_layers = [[256, 7, 3], [256, 7, 3], [256, 3, -1], [256, 3, -1], [256, 3, -1], [256, 3, 3]]
fc_layers = [1024, 1024]
loss = "categorical_crossentropy"

# Parameters
optimizer = "adam"
batch_size = 128
epochs = 7
metrics = [Precision(), Recall()]

# Build and compile model
inputs, outputs = create_model(input_size, len(alphabet), conv_layers, fc_layers, num_of_classes)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.summary()

#%%

# Training
model.fit(training_inputs, training_labels, validation_data=(
    validation_inputs, validation_labels), epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[])

# Export model
model.save(save_model_file)
print("Saved model to disk")

#%%

# Load exported model
loaded_model = load_model(save_model_file)
print("Loaded model from disk")

# Evaluate model on test data
loaded_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
score = loaded_model.evaluate(test_inputs, test_labels, batch_size=batch_size, verbose=1)

# Print metrics-results
for i in range(len(loaded_model.metrics_names)):
    print("%s: %.2f%%" % (loaded_model.metrics_names[i], score[i]*100))

#%%

y_test = np.argmax(test_labels, axis=1) # Convert one-hot to index
y_pred = loaded_model.predict(test_inputs)
y_pred = np.argmax(y_pred,axis=-1)

# Print more Results
print('Classification Report: ')
print(classification_report(y_test, y_pred, labels=list(labels.values()), target_names=list(labels.keys())))
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred, labels=list(labels.values())))

# %%