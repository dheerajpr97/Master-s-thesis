''' Example call: python -m scripts.train_class from /cdtemp/dheerajr/Notebooks/Main '''

import tensorflow as tf
import keras 
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split 
from scripts.model import ResNet34 
from scripts.data import CustomDataGen
from statistics import mean
import os


# Load the training dataset.JSON as a Pandas dataframe
df = pd.read_json('/cdtemp/dheerajr/Datasets/SIFT/Train/Train_hist_filt/Train_histfil_5000.json')

# Prepare the training data and the corresponding labels
patch_width = (np.int32(mean(list(np.round(df['Keypoint Size']))))+15) # define the fixed patch width
train_df, val_df = train_test_split(df, random_state=1, test_size=0.25, shuffle=True)
height = patch_width*2 # height of the patch
width = patch_width*2 # width of the patch
batch_size = 128 # batch size
epochs = 200 # number of epochs
input_shape = (height, width, 3) # set the input shape in the form of (Height, Width, Channel)

traingen = CustomDataGen(train_df,
                         X_col={'path':'Filename', 'cord':'Co-ordinates'},
                         y_col={'size': 'Keypoint Size'},
                         batch_size=batch_size, input_size=input_shape,
                         patch_width=patch_width)

valgen = CustomDataGen(val_df,
                         X_col={'path':'Filename', 'cord':'Co-ordinates'},
                         y_col={'size': 'Keypoint Size'},
                         batch_size=batch_size, input_size=input_shape,
                         patch_width=patch_width)


# Define the loss function, metric to monitor, optimizer with the learning rate 
rmse = tf.keras.metrics.RootMeanSquaredError() # metric for evaluation
loss = tf.keras.losses.mean_absolute_error # set the loss function as MAE
learning_rate=0.0001 # setting the learning rate
optimizer = tf.keras.optimizers.RMSprop(learning_rate, momentum=0.8) # choosing the optimizer as RMSprop


# Saving the best checkpoint with minimum RMSE value
checkpoint_filepath = 'checkpoints/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
    monitor='val_root_mean_squared_error', mode='min', save_best_only=True)
model_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=10, mode='min')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.5, patience=5, 
                                                 min_lr=0.00001) #'val_root_mean_squared_error'

# Setting the tensorboard callback to visualize the training & validation plots
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # set the directory to log the training
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1) # activate tensorboard

# Instantiate, compile & run the model
model = ResNet34(shape=input_shape)
model.compile(loss=loss, optimizer=optimizer, metrics=[rmse])

history = model.fit(traingen, 
                    validation_data=valgen,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[model_checkpoint_callback, tensorboard_callback, reduce_lr, model_earlystop]) #, 

# Save the trained model for further evaluation and use
model.load_weights('checkpoints/')
moddir = os.path.join("saved_models/Training_class/SIFT/", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
model.save(moddir, save_format='h5')
