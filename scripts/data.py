import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size=32,
                 input_size=(58, 58, 3),
                 patch_width=30,
                 shuffle=True,
                 color=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.patch_width = patch_width
        self.color = color
        self.n = len(self.df)

    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_input(self, path, cord):
        
        if self.color==True:
            image = cv2.imread(path)  # Reads the image in BGR form
        else:
            image = cv2.imread(path, 0) # Reads the image in Grayscale
         
        # Adds a border to the image to account for keypoints on the border
        image = cv2.copyMakeBorder(image, self.patch_width*2, self.patch_width*2, self.patch_width*2, self.patch_width*2, 
                               borderType=cv2.BORDER_REPLICATE)         
    
        x, y = cord[1]+self.patch_width*2, cord[0]+self.patch_width*2  
        
        # Extracts the patch around the keypoint
        patch = np.array(image[x-self.patch_width:x+self.patch_width,y-self.patch_width:y+self.patch_width])
        patch = patch.reshape(self.patch_width*2, self.patch_width*2, self.input_size[-1])
      
        
        return patch/255 # Return the normalized keypoint patch

    
    def __get_output(self, label):        
        return np.array(label)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']]
        cord_batch = batches[self.X_col['cord']]
        
        size_batch = batches[self.y_col['size']]

        X_batch = np.asarray([self.__get_input(x, y) for x, y in zip(path_batch, cord_batch)])
        y_batch = np.asarray([self.__get_output(y) for y in size_batch])
       

        return X_batch, y_batch
    
    def __getitem__(self, index):

        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y