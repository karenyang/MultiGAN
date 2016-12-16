import sys
import os
import math
import time
import urllib
import tarfile
import numpy as np
import pandas as pd
from scipy import ndimage, misc
from scipy.misc import imread, imsave, imresize
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

__author__ = 'kaiyuany03@gmail.com'

WORK_DIRECTORY = '.'

def resize_3232(image):
    '''
    maxSize = max(image.shape[0], image.shape[1])
    img_rows = 32
    img_cols = 32

    
    # Size of the resized image, keeping aspect ratio
    imageWidth = int(math.floor(img_rows*image.shape[0]/maxSize))
    imageHeigh = int(math.floor(img_cols*image.shape[1]/maxSize))
    
    # Compute deltas to center image (should be 0 for the largest dimension)
    dRows = (img_rows-imageWidth)//2
    dCols = (img_cols-imageHeigh)//2
                
    imageResized = np.zeros((img_rows, img_cols))
    imageResized[dRows:dRows+imageWidth, dCols:dCols+imageHeigh] = imresize(image, (imageWidth,imageHeigh))
    
    # Fill the empty image with the median value of the border pixels
    # This value should be close to the background color
    val = np.median(np.append(imageResized[dRows,:],
                              (imageResized[dRows+imageWidth-1,:],
                              imageResized[:,dCols],
                              imageResized[:,dCols+imageHeigh-1])))
                              
    # If rows were left blank
    if(dRows>0):
        imageResized[0:dRows,:].fill(val)
        imageResized[dRows+imageWidth:,:].fill(val)
        
    # If columns were left blank
    if(dCols>0):
        imageResized[:,0:dCols].fill(val)
        imageResized[:,dCols+imageHeigh:].fill(val)
        '''
    return imresize(image,[32,32])


def raadimg(path, character, num):
    folder = 'Sample%03d' % (character, )
    sample = os.path.join(path, folder)
    filename = os.path.join(sample, 'img%03d-%03d.png' % (character, num))
    if not os.path.exists(filename):
        print filename
    img = ndimage.imread(filename, flatten=True)
    
    img = resize_3232(img)
    w, h = img.shape

    return img.reshape((w*h))

if os.path.exists('training_data_26U.csv'):
    exit()
path = os.path.join('.', 'English/Hnd/Img/')

training_data = []
training_label = []
for j in range(0, 55):
    
    for i in range(0, 10):
        img = raadimg(path, i+1, j+1)
        training_data.append(img)
        training_label.append(i+1)  
    for i in range(0, 26):
        img = raadimg(path, i+11, j+1)
        training_data.append(img)
        #training_label.append(i+11)
        training_label.append(i+1)
    for i in range(0, 26):
        img = raadimg(path, i+37, j+1)
        training_data.append(img)
        training_label.append(i+37)


training_data = np.array(training_data).astype(np.uint8)
np.savetxt('training_data_mixed.csv', np.c_[training_label, training_data], delimiter=',', fmt ='%d')


dataset = pd.read_csv('training_data_mixed.csv', header=None).values
print 'dataset: ', dataset.shape


labels = dataset[:, 0]
images = dataset[:, 1:].astype(np.int32)

print labels.shape
print images.shape

train_labels = labels[:62*50]
train_images = images[:62*50]
print train_labels.shape


