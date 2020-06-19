import cv2 
import numpy as np 
import os 
from random import shuffle 

# The path to where the images used to train the program are stored 
TRAIN_DIR ="C:/Users/zakry/Downloads/train/train"
# Path of where the images used to test are stored 
TEST_DIR = "C:/Users/zakry/Downloads/test/test"

IMG_SIZE = 50 
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic')

def label_img(img): 
    ''' Function to give labels to cat and dog that the computer can easily recognise later on.'''
    word_label = img.split('.')[-3] 
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog' : return [0,1]
    
    
    
def create_train_data(): 
    '''Using the file that has the training images, this adds it all in an array so that it is easier to use.'''
    training_data = [] 
    for img in os.listdir(TRAIN_DIR): 
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data 


def process_test_data(): 
    '''And now using the test images and storing them in a formal manner in an array'''
    testing_data = []
    for img in os.listdir(TEST_DIR): 
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    np.save('test_data.npy', testing_data) 
    return testing_data 

# Creating the training data and assigning it to train_data
train_data = create_train_data() 
