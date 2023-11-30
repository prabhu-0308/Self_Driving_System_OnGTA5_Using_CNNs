import numpy as np
import pandas as pd
from random import shuffle
import cv2
import os
import time
idx=1
train_data = np.load(os.getcwd() +'\\testing_data\\Testing_data_{}.npy'.format(idx),allow_pickle=True)

df = pd.DataFrame(train_data)
print(df.head())

lefts = []
rights = []
forwards = []

shuffle(train_data)

for data in train_data:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])
        rights.append([cv2.flip(img, 1),[0,0,1]])
    elif choice == [0,1,0]:
        forwards.append([img,choice])
    elif choice == [0,0,1]:
        rights.append([img,choice])
        lefts.append([cv2.flip(img, 1),[1,0,0]])
    else:
        print('no matches')
# Display the image
    # cv2.imshow('Image', img)
    #cv2.imshow('ImageFlip', cv2.flip(img, 1))
    # Wait for a key press and then close the window
    
    

print(len(lefts)," ",len(forwards)," ",len(rights))


forwards = forwards[:len(lefts)]


final_data = forwards + lefts + rights
shuffle(final_data)
print(len(lefts)," ",len(forwards)," ",len(rights))
np.save(os.getcwd() +'\\testing_data\\Testing_data_{}_balanced.npy'.format(idx), final_data)
