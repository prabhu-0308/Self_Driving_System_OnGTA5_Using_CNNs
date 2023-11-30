# create_testing_data.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
from getkeys import key_check
import os


def keys_to_output(keys):
    '''
    Convert keys to a multi-hot array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

idx = 1

file_name = os.getcwd() +'\\testing_data\\Testing_data_{}.npy'.format(idx)
while(True):
    if os.path.isfile(file_name):
        idx+=1
        file_name = os.getcwd() +'\\testing_data\\Testing_data_{}.npy'.format(idx)
    else:
        print('File number ', idx,' created!!!')
        training_data = []
        break

idx = 1
testing_data = []


def main():

    for i in list(range(10))[::-1]:
        print(i+1)
        time.sleep(1)
    print("Rolling!!!")

    paused = False
    while(True):

        if not paused:
            # 1152x864 windowed mode
            screen = grab_screen(region=(0,40,1152,904))
            cv2.imshow('window',cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))
            
            # resize to something a bit more acceptable for a CNN
            keys = key_check()
            output = keys_to_output(keys)
            testing_data.append([screen,output])
            print(len(testing_data))
            if len(testing_data) % 10000 == 0:
                #print(len(testing_data))
                np.save(file_name,testing_data)
                break
                

        keys = key_check()
        if 'B' in keys:
            print(len(testing_data)," number of images stored.")
            np.save(file_name,testing_data)
            break
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()
