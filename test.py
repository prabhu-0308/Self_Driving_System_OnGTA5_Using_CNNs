# test_model.py

import numpy as np
from grabscreen import grab_screen
import cv2
import time
import os
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet2
from getkeys import key_check
import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
MODEL_NAME = os.getcwd() +'\\model\\al2epo12\\GTA-V_Self-Driving_ALN2'

###############  HyperParameters to tune the performance in the game  #############
S_cnt_lim = 3 #2
S_time = 0.14 #0.17
turn_time = 0.02 #0.0098
turn_thresh = 0.3 #0.3
fwd_thresh = 0.65
###################################################################################
def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    ReleaseKey(D)
    PressKey(W)
    PressKey(A)
    time.sleep(turn_time)
    ReleaseKey(A)

def right():
    ReleaseKey(A)
    PressKey(W)
    PressKey(D)
    time.sleep(turn_time)
    ReleaseKey(D)
   
model = alexnet2(WIDTH, HEIGHT, LR,3)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)
        if i == 2:
            PressKey(W)
    S_cnt = 0
    paused = False
    while(True):
        
        if not paused:
            # 800x600 windowed mode
            #screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=(0,40,1152,904))
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (160,120))

            prediction = model.predict([screen.reshape(160,120,1)])[0]
            print(prediction)

            
            
            if prediction[1] > fwd_thresh:
                straight()
                print("Straight")
                
            elif prediction[0] > prediction[2] and prediction[0] > turn_thresh:
                left()
                print("Left")
                
            elif prediction[0] < prediction[2] and prediction[2] > turn_thresh:
                right()
                print("Right")
                
            else:
                straight()
                print("Straight")
            

            
            S_cnt+=1
            if S_cnt == S_cnt_lim:
                PressKey(S)
                time.sleep(S_time)
                ReleaseKey(S)
                S_cnt = 0

        keys = key_check()

        # T to pauses game if want to re-adjust. B to end the loop.
        if 'B' in keys:
            break

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
                PressKey(W)
                time.sleep(2)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()