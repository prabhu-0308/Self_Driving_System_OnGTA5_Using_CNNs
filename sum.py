import numpy as np
import os
from alexnet import alexnet2 as aln2, alexnet as aln1
import pandas as pd
from sklearn.model_selection import train_test_split as tts
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
# MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)
MODEL_NAME = 'GTA-V_Self-Driving'
MODEL_NAME = os.getcwd() +'\\model\\al2epo12\\GTA-V_Self-Driving_ALN2'




model2 = aln2(WIDTH, HEIGHT, LR,3)


model2.summary()

model1 = aln1(WIDTH, HEIGHT, LR)

model1.summary()