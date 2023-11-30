import numpy as np
import os
from alexnet import alexnet

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'GTA-V_Self-Driving'

model = alexnet(WIDTH, HEIGHT, LR)


for i in range(EPOCHS):
    print("A")

    train_data = np.load(os.getcwd() +'\\testing_data\\Testing_data_1_balanced.npy',allow_pickle=True)

    train = train_data[:-100]
    test = train_data[-100:]

    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(os.getcwd() +'\\testing_data\\'+ MODEL_NAME + "_{}".format(i+1) )
