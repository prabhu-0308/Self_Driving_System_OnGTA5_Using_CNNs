import numpy as np
import os
import tensorflow as tf

from alexnet import alexnet as al1, alexnet2 as al2, inception_v3 as glnet
#al1(width, height, lr)   , al2(width, height, lr, output)
#glnet(width, height, frame_count, lr, output=3, model_name )
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'GTA-V_Self-Driving_ALN1'

model1 = al1(WIDTH, HEIGHT, LR)
#model2 = al2(WIDTH, HEIGHT, LR, 3)
# model3 = glnet(WIDTH, HEIGHT,1, LR, 3,MODEL_NAME)

model = model1

class PrintAccuracyCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:
            acc = logs['accuracy']
            print(f'Accuracy at step {batch}: {acc}')


hm_data = 22
for i in range(EPOCHS):
    # print("A")
    for i in range(1,hm_data+1):
        train_data = np.load(os.getcwd() +'\\training_data\\training_data-{}-balanced.npy'.format(i),allow_pickle=True)
        n = len(train_data)
        l = int((-1)*n*0.25)
        train = train_data[:l]
        test = train_data[l:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        model.save(MODEL_NAME)
        # print("B")





