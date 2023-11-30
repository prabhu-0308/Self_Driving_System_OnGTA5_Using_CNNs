import numpy as np
import os
from alexnet import alexnet2
import pandas as pd
from sklearn.model_selection import train_test_split as tts
WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
# MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)
MODEL_NAME = 'GTA-V_Self-Driving'
MODEL_NAME = os.getcwd() +'\\model\\al2epo12\\GTA-V_Self-Driving_ALN2'


hm_data = 22
data = np.load(os.getcwd() +'\\training_data\\training_data-{}-balanced.npy'.format(1),allow_pickle=True)
data = pd.DataFrame(data)
for i in range(2,hm_data+1):
    data_new = pd.DataFrame(np.load(os.getcwd() +'\\training_data\\training_data-{}-balanced.npy'.format(i),allow_pickle=True))
    data = pd.concat([data, data_new], ignore_index=True)
    # print(len(data_new))

model = alexnet2(WIDTH, HEIGHT, LR,3)
model.load(MODEL_NAME)

perc = 0.005
t_range = int(len(data)*perc)
# print("lol")
avg_acc = 0.0
rep_for= 5
idx =1
print("TRAINING DATA ACCURACY")
for i in range(rep_for):
    data = data.sample(frac = 1,ignore_index=True)
    # print(data.head())

    test = data[(-1)*t_range:]
    # test = data[-30:]
    # print(len(test))
    test = test.to_numpy()
    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    pred = []
    for x in test_x:
        prediction = model.predict([x])[0]
        if prediction[0]>prediction[1] and prediction[0]>prediction[2]:
            pred.append([1,0,0])
        elif prediction[1]>prediction[0] and prediction[1]>prediction[2]:
            pred.append([0,1,0])
        elif prediction[2]>prediction[0] and prediction[2]>prediction[1]:
            pred.append([0,0,1])
        else:
            pred.append([0,0,0])
    
    cnt=0.0

    for j in range(t_range):
        # print(pred[i])
        # print(test_y[i])
        if pred[j]==test_y[j]:
            # print("yes")
            cnt+=1
        # else:
        #     print("no")
    # print("\n")
    
    acc=cnt*100/t_range
    print("Iteration ",i,": Accuracy = ",acc)
    avg_acc +=acc 
    idx+=1

avg_acc/=rep_for
print("\n\nAverage accuracy = ",avg_acc)