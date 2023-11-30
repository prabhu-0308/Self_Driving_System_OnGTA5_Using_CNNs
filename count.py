import numpy as np
import os

hm_data = 1
tot = 0
totA = 0
totW = 0
totD = 0
for i in range(1,hm_data+1):
    testing_data = np.load(os.getcwd() +'\\testing_data\\Testing_data_{}.npy'.format(i),allow_pickle=True)
    A =0
    W=0
    D =0

    # print(testing_data[0][])
    for i in range(len(testing_data)):
        if testing_data[i][1] == [0,0,1]:
            D+=1
        elif testing_data[i][1] == [0,1,0]:
            W+=1
        else:
            A+=1
    print(len(testing_data)," ",A," ",W," ",D)
    tot += len(testing_data)
    totA += A
    totW += W
    totD += D
    

print("\n\n",tot," ",totA," ",totW," ",totD)