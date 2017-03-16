import numpy as np

# create new files
label_path = "/home/kyle/Documents/CS-674/HW2/hw2_datasets/hw2_datasets/"
for i in range(5):
    index = i+1
    train_y = np.genfromtxt(label_path+"/dataset"+str(index)+"/train.txt")[:,0]
    test_y = np.genfromtxt(label_path+"/dataset"+str(index)+"/test.txt")[:,0]
    train_X = np.genfromtxt(label_path+"/dataset"+str(index)+"/train.txt")[:,1:]
    test_X = np.genfromtxt(label_path+"/dataset"+str(index)+"/test.txt")[:,1:]
    
    np.savetxt(label_path+"/dataset"+str(index)+"/train_y.txt", train_y)
    np.savetxt(label_path+"/dataset"+str(index)+"/test_y.txt", test_y)
    np.savetxt(label_path+"/dataset"+str(index)+"/train_X.txt", train_X)
    np.savetxt(label_path+"/dataset"+str(index)+"/test_X.txt", test_X)