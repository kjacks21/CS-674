"""
k-Nearest Neighbor with euclidian distance and DTW
"""


import numpy as np
from scipy.stats import mode

home_dir = "/home/kyle/Documents/CS-674/HW1/hw1_datasets/hw1_datasets"

def read_test(test_path):
    """
    Reads in test.txt.
    
    Parameter
    ---------
    test_path: string
    Should follow this format: /dataset1/test.txt
    
    Returns
    -------
    Data and labels in seperate numpy arrays
    """
    path = home_dir + test_path
    data = np.genfromtxt(path)
    test_X = data[:,1:]
    test_y = data[:,0]

    return test_X, test_y
    

def read_train(train_path):
    """
    Reads in train.txt
    
    Parameter
    ---------
    train_path: string
    Should follow this format: /dataset1/train.txt
    
    Returns
    -------
    Data and labels in seperate numpy arrays
    """

    path = home_dir + train_path
    data = np.genfromtxt(path)
    train_X = data[:,1:]
    train_y = data[:,0]

    return train_X, train_y

def euclidean_distance_matrix(train_X, test_X):
    """
    Pre-compute euclidian distance between test and train files.
    
    Returns
    -------
    distance_matrix: numpy array of dimensions N x n, where N is the number of
    time series in test_X and n is the number time series in train_X
    """
    distance_matrix = np.zeros((test_X.shape[0], train_X.shape[0]))
    
    for i, n in enumerate(train_X):
        for I, N in enumerate(test_X):
            distance = (np.sum((n-N)**2))**(1/2)
            distance_matrix[I][i] = distance
    return distance_matrix

def dtw(q, c, w=None):
    """
    Compute dynamic time warping distance between two time series 
    
    Parameters
    ----------
    
    q: numpy array
    Query, one time series
    
    c: numpy array
    Candidate, one time series
    
    w: int, default None
    Warping window constraint. Values are [0,100], where w=10 would mean
    a warping window width of 10% the length of the time series
    """
    # compute initial matrix by row in a q by c matrix
    
    distance_matrix = np.zeros((len(q), len(c)))
    for index_i, i in enumerate(q):
        for index_j, j in enumerate(c):
            sq_dist = (i-j)**2
            distance_matrix[index_i][index_j] = sq_dist
        break
    
    
    
    return dtw_dist
    


def kNN(k=1, train_X=None, train_y=None, test_X=None, distance_measure=None, window = None):
    """
    implementation of k nearest neighbor algorithm with euclidean distance
    and dynamic time warping with paramter w for the size of the warping
    window
    
    parameters
    ----------
        
    k: integer
    number of nearest neighbors used for classification
    
    train_X: numpy array
    time series from train.txt
    
    train_y: 1D numpy array
    label vector from train.txt
    
    test_X: numpy array
    contains the preprocessed text for the kNN method to classify
    
    euclidian_distance_matrix: numpy array
    contains euclidian distances between train and test samples. Each column
    matches to one time series in the train file
    
    distance_measure: must be either {euclidean, dtw}
    contains the labels from the training set
    
    window: None or int
    Defines the size of the dtw window
    
    
    """
    predicted_y = []    
    
    # euclidean distance
    if distance_measure == "euclidean": 
        if k == 1: # 1-NN
            dist_matrix = euclidean_distance_matrix(train_X, test_X)
            print("dist_matrix computed, moving onto classification")
            for i in dist_matrix:
                label = train_y[np.argmin(i)] # get index of min distance between test time series and all time series in train
                predicted_y.append(label)        
            
        elif k > 1: # k-NN
            dist_matrix = euclidean_distance_matrix(train_X, test_X)
            print("dist_matrix computed, moving onto classification")
            for i in dist_matrix:
                smallest_indices = np.argsort(i)[:k]
                labels = np.take(train_y, smallest_indices)
                
                # calculate mode. If mode count is 1, use 1-NN
                # drawback here is that if a tie exists, then the first most frequent label is used
                if mode(labels).count > 1:
                    label = mode(labels).mode[0]
                else:
                    label = train_y[np.argmin(i)] # get index of min distance between test time series and all time series in train 
                predicted_y.append(label)
            
    elif distance_measure == "dtw": # dynamic time warping
        dtw_matrix
            
    return np.array(predicted_y)


def accuracy(predicted_y, test_y):
    """
    Calculate accuracy between predicted and true labels
    
    Parameters
    ----------
    
    predicted_y: numpy array
    predicted labels
    
    test_y: numpy array
    true labels
    """
    if len(predicted_y) != len(test_y):
        raise IndexError("predicted_y and test_y are not the same length")
    return (np.sum(predicted_y == test_y))/len(predicted_y)



# euclidian distance assesment for k=1
k1_euc_accuracy = []
for i in range(4):
    dataset_index = i + 1
    test_X, test_y = read_test("/dataset"+str(dataset_index)+"/test.txt")
    train_X, train_y = read_train("/dataset"+str(dataset_index)+"/train.txt")
    predicted_y = kNN(k=1, train_X = train_X, train_y=train_y, test_X=test_X, distance_measure='euclidean')     
    k1_euc_accuracy.append(accuracy(predicted_y, test_y))
    print(k1_euc_accuracy)
    


# euclidian distance assesment for k=5
k5_euc_accuracy = []
for i in range(4):
    dataset_index = i + 1
    test_X, test_y = read_test("/dataset"+str(dataset_index)+"/test.txt")
    train_X, train_y = read_train("/dataset"+str(dataset_index)+"/train.txt")
    predicted_y = kNN(k=5, train_X = train_X, train_y=train_y, test_X=test_X, distance_measure='euclidean')     
    k5_euc_accuracy.append(accuracy(predicted_y, test_y))
    print(k5_euc_accuracy)
    









