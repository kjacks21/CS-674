"""
Given motifs for each time series, this .py file contains the preprocessing of
those motifs and classification of test motifs.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_words(data):
    """
    Converts letters generated via SAX into words.
    
    Paramter
    --------
    data: numpy array
    Should not have labels
    
    Return
    ------
    List of motifs in the form of a senetence per time series
    """
    
    data_list = []
    for motif in data:
        words = ""
        for i in motif:
            word = "".join([l for l in i if not l.isdigit()])
            words+=word+" "
        data_list.append(words.strip())
    return data_list

def load_data(home_dir, test_path, test_label_path, train_path, 
              train_label_path):
    """
    Reads in test.txt.
    
    Parameter
    ---------
    home_dir: string
    Home directory, should look like "/home/kyle/Documents/CS-674/HW2/"
    
    test_path: string
    Should follow this format: /sax_data/dataset1/test_motifs.txt
    
    test_label_path: string
    Should follow this format: hw2_datasets/hw2_datasets/dataset1/test.txt
    
    train_path: string
    Should follow this format: /sax_data/dataset1/test_motifs.txt
    
    train_label_path: string
    Should follow this format: hw2_datasets/hw2_datasets/dataset1/test.txt
    
    Returns
    -------
    Data and labels in seperate numpy arrays
    """
    # read in data
    test_X = np.array(pd.read_csv(home_dir+test_path, quotechar='"', delimiter="\t", skipinitialspace=True, header=None))
    test_y = np.genfromtxt(home_dir+test_label_path)[:,0]
    train_X = np.array(pd.read_csv(home_dir+train_path, quotechar='"', delimiter="\t", skipinitialspace=True, header=None))
    train_y = np.genfromtxt(home_dir+train_label_path)[:,0]
    
    # return numpy arrays of words
    train_data = clean_words(train_X)
    test_data = clean_words(test_X)
    
    # create tf-idf matrix for train and test
    vectorizer = TfidfVectorizer(lowercase=False)
    train_tfidf = vectorizer.fit_transform(train_data)
    test_tfidf = vectorizer.transform(test_data)

    return test_tfidf, train_tfidf, test_X, test_y, train_y
    

def kNN(testTfidf, trainTfidf, test_file, train_label):
    """
    implementation of k nearest neighbor algorithm using cosine similarity. 
    Only does 1 nearest neighbor.
    
    parameters
    ----------
        
    k: integer
    number of nearest neighbors used for classification.
    
    testTfidf: sparse matrix
    
    trainTfidf: sparse matrix
    
    test_file: must be shape (int,)
    contains the preprocessed text for the kNN method to classify
    
    train_label: must be shape (int,)
    contains the labels from the training set
    
    """
    test_y = []    
    
    # iterate through all lines in the test reviews and classify them
    for index, line in enumerate(test_file):
        # cosine similarity
        cos_similarity = linear_kernel(testTfidf[index:index+1], trainTfidf).flatten()
        # get the indices of nearest neighbors based on k parameter 
        i = np.argmax(cos_similarity, axis=None)
        
        # get a list of labels from the neighbors and sum the list
        test_y.append(train_label[i])        
        print(index)
            
    return np.array(test_y)   

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

# classify
dataset_ac = []
for i in range(5):
    dataset_index = i + 1
    test_tfidf, train_tfidf, test_X, test_y, train_y = load_data(home_dir = "/home/kyle/Documents/CS-674/HW2/",
                                                                 test_path = "/sax_data/dataset"+str(dataset_index)+"/test_motifs.txt",
                                                                 test_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/test.txt",
                                                                 train_path = "/sax_data/dataset"+str(dataset_index)+"/train_motifs.txt",
                                                                 train_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/train.txt")
    predicted_y = kNN(testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=test_X, train_label=train_y)
    ac = accuracy(predicted_y, test_y)
    dataset_ac.append(ac)
    print(ac)


