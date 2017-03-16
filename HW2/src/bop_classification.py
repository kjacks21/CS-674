import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

def letter_to_words(data, w):
    """
    Converts letters generated via SAX into words.
    
    Paramter
    --------
    data: numpy array
    Should not have labels
    
    w: int
    Defines the length of the words from the SAX letters
    
    Return
    ------
    numpy array of each time series SAX representation converted to words
    """
    
    data_list = []
    for ts in data:
        words = ""
        lower = 0
        upper = 0
        for i in range(data.shape[1]):
            if (i+1)%w == 0:
                upper = i+1
                word = "".join(ts[lower:upper])
                words+=word+" "
                lower = i+1
            elif i + 1 == len(ts) and (i+1)%w > w/2:
                word = "".join(ts[-w:])
                words+=word
        data_list.append(words)
    #final_data = np.array(data_list)
    return data_list

def load_data(home_dir, test_path, test_label_path, train_path, 
              train_label_path):
    """
    Loads in data and cleans up SAX words.
    
    Parameter
    ---------
    home_dir: string
    Home directory, should look like "/home/kyle/Documents/CS-674/HW2/"
    
    test_path: string
    Should follow this format: /sax_data/dataset1/test_sax.txt
    
    test_label_path: string
    Should follow this format: hw2_datasets/hw2_datasets/dataset1/test.txt
    
    train_path: string
    Should follow this format: /sax_data/dataset1/test_sax.txt
    
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
    train_data = letter_to_words(train_X, 4)
    test_data = letter_to_words(test_X, 4)
    
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

# alphabet of 5
dataset_ac = []
for i in range(5):
    dataset_index = i + 1
    test_tfidf, train_tfidf, test_X, test_y, train_y = load_data(home_dir = "/home/kyle/Documents/CS-674/HW2/",
                                                                 test_path = "/sax_data/dataset"+str(dataset_index)+"/test_sax.txt",
                                                                 test_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/test.txt",
                                                                 train_path = "/sax_data/dataset"+str(dataset_index)+"/train_sax.txt",
                                                                 train_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/train.txt")
    predicted_y = kNN(testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=test_X, train_label=train_y)
    ac = accuracy(predicted_y, test_y)
    dataset_ac.append(ac)
    print(ac)

# alphabet of 15
dataset_ac_15 = []
for i in range(5):
    dataset_index = i + 1
    test_tfidf, train_tfidf, test_X, test_y, train_y = load_data(home_dir = "/home/kyle/Documents/CS-674/HW2/",
                                                                 test_path = "/sax_data/dataset"+str(dataset_index)+"/test_sax1.txt",
                                                                 test_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/test.txt",
                                                                 train_path = "/sax_data/dataset"+str(dataset_index)+"/train_sax1.txt",
                                                                 train_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/train.txt")
    predicted_y = kNN(testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=test_X, train_label=train_y)
    ac = accuracy(predicted_y, test_y)
    dataset_ac_15.append(ac)
    print(ac)

predicted_y = kNN(testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=test_X, train_label=train_y)
ac = accuracy(predicted_y, test_y)

# alphabet of 20
dataset_ac_20 = []
for i in range(5):
    dataset_index = i + 1
    test_tfidf, train_tfidf, test_X, test_y, train_y = load_data(home_dir = "/home/kyle/Documents/CS-674/HW2/",
                                                                 test_path = "/sax_data/dataset"+str(dataset_index)+"/test_sax2.txt",
                                                                 test_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/test.txt",
                                                                 train_path = "/sax_data/dataset"+str(dataset_index)+"/train_sax2.txt",
                                                                 train_label_path = "hw2_datasets/hw2_datasets/dataset"+str(dataset_index)+"/train.txt")
    predicted_y = kNN(testTfidf=test_tfidf, trainTfidf=train_tfidf, test_file=test_X, train_label=train_y)
    ac = accuracy(predicted_y, test_y)
    dataset_ac_20.append(ac)
    print(ac)



    




