import numpy as np
import pickle

vocab = np.genfromtxt("/media/kyle/My Passport/cs674/vocab.nytimes.txt", dtype='str')


vocab_dictionary = {}
for index,word in enumerate(vocab):
    vocab_dictionary[index] = word

with open("/media/kyle/My Passport/cs674/vocab_nytimes.pickle", "wb") as handle:
    pickle.dump(vocab_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)