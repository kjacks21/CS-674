import numpy as np
from np_to_coomatrix import convert
import pickle
from sklearn.decomposition import LatentDirichletAllocation as LDA
from print_top_words import print_top_words

# todo convert
# D 300000
# W 102660
# NNZ 69679427
# docID wordID count
# docID wordID count
# docID wordID count
# row column data

print("reading in data")
raw_tf = np.load("/media/kyle/My Passport/cs674/nytimes_data.npy")
with open("/media/kyle/My Passport/cs674/vocab_nytimes.pickle", "rb") as handle:
    vocab = pickle.load(handle)

print("converting data to term frequency matrix")
term_freq_matrix = convert(raw_tf)

lda2 = LDA(n_topics=10, learning_method='batch',max_iter=2,n_jobs=-1, verbose=1, random_state=2017)
lda5  = LDA(n_topics=10, learning_method='batch',max_iter=5,n_jobs=-1, verbose=1, random_state=2017)
lda10 = LDA(n_topics=10, learning_method='batch',max_iter=10,n_jobs=-1, verbose=1, random_state=2017)
lda20 = LDA(n_topics=10, learning_method='batch',max_iter=20,n_jobs=-1, verbose=1, random_state=2017)

print("Fitting LDA")
instances = {
    'lda2' : lda2,
    'lda5' : lda5,
    'lda10' : lda10,
    'lda20' : lda20
}

results = {}
for i in [2,5,10,20]:
    print("Starting lda with"+str(i)+"epochs")
    lda_iter = 'lda'+str(i)
    results[lda_iter] = instances[lda_iter].fit(term_freq_matrix)
    print_top_words(instances[lda_iter], vocab, n_top_words = 15)

print("saving results to pickle")
with open("/media/kyle/My Passport/cs674/results.pickle", 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
