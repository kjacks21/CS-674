from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pickle
from print_mnist import print_mnist
import time

matlab_data = loadmat("/media/sf_OneDrive/Documents/GMU Classes/CS 674/hw4/mnist_all.mat")
data = []
for key, value in matlab_data.items():
    if key[:-1] == 'test':
        data.append(value)
data = np.vstack(data)
print("reading in data complete")

lda5  = LDA(n_topics=10, learning_method='batch',max_iter=5,n_jobs=3, verbose=1, random_state=2017)
lda10 = LDA(n_topics=10, learning_method='batch',max_iter=10,n_jobs=3, verbose=1, random_state=2017)
lda20 = LDA(n_topics=10, learning_method='batch',max_iter=20,n_jobs=3, verbose=1, random_state=2017)

print("Fitting LDA")
instances = {
    'lda5' : lda5,
    'lda10' : lda10,
    'lda20' : lda20
}

results = {}
for i in [5,10,20]:
    start = time.time()
    print("Starting lda with %d epochs" % (i))
    lda_iter = 'lda'+str(i)
    results[lda_iter] = instances[lda_iter].fit(data)
    print_mnist(instances[lda_iter], n_top_words = 784, iterations=i)
    end = time.time()
    print("time elapsed:", str(end-start))

print("saving results to pickle")
with open("/media/kyle/My Passport/cs674/mnist_images/mnist_results.pickle", 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)