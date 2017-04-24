from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA

matlab_data = loadmat("/media/sf_OneDrive/Documents/GMU Classes/CS 674/hw4/mnist_all.mat")

data = []
for key, value in matlab_data.items():
    if key[:-1] == 'test':
        data.append(value)
        
data = np.vstack(data)

print("Fitting LDA")
lda = LDA(n_jobs=-1, verbose=1, random_state=2017)
lda.fit(data)

print_topics(lda, vocab, top_words = 15)

