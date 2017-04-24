import numpy as np
from np_to_coomatrix import convert
import pickle
#import gensim
#from sklearn.feature_extraction.text import TfidfTransformer
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

tf = np.load("/media/kyle/My Passport/cs674/nytimes_data.npy")
with open("/media/kyle/My Passport/cs674/vocab_nytimes.pickle", "rb") as handle:
    vocab = pickle.load(handle)

#transformer = TfidfTransformer()
#tfidf_matrix = transformer.fit_transform(tf)

print("Fitting LDA")
lda = LDA(n_jobs=-1, verbose=1, random_state=2017)
lda.fit(tf)

print_top_words(lda, vocab, n_top_words = 15)


"""    
# gensim, tf
data = convert(tf)
corpus = gensim.matutils.Sparse2Corpus(data)

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=vocab, num_topics=10)
lda.print_topics(num_topics=20, num_words=10)
"""