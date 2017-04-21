import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
# http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-topics-extraction-with-nmf-lda-py
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer

# todo convert 
# D 300000
# W 102660
# NNZ 69679427
# docID wordID count
# docID wordID count
# docID wordID count
# row column data

data = np.load("/media/kyle/My Passport/cs674/csr_data.npy")

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(data)
