import numpy as np
from np_to_coomatrix import convert
from print_top_words import print_top_words
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

tf = np.load("/media/kyle/My Passport/cs674/nytimes_data.npy")
data = convert(tf)

# convert term frequency matrix to tf-idf representation
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(data)

lda_tf = LDA(random_state=2017)
lda_tf.fit(tf)


lda_tfidf = LDA(random_state=2017)
lda_tfidf.fit(tfidf)

print("Topics in LDA-tf model:")
tf_feature_names = transformer.get_feature_names()
print_top_words(lda_tf, tf_feature_names, 20)

print("\nTopics in LDA-tfidf model:")
tfidf_feature_names = transformer.get_feature_names()
print_top_words(lda_tfidf, tfidf_feature_names, 20)