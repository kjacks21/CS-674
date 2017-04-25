import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

def print_mnist(model, n_top_words, iterations):
    for topic_idx, topic in enumerate(model.components_):
        pixels = np.sort(topic)[:-n_top_words - 1:-1]
        size = int(math.sqrt(n_top_words))
        pixels = pixels.reshape((size,size))

        plt.title('Topic %d, Epoch %d' % (topic_idx, iterations))
        plt.imshow(pixels, cmap='gray')
        plt.savefig('/media/kyle/My Passport/cs674/mnist_images/T%d-E%d' % (topic_idx, iterations))


if __name__ == '__main__':
    with open("/media/kyle/My Passport/cs674/mnist_images/mnist_results.pickle", 'rb') as handle:
        results = pickle.load(handle)

    for key, model in results.items():
        print_mnist(model, n_top_words=196, iterations = int(key[3:]))