import matplotlib.pyplot as plt
import numpy as np
import pickle

def print_mnist(model, n_top_words, iterations):
    for topic_idx, topic in enumerate(model.components_):
        pixels = [0] * 784
        tracker = []
        [tracker.append(i) for i in topic.argsort()[:-n_top_words - 1:-1]]
        for i in tracker:
            pixels[i] = 255
        pixels = np.array(pixels)
        #normalized = normalize(topic)
        pixels = pixels.reshape((28,28))

        plt.title('Topic %d, Epoch %d' % (topic_idx, iterations))
        plt.imshow(pixels, cmap='gray')
        plt.savefig('/media/kyle/My Passport/cs674/mnist_images/T%d-E%d' % (topic_idx, iterations))

def normalize(topic):
    a = 0
    b = 255
    min = np.min(topic)
    max = np.max(topic)
    new_data = []
    for i in topic:
        new_data.append((b-a)*(i-min)/(max-min)+a)
    return np.array(new_data)

if __name__ == '__main__':
    with open("/media/kyle/My Passport/cs674/mnist_images/mnist_results.pickle", 'rb') as handle:
        results = pickle.load(handle)

    for key, model in results.items():
        print_mnist(model, n_top_words=50, iterations = int(key[3:]))