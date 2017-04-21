import numpy as np


for i in range(1,8):
    if i == 1:
        data = np.load("/media/kyle/My Passport/cs674/%dnytimes_data.npy" % (i))
    else:
        data = np.concatenate((data, np.load("/media/kyle/My Passport/cs674/%dnytimes_data.npy" % (i))))


np.save("/media/kyle/My Passport/cs674/nytimes_data", data)