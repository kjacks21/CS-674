import numpy as np

i=0
skip_rows = 3
data = []
with open("/media/kyle/My Passport/cs674/docword.nytimes.txt") as f:
    for line in f:
        if i < skip_rows:
            pass
        else:
            a = line.split()
            a = [int(s) for s in a]
            data.append(a)
        i += 1

        if i in [10000000,20000000,30000000,40000000,50000000,60000000]:
            print(i)
            data = np.array(data)
            np.save("/media/kyle/My Passport/cs674/"+str(int(i/10000000))+"nytimes_data", data)
            data = []

data = np.array(data)
np.save("/media/kyle/My Passport/cs674/7nytimes_data", data)