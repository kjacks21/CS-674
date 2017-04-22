import numpy as np
import scipy.sparse as sp

def convert(records):
    """
    Convert numpy array raw data to a scipy matrix in prep for conver
    :param records: 
    :return: numpy array
    """

    data = []
    i = [] # row
    j = [] # column

    for n in records:
        i.append(n[0])
        j.append(n[1])
        data.append(n[2])

    return sp.coo_matrix((data, (i,j))).tocsr()

if __name__ == "__main__":
    data_array = np.load("/media/kyle/My Passport/cs674/nytimes_data.npy")
    np.save("/media/kyle/My Passport/cs674/csr_data", convert(data_array))