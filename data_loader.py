import h5py
import os
import numpy as np

def extract_signals(filename, signalName):
    '''
    
    :param filename: .mat data 
    :param signalName: Raw RF data from Ultrasound 
    :return: data in shape (number of samples, channel number, time series length, frame)
    '''
    fData = h5py.File(filename, 'r')
    inData = fData.get(signalName)
    print('loading training data %s in shape: (%d, %d, %d)' % (signalName, inData.shape[0], inData.shape[1], inData.shape[2]))
    data = np.array(inData).reshape(inData.shape[0], inData.shape[1], inData.shape[2], 1)

    return data


