import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math as mt
from PIL import Image

suffix = '1249'
dataCSV = './currents_'+suffix+'.csv'

def plot(signal):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title('Signal')
    ax = plt.gca()
    ax.set_ylim([-80, 80])
    plt.show()
    
data = pd.read_csv(dataCSV, header= None)

sample = data[0][0:140000].values

plot(sample)