import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math as mt
from PIL import Image

dataCSV = './currents.csv'

def save_grayscale(W):
    W = (((W-W.min())/(W.max()-W.min()))*255).astype('uint8')
    print(W.shape)

    img = Image.fromarray(W, mode = "L")

    img.save('./sample.png',"PNG")
    
    img.show()

def save_colormap(W, cmap_name :str):
    W = (W - W.min()) / (W.max() - W.min())

    W.shape

    cmap = plt.get_cmap(cmap_name)

    colored_coeficients = cmap(W)

    img = Image.fromarray(np.uint8(colored_coeficients * 255)).convert('RGB')

    img.save('./sample_'+ cmap_name +'.png',"PNG")

    img.show()

# Preparando a transformada wavelet
Fs = 500 #Frequência de amostragem de 500 Hz
s =  np.arange(17,500) #Escalas utilizadas 17 - 500 (frequências de ~1Hz a ~30 Hz)
wav = 'cmor1.0-1.0' # Wavelet Morlet com B = 1 e C = 1.

data = pd.read_csv(dataCSV, header= None)

sample = data[0][20:420].values # coluna 0, linhas 20 a 420

s = s[:len(sample)]

# print(sample)

W1 = np.ma.zeros(len(sample))
Wv1,_1 = pywt.cwt(sample, s, wav, method = 'fft')
Esc1 = np.abs(Wv1)**2
W1 = W1+Esc1

save_grayscale(W1)

save_colormap(W1, 'inferno')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sample)
plt.title('Original Signal')

plt.show()
