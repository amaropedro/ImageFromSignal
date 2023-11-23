import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math as mt
from PIL import Image

dataCSV = './currents.csv'

def save_grayscale(path :str, W):
    W = (((W-W.min())/(W.max()-W.min()))*255).astype('uint8')

    print(W.shape)

    img = Image.fromarray(W, mode = "L")

    img.save(path+'/sample.png',"PNG")
    
    #img.show()

def save_colormap(path :str, W, cmap_name :str):
    W = (W - W.min()) / (W.max() - W.min())

    #W.shape

    cmap = plt.get_cmap(cmap_name)

    colored_coeficients = cmap(W)

    img = Image.fromarray(np.uint8(colored_coeficients * 255)).convert('RGB')

    img.save(path+'/sample_'+ cmap_name +'.png',"PNG")

    img.show()

def plot_samples(sample_fault, sample_control):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sample_fault)
    plt.title('Signal with Fault')
    ax = plt.gca()
    ax.set_ylim([-80, 80])

    plt.subplot(1, 2, 2)
    plt.plot(sample_control)
    plt.title('Signal Without Fault')
    ax = plt.gca()
    ax.set_ylim([-80, 80])

    plt.show()

def generate_image(sample, path :str):
    # Preparando a transformada wavelet
    Fs = 1/500 #Frequência de amostragem de 500 Hz
    s =  np.arange(0.5,50000) #Escalas utilizadas geram frequências de 0.01Hz a 1 KHz
    wav = 'cmor1.0-1.0' # Wavelet Morlet com B = 1 e C = 1.
    
    s = s[:len(sample)]

    W1 = np.ma.zeros(len(sample))
    Wv1,_1 = pywt.cwt(sample, s, wav, sampling_period = Fs, method = 'fft')
    Esc1 = np.abs(Wv1)**2
    W1 = W1+Esc1

    save_grayscale(path, W1)

    save_colormap(path, W1, 'inferno')

data = pd.read_csv(dataCSV, header= None)

sample_fault = data[0][72000:80000].values # coluna 0, linhas 70000 a 80000 normal. small de 72000:80000
sample_control = data[0][102000:110000].values

generate_image(sample_fault, './.small/fault')
generate_image(sample_control, './.small/control')

plot_samples(sample_fault, sample_control)
