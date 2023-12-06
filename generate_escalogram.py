import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import math as mt
from PIL import Image

# Fault sample intervals for each .csv
# 1231 - 72000:80000
# 1234  - 58500:66500
# 1236  - 33000:41000
# 1239  - 50000:58000
# 1249  - 54000:62000

suffix = '1231'
start = 72000
dataCSV = './currents_'+suffix+'.csv'

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

    colored_coefficients = cmap(W)

    img = Image.fromarray(np.uint8(colored_coefficients * 255)).convert('RGB')

    img.save(path+'/sample_'+ cmap_name +'_'+suffix+'_'+str(i)+'.png',"PNG")

    #img.show()

def plot_samples(sample_fault, sample_control, mode :str):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sample_fault)
    plt.title('Signal With Fault - '+mode)
    ax = plt.gca()
    ax.set_ylim([-80, 80])

    plt.subplot(1, 2, 2)
    plt.plot(sample_control)
    plt.title('Signal Without Fault - '+mode)
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

    #save_grayscale(path, W1)

    save_colormap(path, W1, 'inferno')

def generate_images(sample_fault, sample_control):
    # default
    print('default')
    generate_image(sample_fault, './results/yes')
    generate_image(sample_control, './results/no')
    plot_samples(sample_fault, sample_control, 'default')

    # random_noise
    print('random_noise')
    random_noise = np.random.randint(-8, 8, 8000)
    sample_f = sample_fault+random_noise
    sample_c = sample_control+random_noise
    generate_image(sample_f, './results/yes/random_noise')
    generate_image(sample_c, './results/no/random_noise')
    plot_samples(sample_f, sample_c, 'random noise')

    # flipped
    print('flipped')
    sample_f = np.flip(sample_fault)
    sample_c = np.flip(sample_control)
    generate_image(sample_f, './results/yes/flipped')
    generate_image(sample_c, './results/no/flipped')
    plot_samples(sample_f, sample_c, 'flipped')


# main

data = pd.read_csv(dataCSV, header= None)

for i in [0,1,2]:
    sample_fault = data[i][start:start+8000].values
    sample_control = data[i][102000:110000].values

    generate_images(sample_fault, sample_control)