import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import scale2frequency
import math as mt
from PIL import Image

s =  np.arange(0.5,50000) #Escalas utilizadas geram frequências de 0.01Hz a 1 KHz
wav = 'cmor1.0-1.0' # Wavelet Morlet com B = 1 e C = 1.

f = scale2frequency(wav, s)/(1/500)

print(f.size)
print(f[f.size-1])
print(f[0])