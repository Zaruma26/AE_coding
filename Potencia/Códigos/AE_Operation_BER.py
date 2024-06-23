# ---------------------------------------------------
# Autores: Otavalo D.Andres, Zaruma Samantha 
# Descripcion: Trabajo de titulacion para la implementacion de un 
# auto codificador. Este codigo simula la transmitision de 
# 1000000 de datos, para obtener los valores de BLER en el rango 0-14dB.
# Los valores se guardan en archivos .txt para ser procesados. 
# Fecha: 2024-06-21
# Version: 1.0
# ---------------------------------------------------
import sys
import numpy as np
import math as math
from fractions import Fraction
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from keras.optimizers import SGD,Adam,Adadelta
import random as rn
import time
import os 
from keras import backend as K
from PowerNormalizationLayer import PowerNormalizationLayer
from HelperAEWAWGNLayer import HelperAEWAWGNLayer
import seaborn as sns
from keras.models import load_model
from scipy.special import erfc
import time
tf.keras.utils.get_custom_objects().update({'PowerNormalizationLayer': PowerNormalizationLayer})


#inicializar semilla aleatoria
seed=int(time.time())
np.random.seed(seed)
tf.random.set_seed(seed)


#verificaion de argumentos
iteration = sys.argv[1] if len(sys.argv) > 1 else "default"
output_folder = sys.argv[2] if len(sys.argv) > 2 else "results"


#CARGAR EL MODELO ENTRENADO
autoencoder_loaded = load_model('8_4_AE_power_3dB_sk.keras')

    
N=1000000       #numero de datos transmitir
n_channel =8    #nuermo de canales
M=16             #numero de simbolos
k = np.log2(M)  #numero de bits
R = k/n_channel #tasa de codigo

################### Ruido
EbNodB_range = 20                       #ruido dB
EbNo=10.0**(EbNodB_range/10.0)          #ruido lineal
noise_const = np.sqrt(1/(2*R*EbNo))
##################

# Definir el modelo del codificador
input_signal = autoencoder_loaded.input 
encoded2 = autoencoder_loaded.layers[3].output 
encoder = Model(input_signal, encoded2)
print('modelo del codificador')
print(encoder.summary())

# Definir el modelo del decodificador
encoded_input = Input(shape=(n_channel,))
deco = autoencoder_loaded.layers[-2](encoded_input)
deco = autoencoder_loaded.layers[-1](deco)
decoder = Model(encoded_input, deco)
print('modelo del decodificador')
print(decoder.summary())

'''
#########
num_test_class = N // M
test_data = []

for i in range(M):
    for _ in range(num_test_class):
        temp3 = np.zeros(M,dtype=np.float32)
        temp3[i] = 1
        test_data.append(temp3)

test_data = np.array(test_data,dtype=np.float32)
 #np.random.shuffle(test_data)
test_label = np.array([np.argwhere(vec == 1).flatten()[0] for vec in test_data])
########

'''
#Generando Set DE PRUEBA
test_label = np.random.randint(M,size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
    
test_data = np.array(test_data)
temp_test = 6
print (test_data[temp_test][test_label[temp_test]],test_label[temp_test])



############### BLER

#  secuencia de números en un rango de SNR.
def frange(x, y, jump):
    while x < y:  
        yield x  
        x += jump  

# Genera una lista de valores desde 0 hasta 14, con un paso de 2.
EbNodB_range = list(frange(0, 16, 2))

# Inicializa una lista para almacenar las tasas de error de bit (BER).
ber = [None] * len(EbNodB_range)

# Itera sobre cada valor en 'EbNodB_range' para calcular el BER correspondiente.
for n in range(0, len(EbNodB_range)):
    # Convierte el valor de Eb/No en dB a una relación lineal.
    EbNo_ber = 10.0 ** (EbNodB_range[n] / 10.0)
    
    # Calcula la desviación estándar del ruido con base en la relación Eb/No y la tasa de código 'R'.
    noise_std = np.sqrt(1 / (2 * R * EbNo_ber))
    
    # Media del ruido, asumiendo que es cero.
    noise_mean = 0
    
    # Inicializa el contador de errores.
    no_errors = 0
    
    # Número de símbolos a simular.
    nn = N
    
    # Genera ruido Gaussiano con la desviación estándar calculada.
    noise = noise_std * np.random.randn(nn, n_channel)
    
    # Codifica la señal usando el auto codificador previamente entrenado.
    encoded_signal = encoder.predict(test_data)
    
    # Suma el ruido a la señal codificada.
    final_signal = encoded_signal + noise
    
    # Decodifica la señal ruidosa usando el decodificador previamente entrenado.
    pred_final_signal = decoder.predict(final_signal)
    
    # Predice las etiquetas finales tomando el índice del valor máximo a lo largo del eje 1.
    pred_output = np.argmax(pred_final_signal, axis=1)
    
    # Calcula el número de errores comparando las predicciones con las etiquetas reales.
    no_errors = (pred_output != test_label)
    no_errors = no_errors.astype(int).sum()  # Suma los errores como enteros.
    
    # Calcula (BLER) 
    ber[n] = no_errors / nn

    


data = np.column_stack((EbNodB_range, ber))
output_filename = os.path.join(output_folder, f'datos_AE_8_4_3dB_sk_cod_fd_{iteration}.txt')
np.savetxt(output_filename, data, fmt='%f', delimiter='\t', header='EbNodB_range\tber')



