# ---------------------------------------------------
# Autores: Otavalo D.Andres, Zaruma Samantha 
# Descripcion: Trabajo de titulacion para la implementacion de un 
# auto codificador. Este codigo se generan las graficas de constelacion
# para el auto codificador entrenado. Existe una version cuando se 
# emplear  n=2 y otra para n>2.  
# Fecha: 2024-06-21
# Version: 1.0
# ---------------------------------------------------
import numpy as np
import math as math
from fractions import Fraction
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import random as rn
import time
import os 
from keras import backend as K
from EnergyNormalizationLayer import EnergyNormalizationLayer
import seaborn as sns
import itertools
import umap
from scipy.signal import welch
from keras.models import load_model
from scipy.special import erfc
import time

#inicializar semilla aleatoria
seed=int(time.time())
np.random.seed(seed)
tf.random.set_seed(seed)

#CARGAR EL MODELO ENTRENADO
autoencoder_loaded = load_model('8_4_AE_energy_3dB_relu_coded_fsk.model')

# parametros

M=16            #numero simbolos
k = np.log2(M)  #numero de bits por simbolo
n_channel = 8   #nuermo de canales


R = k/n_channel #tasa de codigo

################### Ruido
EbNodB_range = 10 
EbNo=10.0**(EbNodB_range/10.0)
EbNodB_range2 = 20 
EbNo2=10.0**(EbNodB_range2/10.0)
noise_const = np.sqrt(1/(2*R*k*EbNo))
noise_const2 = np.sqrt(1/(2*R*k*EbNo2))
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


#valores para el set de prueba
N = M
#N=1000000

num_test_class = N // M
test_data = []

for i in range(M):
    for _ in range(num_test_class):
        temp3 = np.zeros(M,dtype=np.float32)
        temp3[i] = 1
        test_data.append(temp3)

test_data = np.array(test_data,dtype=np.float32)
test_label = np.array([np.argwhere(vec == 1).flatten()[0] for vec in test_data])
########
'''

test_label = np.random.randint(M,size=N)
test_data = []

for i in test_label:
    temp = np.zeros(M)
    temp[i] = 1
    test_data.append(temp)
    
test_data = np.array(test_data)
temp_test = 6
print (test_data[temp_test][test_label[temp_test]],test_label[temp_test])
'''


################Grafica de la constelacion unicamente para n=2
######
'''
scatter_plot = []
x_val = []

for i in range(0,M):
    temp = np.zeros(M)
    temp[i] = 1
    scatter_plot.append(encoder.predict(np.expand_dims(temp,axis=0)))
    x_val=scatter_plot
 
scatter_plot = np.array(scatter_plot)
print (scatter_plot.shape)


# Constelacion para M simbolos sin ruido
plt.figure(figsize=(14, 4.5))
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste de los subplots para el espacio del título$
plt.suptitle(f'Diagrama de Constelacion - AE({n_channel},{int(k)})', fontsize=16)
plt.subplot(1,3,1)
scatter_plot = scatter_plot.reshape(M,2,1)
plt.scatter(scatter_plot[:,0],scatter_plot[:,1])
plt.title(f"Sin SNR")
plt.axis((-1.5,1.5,-1.5,1.5))
plt.xlabel("fase (I)")
plt.ylabel("cuadratura (Q)")
plt.grid()


#constelacion para 20*M simbolos con ruido de 10dB
plt.subplot(1,3,2)
plt.grid()
plt.title(f" SNR={EbNodB_range}dB")
plt.axis((-1.5,1.5,-1.5,1.5))
plt.xlabel("fase (I)")
plt.ylabel("cuadratura (Q)")

for rep in range(20):
    deco_scatter_plot = []
    for i in range(0, M):
        noise_scatter = noise_const * np.random.randn(1, n_channel)
        temp2 = np.zeros(M)
        temp2[i] = 1
        decoded_point = encoder.predict(np.expand_dims(temp2, axis=0)) + noise_scatter
        deco_scatter_plot.append(decoded_point)
    deco_scatter_plot = np.array(deco_scatter_plot)

    deco_scatter_plot = deco_scatter_plot.reshape(M, 2, 1)
    plt.scatter(deco_scatter_plot[:, 0], deco_scatter_plot[:, 1],color='red')

#constelacion para 20*M simbolos con ruido de 20dB
plt.subplot(1,3,3)
plt.grid()
plt.title(f"SNR={EbNodB_range2}dB")
plt.axis((-1.5,1.5,-1.5,1.5))
plt.xlabel("fase (I)")
plt.ylabel("cuadratura (Q)")

for rep in range(20):
    deco_scatter_plot2 = []
    for i in range(0, M):
        noise_scatter2 = noise_const2 * np.random.randn(1, n_channel)
        temp3 = np.zeros(M)
        temp3[i] = 1
        decoded_point2 = encoder.predict(np.expand_dims(temp3, axis=0)) + noise_scatter2
        deco_scatter_plot2.append(decoded_point2)
    deco_scatter_plot2 = np.array(deco_scatter_plot2)
    deco_scatter_plot2 = deco_scatter_plot2.reshape(M, 2, 1)
    plt.scatter(deco_scatter_plot2[:, 0], deco_scatter_plot2[:, 1],color='red')

plt.show()
'''
####################
# Ente metodo funcionada para graficar constelaciones cuando n>2
######## metodo t-SNE
#aplicar T-SNE
x_emb = encoder.predict(test_data)
noise = noise_const * np.random.randn(N,n_channel)
X_embedded2 = TSNE(learning_rate=1000,n_components=2, method='exact',random_state=0, perplexity=M/2,early_exaggeration=M,init='random').fit_transform(x_emb)
print (X_embedded2.shape)

#normalizar datos
X_embedded_m2 = X_embedded2 - np.mean(X_embedded2)
mean_squared2=np.mean(np.square(X_embedded_m2))
s2=np.sum(np.square(X_embedded2))
n2 = X_embedded_m2.shape[1]
X_P_normalized2 =  X_embedded_m2 / np.sqrt(2*mean_squared2)
X_E_normalized2 =  X_embedded2 / np.sqrt(s2/(n2))


plt.figure()
plt.subplot(1,2,1)
plt.scatter(X_P_normalized2[:,0],X_P_normalized2[:,1])
plt.title(f"Mapeo TSNE - AE({n_channel},{int(k)})")
plt.axis((-1.5,1.5,-1.5,1.6))
plt.tight_layout()
plt.xlabel("fase (I)")
plt.ylabel("cuadratura (Q)")
plt.grid()


############### P-segundo procedimiento UMAP

# Aplicar UMAP
noise = noise_const * np.random.randn(N, n_channel)
umap_reducer = umap.UMAP(n_components=2, random_state=300, repulsion_strength=5, min_dist=1.0,spread=10,n_neighbors=80)
X_embedded = umap_reducer.fit_transform(x_emb)

print(X_embedded.shape)

# Normalizar los datos
X_embedded_m = X_embedded - np.mean(X_embedded)
mean_squared2 = np.mean(np.square(X_embedded_m))
s = np.sum(np.square(X_embedded))
n = X_embedded_m.shape[1]
X_P_normalized = X_embedded_m / np.sqrt(2*mean_squared2)
X_E_normalized = X_embedded_m / np.sqrt(s/(n/2))


# Graficar la constelación usando UMAP
plt.subplot(1,2,2)
plt.scatter(X_P_normalized[:, 0], X_P_normalized[:, 1])
plt.title(f"Mapeo UMAP - AE({n_channel}, {int(k)})")
plt.axis((-1.5, 1.6, -1.5, 1.5))
plt.tight_layout()
plt.xlabel("Fase (I)")
plt.ylabel("Cuadratura (Q)")
plt.grid()
plt.show()
