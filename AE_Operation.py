import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise,Lambda,Dropout
from keras.models import Model
from keras import regularizers
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from keras.optimizers import SGD,Adam,Adadelta
import random as rn
import os 
from math import sqrt, erfc
from keras import backend as K
#from AE__Training import input_signal, encoded2, n_channel,autoencoder,M,R
import keras
import seaborn as sns
import random
from keras.models import load_model# Deshabilitar el modo seguro de deserialización
from PowerNormalizationLayer import PowerNormalizationLayer
from keras.models import load_model
from sklearn.manifold import TSNE 
keras.config.enable_unsafe_deserialization()
import tensorflow as tf
from PowerNormalizationLayer import PowerNormalizationLayer
import umap



tf.keras.utils.get_custom_objects().update({'PowerNormalizationLayer': PowerNormalizationLayer})

from scipy.special import erfc

def theoretical_ber_qam(EbN0):
    """
    Calcula el BER teórico para 16PSK dada una relación señal a ruido (Eb/N0) específica.
    """
    ##
    ##2/k(1-1/sqrt(M))erfc(((3*log2M)/(2*(M-1)))Eb/N0)
    ##
    
    EbN0_linear = 10**(EbN0/10)  # Convertir de dB a lineal
    ###
    
    #ber =1/2*erfc(np.sqrt(0.5*EbN0_linear))#QPSK
    ber = 3/8 * erfc(np.sqrt(1/10*EbN0_linear))#16QAM
    ####
    return ber

np.random.seed(42)
tf.random.set_seed(42)

#CARGAR EL MODELO ENTRENADO
#autoencoder_loaded = load_model('2_4_symbol_autoencoder_energy.model')
autoencoder_loaded = load_model('3_2_AE_power_3dB.keras')

# Suponiendo que conoces las dimensiones del modelo original
M=4
k = np.log2(M)
n_channel = 3
R = k/n_channel

###################
#EbNodB_range = 20 + 10*math.log10(R)
EbNodB_range = 10 
EbNo=10.0**(EbNodB_range/10.0)

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


#VALORES PARA EL SET DE ENTRENAMIENTO
N = M
#N=1000000
#N=999936 #128-qam
EbNodB_range = 10 
EbNo=10.0**(EbNodB_range/10.0)
noise_const = np.sqrt(1/(2*R*EbNo))

#SET DE PRUEBA
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





################Grafica de la constelacion
###### Grafica AE(2,4)
'''
scatter_plot = []

for i in range(0,M):
    temp = np.zeros(M)
    temp[i] = 1
    scatter_plot.append(encoder.predict(np.expand_dims(temp,axis=0)))
    #print (scatter_plot)
    #deco_scatter_plot = scatter_plot
scatter_plot = np.array(scatter_plot)
print (scatter_plot.shape)
x_emb = encoder.predict(test_data)


# Asegúrate de que x_emb1 sea un array de NumPy
x_emb1 = np.array(x_emb)

# Guardar x_emb1 en un archivo de texto
np.savetxt('x_emb1_values_2_7_12dB.txt', x_emb1, fmt='%.2f')


#numeros_complejos = [complex(real, imag) for real, imag in scatter_plot[:,0]]

#print(scatter_plot[:,0])
#print(numeros_complejos[:,0])
#print(scatter_plot[:,1])

##scatter_plot2 = scatter_plot.reshape(M,2,1)
##scatter_comp = scatter_plot2[:,0]+1j*scatter_plot2[:,1]

##parte_real = scatter_comp.real
##parte_imaginaria = scatter_comp.imag
#print(scatter_comp)
#print(parte_imaginaria)

# Graficar la constelación
#plt.figure(figsize=(8, 8))
#plt.scatter(parte_real, parte_imaginaria, marker='o', color='b')
#plt.xlabel('Parte Real')
#plt.ylabel('Parte Imaginaria')
#plt.title('Constelación')
#plt.grid(True)
#plt.show()


#for i in range(0, n_channel):
#    temp2 = np.zeros(n_channel)
#    temp2[i] = 1
#    deco_scatter_plot.append(decoder.predict(np.expand_dims(temp2, axis=0)))
#deco_scatter_plot = np.array(deco_scatter_plot)
#print (deco_scatter_plot.shape)


# ploting constellation diagram
#import matplotlib.pyplot as plt
#plt.figure(figsize=(12, 5))
plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
scatter_plot = scatter_plot.reshape(M,2,1)
plt.scatter(scatter_plot[:,0],scatter_plot[:,1])
#plt.plot(scatter_comp)
#plt.scatter(scatter_real, scatter_ima)
#plt.axis((-2.5,2.5,-2.5,2.5))
plt.title('Sin SNR')
plt.xlabel('fase (I)')
plt.ylabel('cuadratura (Q)')
plt.legend()
plt.grid(True)
#

plt.subplot(1,3,2)

#plt.grid()
#plt.axis((-2.5,2.5,-2.5,2.5))

EbNodB_range2 = 20 
EbNo2=10.0**(EbNodB_range2/10.0)
noise_const2 = np.sqrt(1/(2*R*EbNo2))

for rep in range(10):
    deco_scatter_plot = []
    deco_scatter_plot2 = []
    for i in range(0, M):
        noise_scatter = noise_const * np.random.randn(1, n_channel)
        temp2 = np.zeros(M)
        temp2[i] = 1
        decoded_point = encoder.predict(np.expand_dims(temp2, axis=0)) + noise_scatter
        deco_scatter_plot.append(decoded_point)

        #SNR=20
        noise_scatter2 = noise_const2 * np.random.randn(1, n_channel)
        temp22 = np.zeros(M)
        temp22[i] = 1
        decoded_point2 = encoder.predict(np.expand_dims(temp22, axis=0)) + noise_scatter2
        deco_scatter_plot2.append(decoded_point2)

        
    deco_scatter_plot = np.array(deco_scatter_plot)
    deco_scatter_plot = deco_scatter_plot.reshape(M, 2, 1)
    plt.subplot(1,3,2)
    plt.scatter(deco_scatter_plot[:, 0], deco_scatter_plot[:, 1],color='red')
    plt.title('SNR=10dB')
    plt.xlabel('fase (I)')
    plt.ylabel('cuadratura (Q)')
    plt.legend()
    plt.grid(True)

    deco_scatter_plot2 = np.array(deco_scatter_plot2)
    deco_scatter_plot2 = deco_scatter_plot2.reshape(M, 2, 1)
    plt.subplot(1,3,3)
    plt.scatter(deco_scatter_plot2[:, 0], deco_scatter_plot2[:, 1],color='red')
    plt.title('SNR=20dB')
    plt.xlabel('fase (I)')
    plt.ylabel('cuadratura (Q)')
    plt.legend()
    plt.grid(True)

    
# Agregando un título general
plt.suptitle(f"Diagrama de Constelación - AE({n_channel},{int(k)})", fontsize=16)

#plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste del diseño para no solapar con el título
plt.tight_layout()
plt.show()
'''
####################
######## GRAFICA AE(n,4)

x_emb = encoder.predict(test_data)


# Asegúrate de que x_emb1 sea un array de NumPy
x_emb1 = np.array(x_emb)

# Guardar x_emb1 en un archivo de texto
np.savetxt('x_emb1_values_3_2_3dB.txt', x_emb1, fmt='%.2f')



x_emb = encoder.predict(test_data)
#noise_std = np.sqrt(1/(2*R*EbNo_train))
noise = noise_const * np.random.randn(N,n_channel)
x_emb = x_emb 
X_embedded = TSNE(learning_rate=1000,n_components=2, method='exact',random_state=0, perplexity=M/2,early_exaggeration=M,init='random').fit_transform(x_emb)
print (X_embedded.shape)
X_embedded1 = X_embedded

#X_embedded_m = X_embedded - tf.reduce_mean(X_embedded)
X_embedded_m = X_embedded - np.mean(X_embedded)
mean_squared2=np.mean(np.square(X_embedded_m))

s=np.sum(np.square(X_embedded))
n = X_embedded_m.shape[1]
X_P_normalized =  X_embedded_m / np.sqrt(2*mean_squared2)
X_E_normalized =  X_embedded_m / np.sqrt(s/(n/2))


plt.figure(figsize=(10, 5))
#plt.subplot(2,2,2)
plt.subplot(1, 2, 1)
#plt.figure(figsize=(5, 5))
plt.scatter(X_P_normalized[:,0],X_P_normalized[:,1], c=test_label, cmap='viridis')
#plt.title('TSNE normalizado Potencia promedio')
plt.title(f"Mapeo TSNE - AE({n_channel},{int(k)})")
plt.axis((-1.5,1.5,-1.5,1.5))
plt.tight_layout()
plt.xlabel("fase (I)")
plt.ylabel("cuadratura (Q)")
plt.grid(True)

"""
plt.figure(figsize=(5, 5))
#plt.subplot(2,2,3)
plt.scatter(X_E_normalized[:,0],X_E_normalized[:,1], c=test_label, cmap='viridis')
#plt.title('TSNE normalizado Energia')
plt.title(f"Constelacion - AE({n_channel},{int(k)})")
plt.axis((-1.5,1.5,-1.5,1.5))
plt.grid()
plt.xlabel("Fase (I)")
plt.ylabel("Cuadratura (Q)")
"""


######################
####################
######## GRAFICA AE(n,4)
# Aplicar UMAP en lugar de t-SNE

x_emb = encoder.predict(test_data)
noise = noise_const * np.random.randn(N, n_channel)
x_emb = x_emb 
#umap_reducer = umap.UMAP(n_components=2, random_state=0, repulsion_strength=8, min_dist=0.8,spread=8,n_neighbors=80)
umap_reducer = umap.UMAP(n_components=2, random_state=300, repulsion_strength=5, min_dist=1.0,spread=10,n_neighbors=80)
X_embedded = umap_reducer.fit_transform(x_emb)

print(X_embedded.shape)



# Normalizar los datos
X_embedded_m = X_embedded - np.mean(X_embedded)
mean_squared2 = np.mean(np.square(X_embedded_m))
s = np.sum(np.square(X_embedded))
n = X_embedded_m.shape[1]
X_P_normalized = X_embedded_m / np.sqrt(2*mean_squared2)
#X_P_normalized=X_embedded
average_energy=np.mean(np.square(np.abs(X_P_normalized)))
normalization_factor=np.sqrt(average_energy)
#X_P_normalized= X_P_normalized/ normalization_factor
X_E_normalized = X_embedded_m / np.sqrt(s/(n/2))


# Graficar la constelación usando UMAP
#plt.figure(figsize=(5, 5))
plt.subplot(1, 2, 2)
plt.scatter(X_P_normalized[:, 0], X_P_normalized[:, 1], c=test_label, cmap='viridis')
plt.title(f"Mapeo UMAP - AE({n_channel}, {int(k)})")
#plt.axis((-1.5, 1.5, -1.5, 1.5))
plt.tight_layout()
plt.xlabel("fase (I)")
plt.ylabel("cuadratura (Q)")
plt.grid()
plt.show()

