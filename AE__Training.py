import numpy as np
import math as math
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise,BatchNormalization, Lambda,Dropout,Normalization
from keras.models import Model
from keras import regularizers
from keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from keras.optimizers import Adam,SGD,Adadelta
import random as rn
import os 
import time  # Importar el módulo time
from keras import backend as K
#from EnergyNormalizationLayer import EnergyNormalizationLayer
from PowerNormalizationLayer import PowerNormalizationLayer
#from HelperAEWAWGNLayer import HelperAEWAWGNLayer
from tensorflow.keras.callbacks import EarlyStopping
# Registrar la capa personalizada
tf.keras.utils.get_custom_objects().update({'PowerNormalizationLayer': PowerNormalizationLayer})


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



np.random.seed(42)
tf.random.set_seed(42)


    # DEFINIR PARAMETROS
tf.random.set_seed(42) 
A=1   
M = 128
k = np.log2(M)
k = int(k)
print ('M:',M,'k:',k)

#generando datos de tamaño N
#N = 10000 * M
N = 1000000

#SET PARA ENTRENAMIENTO
num_per_class = N // M
data = []
for i in range(M):
    for _ in range(num_per_class):
        temp = np.zeros(M,dtype=np.float32)
        temp[i] = 1
        data.append(temp)
data = np.array(data,dtype=np.float32)
np.random.shuffle(data)
#print(data)
unique, counts = np.unique(data, axis=0, return_counts=True)
for i, u in enumerate(unique):
    print(f"Combinación {u}: {counts[i]} veces")

# tasa de código
n_channel = 14 #64- 2,6,8,12 #32-2,5,10 #128-2,7,14 #4-2,4 #16-2,4,8
R = k/n_channel
print (int(k/R))

#MODELO DE AUTOENCODER
#CAPA DE ENTRADA, cada muestra de entrada es un vector de longitud M.
input_signal = Input(shape=(M,)) 
#primera capa de codificación: CAPA DENSA CON 16 NEURONAS, ACTIVACION ReLu
encoded = Dense(M, activation='relu')(input_signal)
#SEGUNDA capa de codificación: CAPA DENSA CON 7 NEURONAS, ACTIVACION LINEAL
encoded1 = Dense(n_channel, activation='linear')(encoded)
#TERCERA capa de codificación: 
#normalización por lotes a la salida de la capa anterior. 
#encoded2 = EnergyNormalizationLayer(A,n_channel)(encoded1)
encoded2 = PowerNormalizationLayer(A,n_channel)(encoded1)

#Añade ruido gaussiano a la señal, simulando así el efecto del canal de comunicación ruidoso
#EbNo_train_dB = 3 + 10*math.log10(R)
EbNo_train_dB = 12 #5-32, 10-64, 12-128, 3-16
EbNo_train=10.0**(EbNo_train_dB/10.0)
#encoded3 = GaussianNoise(np.sqrt(1/(2*R*k*EbNo_train)))(encoded2)
encoded3 = GaussianNoise(np.sqrt(1/(2*R*EbNo_train)))(encoded2)
# -------------- DECODIFICADOR
#primera capa de decodificación: CAPA DENSA CON 16 NEURONAS, ACTIVACION ReLu
decoded = Dense(M, activation='relu')(encoded3)
#SEGUNDA capa de decodificación: CAPA DENSA CON 16 NEURONAS, ACTIVACION SOFTMAX
#softmax como función de activación para obtener una distribución de probabilidad sobre las M clases posibles, 
decoded1 = Dense(M, activation='softmax')(decoded)

#Crea el modelo de autoencoder especificando la señal de entrada y la salida reconstruida.
autoencoder = Model(input_signal, decoded1)
#sgd = SGD(lr=0.08)
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Imprime un resumen del modelo, incluyendo el número de parámetros y la arquitectura del autoencoder.                    
print (autoencoder.summary())


#SET PARA VALIDACION
#N_val =250 * M
N_val =100000
num_val_class = N_val // M
val_data = []

for i in range(M):
    for _ in range(num_val_class):
        temp2 = np.zeros(M,dtype=np.float32)
        temp2[i] = 1
        val_data.append(temp2)

val_data = np.array(val_data,dtype=np.float32)
np.random.shuffle(val_data)
#print(data)
unique_val, counts_val = np.unique(val_data, axis=0, return_counts=True)

for i, u in enumerate(unique_val):
    print(f"validacion Combinación {u}: {counts_val[i]} veces")

# Callback EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

time_callback=TimeHistory()

# Medir el tiempo de entrenamiento
start_time = time.time()

#ENTRENAMIENTO
historial = autoencoder.fit(data, data,
                epochs=50,
                batch_size=1024,
                validation_data=(val_data, val_data),
                callbacks=[early_stopping,time_callback])

total_training_time=sum(time_callback.times)
print(f"Tiempo total de entrenamiento:{total_training_time:.2f} segundos")

end_time = time.time()
training_time = end_time - start_time
print(f"Tiempo de entrenamiento: {training_time:.2f} segundos")

# Accediendo a los valores de la función de pérdida
loss = historial.history['loss']
val_loss = historial.history['val_loss']
accuracy = historial.history['accuracy']
val_accuracy = historial.history['val_accuracy']

    # Creando el rango de épocas

epochs = range(1, len(loss) + 1)


# Graficando la función de pérdida del entrenamiento
plt.figure(figsize=(12, 5))
# Graficando pérdida
plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primer subplot
plt.plot(epochs, loss, 'bo', label='Valor de Pérdida')
plt.plot(epochs, val_loss, 'b', label='Valor de Validación')
plt.title('Función de Pérdida en el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Graficando precisión
plt.subplot(1, 2, 2)  # 1 fila, 2 columnas, segundo subplot
plt.plot(epochs, accuracy, 'ro', label='Valor de Precisión')  # 'ro' indica "red dot"
plt.plot(epochs, val_accuracy, 'r', label='Valor de Validación')  # 'r' indica "solid red line"
plt.title('Precisión en el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Agregando un título general
plt.suptitle(f"Resultados del Entrenamiento del Modelo - AE({n_channel}, {int(k)})", fontsize=16) 

#plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuste del diseño para no solapar con el título
plt.show()


#plt.tight_layout()  
#plt.show()
tf.keras.utils.get_custom_objects().update({'PowerNormalizationLayer': PowerNormalizationLayer})
autoencoder.save('14_7_AE_power_12dB_sk.keras')
