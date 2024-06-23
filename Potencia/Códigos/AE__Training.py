# ---------------------------------------------------
# Autores: Otavalo D.Andres, Zaruma Samantha 
# Descripcion: Trabajo de titulacion para la implementacion de un 
# auto codificador. Este codigo se define la arquitectura, parametros y
# datos para el entrenamiento. 
# Fecha: 2024-06-21
# Version: 1.0
# ---------------------------------------------------
import numpy as np
import math as math
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import random as rn
import os 
import time
from keras import backend as K
from EnergyNormalizationLayer import EnergyNormalizationLayer
from tensorflow.keras.callbacks import EarlyStopping

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#inicializar semilla
np.random.seed(42)
tf.random.set_seed(42)


    # DEFINIR PARAMETRO
A=1   #amplitud E
M = 2 # numero de simbolos
k = np.log2(M)
# tasa de código
n_channel = 3
R = k/n_channel
print ('M:',M,'k:',k)
print ('R:',int(k/R))


#generando datos de tamaño N
N = 25000*M

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
unique, counts = np.unique(data, axis=0, return_counts=True)
for i, u in enumerate(unique):
    print(f"Combinación {u}: {counts[i]} veces")



#MODELO DE AUTOENCODER
#CAPA DE ENTRADA, vector de longitud M.
input_signal = Input(shape=(M,)) 
#primera capa de codificación: CAPA DENSA CON M NEURONAS, ACTIVACION ReLu
encoded = Dense(M, activation='relu')(input_signal)
#SEGUNDA capa de codificación: CAPA DENSA CON n NEURONAS, ACTIVACION LINEAL
encoded1 = Dense(n_channel, activation='linear')(encoded)
#TERCERA capa de codificación: normalización de energia 
encoded2 = EnergyNormalizationLayer(A,n_channel)(encoded1)

#Añade ruido gaussiano a la señal, simulando así el efecto del canal de comunicación ruidoso
EbNo_train_dB = 3
EbNo_train=10.0**(EbNo_train_dB/10.0)
encoded3 = GaussianNoise(np.sqrt(1/(2*R*EbNo_train)))(encoded2)
# -------------- DECODIFICADOR
#primera capa de decodificación: CAPA DENSA CON M NEURONAS, ACTIVACION ReLu
decoded = Dense(M, activation='relu')(encoded3)
#SEGUNDA capa de decodificación: CAPA DENSA M 16 NEURONAS, ACTIVACION SOFTMAX
decoded1 = Dense(M, activation='softmax')(decoded)

#Crea el modelo de autoencoder especificando la señal de entrada y la salida reconstruida.
autoencoder = Model(input_signal, decoded1)
# Compilamos el modelo autoencoder con los siguientes parámetros:
# - Optimizador: 'adam', que ajusta dinámicamente las tasas de aprendizaje de los parámetros.
# - Función de pérdida: 'categorical_crossentropy', adecuada para problemas de clasificación multiclase.
# - Métricas: ['accuracy'], para medir la fracción de predicciones correctas.
autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Imprime un resumen del modelo, incluyendo el número de parámetros y la arquitectura del autoencoder.                    
print (autoencoder.summary())


#SET PARA VALIDACION
N_val =200*M
num_val_class = N_val // M
val_data = []

for i in range(M):
    for _ in range(num_val_class):
        temp2 = np.zeros(M,dtype=np.float32)
        temp2[i] = 1
        val_data.append(temp2)

val_data = np.array(val_data,dtype=np.float32)
np.random.shuffle(val_data)
unique_val, counts_val = np.unique(val_data, axis=0, return_counts=True)

for i, u in enumerate(unique_val):
    print(f"validacion Combinación {u}: {counts_val[i]} veces")

# Callback EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
# Define tu TimeHistory callback
time_callback = TimeHistory()

#ENTRENAMIENTO
historial = autoencoder.fit(data, data,
                epochs=50,
                batch_size=64,
                validation_data=(val_data, val_data),
                callbacks=[early_stopping, time_callback])

total_training_time = sum(time_callback.times)
print(f"Tiempo total de entrenamiento: {total_training_time:.2f} segundos")
# Accediendo a los valores de la función de pérdida y precision
loss = historial.history['loss']
val_loss = historial.history['val_loss']
accuracy = historial.history['accuracy']
val_accuracy = historial.history['val_accuracy']

final_epoch = len(historial.history['loss']) -1
print(f"Numero de epocas: {final_epoch+1}")

# Recuperar los valores finales de pérdida para entrenamiento y validación
final_train_loss = historial.history['loss'][final_epoch]
final_val_loss = historial.history['val_loss'][final_epoch]
final_train_accuracy = historial.history['accuracy'][final_epoch]
final_val_accuracy = historial.history['val_accuracy'][final_epoch]

print(f"Valor final de la pérdida de entrenamiento: {final_train_loss:.4f}")
print(f"Valor final de la pérdida de validación: {final_val_loss:.4f}")
print(f"Valor final de la precisión de entrenamiento: {final_train_accuracy:.4f}")
print(f"Valor final de la precisión de validación: {final_val_accuracy:.4f}")

# Creando el rango de épocas
epochs = range(1, len(loss) + 1)


# Graficando la función de pérdida del entrenamiento
plt.figure(figsize=(12, 5))
# Graficando pérdida
plt.subplot(1, 2, 1)  # 1 fila, 2 columnas, primer subplot
plt.plot(epochs, loss, 'bo', label='valor de Pérdida')
plt.plot(epochs, val_loss, 'b', label='valor de validación')
plt.title('Funcion de Pérdida en  el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Graficando precisión
plt.subplot(1, 2, 2) 
plt.plot(epochs, accuracy, 'ro', label='valor de Precisión ')  
plt.plot(epochs, val_accuracy, 'r', label='valor de Validación ')  
plt.title('Precisión en el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()  
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.suptitle(f'Resultados del Entrenamiento del Modelo - AE({n_channel},{int(k)})', fontsize=16)

plt.show()
#Guarda el modelo entrenado
autoencoder.save('3_1_AE_energy_3dB_relu_coded_fsk.model')

