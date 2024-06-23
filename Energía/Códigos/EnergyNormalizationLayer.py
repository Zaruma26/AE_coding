# ---------------------------------------------------
# Autores: Otavalo D.Andres, Zaruma Samantha 
# Descripcion: Trabajo de titulacion para la implementacion de un 
# auto codificador. Este codigo se desarrolla se implementa una capa
# para el proceso de normalizacion en Energia. 
# Fecha: 2024-06-21
# Version: 1.0
# ---------------------------------------------------
import tensorflow as tf
import numpy as np
import math as math
import matplotlib.pyplot as plt
import seaborn as sns

class EnergyNormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, A, n):
        super(EnergyNormalizationLayer, self).__init__()
        self.A = A
        self.n = n
        
    def call(self, inputs):
        
        n = inputs.shape[1]  # Obtén la longitud del vector de entrada
        #valores para normalizar
        s=tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True) / (n/2)
        #normalizacion
        normalized_inputs = np.sqrt(self.A) *(inputs / tf.sqrt(s))
        #energia
        E=tf.reduce_sum(tf.square(normalized_inputs), axis=1, keepdims=True)
        
       
        return normalized_inputs
