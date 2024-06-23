# ---------------------------------------------------
# Autores: Otavalo D.Andres, Zaruma Samantha 
# Descripcion: Trabajo de titulacion para la implementacion de un 
# auto codificador. Este codigo ejecutar las 10 simulaciones 
# datos para el entrenamiento. 
# Fecha: 2024-06-21
# Version: 1.0
# ---------------------------------------------------

from scipy.special import erfc
import numpy as np
import os
import pandas as pd
import scipy.stats as stats
import subprocess
import matplotlib.pyplot as plt

from fractions import Fraction
def main():
    # Nombre de la carpeta donde se guardarán los resultados
    output_folder = "datos_AEn4"
    
    # Asegurarse de que la carpeta de salida exista
    os.makedirs(output_folder, exist_ok=True)
    script_name = 'AE_Operation_BER.py'
    param2 = output_folder
   
    
    for i in range(1, 11):
        print(f'Iteracion {i}')

        # Ejecutar el comando
        subprocess.run(['python', 'AE_Operation_BER.py', str(i), output_folder])

if __name__ == "__main__":
    main()




