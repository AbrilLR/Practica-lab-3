# Problema del coctel 
## Descripción 

En este laboratorio se aborda el problema de la "fiesta de coctel", el cual consiste en la separación de señales de voz capturadas simultáneamente por varios micrófonos en un entorno con múltiples emisores de sonido. El objetivo es simular una situación real donde varias personas hablan al mismo tiempo y se desea extraer la voz de un participante específico, a pesar del ruido ambiente y la mezcla de voces.
Se utilizaron tres micrófonos distribuidos dentro de una sala, donde los tres integrantes del grupo hablan al mismo tiempo. Cada micrófono captura una mezcla de las voces y del ruido del entorno. Posteriormente, las señales grabadas son procesadas mediante análisis temporal y espectral para identificar las características principales de cada fuente sonora.
Además, se aplican técnicas de separación de fuentes, como el Análisis de Componentes Independientes (ICA), con el fin de aislar la señal de interés. Finalmente, se evalúa el SNR, comparando las señales aisladas con las originales.

## Captura de la señal 

## Análisis temporal y espectral
Para la señal capturada por cada uno de los micrófonos se realizó una gráfica en función del tiempo para el análisis temporal, además se utilizó la transformada rápida de Fourier
(FFT) para obtener cada señal en función de la frecuencia, esta nos ayuda a identificar la intensidad de las diferentes frecuencias en la señal.

```python
#Audio 1
duracion = len(audio_voces) / sr_voces
print(f"Duración de la señal 1: ", duracion ," segundos")
tiempo = np.linspace(0, duracion, num=len(audio_voces))
plt.figure(figsize=(12, 6))
plt.plot(tiempo, audio_voces, color="b", alpha=0.7)
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Señal en el Dominio del Tiempo")
plt.grid()
plt.show()
print(f"Cantidad de muestras: " ,len(audio_voces))
print(f"Frecuencia de muestreo: " , sr_voces , " Hz")

#Fourier
fxt = np.fft.fft(audio_voces)
frecuencias = np.fft.fftfreq(len(audio_voces), d=1/sr_voces)
limite_frecuencia = 1000
mask = (frecuencias >= 0) & (frecuencias <= limite_frecuencia)
plt.figure(figsize=(10, 5))

# Magnitud de la FFT
plt.plot(frecuencias[mask], np.abs(fxt[mask]))
plt.title("FFT")
plt.ylabel("Magnitud")
plt.xlabel("Frecuencia (Hz)")
plt.grid()
plt.xlim(0, limite_frecuencia)  
plt.show()
```

![FFT](https://github.com/user-attachments/assets/1cf3ed33-85b6-4c74-a5f9-1fa2c5502331)

## Métodos de aislamiento de señales
Para el aislamiento de la voz de interés, se estudiaron 2 métodos: el analisís de componentes independientes (ICA) y el Beamforming
