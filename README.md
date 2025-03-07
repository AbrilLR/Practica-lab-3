# Problema del coctel 
## Descripción 

En este laboratorio se aborda el problema de la "fiesta de coctel", el cual consiste en la separación de señales de voz capturadas simultáneamente por varios micrófonos en un entorno con múltiples emisores de sonido. El objetivo es simular una situación real donde varias personas hablan al mismo tiempo y se desea extraer la voz de un participante específico, a pesar del ruido ambiente y la mezcla de voces.
Se utilizaron tres micrófonos distribuidos dentro de una sala, donde los tres integrantes del grupo hablan al mismo tiempo. Cada micrófono captura una mezcla de las voces y del ruido del entorno. Posteriormente, las señales grabadas son procesadas mediante análisis temporal y espectral para identificar las características principales de cada fuente sonora.
Además, se aplican técnicas de separación de fuentes, como el Análisis de Componentes Independientes (ICA), con el fin de aislar la señal de interés. Finalmente, se evalúa el SNR, comparando las señales aisladas con las originales.

## Captura de la señal 
Para el experimento, se utilizaron tres fuentes de sonido correspondientes a tres voces diferentes. La captura de la señal fue realizada con los siguientes dispositivos: iPad, iPad pro y un iPhone.
Adicionalmente a los 3 audios en los cuales se captaron a las tres fuentes de sonido, también se capturó el ruido ambiente del espacio en el cual se realizó la grabación con el fin de calcular la relación señal-ruido de las grabaciones de audio.
Los participantes se distribuyeron en diferentes puntos de la sala, mientras que los dispositivos de grabación se ubicaron de la siguiente manera:
Los dos las dos tablets se colocaron sobre la mesa central y el teléfono se situó en el piso. Con el fin de minimizar posibles interferencias durante la captura de las señales, se procuró que los micrófonos no apuntaran directamente hacia el participante más cercano o que apuntaran en la misma dirección, buscando disminuir la influencia predominante de una sola fuente sonora en cada micrófono o que dos micrófonos llegaran a captar la misma fuente

![Distancias](https://github.com/user-attachments/assets/2b37c924-774b-48f6-b2b7-45ecbdc36f7c)


En la imagen se muestra la disposición de cada uno los elementos y las distancias entre cada fuente de sonido y cada micrófono. 
El ambiente en el cual se realizaron las grabaciones era un espacio cerrado, pero en el cual aun así se pueden apreciar interferencias de sonidos del ambiente las cuales provienen del exterior.



## Calculo del SNR
Para el calculo del SNR de las señales, en primer lugar se cargaron los audios mediante la librería Librosa, las señales tambien fueron graficadas en el dominio del tiempo con el fin de  visualizar cómo varía la amplitud de cada una a lo largo del tiempo.


![vocesgraf](https://github.com/user-attachments/assets/7177d7be-7402-4af7-813e-45962e5f5f78)



Posteriormente se igualaron las duraciones de los audios con el fin evitar errores de cálculo o de código debido a la diferencia de datos entre ambas señales.
Finalmente, se calculó la potencia de cada señal y se realizó la operación de la siguiente manera.

```python
# Cargar audios 
audio_voces, sr = librosa.load(r'C:\Users\ACER\Documents\Antecedentes\Señales\Lab3\voces.wav', sr=None, mono=True)
audio_ruido, sr = librosa.load(r'C:\Users\ACER\Documents\Antecedentes\Señales\Lab3\Nueva.wav', sr=None, mono=True)
audio_voces2, sr = librosa.load(r'C:\Users\ACER\Documents\Antecedentes\Señales\Lab3\Universidad.wav', sr=None, mono=True)
audio_voces3, sr = librosa.load(r'C:\Users\ACER\Documents\Antecedentes\Señales\Lab3\voz5.wav', sr=None, mono=True)

# Igualar duración
min_len = min(len(audio_voces2), len(audio_voces3), len(audio_voces), len(audio_ruido))
audio1 = audio_voces[:min_len]
ruido = audio_ruido[:min_len]
audio2 = audio_voces2[:min_len]
audio3 = audio_voces3[:min_len]

# Calcular potencias
potencia_voces = np.mean(audio1 ** 2)
potencia_ruido = np.mean(ruido ** 2)
potencia_voces2 = np.mean(audio2 ** 2)
potencia_voces3 = np.mean(audio3 ** 2)

# Calcular SNR
snr = 10 * np.log10(potencia_voces / potencia_ruido)
snr2 = 10 * np.log10(potencia_voces2 / potencia_ruido)
snr3 = 10 * np.log10(potencia_voces3 / potencia_ruido)


print("SNR audio 1 antes de ser procesado: ", snr)
print("SNR audio 2 antes de ser procesado: ", snr2)
print("SNR audio 3 antes de ser procesado: ", snr3)
```
![image](https://github.com/user-attachments/assets/71e80ef0-0edd-4bec-9658-865ba977e884)
Despues de realizado el calculo el SNR de cada audio dio como resultado lo mostrado en la imagen, estos resultados reflejan que las grabaciones poseen una calidad buena, ya que se encuentran dentro de un rango donde el ruido es bajo y no interfiere de manera significativa con la señal. Específicamente, el audio 2 muestra la mejor calidad, con un SNR cercano a los 40 dB, lo que indica una presencia mínima de ruido.

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
En las graficas con respecto al tiempo podemos ver un pico de mayor amplitud aproximadamente a los 10 segundos el cual corresponde a un ruido proveniente del ambiente, en el resto del tiempo vemos una magnitud constante proveniente de las voces. En las gráficas de frecuencia podemos observar picos más altos en las correspondientes a las voces humanas, frecuencias más bajas para la voz masculina y frecuencias mas altas para las voces femeninas, además de otros picos causados por el ruido del ambiente.

![FFT](https://github.com/user-attachments/assets/1cf3ed33-85b6-4c74-a5f9-1fa2c5502331)


## Métodos de aislamiento de señales
Para el aislamiento de la voz de interés, se estudiaron 2 métodos: el analisís de componentes independientes (ICA) y el Beamforming
