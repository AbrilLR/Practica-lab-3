# Problema del coctel 
## Descripción 

En este laboratorio se aborda el problema de la "fiesta de coctel", el cual consiste en la separación de señales de voz capturadas simultáneamente por varios micrófonos en un entorno con múltiples emisores de sonido. El objetivo es simular una situación real donde varias personas hablan al mismo tiempo y se desea extraer la voz de un participante específico, a pesar del ruido ambiente y la mezcla de voces.
Se utilizaron tres micrófonos distribuidos dentro de una sala, donde los tres integrantes del grupo hablan al mismo tiempo. Cada micrófono captura una mezcla de las voces y del ruido del entorno. Posteriormente, las señales grabadas son procesadas mediante análisis temporal y espectral para identificar las características principales de cada fuente sonora.
Además, se aplican técnicas de separación de fuentes, como el Análisis de Componentes Independientes (ICA), con el fin de aislar la señal de interés. Finalmente, se evalúa el SNR, comparando las señales aisladas con las originales.

## Captura de la señal 
Para el experimento, se utilizaron tres fuentes de sonido correspondientes a tres voces diferentes. La captura de la señal fue realizada con los siguientes dispositivos: iPad, iPad pro y un iPhone.
Adicionalmente a los 3 audios en los cuales se captaron a las tres fuentes de sonido, también se capturó el ruido ambiente del espacio en el cual se realizó la grabación con el fin de calcular la relación señal-ruido de las grabaciones de audio.

Los participantes se distribuyeron en diferentes puntos de la sala, mientras que los dispositivos de grabación se ubicaron de la siguiente manera:
Los dos las dos tablets se colocaron sobre la mesa central y el teléfono se situó en el piso. Con el fin de minimizar posibles interferencias durante la captura de las señales, se procuró que los micrófonos no apuntaran directamente hacia el participante más cercano o que apuntaran en la misma dirección, buscando disminuir la influencia predominante de una sola fuente sonora en cada micrófono o que dos micrófonos llegaran a captar la misma fuente.

![Distancias](https://github.com/user-attachments/assets/2b37c924-774b-48f6-b2b7-45ecbdc36f7c)


En la imagen se muestra la disposición de cada uno los elementos y las distancias entre cada fuente de sonido y cada micrófono. 
El ambiente en el cual se realizaron las grabaciones era un espacio cerrado, pero en el cual aun así se pueden apreciar interferencias de sonidos del ambiente las cuales provienen del exterior.

### Digitalización 

Se utilizó una frecuencia de muestreo de 44,100 Hz. Esta frecuencia cumple con el criterio de Nyquist para señales de audio, permitiendo capturar adecuadamente señales con componentes de frecuencia de hasta 22,050 Hz, lo cual es suficiente para abarcar el rango audible por el ser humano el cual va de 20 a 20k Hz aproximadamente.

 ![image](https://github.com/user-attachments/assets/042ef023-7310-4185-9825-0a37272b0c02)



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
Despues de realizado el calculo el SNR de cada audio dio como resultado lo mostrado en la imagen, estos resultados reflejan que las grabaciones poseen una calidad buena, ya que se encuentran dentro de un rango donde el ruido es bajo y no interfiere de manera significativa con la señal. Específicamente, el audio 2 muestra la mejor calidad, con un SNR cercano a los 40 dB, lo que indica poca presencia  de ruido.

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
### Analisís de componentes independientes (ICA)
Este método se basa en la separación de distintas fuentes suponiendo que son estadisticamente independientes, por lo que es util cuando las señales se graban en distintos microfonos, sin embargo tiené problemas si las señales presentan alta correlación entre si, el código implementado crea un arreglo en donde se mezclan los audios y se utiliza la libreria FastICA para aplicar el método, por ultimo se gráfican las señales obtenidas, se guardan los audios en archivos .wav y se imprimen nuevos snr que nos mostraran si hubo una mejora en la calidad de la señal.

```python
X = np.vstack([audio1, audio2, audio3]).T

# Aplicar ICA
ica = FastICA(n_components=3)
sources = ica.fit_transform(X)

# Guardar los audios separados por ICA
for i in range(3):
    sf.write(f"fuente_{i+1}.wav", sources[:, i], sr)


# ----------- Graficar las señales separadas por ICA -----------

plt.figure(figsize=(12, 8))

tiempo = np.linspace(0, len(sources) / sr, num=len(sources))

for i in range(3):
    plt.subplot(3, 1, i + 1)
    plt.plot(tiempo, sources[:, i], alpha=0.7)
    plt.title(f'Señal separada por ICA {i + 1}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid()

plt.tight_layout()
plt.show()
```
![Figure_1](https://github.com/user-attachments/assets/c1bb572a-6fae-4b16-bb74-aa3ecefd92c2)

Podemos observar que la amplitud de las señales 1 y 2 se ve afectada por un ruido de tipo impulso, sin embargo al no ser algo repetitivo no representa algo relevante en el tratamiento de la señal.

```python

#Snr por ICA
ica_sen1 = sources[:min_len, 0]
ica_sen2 = sources[:min_len, 1]
ica_sen3 = sources[:min_len, 2]

print ("SNR señales aisladas por ICA \n")

potencia_ruido = np.mean(r_sig ** 2)

potencia_ica_sen1 = np.mean(ica_sen1 ** 2)
snr_ica1 = 10 * np.log10(potencia_ica_sen1 / potencia_ruido)
print(f"SNR de la señal separada por ICA 1:", snr_ica1, "dB")

potencia_ica_sen2 = np.mean(ica_sen2 ** 2)
snr_ica2 = 10 * np.log10(potencia_ica_sen2 / potencia_ruido)
print("SNR de la señal separada por ICA 2:", snr_ica2, "dB")

potencia_ica_sen3 = np.mean(ica_sen3 ** 2)
snr_ica3 = 10 * np.log10(potencia_ica_sen3 / potencia_ruido)
print("SNR de la señal separada por ICA 3:", snr_ica3, "dB")
```
![Captura de pantalla 2025-03-06 234500](https://github.com/user-attachments/assets/5eb45377-e405-4d82-9df9-1bc899e4af82)

El snr de las señales mejora ligeramente por lo que la potencia de las 3 es mayor a lo que era antes del ICA, sin embargo al escuchar el audio no hay un aislamiento claro, sino que se escuchan señales bastante saturadas, esto debido a que el método está diseñado para microfonos aislados, no un arreglo como el que se utilizó en este caso.

### Beamforming
es un método que explota la información espacial de las fuentes y los micrófonos para enfocar la captación del sonido en una dirección específica. Es decir, usa la diferencia en los tiempos de llegada del sonido a cada micrófono para mejorar la captura de la fuente deseada, es un método util para un arreglo de microfonos con fuentes de sonido en posiciones medidas, por esto mismo se crea el eje de cooordenadas tomando el microfono del celular como la posición (0,0) y gracias a las fotos tomadas, se logra aproximar la posición de los otros elementos respecto a la refencia.
```python
# Preparar las señales originales para beamforming
sigs = [audio1, audio2, audio3]

# Parámetros físicos
c = 343  # m/s

# Posiciones
mic_positions = np.array([[0, 0], [1.52, 2.16], [2.20, 2.10]])
source_positions = np.array([[1.52, 0], [0, 3.85], [3.86, 0]])

# Función de alineación
def align_sigs(sigs, mic_positions, source_positions, fs):
    num_sources = len(source_positions)
    aligned_sigs = np.zeros((num_sources, len(sigs[0])))

    for i, src_pos in enumerate(source_positions):
        delays = np.zeros(len(mic_positions))
        for j, mic_pos in enumerate(mic_positions):
            distance = np.linalg.norm(mic_pos - src_pos)
            delays[j] = distance / c
        delay_samples = np.round(delays * fs).astype(int)

        for j in range(len(sigs)):
            sig = sigs[j]
            shifted_sig = np.roll(sig, -delay_samples[j])
            aligned_sigs[i] += shifted_sig[:len(sigs[0])]

    return aligned_sigs

# Aplicar beamforming
aligned_sigs = align_sigs(sigs, mic_positions, source_positions, sr)

# Guardar resultados
for i in range(aligned_sigs.shape[0]):
    sf.write(f'beamformed_{i+1}.wav', aligned_sigs[i], sr)

# Graficar
plt.figure(figsize=(12, 8))
for i in range(aligned_sigs.shape[0]):
    plt.subplot(3, 1, i + 1)
    librosa.display.waveshow(aligned_sigs[i], sr=sr)
    plt.title(f'Señal separada por Beamforming {i+1}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid()

plt.tight_layout()
plt.show()
```
![Figure_2](https://github.com/user-attachments/assets/d1b114f9-14d5-4047-8fb7-035164cc3782)

La amplitud y forma de las señales separadas es bastante similar entre las 3 señales.
```python
r_sig, _ = librosa.load(r'C:\Users\sebas\OneDrive\Escritorio\Johitan\Labs de señales\Lab3fin\Nueva.wav', sr=sr, mono=True)


min_len = min(len(r_sig), aligned_sigs.shape[1])
r_sig = r_sig[:min_len]

#Snr por beanforming
sen1 = aligned_sigs[0][:min_len]
sen2 = aligned_sigs[1][:min_len]
sen3 = aligned_sigs[2][:min_len]

print (f"SNR señales aisladas por beamforming \n")
potencia_ruido = np.mean(r_sig ** 2)

potencia_sen1 = np.mean(sen1 ** 2)
snr1 = 10 * np.log10(potencia_sen1 / potencia_ruido)
print("SNR de la señal separada 1:", snr1, "dB")

potencia_sen2 = np.mean(sen2 ** 2)
snr2 = 10 * np.log10(potencia_sen2 / potencia_ruido)
print("SNR de la señal separada 2:", snr2, "dB")

potencia_sen3 = np.mean(sen3 ** 2)
snr3 = 10 * np.log10(potencia_sen3 / potencia_ruido)
print("SNR de la señal separada 3:", snr3, "dB \n")

```
![Captura de pantalla 2025-03-06 234522](https://github.com/user-attachments/assets/ce1e586d-ce22-4698-88c2-75baa5320590)

El snr de todas las señales mejora considerablemente, reafirmando que el método por beamforming resulta dar mejores resultados para el experimento del coctel, sin embargo no es suficiente para escuchar una voz completamente aislada, sin embargo el audio "beamformed 3" es donde más se llega a notar que la voz de interes se oye de manera más clara a las otras, además estás otras voces se escuchan más bajas, como si fueran ruido de fondo. Estos resultados se pueden atribuir al hecho de que las distancias en el eje de coordenadas se encuentran bastante aproximadas debido al entorno en el que se realizó el arreglo de microfonos, además el hecho de que los diferentes microfonos sean de diferentes tipos de dispositivos tambien influye en el resultado del experimento.





### Requisitos 
* Pyton 3.9.0 ó superior
* 3 micrófonos
* 3 voces o fuentes de sonido
### Librerias
* FastICA
* numpy
* matplotlib
* scipy.stats
* librosa
