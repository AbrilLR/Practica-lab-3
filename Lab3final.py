from scipy.io import wavfile
import numpy as np
from scipy.fft import fft, fftfreq
import soundfile as sf
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import librosa
import librosa.display
import os


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

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# ----------- Mostrar información de cada audio -----------

# Audio 1
duracion1 = len(audio_voces) / sr
print(f"Audio 1:")
print(f"- Duración: {duracion1:.2f} segundos")
print(f"- Cantidad de muestras: {len(audio_voces)}")
print(f"- Frecuencia de muestreo: {sr} Hz\n")

# Audio 2
duracion2 = len(audio_voces2) / sr
print(f"Audio 2:")
print(f"- Duración: {duracion2:.2f} segundos")
print(f"- Cantidad de muestras: {len(audio_voces2)}")
print(f"- Frecuencia de muestreo: {sr} Hz\n")

# Audio 3
duracion3 = len(audio_voces3) / sr
print(f"Audio 3:")
print(f"- Duración: {duracion3:.2f} segundos")
print(f"- Cantidad de muestras: {len(audio_voces3)}")
print(f"- Frecuencia de muestreo: {sr} Hz\n")



plt.figure(figsize=(12, 10))

# Señal 1
tiempo1 = np.linspace(0, duracion1, num=len(audio_voces))
plt.subplot(3, 1, 1)
plt.plot(tiempo1, audio_voces, color="b", alpha=0.7)
plt.title("Señal 1 en el Dominio del Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

# Señal 2
tiempo2 = np.linspace(0, duracion2, num=len(audio_voces2))
plt.subplot(3, 1, 2)
plt.plot(tiempo2, audio_voces2, color="g", alpha=0.7)
plt.title("Señal 2 en el Dominio del Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

# Señal 3
tiempo3 = np.linspace(0, duracion3, num=len(audio_voces3))
plt.subplot(3, 1, 3)
plt.plot(tiempo3, audio_voces3, color="r", alpha=0.7)
plt.title("Señal 3 en el Dominio del Tiempo")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

plt.tight_layout()
plt.show()



plt.figure(figsize=(12, 10))

# FFT de la señal 1
fxt1 = np.fft.fft(audio_voces)
frecuencias1 = np.fft.fftfreq(len(audio_voces), d=1/sr)
mask1 = (frecuencias1 >= 0) & (frecuencias1 <= 1000)
plt.subplot(3, 1, 1)
plt.plot(frecuencias1[mask1], np.abs(fxt1[mask1]), color="b")
plt.title("FFT de la Señal 1")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

# FFT de la señal 2
fxt2 = np.fft.fft(audio_voces2)
frecuencias2 = np.fft.fftfreq(len(audio_voces2), d=1/sr)
mask2 = (frecuencias2 >= 0) & (frecuencias2 <= 1000)
plt.subplot(3, 1, 2)
plt.plot(frecuencias2[mask2], np.abs(fxt2[mask2]), color="g")
plt.title("FFT de la Señal 2")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

# FFT de la señal 3
fxt3 = np.fft.fft(audio_voces3)
frecuencias3 = np.fft.fftfreq(len(audio_voces3), d=1/sr)
mask3 = (frecuencias3 >= 0) & (frecuencias3 <= 1000)
plt.subplot(3, 1, 3)
plt.plot(frecuencias3[mask3], np.abs(fxt3[mask3]), color="r")
plt.title("FFT de la Señal 3")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid()

plt.tight_layout()
plt.show()

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


r_sig, _ = librosa.load(r'C:\Users\ACER\Documents\Antecedentes\Señales\Lab3\Nueva.wav', sr=sr, mono=True)


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



