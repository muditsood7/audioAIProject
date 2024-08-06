import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "PinkPanther60.wav"

signal, sr = librosa.load(file, sr=22050) # stores samles for each second of the file in signal array. length * seconds
librosa.display.waveshow(signal, sr=sr)
#plt.xlabel("Time")
#plt.ylabel("Amplitude")
#plt.show()

# perform fast fourier transform
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency) / 2)] # split in half because plot is symmetrical
left_magnitude = magnitude[:int(len(magnitude) / 2)]


# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# perform stft to get spectrogram

n_fft = 2048 # this amount of samples for each fourier transform
hop_length = 512 # shift right after each transform

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)

#librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
#plt.xlabel("Time")
#plt.ylabel("Frequency")
#plt.colorbar()
#plt.show()

# Calculate MFCCs

MFCCs = librosa.feature.mfcc(y=signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()