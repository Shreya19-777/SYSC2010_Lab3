import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy.signal import butter

# 2.2 Generating ECG signal
ecg_sig = nk.ecg_simulate(duration = 10, sampling_rate = 1000)
#Cunstructing time axis
time = np.arange(len(ecg_sig))/1000 #which is 10 seconds
#plotting the ECG signal
plt.figure(figsize=(12,6))
plt.plot(time, ecg_sig)
plt.title("Simulated ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(0,11,0.5))
plt.show()

#3.2 Downsampling the ECG signal
#500Hz
ecg_sig_500 = nk.ecg_simulate(duration = 10, sampling_rate = 500)
#Cunstructing time axis
time = np.arange(len(ecg_sig_500))/500 #which is 10 seconds
#plotting the ECG signal
plt.figure(figsize=(12,6))
plt.plot(time, ecg_sig_500)
plt.title("Simulated ECG Signal(500Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(0,11,0.5))
plt.show()

#  250 Hz
ecg_sig_250 = nk.ecg_simulate(duration = 10, sampling_rate = 250)
#Constructing time axis
time = np.arange(len(ecg_sig_250))/250 #which is 10 seconds
#plotting the ECG signal
plt.figure(figsize=(12,6))
plt.plot(time, ecg_sig_250)
plt.title("Simulated ECG Signal (250Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(0,11,0.5))
plt.show()
# 125 Hz
ecg_sig_125 = nk.ecg_simulate(duration = 10, sampling_rate = 125)
#Constructing time axis
time = np.arange(len(ecg_sig_125))/125 #which is 10 seconds
#plotting the ECG signal
plt.figure(figsize=(12,6))
plt.plot(time, ecg_sig_125)
plt.title("Simulated ECG Signal (125Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(0,11,0.5))
plt.show()

#Section 4 FFT Transform
#4.2
fft_values = np.fft.rfft(ecg_sig)
magnitude = np.abs(fft_values)
sampling_rate = 250
freq_axis = np.fft.rfftfreq(len(ecg_sig), d=1/sampling_rate)

# Plot
plt.figure()
plt.plot(freq_axis, magnitude)
plt.title("Frequency Spectrum of ECG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 50)
plt.grid()
plt.show()

#5.2 Adding noise to the ECG signal
noise = np.random.randn(len(ecg_sig))*0.1
noisy_ecg = ecg_sig + noise
time = np.arange(len(noisy_ecg))/1000 #which is 10 seconds
#plotting the ECG signal
plt.figure(figsize=(12,6))
plt.plot(time, noisy_ecg)
plt.title("Simulated ECG Signal with Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xticks(np.arange(0,11,0.5))
plt.show()

#6.2

#7.2 Low Pass Filter 
butter(4, 40, btype='low', fs=1000)