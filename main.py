import matplotlib.pyplot as plt
import numpy as np
import neurokit2 as nk
from scipy.signal import butter, filtfilt


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
plt.figure(figsize=(12,6))
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
noisy_fft_values = np.fft.rfft(noisy_ecg)
noisy_magnitude = np.abs(noisy_fft_values)

# 2. Plotting the comparison
plt.figure(figsize=(12, 6))

# Plot 1: Clean ECG Spectrum
plt.subplot(2, 1, 1)
plt.plot(freq_axis, magnitude, color='blue')
plt.title("Frequency Spectrum: Clean ECG")
plt.xlim(0, 100) # ECG plus a bit of room for noise observation
plt.ylabel("Magnitude")
plt.grid(True)

# Subplot 2: Noisy ECG Spectrum
plt.subplot(2, 1, 2)
plt.plot(freq_axis, noisy_magnitude, color='red')
plt.title("Frequency Spectrum: Noisy ECG")
plt.xlim(0, 100)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()


#7.2 Low Pass Filter 
def butter_lp(data, cutoff, fs, order=5):
    nyquist_f = 0.5 * fs
    normal_cutoff = cutoff / nyquist_f
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Applying filter with 40 Hz cutoff
cutoff_freq = 40
filtered_ecg = butter_lp(noisy_ecg, cutoff_freq, sampling_rate)

plt.figure(figsize=(12, 6))
plt.plot(noisy_ecg, label='Noisy ECG', color='blue')
plt.plot(filtered_ecg, label='Filtered ECG', color='green')
plt.title("Noisy vs. Filtered ECG")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

#Plotting the FFT
filtered_fft = np.fft.rfft(filtered_ecg)
filtered_magnitude = np.abs(filtered_fft)

plt.figure(figsize=(12, 6))
plt.plot(freq_axis, noisy_magnitude, label='Noisy signal', alpha=0.4, color='red')
plt.plot(freq_axis, filtered_magnitude, label='Filtered signal', color='blue')
plt.title("FFT of Filtered Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 60)
plt.legend()
plt.grid(True)
plt.show()