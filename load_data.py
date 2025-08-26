import wfdb
import pylab as pylab
import scipy
import biosppy
import matplotlib.pyplot as plt

path = "/Users/toucanfirm/Documents/Zibra/dev/Cortrium/lead_gen/data/physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/01/010/JS00001"

signal = wfdb.rdrecord(path)
#annotation = wfdb.rdann(path, extension="hea")

mat = scipy.io.loadmat(path)

r_peaks = biosppy.signals.ecg.ASI_segmenter(signal.p_signal[:,6], sampling_rate=500)

def butter(low, high):
    b, a = scipy.signal.butter(5, [low, high], btype='bandpass', fs=500)
    return scipy.signal.lfilter(b, a, signal.p_signal[:,6])

filtered_1 = butter(0.5,150)
filtered_2 = butter(1,120)
filtered_3 = butter(3,100)
filtered_4 = butter(3,70)
filtered_5 = butter(3,50)
filtered_6 = butter(5,50)





fig, axs = plt.subplots(7,1)

axs[0].plot(signal.p_signal[:,6])
axs[1].plot(filtered_1)
axs[2].plot(filtered_2)
axs[3].plot(filtered_3)
axs[4].plot(filtered_4)
axs[5].plot(filtered_5)
axs[6].plot(filtered_6)

for peak in r_peaks[0]:
    plt.axvline(x=peak)


plt.show()