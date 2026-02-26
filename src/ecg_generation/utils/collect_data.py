import wfdb
import mne
import matplotlib.pyplot as plt

path_scp = "/home/rose/Cortrium/12-lead-ecg-generation/data/SCP/test3"
path_edf = "/home/rose/Cortrium/12-lead-ecg-generation/data/C3/rose.EDF"

signal = wfdb.rdrecord(path_scp)
#annotation = wfdb.rdann(path, extension="hea")

raw = mne.io.read_raw_edf(path_edf, preload=True)

fig, axs = plt.subplots(12)
for i in range (12):
    axs[i].plot(signal.p_signal[:,i])
plt.savefig("data.png")