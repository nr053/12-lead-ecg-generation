import wfdb
import glob
from tqdm import tqdm
import statistics

filenames = glob.glob("/home/rose/Cortrium/12-lead-ecg-generation/data/PTB" + '/**/*.dat', recursive=True) #find all the filenames with .dat file extension
times = []

for file in tqdm(filenames, desc="Processing"):
    tqdm.write(f"loading file: {file}")
    
    #load the file into a WFDB record object
    signal = wfdb.rdrecord(file.removesuffix(".dat"))
    annotation = wfdb.rdann(file.removesuffix(".dat"), extension="hea") 
    times.append(signal.__dict__["sig_len"] / signal.__dict__["fs"])


print(f"min: {min(times)}")
print(f"max: {max(times)}")
print(f"mean: {statistics.mean(times)}")