import time
from tqdm import tqdm


for i in tqdm(range(100), desc="training"):
    tqdm.write(f"Epoch number: 1")
    tqdm.write(f"Batch loss {i}: {3.5 - (i*0.05)}")
    tqdm.write(f"")
    time.sleep(0.3)