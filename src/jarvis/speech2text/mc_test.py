import sounddevice as sd
import numpy as np

def callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print("|" * int(volume_norm))

with sd.InputStream(callback=callback):
    input("Speak into the mic. Press Enter to stop...\n")
