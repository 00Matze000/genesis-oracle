import numpy as np
import matplotlib.pyplot as plt
import os
import random

def main():
    # 1. Dynamisches Wellensignal generieren
    t = np.linspace(0, 10, 500)
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(500)
    
    # 2. Secret Malfunction einführen (Clipping Artefakt)
    malfunction_start = random.randint(150, 350)
    malfunction_end = malfunction_start + 30
    
    # Hässliches hochfrequentes Rauschen / Clipping
    signal[malfunction_start:malfunction_end] += 2.0 * np.sin(2 * np.pi * 25 * t[malfunction_start:malfunction_end])
    signal = np.clip(signal, -1.5, 1.5)
    
    # 3. Plot erstellen
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, color='teal', linewidth=1.2)
    plt.title("Sensor Array Output")
    plt.xlabel("Timestep")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 4. Ohne Timestamp speichern
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    save_path = os.path.join(data_dir, 'audit_target.png')
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
