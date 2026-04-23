import numpy as np
import matplotlib.pyplot as plt
import os

def generate_data():
    # Parameters based on Student ID 310
    T_val = 10
    omega_0 = 2 * np.pi / T_val
    R = 500  # 0.5 kOhm
    C = 1310 * 1e-6  # 1310 uF
    tau = R * C
    
    # Time vector: 100 periods, 1000 points per period for smoothness
    t = np.linspace(0, 100 * T_val, 100000)
    
    # 1. Fourier Synthesis (Square Wave)
    # f(t) = 4/pi * sum(sin(k*omega_0*t)/k) for k = 1, 3, 5, ..., 17
    harmonics = np.arange(1, 18, 2)
    signal = np.zeros_like(t)
    filtered_signal = np.zeros_like(t)
    
    for k in harmonics:
        omega_k = k * omega_0
        amplitude_k = 4 / (np.pi * k)
        
        # Original component
        component = amplitude_k * np.sin(omega_k * t)
        signal += component
        
        # 2. RC Filter Application
        # H(w) = 1 / (1 + j*w*R*C)
        # Magnitude |H| = 1 / sqrt(1 + (w*tau)^2)
        # Phase phi = -arctan(w*tau)
        mag_h = 1 / np.sqrt(1 + (omega_k * tau)**2)
        phase_h = -np.arctan(omega_k * tau)
        
        filtered_component = amplitude_k * mag_h * np.sin(omega_k * t + phase_h)
        filtered_signal += filtered_component

    # 3. Noise & Sabotage
    # Gaussian noise
    noise = np.random.normal(0, 0.05, size=t.shape)
    noisy_signal = filtered_signal + noise
    
    # Anomaly injection between period 70 and 75
    # Period 70 is at t = 700, Period 75 is at t = 750
    spike_mask = (t >= 70 * T_val) & (t <= 75 * T_val)
    # High frequency spike
    noisy_signal[spike_mask] += 2.0 * np.sin(50 * omega_0 * t[spike_mask])

    # 4. Save Data
    os.makedirs("data", exist_ok=True)
    np.save("data/datastream.npy", noisy_signal)
    
    # 5. Plotting
    plt.figure(figsize=(12, 6))
    
    # Normal window (e.g., period 10-15)
    plt.subplot(2, 1, 1)
    mask_normal = (t >= 10 * T_val) & (t <= 15 * T_val)
    plt.plot(t[mask_normal], noisy_signal[mask_normal], label="Filtered + Noise", color="blue")
    plt.plot(t[mask_normal], signal[mask_normal], label="Original Fourier", alpha=0.3, linestyle="--", color="gray")
    plt.title("Normal Signal Window (Periods 10-15)")
    plt.legend()
    plt.grid(True)
    
    # Anomaly window (Periods 65-80)
    plt.subplot(2, 1, 2)
    mask_anomaly = (t >= 65 * T_val) & (t <= 80 * T_val)
    plt.plot(t[mask_anomaly], noisy_signal[mask_anomaly], label="Signal with Anomaly", color="red")
    plt.axvspan(70 * T_val, 75 * T_val, color='yellow', alpha=0.2, label="Anomaly Region")
    plt.title("Anomaly Injection (Periods 70-75)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("data_feed.png")
    print("Data generated, saved to data/datastream.npy and plot saved as data_feed.png")

if __name__ == "__main__":
    generate_data()
