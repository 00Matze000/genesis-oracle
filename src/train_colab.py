import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from src.architecture import PhysicsAutoencoder, prepare_data

# 1. Load data
try:
    # Use the real data from Project 2
    signal = np.load('data/datastream.npy') 
    print(f"Signal data loaded successfully. Shape: {signal.shape}")
except FileNotFoundError:
    print("Warning: data/datastream.npy not found.")
    # Fallback only if absolutely necessary
    signal = np.sin(np.linspace(0, 100, 10000))

# 2. Prepare data for the Autoencoder
# Normal data before period 60, rest for testing
# window_size = 50
train_windows, test_windows = prepare_data(signal, window_size=50, split_period=600) # period 60 roughly index 600 if dt=0.1
all_windows = np.concatenate([train_windows, test_windows], axis=0)

# 3. Initialize and Compile the Oracle
model = PhysicsAutoencoder(latent_dim=8, original_dim=50)
model.compile(optimizer='adam', loss='mse')

# 4. Training (on normal data only)
print("Training the Oracle on normal data...")
history = model.fit(
    train_windows, train_windows, 
    epochs=30, 
    batch_size=16, 
    validation_split=0.1,
    verbose=1
)

# 5. Detection Run
print("Calculating reconstruction loss...")
reconstructions = model.predict(all_windows)
# MAE between raw input and reconstructed prediction
mae_loss = np.mean(np.abs(reconstructions - all_windows), axis=1)

# 6. Plotting
plt.figure(figsize=(12, 6))
plt.plot(mae_loss, label='Reconstruction Loss (MAE)')

# Anomaly Threshold (e.g., mean + 3*std of training loss)
threshold = np.mean(mae_loss[:len(train_windows)]) + 3 * np.std(mae_loss[:len(train_windows)])
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Anomaly Threshold ({threshold:.4f})')

plt.title('Reconstruction Loss over Time')
plt.xlabel('Window Index')
plt.ylabel('MAE Loss')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('reconstruction_loss.png')
print("Plot saved as reconstruction_loss.png")
plt.show()
