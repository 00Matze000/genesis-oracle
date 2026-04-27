import numpy as np
import tensorflow as tf
from tensorflow import keras

def prepare_data(signal, window_size=50, split_period=60):
    """
    Slices a 1D signal into overlapping 2D matrices (window_size=50 timesteps).
    Splits the data: use the normal data (before period 60) for training,
    and the rest for testing.
    """
    # Create overlapping windows
    windows = []
    for i in range(len(signal) - window_size + 1):
        windows.append(signal[i : i + window_size])
    windows = np.array(windows)
    
    # We assume 'split_period' corresponds to the index where the anomaly starts.
    # Depending on the exact data structure, this might need an adjustment to map 'period 60' to exact index.
    train_windows = windows[:split_period]
    test_windows = windows[split_period:]
    
    return train_windows, test_windows

class SignalCompression(keras.layers.Layer):
    """
    Custom encoder layer using subclassing.
    Reduces the 50-timestep window down to a latent dimension of 8.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super(SignalCompression, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.dense = keras.layers.Dense(self.latent_dim, activation='relu')
        super(SignalCompression, self).build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

class SignalExpansion(keras.layers.Layer):
    """
    Custom decoder layer using subclassing.
    Reconstructs 8 dimensions back to 50.
    """
    def __init__(self, original_dim=50, **kwargs):
        super(SignalExpansion, self).__init__(**kwargs)
        self.original_dim = original_dim

    def build(self, input_shape):
        # Typically linear or sigmoid for reconstruction depending on data scaling
        self.dense = keras.layers.Dense(self.original_dim)
        super(SignalExpansion, self).build(input_shape)

    def call(self, inputs):
        return self.dense(inputs)

class PhysicsAutoencoder(keras.Model):
    """
    Autoencoder model chaining SignalCompression and SignalExpansion.
    """
    def __init__(self, latent_dim=8, original_dim=50, **kwargs):
        super(PhysicsAutoencoder, self).__init__(**kwargs)
        self.encoder = SignalCompression(latent_dim=latent_dim)
        self.decoder = SignalExpansion(original_dim=original_dim)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
