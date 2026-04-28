import numpy as np
import keras
from keras import ops

def prepare_data(signal, window_size=50, train_ratio=0.6):
    """
    Slices a 1D signal into overlapping 2D matrices (window_size=50 timesteps).
    Splits at int(len(signal) * train_ratio): the first share is training
    (normal physics, before the injected anomaly), the rest is test data.
    With 100k samples and train_ratio=0.6, this matches "before period 60"
    of the 100-period source signal (1000 points/period).
    """
    windows = []
    for i in range(len(signal) - window_size + 1):
        windows.append(signal[i : i + window_size])
    windows = np.array(windows)

    split_idx = int(len(signal) * train_ratio)
    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]

    return train_windows, test_windows

class SignalCompression(keras.layers.Layer):
    """
    Custom encoder layer using Conv1D for better temporal pattern recognition.
    Input: (batch, 50, 1) - assumed window_size=50 with 1 feature.
    """
    def __init__(self, latent_dim=8, **kwargs):
        super(SignalCompression, self).__init__(**kwargs)
        self.latent_dim = latent_dim

    def build(self, input_shape):
        # Shape: (batch, 50, 1) -> (batch, 25, 16) -> (batch, 1, latent_dim)
        self.conv1 = keras.layers.Conv1D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')
        self.pool = keras.layers.GlobalAveragePooling1D()
        self.dense = keras.layers.Dense(self.latent_dim, activation='relu')
        super(SignalCompression, self).build(input_shape)

    def call(self, inputs):
        # Add feature dimension if missing: (batch, 50) -> (batch, 50, 1)
        if len(inputs.shape) == 2:
            inputs = ops.expand_dims(inputs, axis=-1)
        x = self.conv1(inputs) # (batch, 25, 16)
        x = self.pool(x)       # (batch, 16)
        return self.dense(x)   # (batch, 8)

class SignalExpansion(keras.layers.Layer):
    """
    Custom decoder layer using Conv1DTranspose to reconstruct the signal.
    Input: (batch, 8) -> Output: (batch, 50)
    """
    def __init__(self, original_dim=50, **kwargs):
        super(SignalExpansion, self).__init__(**kwargs)
        self.original_dim = original_dim

    def build(self, input_shape):
        # (batch, 8) -> (batch, 1, 8) -> (batch, 50, 1)
        self.dense = keras.layers.Dense(25 * 16, activation='relu')
        self.reshape = keras.layers.Reshape((25, 16))
        self.deconv = keras.layers.Conv1DTranspose(filters=1, kernel_size=3, strides=2, padding='same')
        super(SignalExpansion, self).build(input_shape)

    def call(self, inputs):
        x = self.dense(inputs)   # (batch, 400)
        x = self.reshape(x)     # (batch, 25, 16)
        x = self.deconv(x)      # (batch, 50, 1)
        return ops.squeeze(x, axis=-1) # (batch, 50)

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
