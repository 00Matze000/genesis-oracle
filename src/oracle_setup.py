import os

# Set Keras backend to JAX before importing keras
os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax.numpy as jnp

def verify_setup():
    print(f"Keras version: {keras.__version__}")
    print(f"Keras backend: {keras.backend.backend()}")
    
    # Create a basic Keras random tensor
    random_tensor = keras.random.normal((3, 3))
    print(f"Random tensor type: {type(random_tensor)}")
    print(f"Tensor data:\n{random_tensor}")

if __name__ == "__main__":
    verify_setup()
