import sys
import os
import jax
import jax.numpy as jnp

# Add src to system path to allow imports from subfolder
sys.path.append(os.path.join(os.getcwd(), 'src'))

from fabric_pinn import init_model, train_step, optimizer, compute_loss, visualize_pinn
from pinn_data import generate_data

def main():
    # Setup
    key = jax.random.PRNGKey(42)
    model, params = init_model(key)
    
    print("--- Generating Physics Domain ---")
    batch = generate_data(key)
    
    # Training Configuration
    opt_state = optimizer.init(params)
    epochs = 10000
    
    print(f"--- Starting High-Octane Training ({epochs} Epochs) ---")
    for epoch in range(epochs + 1):
        params, opt_state, loss = train_step(params, opt_state, batch)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6e}")
            
    # Final Visualization
    print("--- Generating 3D Interactive Fabric ---")
    visualize_pinn(params, filename="data/pinn_3d_fabric.html")
    print("--- Success! Download results from data/pinn_3d_fabric.html ---")

if __name__ == "__main__":
    main()
