import jax
import jax.numpy as jnp

def generate_data(key, num_colloc=5000, num_ic=500, num_bc=500):
    """
    Generates training data for a Physics-Informed Neural Network (PINN).
    Domain: x in [-1, 1], t in [0, 1]
    """
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # 1. Collocation Points (PDE Domain)
    x_colloc = jax.random.uniform(k1, (num_colloc, 1), minval=-1.0, maxval=1.0)
    t_colloc = jax.random.uniform(k2, (num_colloc, 1), minval=0.0, maxval=1.0)
    colloc_points = jnp.hstack([x_colloc, t_colloc])

    # 2. Initial Condition Points (t = 0)
    x_ic = jax.random.uniform(k3, (num_ic, 1), minval=-1.0, maxval=1.0)
    t_ic = jnp.zeros((num_ic, 1))
    ic_points = jnp.hstack([x_ic, t_ic])
    u_ic = -jnp.sin(jnp.pi * x_ic)

    # 3. Boundary Condition Points (x = -1 and x = 1)
    num_bc_half = num_bc // 2
    # Left boundary: x = -1
    t_bc_left = jax.random.uniform(k4, (num_bc_half, 1), minval=0.0, maxval=1.0)
    x_bc_left = -jnp.ones((num_bc_half, 1))
    bc_left_points = jnp.hstack([x_bc_left, t_bc_left])
    u_bc_left = jnp.zeros((num_bc_half, 1))

    # Right boundary: x = 1
    t_bc_right = jax.random.uniform(k5, (num_bc_half, 1), minval=0.0, maxval=1.0)
    x_bc_right = jnp.ones((num_bc_half, 1))
    bc_right_points = jnp.hstack([x_bc_right, t_bc_right])
    u_bc_right = jnp.zeros((num_bc_half, 1))

    # Combine BCs
    bc_points = jnp.vstack([bc_left_points, bc_right_points])
    u_bc = jnp.vstack([u_bc_left, u_bc_right])

    return colloc_points, (ic_points, u_ic), (bc_points, u_bc)

if __name__ == "__main__":
    # Test datagen script and deterministic chaos via PRNGKey
    key = jax.random.PRNGKey(42)
    colloc, ic, bc = generate_data(key)
    
    print("--- PINN Data Generation Successful ---")
    print(f"Domain Shape: {colloc.shape} (Collocation Points)")
    print(f"IC Points Shape: {ic[0].shape}, IC Values Shape: {ic[1].shape}")
    print(f"BC Points Shape: {bc[0].shape}, BC Values Shape: {bc[1].shape}")
