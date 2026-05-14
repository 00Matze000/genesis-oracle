import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import plotly.graph_objects as go
from functools import partial
import os

class HeatSurrogate(nn.Module):
    """
    A Multi-Layer Perceptron for approximating the 1D Heat Equation solution.
    Using tanh activation for smooth derivatives.
    """
    @nn.compact
    def __call__(self, x, t):
        # Concatenate x and t into a single 2D input vector
        inputs = jnp.hstack([x, t])
        
        # 4 Hidden layers with 32 neurons each
        z = nn.Dense(features=32)(inputs)
        z = jnp.tanh(z)
        z = nn.Dense(features=32)(z)
        z = jnp.tanh(z)
        z = nn.Dense(features=32)(z)
        z = jnp.tanh(z)
        z = nn.Dense(features=32)(z)
        z = jnp.tanh(z)
        
        # Output layer: scalar temperature u
        u = nn.Dense(features=1)(z)
        return u

def init_model(key):
    model = HeatSurrogate()
    # Dummy inputs for initialization (x, t) - each is (Batch, 1)
    # The model expects hstack([x, t]) which results in (Batch, 2)
    x_dummy = jnp.ones((1, 1))
    t_dummy = jnp.ones((1, 1))
    params = model.init(key, x_dummy, t_dummy)
    return model, params

# ==========================================
# Autodiff & Physics Loss (Exercise 3)
# ==========================================

alpha = 0.05 # Thermal diffusivity

def predict_u(params, x, t):
    """Forward function returning a scalar temperature. Required for jax.grad"""
    model = HeatSurrogate()
    u = model.apply(params, x, t)
    return u[0] # Return scalar

# First derivative w.r.t time t
def u_t(params, x, t):
    grad_t = jax.grad(predict_u, argnums=2)(params, x, t)
    return grad_t[0]

# First derivative w.r.t space x
def u_x(params, x, t):
    grad_x = jax.grad(predict_u, argnums=1)(params, x, t)
    return grad_x[0]

# Second derivative w.r.t space x
def u_xx(params, x, t):
    grad_xx = jax.grad(u_x, argnums=1)(params, x, t)
    return grad_xx[0]

def physics_loss_single(params, x, t):
    """Computes the squared PDE residual for a single point."""
    ut = u_t(params, x, t)
    uxx = u_xx(params, x, t)
    residual = ut - alpha * uxx
    return residual ** 2

# Vectorize the physics loss to evaluate all collocation points in parallel
batched_physics_loss = jax.vmap(physics_loss_single, in_axes=(None, 0, 0))

def compute_loss(params, batch):
    """
    Computes the Total Loss: Physics_Loss + IC_Loss + BC_Loss
    """
    colloc, (ic_points, ic_u), (bc_points, u_bc) = batch
    
    # 1. Physics Loss (PDE Residual)
    colloc_x = colloc[:, 0:1]
    colloc_t = colloc[:, 1:2]
    phys_loss = jnp.mean(batched_physics_loss(params, colloc_x, colloc_t))
    
    # 2. Initial Condition Loss
    model = HeatSurrogate()
    ic_x = ic_points[:, 0:1]
    ic_t = ic_points[:, 1:2]
    pred_ic = model.apply(params, ic_x, ic_t)
    ic_loss = jnp.mean((pred_ic - ic_u) ** 2)
    
    # 3. Boundary Condition Loss
    x_bc = bc_points[:, 0:1]
    t_bc = bc_points[:, 1:2]
    pred_bc = model.apply(params, x_bc, t_bc)
    bc_loss = jnp.mean((pred_bc - u_bc) ** 2)
    
    total_loss = phys_loss + ic_loss + bc_loss
    return total_loss

@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(compute_loss)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

optimizer = optax.adam(learning_rate=1e-3)

def visualize_pinn(params, filename="data/pinn_3d_fabric.html"):
    """
    Generates a high-resolution 3D surface plot of the predicted temperature field.
    """
    x = jnp.linspace(-1, 1, 100)
    t = jnp.linspace(0, 1, 100)
    X, T = jnp.meshgrid(x, t)
    
    # Flatten for batch prediction
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    
    model = HeatSurrogate()
    U_flat = model.apply(params, x_flat, t_flat)
    U = U_flat.reshape(X.shape)
    
    fig = go.Figure(data=[go.Surface(x=X, y=T, z=U, colorscale='inferno')])
    fig.update_layout(
        title='PINN Predicted Temperature Field: Project Genesis',
        scene=dict(
            xaxis_title='Space (x)',
            yaxis_title='Time (t)',
            zaxis_title='Temperature (u)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.write_html(filename)
    print(f"--- 3D Visualization saved to {filename} ---")

if __name__ == "__main__":
    from pinn_data import generate_data
    
    key = jax.random.PRNGKey(42)
    model, params = init_model(key)
    
    # Generate Training Data
    print("--- Generating Training Data ---")
    colloc, ic, bc = generate_data(key)
    # Package into format expected by compute_loss
    batch = (colloc, (ic[0], jnp.zeros_like(ic[0]), ic[1]), (bc[0][:,0:1], bc[0][:,1:2], bc[1]))
    
    # Initial Test
    loss_val = compute_loss(params, batch)
    print(f"Initial Total Loss: {loss_val:.4f}")
    
    # Training Loop (Local Test - only 100 epochs, full training in Colab/GPU)
    print("--- Starting Local Mini-Training (100 Epochs) ---")
    opt_state = optimizer.init(params)
    for epoch in range(101):
        params, opt_state, loss = train_step(params, opt_state, batch)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")
            
    # Final Visualization
    visualize_pinn(params)
