import numpy as jnp
import matplotlib.pyplot as plt
import os

def mandelbrot_kernel(C, max_iters):
    z = jnp.zeros_like(C)
    count = jnp.zeros_like(C, dtype=jnp.int32)
    active = jnp.ones_like(C, dtype=bool)
    
    for i in range(max_iters):
        z[active] = z[active]**2 + C[active]
        active[active] = jnp.abs(z[active]) <= 2.0
        count[active] += 1
        if not jnp.any(active):
            break
            
    return count

def run_simulation(center_real, center_imag, zoom, resolution=50, max_iterations=50):
    # Map resolution coordinates to complex plane
    width, height = resolution, resolution
    r = jnp.linspace(center_real - 1.5 / zoom, center_real + 1.5 / zoom, width)
    i = jnp.linspace(center_imag - 1.5 / zoom, center_imag + 1.5 / zoom, height)
    R, I = jnp.meshgrid(r, i)
    C = R + 1j * I

    # Execute high-performance calculation on accelerator
    counts = mandelbrot_kernel(C.flatten(), max_iterations)
    counts = counts.reshape((height, width))

    # Calculate Shannon Entropy of escape times
    hist, _ = jnp.histogram(counts, bins=20)
    hist_prob = hist / jnp.sum(hist)
    hist_prob = jnp.where(hist_prob > 0, hist_prob, 1.0)  # Avoid log(0)
    entropy = -jnp.sum(hist_prob * jnp.log(hist_prob))

    # Calculate Boundary Complexity (ratio of boundary pixels)
    boundary_pixels = jnp.sum((counts > 0) & (counts < max_iterations))
    boundary_ratio = boundary_pixels / (width * height)

    return counts, {
        "entropy": float(entropy),
        "boundary_complexity": float(boundary_ratio),
        "center_real": float(center_real),
        "center_imag": float(center_imag),
        "zoom": float(zoom),
        "max_iterations": int(max_iterations)
    }

def save_plot(counts, center_real, center_imag, zoom, filepath):
    plt.figure(figsize=(6, 6))
    extent = [
        center_real - 1.5 / zoom, center_real + 1.5 / zoom,
        center_imag - 1.5 / zoom, center_imag + 1.5 / zoom
    ]
    plt.imshow(counts, cmap='twilight_shifted', extent=extent, origin='lower')
    plt.colorbar(label='Iterations until escape')
    plt.title(f'Mandelbrot View\nCenter: {center_real:.4f} + {center_imag:.4f}i, Zoom: {zoom:.1f}x')
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
