import jax
import jax.numpy as jnp
import time
import numpy as np

def oscillator_step(x, v, w, gamma=0.1, dt=0.01):
    """
    Berechnet den nächsten Zustand für einen einzelnen Oszillator (Pure Function).
    """
    # Beschleunigung: a = -w^2 * x - gamma * v
    a = -(w**2) * x - gamma * v
    
    # Update v und x (Euler-Integration)
    v_new = v + a * dt
    x_new = x + v_new * dt
    
    return x_new, v_new

# Vektorisierung über die Oszillatoren (Parallel Universes)
# vmap wandelt die Skalar-Funktion in eine Batch-Funktion um.
# Wir vektorisieren über x, v und w (in_axes=0 für alle drei).
vmapped_step = jax.vmap(oscillator_step, in_axes=(0, 0, 0))

@jax.jit
def run_simulation_jax(x, v, w, n_steps=1000):
    """
    Führt die gesamte Simulation auf dem Accelerator aus (Hardware Fusion).
    Wir nutzen jax.lax.fori_loop, um Tracer-Probleme mit Python-Schleifen zu vermeiden.
    """
    def body_fun(i, state):
        x, v = state
        return vmapped_step(x, v, w)

    # fori_loop(lower, upper, body_fun, init_val)
    # n_steps wird hier als dynamischer Wert im Graphen behandelt.
    final_x, final_v = jax.lax.fori_loop(0, n_steps, body_fun, (x, v))
    
    return final_x, final_v

def main():
    n_oscillators = 100_000
    n_steps = 1_000
    
    # Initialisierung (NumPy -> JAX DeviceArray)
    w = jnp.array(np.random.uniform(0.5, 2.0, size=n_oscillators))
    x = jnp.ones(n_oscillators)
    v = jnp.zeros(n_oscillators)
    
    print(f"--- JAX Simulation (vmap + jit) ---")
    print(f"Oszillatoren: {n_oscillators}")
    
    # 1. Lauf: Warm-up (The Tracing Phenomenon)
    print("Starte Warm-up (Kompilierung)...")
    start_warmup = time.time()
    _ = run_simulation_jax(x, v, w, n_steps)
    jax.block_until_ready(_) # Wichtig für asynchrone JAX-Ausführung
    end_warmup = time.time()
    print(f"Warm-up Zeit: {end_warmup - start_warmup:.4f} Sekunden")
    
    # 2. Lauf: Messung
    print("Starte optimierten Run...")
    start_run = time.time()
    _ = run_simulation_jax(x, v, w, n_steps)
    jax.block_until_ready(_)
    end_run = time.time()
    
    total_time = end_run - start_run
    print(f"Simulation abgeschlossen.")
    print(f"Gesamtzeit (JAX 2nd Run): {total_time:.4f} Sekunden")
    
    return total_time

if __name__ == "__main__":
    main()
