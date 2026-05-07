import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np

class SimpleMLP(nn.Module):
    """
    Ein einfaches Multi-Layer Perceptron in Flax.
    Demonstriert die funktionale Reinheit: Das Modul selbst speichert KEINE Gewichte.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

def run_flax_demonstration():
    print("--- Flax: Agentic Refactoring for the Horizon ---")
    
    # 1. Initialisierung des Modells
    model = SimpleMLP()
    key = jax.random.PRNGKey(0)  # Expliziter Zufallsschlüssel für JAX
    
    # Beispiel-Input
    dummy_input = jnp.ones((1, 10))
    
    # 2. Modell-Initialisierung (model.init)
    # Hier werden die Gewichte separat vom Modell-Objekt erstellt (stateless)
    print("Initialisiere Modellparameter...")
    variables = model.init(key, dummy_input)
    params = variables['params']
    
    # 3. Forward Pass (model.apply)
    # Wir übergeben die Parameter explizit bei jedem Aufruf.
    print("Führe Forward Pass aus...")
    output = model.apply({'params': params}, dummy_input)
    
    print(f"\nErgebnis-Shape: {output.shape}")
    print(f"Output-Wert: {output[0, 0]:.4f}")
    
    print("\n[Erklärung]")
    print("Im Gegensatz zu Keras (OOP) speichert ein Flax-Modul keine Zustände.")
    print("Die Gewichte (params) leben in einem separaten Dictionary.")
    print("Dies ermöglicht perfekte Kompatibilität mit jax.jit, jax.vmap und jax.grad.")

if __name__ == "__main__":
    run_flax_demonstration()
