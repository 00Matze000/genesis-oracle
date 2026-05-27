import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Stochastische Parameter (als globale Konstanten für Subagent-Kompatibilität)
# =============================================================================
MARKET_DEMAND_MU = 1000.0
MARKET_DEMAND_SIGMA = 150.0

ASSET_COST_MU = 5.5
ASSET_COST_SIGMA = 0.3  # <-- Wird von Subagent-Alpha modifiziert!

PENALTY_RATE_MIN = 0.05
PENALTY_RATE_MAX = 0.25

REVENUE_MULTIPLIER = 150.0

def simulate_path(key):
    """
    Berechnet den Netto-Einnahmenwert für einen stochastischen Pfad.
    Konsumiert einen PRNGKey und splittet ihn sauber funktional auf.
    """
    # Key aufspalten in 3 Subkeys für D, C, R
    key_D, key_C, key_R = jax.random.split(key, 3)
    
    # 1. Marktnachfrage D ~ N(mu, sigma^2)
    D = jax.random.normal(key_D) * MARKET_DEMAND_SIGMA + MARKET_DEMAND_MU
    
    # 2. Produktionsanlagenkosten C ~ Log-Normal(mu, sigma^2)
    C = jnp.exp(jax.random.normal(key_C) * ASSET_COST_SIGMA + ASSET_COST_MU)
    
    # 3. Regulatorischer Strafsatz R ~ U(min, max)
    R = jax.random.uniform(key_R, minval=PENALTY_RATE_MIN, maxval=PENALTY_RATE_MAX)
    
    # Nettoprofitgleichung
    revenue = (D * REVENUE_MULTIPLIER) - C * (1.0 - R)
    return revenue

def main():
    N_SIMULATIONS = 1_000_000
    
    print(f"--- Starting JAX Monte Carlo Engine with {N_SIMULATIONS:,} runs ---")
    
    # Master-Key initialisieren
    master_key = jax.random.PRNGKey(42)
    
    # Zeitmessung vor der JAX-Kompilierung (1. Ausführung inkl. Kompilierung)
    start_time_all = time.perf_counter()
    
    # Generiere 1 Million eindeutige Subkeys
    print("Generating unique JAX subkeys...")
    subkeys = jax.random.split(master_key, N_SIMULATIONS)
    
    # Vektorisierte Ausführung via jax.vmap
    print("Running parallelized simulation (vmap)...")
    vmapped_simulate = jax.vmap(simulate_path)
    
    # Simulation ausführen (1st Run - warm-up / compile included)
    start_run1 = time.perf_counter()
    revenues = vmapped_simulate(subkeys)
    # Blocken, um JAX asynchrone Ausführung zu zwingen und reale Zeit zu messen
    revenues.block_until_ready()
    time_run1 = time.perf_counter() - start_run1
    
    # 2nd Run - Warm execution (reiner Maschinencode)
    start_run2 = time.perf_counter()
    revenues_warm = vmapped_simulate(subkeys)
    revenues_warm.block_until_ready()
    time_run2 = time.perf_counter() - start_run2
    
    # Statistiken berechnen
    mean_revenue = float(jnp.mean(revenues))
    # VaR_95% entspricht dem 5. Perzentil (5% schlechteste Ergebnisse)
    var_95 = float(jnp.percentile(revenues, 5.0))
    
    print("\n--- Simulation Results ---")
    print(f"Expected Revenue E[Revenue]:  {mean_revenue:,.2f} EUR")
    print(f"Value-at-Risk (VaR_95%):       {var_95:,.2f} EUR")
    print(f"1st Run Time (incl. Compile): {time_run1:.6f} seconds")
    print(f"2nd Run Time (Warm Execution): {time_run2:.6f} seconds")
    print(f"JAX Acceleration Speedup:     {time_run1 / time_run2:.2f}x")
    
    # -------------------------------------------------------------------------
    # Visualisierung (Histogramm der Einnahmenverteilung)
    # -------------------------------------------------------------------------
    print("\nGenerating distribution plot...")
    plt.figure(figsize=(10, 6))
    
    # Histogramm zeichnen
    # Konvertiere in Numpy-Array für Plotting
    revenues_np = np.array(revenues)
    plt.hist(revenues_np, bins=100, color='royalblue', edgecolor='navy', alpha=0.7, density=True, label='Revenue Paths')
    
    # Vertikale Linien einzeichnen
    plt.axvline(mean_revenue, color='black', linestyle='-', linewidth=2.5, label=f"E[Revenue] = {mean_revenue:,.2f} EUR")
    plt.axvline(var_95, color='crimson', linestyle='--', linewidth=2.5, label=f"VaR_95% = {var_95:,.2f} EUR")
    
    # Ästhetik-Optimierung
    plt.title(f"Annual Net Revenue Distribution (N = {N_SIMULATIONS:,})\nStochastic Business Simulation", fontsize=14, fontweight='bold')
    plt.xlabel("Net Revenue (EUR)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left', frameon=True, shadow=False, fontsize=10)
    
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    os.makedirs('data', exist_ok=True)
    
    # Speichern
    output_path = 'data/revenue_dist.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved revenue distribution to {output_path}")

if __name__ == "__main__":
    main()
