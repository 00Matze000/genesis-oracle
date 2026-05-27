import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("--- Initializing Macro-Economic Markov Simulation ---")
    
    # -------------------------------------------------------------------------
    # 1. Baseline Übergangsmatrix P
    # -------------------------------------------------------------------------
    P_base = jnp.array([
        [0.85, 0.12, 0.03],  # State 0: Bull Market
        [0.10, 0.75, 0.15],  # State 1: Stagnation
        [0.05, 0.20, 0.75]   # State 2: Catastrophic Recession
    ])
    
    # -------------------------------------------------------------------------
    # 2. Berechnung der Schock-Matrix P_shock (Tage 180 bis 189 inclusive)
    # -------------------------------------------------------------------------
    # State 0:
    p0_2_shock = 0.80
    sum_p0_rest = P_base[0, 0] + P_base[0, 1]
    p0_0_shock = 0.20 * (P_base[0, 0] / sum_p0_rest)
    p0_1_shock = 0.20 * (P_base[0, 1] / sum_p0_rest)
    p0_shock = jnp.array([p0_0_shock, p0_1_shock, p0_2_shock])
    
    # State 1:
    p1_2_shock = 0.80
    sum_p1_rest = P_base[1, 0] + P_base[1, 1]
    p1_0_shock = 0.20 * (P_base[1, 0] / sum_p1_rest)
    p1_1_shock = 0.20 * (P_base[1, 1] / sum_p1_rest)
    p1_shock = jnp.array([p1_0_shock, p1_1_shock, p1_2_shock])
    
    # State 2 bleibt unverändert auf Baseline
    p2_shock = P_base[2, :]
    
    # Schock-Matrix zusammensetzen
    P_shock = jnp.stack([p0_shock, p1_shock, p2_shock])
    
    print("\n--- Transition Probability Matrices ---")
    print("Baseline Matrix P_base:")
    print(P_base)
    print("\nShock Matrix P_shock (Tage 180-189):")
    print(P_shock)
    
    # -------------------------------------------------------------------------
    # 3. Vorbereitung der Zeitreihe (365 Tage)
    # -------------------------------------------------------------------------
    # JAX Arrays sind unveränderlich, wir erstellen die Historie mit .at[]
    P_history = jnp.tile(P_base, (365, 1, 1))
    
    # Sabotage: Tage 180 bis 189 (genau 10 Tage) durch die Schock-Matrix ersetzen
    # Slicing 180:190 in Python umfasst die Indizes 180, 181, 182, 183, 184, 185, 186, 187, 188, 189
    P_history = P_history.at[180:190].set(P_shock)
    
    # Anfangsvektor: Start im Bull Market (Zustand 0)
    v0 = jnp.array([1.0, 0.0, 0.0])
    
    # -------------------------------------------------------------------------
    # 4. Simulation via jax.lax.scan
    # -------------------------------------------------------------------------
    def transition_step(v, P_t):
        v_next = jnp.dot(v, P_t)
        return v_next, v_next
    
    print("\nPropagating state vector over 365 days via jax.lax.scan...")
    _, v_history = jax.lax.scan(transition_step, v0, P_history)
    
    # Anfangszustand hinzufügen, um Tag 0 mit abzubilden (Gesamt: 366 Tage)
    v_all = jnp.vstack([v0, v_history])
    
    # Umrechnung in Prozent
    v_all_pct = v_all * 100.0
    
    # -------------------------------------------------------------------------
    # 5. Visualisierung der Ergebnisse
    # -------------------------------------------------------------------------
    days = np.arange(366)
    
    plt.figure(figsize=(12, 6.5))
    
    # Linien für die drei Zustände zeichnen
    plt.plot(days, v_all_pct[:, 0], color='forestgreen', linewidth=2.0, label='State 0: Bull Market')
    plt.plot(days, v_all_pct[:, 1], color='orange', linewidth=2.0, label='State 1: Stagnation')
    plt.plot(days, v_all_pct[:, 2], color='crimson', linewidth=2.0, label='State 2: Catastrophic Recession')
    
    # Krisenzeitraum hervorheben (Tage 180 bis 190)
    plt.axvspan(180, 190, color='red', alpha=0.15, label='Black Swan Crisis (Days 180-190)')
    
    # Ästhetik-Optimierung
    plt.title("Macro-Economic State Distribution Timeline (365 Days)\nDeterministic Markov Chain Propagation (lax.scan)", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline (Days)", fontsize=12)
    plt.ylabel("Probability Distribution (%)", fontsize=12)
    plt.xlim(0, 365)
    plt.ylim(0, 100)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, shadow=False, fontsize=10)
    
    # Wichtige Phasen markieren
    plt.text(90, 8, "Baseline Phase", fontsize=10, style='italic', ha='center')
    plt.text(185, 93, "Shock", fontsize=10, color='darkred', weight='bold', ha='center')
    plt.text(270, 8, "Recovery Phase", fontsize=10, style='italic', ha='center')
    
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    os.makedirs('data', exist_ok=True)
    
    # Speichern
    output_path = 'data/markov_states.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved Markov simulation plot to {output_path}")
    
    # -------------------------------------------------------------------------
    # 6. Kontrollausgaben
    # -------------------------------------------------------------------------
    print("\n--- Key Timeline Checkpoints (State Probabilities in %) ---")
    print(f"Day 0 (Initial):   Bull: {v_all_pct[0,0]:.2f}%, Stagnation: {v_all_pct[0,1]:.2f}%, Recession: {v_all_pct[0,2]:.2f}%")
    print(f"Day 179 (Pre-Shock): Bull: {v_all_pct[179,0]:.2f}%, Stagnation: {v_all_pct[179,1]:.2f}%, Recession: {v_all_pct[179,2]:.2f}%")
    print(f"Day 180 (1st Shock Day): Bull: {v_all_pct[180,0]:.2f}%, Stagnation: {v_all_pct[180,1]:.2f}%, Recession: {v_all_pct[180,2]:.2f}%")
    print(f"Day 189 (Last Shock Day): Bull: {v_all_pct[189,0]:.2f}%, Stagnation: {v_all_pct[189,1]:.2f}%, Recession: {v_all_pct[189,2]:.2f}%")
    print(f"Day 190 (1st Recovery Day): Bull: {v_all_pct[190,0]:.2f}%, Stagnation: {v_all_pct[190,1]:.2f}%, Recession: {v_all_pct[190,2]:.2f}%")
    print(f"Day 365 (Final State): Bull: {v_all_pct[365,0]:.2f}%, Stagnation: {v_all_pct[365,1]:.2f}%, Recession: {v_all_pct[365,2]:.2f}%")

if __name__ == "__main__":
    main()
