import os
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Parameter
    N = 5_000_000
    subset_size = 10_000
    
    print(f"--- Starting Monte Carlo Pi Estimation with {N:,} points ---")
    
    # Zeitmessung starten
    start_time = time.perf_counter()
    
    # Gleichmäßig verteilte Zufallspunkte in [0, 1]x[0, 1] generieren
    x = np.random.uniform(0.0, 1.0, N)
    y = np.random.uniform(0.0, 1.0, N)
    
    # Euklidischen Abstand quadratisch berechnen (vermeidet teure Wurzelberechnung)
    dist_sq = x**2 + y**2
    
    # Zählen wie viele Punkte innerhalb des Einheitskreises liegen
    inside_mask = dist_sq <= 1.0
    num_inside = np.sum(inside_mask)
    
    # Pi schätzen
    pi_estimation = 4.0 * num_inside / N
    
    # Zeitmessung stoppen
    execution_time = time.perf_counter() - start_time
    
    print(f"Estimation of Pi: {pi_estimation:.6f}")
    print(f"Execution Time:   {execution_time:.6f} seconds")
    
    # -------------------------------------------------------------------------
    # Visualisierung (Teilmenge von 10.000 Punkten extrahieren)
    # -------------------------------------------------------------------------
    print(f"--- Extracting random subset of {subset_size:,} points for plotting ---")
    
    # Zufällige Indizes für Stichprobe wählen
    indices = np.random.choice(N, subset_size, replace=False)
    x_sub = x[indices]
    y_sub = y[indices]
    inside_sub = inside_mask[indices]
    
    # Plot konfigurieren
    plt.figure(figsize=(8, 8))
    
    # Punkte zeichnen (Blau für innen, Rot für außen)
    plt.scatter(x_sub[inside_sub], y_sub[inside_sub], color='blue', s=1, alpha=0.6, label='Inside Circle')
    plt.scatter(x_sub[~inside_sub], y_sub[~inside_sub], color='red', s=1, alpha=0.6, label='Outside Circle')
    
    # Viertelkreis-Grenzlinie einzeichnen
    x_arc = np.linspace(0.0, 1.0, 500)
    y_arc = np.sqrt(1.0 - x_arc**2)
    plt.plot(x_arc, y_arc, color='darkred', linewidth=2.5, label='Quarter Circle Boundary')
    
    # Ästhetik-Optimierung
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Monte Carlo Pi Estimation (N = {N:,})\nCalculated $\\pi \\approx {pi_estimation:.6f}$ in {execution_time:.4f}s", fontsize=12)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("y", fontsize=10)
    plt.legend(loc='upper right', frameon=True, shadow=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    os.makedirs('data', exist_ok=True)
    
    # Speichern
    output_path = 'data/classical_pi_disp.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved scatter plot to {output_path}")

if __name__ == "__main__":
    main()
