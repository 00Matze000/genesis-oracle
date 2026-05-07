import numpy as np
import time

def simulate_legacy_swarm(n_oscillators=100_000, n_steps=1_000, dt=0.01):
    """
    Simuliert ein Array von gedämpften harmonischen Oszillatoren sequentiell.
    Nutzt NumPy für Array-Operationen, aber eine Python-Schleife für die Zeitschritte.
    """
    # 1. Initialisierung
    # Zufällige natürliche Frequenzen (omega) zwischen 0.5 und 2.0
    w = np.random.uniform(0.5, 2.0, size=n_oscillators)
    
    # Startpositionen (x) auf 1.0 und Geschwindigkeiten (v) auf 0.0
    x = np.ones(n_oscillators)
    v = np.zeros(n_oscillators)
    
    # Dämpfungskonstante (gamma)
    gamma = 0.1

    print(f"--- Legacy Simulation (NumPy) ---")
    print(f"Oszillatoren: {n_oscillators}")
    print(f"Zeitschritte: {n_steps}")
    print(f"Starte Simulation...")
    
    # 2. Zeitmessung: Start
    start_time = time.time()

    # 3. Die Zeitschleife (The Sequential Swarm)
    for step in range(n_steps):
        # Explizite Euler-Integration:
        # Beschleunigung: a = -w^2 * x - gamma * v
        a = -(w**2) * x - gamma * v
        
        # Update Geschwindigkeit und Position
        v = v + a * dt
        x = x + v * dt

    # 2. Zeitmessung: Ende
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Simulation abgeschlossen.")
    print(f"Gesamtzeit (Legacy): {total_time:.4f} Sekunden")
    
    return total_time

if __name__ == "__main__":
    # Wir führen die Simulation einmal aus
    simulate_legacy_swarm()
