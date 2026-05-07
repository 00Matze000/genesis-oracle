import jax
import jax.numpy as jnp

def projectile_loss(v_initial, target_distance=150.0, flight_time=5.0, g=9.81):
    """
    Simuliert die Distanz eines Projektils und berechnet den Mean Squared Error (MSE).
    Distanz d = v_initial * t - 0.5 * g * t^2 (vereinfacht für den Test, 
    da die Aufgabe von 'final distance' spricht).
    """
    # Physikalische Formel: s = v0 * t (Horizontalbewegung)
    # Oder falls vertikal/parabolisch: Hier nehmen wir die horizontale Distanz an.
    simulated_distance = v_initial * flight_time
    
    # Mean Squared Error zum Ziel
    loss = (simulated_distance - target_distance)**2
    return loss

# Automatische Differenzierung: Erzeuge eine Funktion, die den Gradienten berechnet
grad_projectile_loss = jax.grad(projectile_loss)

def optimize_velocity():
    # 3. The Optimization Loop
    v_initial = 10.0  # Startwert (Random Guess)
    learning_rate = 0.1
    n_iterations = 20
    
    print(f"--- JAX Gradient Optimization ---")
    print(f"Ziel-Distanz: 150.0m")
    print(f"Start-v: {v_initial}")
    
    for i in range(n_iterations):
        # Berechne aktuellen Loss und Gradienten
        current_loss = projectile_loss(v_initial)
        gradient = grad_projectile_loss(v_initial)
        
        # Update v mittels Gradient Descent: v = v - lr * gradient
        v_initial = v_initial - learning_rate * gradient
        
        if (i + 1) % 5 == 0 or i == 0:
            print(f"Iteration {i+1:2d}: v = {v_initial:6.2f}, Loss = {current_loss:10.4f}")
            
    print(f"\nOptimierte Anfangsgeschwindigkeit: {v_initial:.4f} m/s")
    print(f"Finale Distanz: {v_initial * 5.0:.2f} m")
    
    return v_initial

if __name__ == "__main__":
    optimize_velocity()
