import os
from src.mandelbrot_core import run_simulation, save_plot

def simulate_mandelbrot(center_real: float, center_imag: float, zoom: float, max_iterations: int = 500) -> dict:
    print(f"\n[Tool Call] simulate_mandelbrot(center_real={center_real}, center_imag={center_imag}, zoom={zoom})")
    counts, metrics = run_simulation(center_real, center_imag, zoom, max_iterations=max_iterations)
    
    step = metrics.get('zoom', 1.0)
    plot_path = rf"C:\Users\Reyma\SynologyDrive\1.BTU\Semester 6\Angewante Modellierung und Systemsimulation\Projekte\Projekt9\Latex\Bilder\react_step_z{step:.1f}.png"
    save_plot(counts, center_real, center_imag, zoom, plot_path)
    return metrics

def run_autonomous_agent(target_description):
    print("Initializing GenAI client (Mocked)...")
    print("Starting ReAct loop...")
    
    mock_calls = [
        {"center_real": -0.7, "center_imag": 0.1, "zoom": 10},
        {"center_real": -0.74, "center_imag": 0.13, "zoom": 100},
        {"center_real": -0.7436, "center_imag": 0.1318, "zoom": 1000},
        {"center_real": -0.7436, "center_imag": 0.1318, "zoom": 20000}
    ]
    
    iteration = 1
    for call in mock_calls:
        print(f"\n--- ReAct Iteration {iteration} ---")
        print(f"Thought: I need to zoom closer to Seahorse Valley. I will call simulate_mandelbrot with {call}")
        
        metrics = simulate_mandelbrot(**call)
        
        if metrics.get("zoom", 0) >= 15000:
            print("\nTarget zoom reached. Agent successfully converged on Seahorse Valley!")
            return metrics
            
        iteration += 1

if __name__ == "__main__":
    target = ("Find the Seahorse Valley in the Mandelbrot set...")
    run_autonomous_agent(target)