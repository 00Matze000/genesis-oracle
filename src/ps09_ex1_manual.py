import os
import json
from src.mandelbrot_core import run_simulation, save_plot

def run_manual_cartographer():
    # Initial state
    center_real = -0.5
    center_imag = 0.0
    zoom = 1.5
    
    print("Running initial global view...")
    counts, metrics = run_simulation(center_real, center_imag, zoom)
    
    base_plot_path = r"C:\Users\Reyma\SynologyDrive\1.BTU\Semester 6\Angewante Modellierung und Systemsimulation\Projekte\Projekt9\Latex\Bilder\mandelbrot_base.png"
    save_plot(counts, center_real, center_imag, zoom, base_plot_path)
    print(f"Base plot saved to {base_plot_path}")
    print(f"Initial metrics: {metrics}")
    
    mock_responses = [
        {"center_real": -0.7, "center_imag": 0.1, "zoom": 10},
        {"center_real": -0.74, "center_imag": 0.13, "zoom": 100},
        {"center_real": -0.7436, "center_imag": 0.1318, "zoom": 1000}
    ]
    
    for step in range(1, 4):
        print(f"\n--- Step {step} ---")
        suggestion = mock_responses[step - 1]
        print(f"LLM Suggestion: {json.dumps(suggestion)}")
        
        center_real = suggestion["center_real"]
        center_imag = suggestion["center_imag"]
        zoom = suggestion["zoom"]
        
        counts, metrics = run_simulation(center_real, center_imag, zoom)
        
        plot_path = rf"C:\Users\Reyma\SynologyDrive\1.BTU\Semester 6\Angewante Modellierung und Systemsimulation\Projekte\Projekt9\Latex\Bilder\mandelbrot_step_{step}.png"
        save_plot(counts, center_real, center_imag, zoom, plot_path)
        print(f"Plot saved to {plot_path}")
        print(f"New metrics: {metrics}")

if __name__ == "__main__":
    run_manual_cartographer()
