import os
from src.mandelbrot_core import run_simulation, save_plot
from google import genai
from google.genai import types

def simulate_mandelbrot(center_real: float, center_imag: float, zoom: float, max_iterations: int = 50) -> dict:
    """
    Runs a JAX-accelerated Mandelbrot simulation on the specified center coordinates and zoom factor.
    Returns visual complexity and Shannon entropy metrics.
    """
    print(f"\n[Tool Call] simulate_mandelbrot(center_real={center_real}, center_imag={center_imag}, zoom={zoom})")
    counts, metrics = run_simulation(center_real, center_imag, zoom, max_iterations=max_iterations)
    
    # Save intermediate plots during ReAct loop
    step = metrics.get('zoom', 1.0)
    plot_path = rf"C:\Users\Reyma\SynologyDrive\1.BTU\Semester 6\Angewante Modellierung und Systemsimulation\Projekte\Projekt9\Latex\Bilder\react_step_z{step:.1f}.png"
    save_plot(counts, center_real, center_imag, zoom, plot_path)
    
    return metrics

def run_autonomous_agent(target_description):
    print("Initializing GenAI client...")
    client = genai.Client()
    
    tools = [simulate_mandelbrot]
    
    history = [
        types.Content(role="user", parts=[types.Part.from_text(text=target_description)])
    ]
    
    print("Starting ReAct loop...")
    iteration = 1
    while True:
        print(f"\n--- ReAct Iteration {iteration} ---")
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=history,
            config=types.GenerateContentConfig(
                tools=tools,
            )
        )
        
        # Add model's response to history
        history.append(response.candidates[0].content)
        
        if response.function_calls:
            for function_call in response.function_calls:
                if function_call.name == "simulate_mandelbrot":
                    args = dict(function_call.args) if function_call.args else {}
                    metrics = simulate_mandelbrot(**args)
                    
                    history.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name="simulate_mandelbrot",
                                    response={"result": metrics}
                                )
                            ]
                        )
                    )
                    
                    if metrics.get("zoom", 0) >= 15000:
                        print("Target zoom reached. Agent successfully converged on Seahorse Valley!")
                        return metrics
        else:
            print("Model generated text response:")
            print(response.text)
            break
            
        iteration += 1

if __name__ == "__main__":
    target = ("Find the Seahorse Valley in the Mandelbrot set. The Seahorse Valley is located roughly at "
              "center_real = -0.7436, center_imag = 0.1318. Start from a global view (center_real=-0.5, center_imag=0.0, zoom=1.5) and "
              "use the `simulate_mandelbrot` tool to iteratively explore and zoom in. Each step, increase the zoom factor by a multiple (e.g. 10x or 100x). "
              "Your goal is to reach a zoom level of at least 15000x on the Seahorse Valley.")
    run_autonomous_agent(target)