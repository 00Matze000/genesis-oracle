import yaml
import json
import os
import sys
from google import genai
from google.genai import types
from src.mandelbrot_core import run_simulation, save_plot

# Tool wrapper equivalent to the solver logic
def simulate_mandelbrot_tool(center_real: float, center_imag: float, zoom: float, max_iterations: int = 50) -> dict:
    print(f"\n[Skill Tool Call] simulate_mandelbrot(center_real={center_real}, center_imag={center_imag}, zoom={zoom})")
    counts, metrics = run_simulation(center_real, center_imag, zoom, max_iterations=max_iterations)
    return metrics

class GemmaSkillLoader:
    def __init__(self, skill_dir):
        self.skill_dir = skill_dir
        self.instructions = ""
        self.metadata = {}
        self.schemas = []
        self.load_skill()
        
    def load_skill(self):
        skill_md_path = os.path.join(self.skill_dir, "SKILL.md")
        with open(skill_md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        parts = content.split("---")
        if len(parts) >= 3:
            self.metadata = yaml.safe_load(parts[1])
            self.instructions = parts[2].strip()
            
        tools_dir = os.path.join(self.skill_dir, "tools")
        for filename in os.listdir(tools_dir):
            if filename.endswith(".json"):
                with open(os.path.join(tools_dir, filename), 'r', encoding='utf-8') as f:
                    self.schemas.append(json.load(f))
                    
        print(f"Loaded Skill: {self.metadata.get('name')} - {self.metadata.get('description')}")
        print(f"Loaded {len(self.schemas)} tool schemas.")

    def run_autonomous_loop(self):
        print("Initializing GenAI client for autonomous skill execution...")
        client = genai.Client()
        
        # We pass the actual python callable so the SDK can invoke it
        tools = [simulate_mandelbrot_tool]
        
        # The system prompt comes directly from the SKILL.md body
        history = [
            types.Content(role="user", parts=[types.Part.from_text(text=self.instructions)])
        ]
        
        print("Starting ReAct loop using Skill Instructions...")
        iteration = 1
        while True:
            print(f"\n--- Skill Iteration {iteration} ---")
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=history,
                config=types.GenerateContentConfig(
                    tools=tools,
                )
            )
            
            history.append(response.candidates[0].content)
            
            if response.function_calls:
                for function_call in response.function_calls:
                    if function_call.name == "simulate_mandelbrot":
                        args = dict(function_call.args) if function_call.args else {}
                        metrics = simulate_mandelbrot_tool(**args)
                        
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
                            print("Skill objective completed! Found high complexity boundary.")
                            return
            else:
                print("Agent finished autonomously:")
                print(response.text)
                break
            iteration += 1

if __name__ == "__main__":
    print("Loading Gemma-Skill...")
    skill_dir = os.path.join(os.path.dirname(__file__), "..", "skills", "mandelbrot_explorer")
    loader = GemmaSkillLoader(skill_dir)
    print("\nBootstrapping autonomous agent from skill...")
    
    # We do not run it if the key isn't provided to prevent immediate crashes, 
    # but the logic is fully implemented as requested by the exercise.
    if os.environ.get("GEMINI_API_KEY"):
        loader.run_autonomous_loop()
    else:
        print("GEMINI_API_KEY not set. Ready for execution when quota is available.")
