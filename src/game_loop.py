import os
import json
from google import genai
from sandbox_env import ThermalDampener, ControlDecision

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Fehler: GEMINI_API_KEY Umgebungsvariable ist nicht gesetzt.")
        return

    client = genai.Client()
    
    # Hoch volatiles Start-Szenario (Kappa extrem niedrig -> FREEZING)
    dampener = ThermalDampener(initial_kappa=10.0)
    
    print("=== BEGINNING CLOSED-LOOP TUNING (MAX 5 ITERATIONS) ===\n")
    
    for turn in range(1, 6):
        print(f"--- TURN {turn} ---")
        telemetry = dampener.get_temperature_log()
        print(f"System Log: {telemetry}")
        print(f"Current Kappa: {dampener.kappa:.2f}")
        
        if "Status is PERFECT" in telemetry:
            print(">>> System has reached PERFECT state. Closed-loop successful. <<<")
            break

        prompt = (
            f"You are an automated regulatory AI for a thermal dampener. "
            f"Here is the latest telemetry: {telemetry}\n"
            f"The PERFECT temperature zone is strictly between 90.0K and 110.0K (Target ~100.0K). "
            f"Kappa directly scales the temperature (Temperature = Kappa * 2). "
            f"Analyze the state, decide whether to 'INCREASE', 'DECREASE' or 'HOLD' Kappa, "
            f"and provide the exact 'delta_value' to add to Kappa to reach 100.0K. "
            f"(e.g. if you need to increase temp by 40, Kappa must increase by 20)."
        )
        
        # Aufruf mit Pydantic Structured JSON schema
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': ControlDecision,
                'temperature': 0.1,
            },
        )
        
        # JSON parsen und anwenden
        try:
            decision_data = json.loads(response.text)
            print(f"Model Decision (Structured JSON):\n{json.dumps(decision_data, indent=2)}")
            
            delta = float(decision_data.get('delta_value', 0.0))
            action = decision_data.get('adjustment_action', 'HOLD')
            
            # Sicherheitscheck, falls das Modell einen positiven Delta bei DECREASE ausgibt
            if action == 'DECREASE' and delta > 0:
                delta = -delta
                
            print(f"-> Action: {action}, Applying delta: {delta}")
            dampener.apply_adjustment(delta)
            
        except Exception as e:
            print(f"Failed to parse or apply JSON decision: {e}")
            
        print()
    
    print("=== END OF TUNING ===")

if __name__ == "__main__":
    main()
