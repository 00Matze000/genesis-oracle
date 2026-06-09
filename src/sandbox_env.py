from pydantic import BaseModel, Field

class ControlDecision(BaseModel):
    system_state: str = Field(description="Must be 'FREEZING', 'BOILING', or 'PERFECT'")
    adjustment_action: str = Field(description="Must be 'INCREASE', 'DECREASE', or 'HOLD'")
    delta_value: float = Field(description="The exact numerical change to apply to Kappa")
    confidence_score: float

class ThermalDampener:
    def __init__(self, initial_kappa: float):
        self.kappa = initial_kappa
        self.target_temp = 100.0
    
    def get_temperature_log(self) -> str:
        # Simples physikalisches Modell: Temperatur ist Kappa * 2
        temp = self.kappa * 2.0
        
        state = "PERFECT"
        if temp < 90.0:
            state = "FREEZING"
        elif temp > 110.0:
            state = "BOILING"
            
        return f"[TELEMETRY] Current dampener temperature: {temp:.1f}K. Status is {state}."

    def apply_adjustment(self, delta: float):
        self.kappa += delta
