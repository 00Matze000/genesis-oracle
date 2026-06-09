import os
from PIL import Image
from google import genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Fehler: GEMINI_API_KEY Umgebungsvariable ist nicht gesetzt.")
        return

    client = genai.Client()
    
    # Bildpfad ermitteln
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(base_dir, 'data', 'audit_target.png')
    
    if not os.path.exists(image_path):
        print(f"Fehler: Konnte das Bild unter {image_path} nicht finden. Bitte führe zuerst generate_signals.py aus.")
        return

    # Bild laden
    img = Image.open(image_path)

    # Prompt definieren
    prompt = (
        "You are a Visual Detective auditing a system telemetry plot. "
        "1. Find the visual anomaly (high-frequency clipping artifact) in this dynamic wave signal. "
        "2. Guess the exact X-axis region (timestep) where the malfunction happened. "
        "3. Write a short, funny poem mocking the engineering team that allowed this bug to pass."
    )

    print("Übergebe Plot an das Gemini Vision Modell...")
    
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=[img, prompt]
    )

    print("\n--- Visual Detective Report ---")
    print(response.text.strip())

if __name__ == "__main__":
    main()
