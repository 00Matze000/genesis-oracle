import os
from google import genai

def main():
    # Prüfe, ob der API Key gesetzt ist
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Fehler: GEMINI_API_KEY Umgebungsvariable ist nicht gesetzt.")
        print("Bitte setze sie mit: $env:GEMINI_API_KEY=\"dein_api_key\"")
        return

    # Client instanziieren
    client = genai.Client()

    # Prompt für die Oracle-Frage definieren
    prompt = "Explain the difference between a stateful NumPy random generation process and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."

    print("Sende Anfrage an das Orakel (gemini-3.5-flash)...")
    
    # API Call
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=prompt
    )

    print("\n--- Oracle Response ---")
    print(response.text.strip())

if __name__ == "__main__":
    main()
