"""Parameter-Extraktion aus wissenschaftlichen Texten (Problem Set 11, Ex4).

Der Extraktor arbeitet rein regelbasiert (Regex) und meldet ausschliesslich
Groessen, die im uebergebenen Text tatsaechlich vorkommen. Es werden keine
Werte erfunden oder aus Vorwissen ergaenzt (Anti-Halluzination): Fehlt eine
Groesse im Abstract, bleibt das Feld `None`.
"""

import re

# Bekannte Reaktor-/Brennstoffmaterialien, nach denen im Text gesucht wird.
_MATERIALS = [
    ("Uranium Dioxide (UO2)", r"\bUO2\b|uranium dioxide"),
    ("Beryllium Oxide (BeO)", r"\bBeO\b|beryllium oxide"),
    ("Plutonium-240", r"\b240Pu\b|plutonium-?240"),
    ("Silicon Carbide (SiC)", r"\bSiC\b|silicon carbide"),
    ("Zircaloy", r"zircaloy|\bZr\b"),
]

# Zahl (auch wissenschaftliche Notation und negative Exponenten wie 10-9).
_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:\s*[×x*]\s*10[-−]?\d+|[eE][-+]?\d+)?"


def _to_float(raw: str):
    """Konvertiert einen erkannten Zahlenstring nach float, sonst None."""
    s = raw.replace(",", ".").replace(" ", "")
    s = s.replace("×10", "e").replace("x10", "e").replace("*10", "e")
    s = s.replace("−", "-")
    try:
        return float(s)
    except ValueError:
        return None


def _first(pattern: str, text: str):
    """Erste Regex-Gruppe (Zahl) als float, sonst None; plus Rohtreffer."""
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None, None
    return _to_float(m.group(1)), m.group(0).strip()


def extract_parameters_from_text(text: str) -> dict:
    """Extrahiert thermodynamische und Materialparameter aus einem Text.

    Args:
        text: Textausschnitt (z. B. der Abstract einer Publikation), der die
            zu extrahierenden Groessen enthaelt.

    Returns:
        Ein Dictionary mit den Feldern:
            material (str | None): erkanntes Hauptmaterial.
            thermal_conductivity_W_mK (float | None): Waermeleitfaehigkeit in
                W/(m*K), falls im Text als Zahl mit Einheit angegeben.
            interfacial_thermal_resistance_m2K_W (float | None): Grenzflaechen-
                Waermewiderstand in m^2*K/W.
            melting_point_K (float | None): Schmelzpunkt in Kelvin.
            density_g_cm3 (float | None): Dichte in g/cm^3.
            temperature_range_C (list[float] | None): erkanntes Temperatur-
                intervall in Grad Celsius.
            raw_matches (dict): die jeweiligen Rohtreffer zur Nachvollziehbarkeit.
            source_preview (str): erste 200 Zeichen des Quelltexts.
    """
    result = {
        "material": None,
        "thermal_conductivity_W_mK": None,
        "interfacial_thermal_resistance_m2K_W": None,
        "melting_point_K": None,
        "density_g_cm3": None,
        "temperature_range_C": None,
        "raw_matches": {},
    }

    # Material erkennen.
    for name, pat in _MATERIALS:
        if re.search(pat, text, flags=re.IGNORECASE):
            result["material"] = name
            break

    # Waermeleitfaehigkeit: Zahl gefolgt von W/(m K)-Einheit in diversen Schreibweisen.
    k_unit = r"W\s*[/·*]?\s*\(?\s*m\s*[·*\s]\s*K\)?|W\s*m-?1\s*K-?1"
    val, raw = _first(rf"({_NUM})\s*(?:{k_unit})", text)
    if val is not None:
        result["thermal_conductivity_W_mK"] = val
        result["raw_matches"]["thermal_conductivity"] = raw

    # Grenzflaechen-Waermewiderstand (m^2 K / W), inkl. Zehnerpotenz-Notation "10-9".
    m = re.search(r"(10[-−]\d+)\s*m2\s*K\s*/\s*W", text, flags=re.IGNORECASE)
    if m:
        result["interfacial_thermal_resistance_m2K_W"] = _to_float(
            m.group(1).replace("10", "1e").replace("−", "-")
        )
        result["raw_matches"]["interfacial_thermal_resistance"] = m.group(0).strip()

    # Schmelzpunkt.
    val, raw = _first(rf"melting point[^\d]*({_NUM})\s*K", text)
    if val is not None:
        result["melting_point_K"] = val
        result["raw_matches"]["melting_point"] = raw

    # Dichte.
    val, raw = _first(rf"({_NUM})\s*g\s*/?\s*cm\s*3", text)
    if val is not None:
        result["density_g_cm3"] = val
        result["raw_matches"]["density"] = raw

    # Temperaturintervall der Form "900-1200C".
    m = re.search(r"(\d{2,4})\s*[-–]\s*(\d{2,4})\s*°?C", text)
    if m:
        result["temperature_range_C"] = [float(m.group(1)), float(m.group(2))]
        result["raw_matches"]["temperature_range"] = m.group(0).strip()

    result["source_preview"] = text[:200]
    return result
