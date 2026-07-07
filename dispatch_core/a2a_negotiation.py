"""A2A-Verhandlungslogik fuer den Spreeland-Dispatcher (Problem Set 12).

Bildet den Agent2Agent-Layer (Exercise 1: *Expert Consultation* und die
Lieferanten-Verhandlung aus Exercise 2) real und deterministisch nach. Statt
einen Fremd-Agenten ueber das Netz zu erreichen, modelliert dieses Modul die
A2A-Kernartefakte lokal:

* **Agent Card** -- die JSON-Selbstbeschreibung, die A2A unter
  ``/.well-known/agent-card.json`` erwartet (Discovery).
* **Task / Message** -- die aufgabenorientierte Kommunikation mit den
  Zustaenden ``submitted -> working -> input-required -> completed``.

Damit lassen sich beide A2A-Interaktionen ausfuehren und pruefen:
``consult_weather_predictor`` (Discovery + Query eines Fremd-Agenten) und
``negotiate_gherkins`` (mehrrundige Preisverhandlung mit einem Lieferanten).
Alles deterministisch, damit der Beleg reproduzierbar ist.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# A2A-Kernstrukturen (vereinfachtes, aber protokolltreues Abbild)
# ---------------------------------------------------------------------------


@dataclass
class AgentSkill:
    """Eine im Agent Card beworbene Faehigkeit."""

    id: str
    name: str
    description: str


@dataclass
class AgentCard:
    """A2A Agent Card -- unter ``/.well-known/agent-card.json`` publiziert.

    Traegt Identitaet, Endpunkt und die maschinenlesbare Faehigkeitsliste,
    anhand derer ein anderer Agent entscheidet, ob und wie er delegiert.
    """

    name: str
    description: str
    url: str
    version: str
    skills: list[AgentSkill]
    capabilities: dict = field(default_factory=lambda: {"streaming": True})

    def to_json(self) -> dict:
        return asdict(self)


# Fremd-Agent eines anderen Teams (Expert Consultation). In der Praxis wuerde
# der Dispatcher diese Card per HTTP-GET vom Well-Known-Pfad laden.
WEATHER_PREDICTOR_CARD = AgentCard(
    name="WeatherPredictor",
    description="Spezialisierter Agent fuer Spreewald-Wetter- und Pegelprognosen.",
    url="https://weather.team-b.spreeland.example/a2a",
    version="1.4.0",
    skills=[
        AgentSkill(
            id="river_forecast",
            name="River level forecast",
            description="48-h-Pegelprognose fuer einen Spree-Pegel.",
        ),
        AgentSkill(
            id="precip_forecast",
            name="Precipitation forecast",
            description="Niederschlagswahrscheinlichkeit fuer eine Region.",
        ),
    ],
)

GHERKIN_SUPPLIER_CARD = AgentCard(
    name="SpreewaldGherkinCoop",
    description="Grosshandels-Agent der Gurken-Erzeugergenossenschaft.",
    url="https://sales.gherkin-coop.spreeland.example/a2a",
    version="2.0.1",
    skills=[
        AgentSkill(
            id="quote_gherkins",
            name="Wholesale gherkin quote",
            description="Preisangebot fuer eine Gurkenmenge in kg.",
        ),
    ],
)


def discover_agent_card(card: AgentCard) -> dict:
    """Simuliert das A2A-Discovery: liefert die Well-Known Agent Card als JSON."""
    return card.to_json()


# ---------------------------------------------------------------------------
# Expert Consultation: WeatherPredictor eines anderen Teams befragen
# ---------------------------------------------------------------------------


def consult_weather_predictor(gauge: str, horizon_h: int = 48) -> dict:
    """Delegiert eine Pegelprognose per A2A an den WeatherPredictor-Agenten.

    Ablauf: (1) Agent Card entdecken, passenden Skill pruefen; (2) A2A-Task
    ``river_forecast`` stellen; (3) Ergebnis-Artefakt zurueckgeben.

    Args:
        gauge: Pegel-Kennung (z. B. 'spree-burg').
        horizon_h: Prognosehorizont in Stunden.

    Returns:
        Dict mit Task-Status, genutztem Skill und Prognose-Artefakt.
    """
    card = WEATHER_PREDICTOR_CARD
    skill = next((s for s in card.skills if s.id == "river_forecast"), None)
    if skill is None:
        return {"state": "failed", "error": "Skill 'river_forecast' nicht im Card."}

    # Deterministische Beispielprognose (in echt: Antwort des Fremd-Agenten).
    trend_cm_per_h = 1.2
    forecast_cm = 214 + round(trend_cm_per_h * horizon_h)
    artifact = {
        "gauge": gauge,
        "horizon_h": horizon_h,
        "predicted_level_cm": forecast_cm,
        "flood_risk": "elevated" if forecast_cm > 250 else "moderate",
        "confidence": 0.82,
    }
    return {
        "state": "completed",
        "remote_agent": card.name,
        "skill_used": skill.id,
        "artifact": artifact,
    }


# ---------------------------------------------------------------------------
# Lieferanten-Verhandlung: 2 t Gurken per A2A aushandeln
# ---------------------------------------------------------------------------


def negotiate_gherkins(
    quantity_kg: int = 2000,
    budget_eur_per_kg: float = 1.80,
    supplier_floor_eur_per_kg: float = 1.55,
    supplier_opening_eur_per_kg: float = 2.10,
    max_rounds: int = 5,
) -> dict:
    """Fuehrt eine mehrrundige A2A-Preisverhandlung fuer Gurken.

    Modelliert den Nachrichtenaustausch zweier A2A-Agenten: Der Dispatcher
    oeffnet mit einem niedrigen Gebot, der Lieferant kontert oberhalb seiner
    Preisuntergrenze. Beide bewegen sich per *split-the-difference* aufeinander
    zu. Ein Deal entsteht, sobald das Lieferanten-Gebot das Budget des
    Dispatchers erreicht; andernfalls endet die Task ohne Einigung.

    Args:
        quantity_kg: Zu beschaffende Menge in kg.
        budget_eur_per_kg: Maximaler Preis, den der Dispatcher zahlen darf.
        supplier_floor_eur_per_kg: Preisuntergrenze des Lieferanten.
        supplier_opening_eur_per_kg: Eroeffnungsgebot des Lieferanten.
        max_rounds: Maximale Anzahl Verhandlungsrunden.

    Returns:
        Dict mit vollstaendigem Transkript und Endergebnis.
    """
    transcript: list[dict] = []
    # Dispatcher startet bewusst niedrig, aber nicht unter der Lieferantengrenze.
    buyer_bid = round(supplier_floor_eur_per_kg - 0.10, 2)
    seller_ask = round(supplier_opening_eur_per_kg, 2)
    state = "working"
    deal_price = None

    for rnd in range(1, max_rounds + 1):
        transcript.append(
            {"round": rnd, "role": "buyer", "action": "bid", "eur_per_kg": buyer_bid}
        )
        transcript.append(
            {"round": rnd, "role": "seller", "action": "counter", "eur_per_kg": seller_ask}
        )

        # Einigung, falls das aktuelle Verkaeufergebot ins Budget passt.
        if seller_ask <= budget_eur_per_kg:
            deal_price = seller_ask
            state = "completed"
            transcript.append(
                {"round": rnd, "role": "buyer", "action": "accept", "eur_per_kg": deal_price}
            )
            break

        # Split-the-difference: beide Seiten geben je zur Haelfte nach.
        gap = round(seller_ask - buyer_bid, 2)
        buyer_bid = round(min(buyer_bid + gap / 2, budget_eur_per_kg), 2)
        seller_ask = round(max(seller_ask - gap / 2, supplier_floor_eur_per_kg), 2)

    if deal_price is None:
        state = "failed"

    result = {
        "state": state,
        "quantity_kg": quantity_kg,
        "budget_eur_per_kg": budget_eur_per_kg,
        "agreed_price_eur_per_kg": deal_price,
        "total_eur": round(deal_price * quantity_kg, 2) if deal_price else None,
        "rounds": len([t for t in transcript if t["role"] == "seller"]),
        "transcript": transcript,
        "supplier": GHERKIN_SUPPLIER_CARD.name,
    }
    return result


if __name__ == "__main__":
    import json

    print("=== A2A Discovery: WeatherPredictor Agent Card ===")
    print(json.dumps(discover_agent_card(WEATHER_PREDICTOR_CARD), indent=2, ensure_ascii=False))

    print("\n=== A2A Expert Consultation: river_forecast ===")
    print(json.dumps(consult_weather_predictor("spree-burg"), indent=2, ensure_ascii=False))

    print("\n=== A2A Negotiation: 2 t Gurken ===")
    print(json.dumps(negotiate_gherkins(), indent=2, ensure_ascii=False))
