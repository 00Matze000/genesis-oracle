"""SpreelandDispatcher: Multi-Agent-Orchestrierung (Problem Set 12, Exercise 2).

Der Dispatcher entdeckt seine Infrastruktur-Tools ueber MCP (Bridge-Server,
siehe ``bridge_mcp_server.py``) und traegt zusaetzlich die Bruecke zu den
uebrigen Protokoll-Layern des Agentic Protocol Stack als gebundene Funktionen:
A2A-Verhandlung, AP2-Autorisierung und A2UI-Kartenausgabe. ``root_agent`` wird
von der ADK-Web-UI erkannt; der Reasoning-Stream (AG-UI-Muster) ist dort live
sichtbar.

Start der Web-UI (Repo-Wurzel):  uv run --native-tls adk web
"""

import sys
from pathlib import Path

from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

from scholar_core import tls_fix

# TLS-Proxy-Fix aktivieren, bevor ADK/google-genai die Gemini-API kontaktiert
# (BTU-Netz, siehe scholar_core/tls_fix.py).
tls_fix.apply()

from dispatch_core import a2a_negotiation, a2ui_card, ap2_mandate

_HERE = Path(__file__).resolve().parent
_BRIDGE_SERVER = _HERE / "bridge_mcp_server.py"

# 1. Infrastruktur-Daten via MCP: der Dispatcher spricht den lokalen
#    Bridge-MCP-Server ueber stdio an und entdeckt dessen Tools zur Laufzeit.
infra_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[str(_BRIDGE_SERVER)],
        )
    )
)


# 2. A2A / AP2 / A2UI als gebundene Funktions-Tools -----------------------------


def consult_weather(gauge: str, horizon_h: int = 48) -> dict:
    """Fragt per A2A den WeatherPredictor-Agenten eines anderen Teams (Pegelprognose)."""
    return a2a_negotiation.consult_weather_predictor(gauge, horizon_h)


def negotiate_supply(quantity_kg: int = 2000, budget_eur_per_kg: float = 1.80) -> dict:
    """Verhandelt per A2A den Gurken-Grosseinkauf mit dem Lieferanten-Agenten."""
    return a2a_negotiation.negotiate_gherkins(
        quantity_kg=quantity_kg, budget_eur_per_kg=budget_eur_per_kg
    )


def authorize_purchase(quantity_kg: int, price_eur_per_kg: float) -> dict:
    """Erstellt und signiert das AP2-Mandat fuer den Kauf (autorisiert + verifizierbar)."""
    return ap2_mandate.run_secure_fulfillment(quantity_kg, price_eur_per_kg)


def render_status_card(route: str, eta: str, price_total_eur: float) -> dict:
    """Erzeugt die deklarative A2UI-Karte fuer den Lieferstatus (fuer den AG-UI-Stream)."""
    return a2ui_card.build_delivery_card(
        route=route,
        bridge_rows=[
            ("Hauptbruecke Burg", "maintenance"),
            ("Umleitungsbruecke Muellrose", "open"),
            ("Nordbruecke Cottbus", "open"),
        ],
        gauge_series=[208, 210, 214, 219, 223],
        eta=eta,
        price_total_eur=price_total_eur,
    )


spreeland_dispatcher = Agent(
    model="gemini-3.5-flash",
    name="Spreeland_Dispatcher",
    description=(
        "Logistics dispatcher that orchestrates a resilient Spreeland supply "
        "chain across the agentic protocol stack (MCP, A2A, AP2, A2UI)."
    ),
    instruction=(
        "You coordinate logistics in the Spreeland region, moving organic "
        "harvest to Cottbus despite bridge maintenance and fluctuating river "
        "levels. Work through the agentic protocol stack in this order and "
        "explain each step to the user as you go:\n"
        "1. INFRASTRUCTURE (MCP): call the bridge tools (list_bridges, "
        "get_bridge_status, get_river_level) to find an open route. If the "
        "main bridge is under maintenance, use its detour.\n"
        "2. FORECAST (A2A): call consult_weather to ask the external "
        "WeatherPredictor agent for the river-level outlook.\n"
        "3. NEGOTIATE (A2A): call negotiate_supply to agree a gherkin price "
        "within budget.\n"
        "4. AUTHORIZE (AP2): call authorize_purchase with the agreed quantity "
        "and price; only proceed if the mandate is authorized and its "
        "signature is valid.\n"
        "5. VISUALIZE (A2UI): call render_status_card to emit the delivery "
        "status card.\n"
        "Perform each tool call at most once unless it returns an error. "
        "Ground every statement in a tool result and never invent bridge, "
        "price, or weather data."
    ),
    tools=[infra_tools, consult_weather, negotiate_supply, authorize_purchase, render_status_card],
)

# ADK-Web-UI-Discovery.
root_agent = spreeland_dispatcher
