"""Bridge-MCP-Server: Infrastruktur-Daten der Stadt Cottbus via MCP (Problem Set 12).

Dieser stdio-MCP-Server steht stellvertretend fuer die im Aufgabenblatt
genannte PostgreSQL-Datenbank der Stadt ("@spreeland/bridge-mcp-server"). Statt
eine externe npm-Abhaengigkeit zu erfinden, kapselt er eine reale, lokale
Bruecken-/Pegel-Tabelle und exportiert sie ueber das echte Model Context
Protocol. So kann der ADK-Dispatcher die Tools per MCP *discoveren* und
aufrufen -- exakt der in Exercise 1 (Infrastructure Discovery) beschriebene
MCP-Layer, nur real ausfuehrbar.

Start (stdio):  uv run --native-tls python -m dispatch_core.bridge_mcp_server
"""

from mcp.server.fastmcp import FastMCP

# In-Memory-Abbild der "City PostgreSQL"-Tabellen. In einem echten Deployment
# wuerde der Server hier per psycopg gegen die Datenbank sprechen; die
# MCP-Schnittstelle nach aussen bliebe identisch.
_BRIDGES = {
    "burg-01": {
        "name": "Hauptbruecke Burg (Spree)",
        "status": "maintenance",
        "clearance_t": 3.5,
        "reopen_eta": "2026-07-08T06:00",
        "detour": "burg-03",
    },
    "burg-03": {
        "name": "Umleitungsbruecke Muellrose",
        "status": "open",
        "clearance_t": 7.5,
        "reopen_eta": None,
        "detour": None,
    },
    "cottbus-07": {
        "name": "Nordbruecke Cottbus (B168)",
        "status": "open",
        "clearance_t": 40.0,
        "reopen_eta": None,
        "detour": None,
    },
    "lehde-02": {
        "name": "Kahnfaehre Lehde",
        "status": "closed",
        "clearance_t": 0.0,
        "reopen_eta": "2026-07-12T00:00",
        "detour": "burg-03",
    },
}

_GAUGES = {
    "spree-burg": {"level_cm": 214, "trend": "rising", "flood_mark_cm": 260},
    "spree-cottbus": {"level_cm": 187, "trend": "steady", "flood_mark_cm": 240},
}

mcp = FastMCP("spreeland-bridge-server")


@mcp.tool()
def list_bridges() -> list[dict]:
    """Listet alle Bruecken der Region mit Kennung, Name und aktuellem Status."""
    return [
        {"bridge_id": bid, "name": b["name"], "status": b["status"]}
        for bid, b in _BRIDGES.items()
    ]


@mcp.tool()
def get_bridge_status(bridge_id: str) -> dict:
    """Liefert den Echtzeit-Status einer Bruecke.

    Args:
        bridge_id: Kennung der Bruecke (z. B. 'burg-01').

    Returns:
        Status-Datensatz mit Zustand, Traglast (t), Wiedereroeffnung und
        Umleitungsempfehlung; ein Fehlerfeld, falls die Kennung unbekannt ist.
    """
    b = _BRIDGES.get(bridge_id)
    if b is None:
        return {"error": f"Unbekannte Bruecken-Kennung '{bridge_id}'."}
    return {"bridge_id": bridge_id, **b}


@mcp.tool()
def get_river_level(gauge: str) -> dict:
    """Liefert Pegelstand und Trend eines Flusspegels.

    Args:
        gauge: Pegel-Kennung (z. B. 'spree-burg').
    """
    g = _GAUGES.get(gauge)
    if g is None:
        return {"error": f"Unbekannter Pegel '{gauge}'."}
    return {"gauge": gauge, **g}


if __name__ == "__main__":
    # stdio-Transport: liest MCP-JSON-RPC von stdin, antwortet auf stdout.
    mcp.run(transport="stdio")
