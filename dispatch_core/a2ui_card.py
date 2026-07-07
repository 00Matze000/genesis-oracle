"""A2UI-Kartenbau + Validierung (Problem Set 12, Dynamic Visualization).

A2UI (Agent-to-UI) beschreibt Oberflaechen *deklarativ*: Der Agent sendet einen
Baum aus rund 18 Primitiven (Layout: Card/Column/Row/Divider, Display:
Text/Image/Table/Chart, Action: Button/Link), den der Host nativ rendert -- ohne
dass der Agent React/Flutter-Code schreibt. Dieses Modul erzeugt die im
Aufgabenblatt geforderte A2UI-Message fuer eine Liefer-Status-Karte und
validiert sie gegen die zugelassenen Primitive.

Der AG-UI-Layer (SSE-Event-Stream) transportiert diese Message spaeter als
Event ``{"type": "a2ui.render", "payload": <card>}`` an das Frontend.
"""

from __future__ import annotations

# Zugelassene deklarative A2UI-Komponenten-Primitive.
A2UI_PRIMITIVES = {
    "Card", "Column", "Row", "Divider",       # Layout
    "Text", "Image", "Table", "Chart",         # Display
    "Button", "Link",                          # Action
}


def build_delivery_card(
    route: str,
    bridge_rows: list[tuple[str, str]],
    gauge_series: list[int],
    eta: str,
    price_total_eur: float,
) -> dict:
    """Baut die A2UI-Message fuer eine Liefer-Status-Karte.

    Args:
        route: Bezeichnung der Lieferroute.
        bridge_rows: Liste aus (Brueckenname, Status) fuer die Statustabelle.
        gauge_series: Pegelwerte (cm) fuer das Verlaufs-Diagramm.
        eta: Voraussichtliche Ankunftszeit (ISO).
        price_total_eur: Gesamtpreis der Beschaffung.

    Returns:
        A2UI-Message als Dict (Typ ``a2ui/v0.8``).
    """
    card = {
        "type": "Card",
        "id": "delivery-status",
        "children": [
            {"type": "Text", "variant": "title", "value": "Lieferstatus Spreeland"},
            {"type": "Text", "variant": "subtitle", "value": route},
            {"type": "Divider"},
            {
                "type": "Table",
                "columns": ["Bruecke", "Status"],
                "rows": [list(r) for r in bridge_rows],
            },
            {
                "type": "Chart",
                "chartType": "line",
                "title": "Pegel Spree-Burg (cm)",
                "series": [{"name": "level_cm", "data": gauge_series}],
            },
            {
                "type": "Row",
                "children": [
                    {"type": "Text", "variant": "label", "value": "ETA"},
                    {"type": "Text", "variant": "value", "value": eta},
                    {"type": "Text", "variant": "label", "value": "Summe"},
                    {"type": "Text", "variant": "value", "value": f"{price_total_eur:.2f} EUR"},
                ],
            },
            {
                "type": "Row",
                "children": [
                    {"type": "Button", "action": "confirm_dispatch", "label": "Disposition freigeben"},
                    {"type": "Button", "action": "reroute", "label": "Umleiten", "style": "secondary"},
                ],
            },
        ],
    }
    return {
        "protocol": "a2ui/v0.8",
        "messageType": "render",
        "surface": card,
    }


def validate_a2ui(message: dict) -> list[str]:
    """Prueft eine A2UI-Message auf ausschliesslich zugelassene Primitive.

    Returns:
        Liste der Fehlermeldungen; leer, wenn die Message gueltig ist.
    """
    errors: list[str] = []

    def walk(node: dict, path: str) -> None:
        t = node.get("type")
        if t not in A2UI_PRIMITIVES:
            errors.append(f"{path}: unbekanntes Primitiv '{t}'")
        for i, child in enumerate(node.get("children", [])):
            walk(child, f"{path}.children[{i}]")

    surface = message.get("surface")
    if not isinstance(surface, dict):
        errors.append("root: 'surface' fehlt oder ist kein Objekt")
    else:
        walk(surface, "surface")
    return errors


if __name__ == "__main__":
    import json

    msg = build_delivery_card(
        route="Burg -> Cottbus (via burg-03)",
        bridge_rows=[
            ("Hauptbruecke Burg", "maintenance"),
            ("Umleitungsbruecke Muellrose", "open"),
            ("Nordbruecke Cottbus", "open"),
        ],
        gauge_series=[208, 210, 214, 219, 223],
        eta="2026-07-08T09:30",
        price_total_eur=3560.00,
    )
    errs = validate_a2ui(msg)
    print("Validierung:", "OK" if not errs else errs)
    print(json.dumps(msg, indent=2, ensure_ascii=False))
