"""A2UI-Referenz-Renderer (Problem Set 12, Dynamic Visualization).

Liest die deklarative A2UI-Message (``a2ui_delivery_card.json``) und rendert sie
mit Matplotlib zu einer PNG. Das ist der *Host-Renderer*, den A2UI voraussetzt:
Der Agent liefert nur den Komponentenbaum, der Host erzeugt die native Ansicht
-- ohne React/Flutter-Code im Agenten. Belegt, dass das JSON-Schema tatsaechlich
eine Oberflaeche ergibt.

Lauf:  uv run --native-tls python -m dispatch_core.a2ui_render
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_JSON = _HERE / "a2ui_delivery_card.json"
_OUT = (
    _HERE.parent.parent
    / "Projekt12" / "Latex" / "Bilder" / "a2ui_card.png"
)

_C_CARD = "#f4f6f8"
_C_ACCENT = "#2f6f4f"
_C_BTN = "#2f6f4f"
_C_BTN2 = "#c9d3cc"


def render(message: dict, out_path: Path) -> None:
    surface = message["surface"]
    children = surface["children"]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.6, 14)
    ax.axis("off")
    # Kartenhintergrund.
    ax.add_patch(plt.Rectangle((0.2, 3.8), 9.6, 10.0, facecolor=_C_CARD,
                               edgecolor="#cbd3da", linewidth=1.5, zorder=0))

    y = 13.0
    for node in children:
        t = node["type"]
        if t == "Text":
            var = node.get("variant")
            if var == "title":
                ax.text(0.6, y, node["value"], fontsize=15, fontweight="bold",
                        color=_C_ACCENT); y -= 0.9
            elif var == "subtitle":
                ax.text(0.6, y, node["value"], fontsize=11, color="#556"); y -= 0.9
        elif t == "Divider":
            ax.plot([0.6, 9.4], [y, y], color="#cbd3da", lw=1); y -= 0.6
        elif t == "Table":
            ax.text(0.6, y, "  ".join(f"{c:<26}" for c in node["columns"]),
                    fontsize=10, fontweight="bold", family="monospace"); y -= 0.7
            for row in node["rows"]:
                status = row[1]
                col = {"open": "#2f8f4f", "maintenance": "#c07a00",
                       "closed": "#b23b3b"}.get(status, "#333")
                ax.text(0.6, y, f"{row[0]:<26}", fontsize=10, family="monospace")
                ax.text(5.2, y, status, fontsize=10, family="monospace", color=col,
                        fontweight="bold"); y -= 0.6
            y -= 0.3
        elif t == "Chart":
            data = node["series"][0]["data"]
            ax.text(0.6, y, node["title"], fontsize=10, fontweight="bold"); y -= 1.8
            # Mini-Sparkline in den freigehaltenen Streifen zeichnen.
            xs = [0.8 + i * (8.4 / (len(data) - 1)) for i in range(len(data))]
            lo, hi = min(data), max(data)
            span = (hi - lo) or 1
            ys = [y + 0.2 + 1.4 * (v - lo) / span for v in data]
            ax.plot(xs, ys, color=_C_ACCENT, lw=2, marker="o", ms=4)
            ax.text(9.3, ys[-1], f"{data[-1]}", fontsize=9, color=_C_ACCENT,
                    ha="right", va="bottom")
            y -= 0.7
        elif t == "Row":
            kids = node["children"]
            if all(k["type"] == "Text" for k in kids):
                parts = "   ".join(k["value"] for k in kids)
                ax.text(0.6, y, parts, fontsize=10); y -= 0.8
            else:  # Button-Reihe
                x = 0.6
                for k in kids:
                    if k["type"] == "Button":
                        secondary = k.get("style") == "secondary"
                        w = 0.14 * len(k["label"]) + 0.6
                        ax.add_patch(plt.Rectangle((x, y - 0.5), w, 0.7,
                                     facecolor=_C_BTN2 if secondary else _C_BTN,
                                     edgecolor="none", zorder=2))
                        ax.text(x + w / 2, y - 0.15, k["label"], fontsize=9,
                                ha="center", va="center", zorder=3,
                                color="#223" if secondary else "white")
                        x += w + 0.4
                y -= 1.0

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    msg = json.loads(_JSON.read_text(encoding="utf-8"))
    render(msg, _OUT)
    print(f"A2UI-Karte gerendert -> {_OUT}")
