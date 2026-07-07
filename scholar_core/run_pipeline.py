"""Extraktions-Pipeline fuer Scholar-Prime (Problem Set 11, Ex4).

Kette: arXiv-Suche -> relevantestes Paper waehlen -> Abstract an den Extraktor
uebergeben -> strukturiertes JSON mit den Parametern und der DOI-Referenz des
Quellpapers schreiben.

Aufruf:
    uv run --native-tls python -m scholar_core.run_pipeline
"""

import json
from pathlib import Path

from scholar_core.agent import search_arxiv_papers
from scholar_core.extraction import extract_parameters_from_text

_QUERY = "thermal conductivity uranium dioxide UO2 nuclear fuel"
_OUT = Path(__file__).resolve().parent.parent / "docs" / "simulation_parameters.json"


def _param_richness(params: dict) -> int:
    """Anzahl tatsaechlich extrahierter (nicht-leerer) Messgroessen."""
    keys = [
        "thermal_conductivity_W_mK",
        "interfacial_thermal_resistance_m2K_W",
        "melting_point_K",
        "density_g_cm3",
        "temperature_range_C",
    ]
    return sum(1 for k in keys if params.get(k) is not None)


def main() -> None:
    papers = search_arxiv_papers(_QUERY, max_results=5)
    if isinstance(papers, str):
        raise SystemExit(f"arXiv-Suche fehlgeschlagen: {papers}")
    if not papers:
        raise SystemExit("Keine Treffer.")

    # "Relevantestes" Paper = das, aus dessen Abstract der Extraktor die meisten
    # Simulationsparameter gewinnt; bei Gleichstand entscheidet die arXiv-
    # Relevanzreihenfolge (Listenposition).
    scored = []
    for idx, p in enumerate(papers):
        params = extract_parameters_from_text(p.get("summary", ""))
        scored.append((_param_richness(params), -idx, p, params))
    scored.sort(reverse=True)
    _, _, best, params = scored[0]

    record = {
        "extracted_parameters": params,
        "source_paper": {
            "title": best.get("title"),
            "authors": best.get("authors"),
            "arxiv_id": best.get("id"),
            "doi": best.get("doi"),
            "doi_url": (
                f"https://doi.org/{best['doi']}" if best.get("doi") else None
            ),
            "pdf_url": best.get("pdf_url"),
            "primary_category": best.get("primary_category"),
            "published": best.get("published"),
        },
        "provenance": {
            "database": "arXiv (via DeepMind science-skills literature_search_arxiv)",
            "query": _QUERY,
            "extractor": "scholar_core.extraction.extract_parameters_from_text",
        },
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Gewaehltes Paper: {best.get('title')}")
    print(f"DOI/arXiv: {best.get('doi') or best.get('id')}")
    print(f"Extrahierte Parameter: {params['raw_matches']}")
    print(f"JSON geschrieben nach: {_OUT}")


if __name__ == "__main__":
    main()
