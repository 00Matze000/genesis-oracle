"""Scholar-Prime: Literatur-Recherche-Agent (Problem Set 11).

Baut auf dem ADK-Setup aus Projekt 10 (Observer-Prime) auf, ist aber ein
eigenstaendiger Agent mit Zugriff auf die DeepMind science-skills. Das
`search_arxiv`-Tool ruft das arXiv-CLI-Skript aus dem geklonten Repo auf und
gibt dem Agenten die Trefferliste als Klartext zurueck.
"""

import json
import os
import subprocess
from pathlib import Path

from google.adk.agents.llm_agent import Agent

from scholar_core import tls_fix

# TLS-Proxy-Fix aktivieren, bevor ADK/google-genai die erste HTTPS-Verbindung
# zur Gemini-API aufbaut (BTU-Netz, siehe scholar_core/tls_fix.py).
tls_fix.apply()

# Pfade relativ zu dieser Datei, damit das Tool unabhaengig vom
# Arbeitsverzeichnis der ADK-Web-UI funktioniert.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ARXIV_CLI = (
    _REPO_ROOT
    / "science-skills"
    / "skills"
    / "literature_search_arxiv"
    / "scripts"
    / "search_arxiv.py"
)
_RUNNER = _REPO_ROOT / "scholar_core" / "sci_runner.py"


def _parse_arxiv_stdout(stdout: str):
    """Extrahiert die Trefferliste aus der arXiv-CLI-Ausgabe.

    Das CLI-Skript gibt seinen JSON-Wrapper {"status", "results_count",
    "papers": [...]} nach jedem gefundenen Eintrag erneut aus. Der stdout-Puffer
    enthaelt daher mehrere aneinandergereihte JSON-Objekte; das letzte ist das
    vollstaendige. Diese Funktion dekodiert alle Objekte und liefert die
    `papers`-Liste des letzten. Gibt None zurueck, wenn kein JSON erkennbar ist.
    """
    decoder = json.JSONDecoder()
    idx, last = 0, None
    text = stdout.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        last = obj
        idx = end
    if last is None:
        return None
    if isinstance(last, dict):
        return last.get("papers", [])
    if isinstance(last, list):
        return last
    return None


def search_arxiv_papers(query: str, max_results: int = 5):
    """Ruft das arXiv-CLI auf und gibt die Treffer als Liste von Dicts zurueck.

    Gemeinsam von der Tool-Funktion `search_arxiv` und der Extraktions-Pipeline
    genutzt. Bei Fehlern wird ein Fehlerstring zurueckgegeben.

    Args:
        query: Suchbegriff bzw. arXiv-Query.
        max_results: Maximale Anzahl Treffer.

    Returns:
        list[dict] mit Paper-Metadaten, oder str im Fehlerfall.
    """
    if not _ARXIV_CLI.exists():
        return f"Error: arXiv-CLI nicht gefunden unter {_ARXIV_CLI}"

    cmd = [
        "uv", "run", "--native-tls", "--with", "polite-http",
        "python", str(_RUNNER), str(_ARXIV_CLI),
        "--query", query, "--max_results", str(max_results),
    ]
    try:
        res = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        return "Error: arXiv-Anfrage hat das Zeitlimit ueberschritten."
    if res.returncode != 0:
        return f"CLI Error: {res.stderr.strip()}"

    papers = _parse_arxiv_stdout(res.stdout)
    if papers is None:
        return res.stdout.strip()
    return papers


def search_arxiv(query: str, max_results: int = 5) -> str:
    """Durchsucht arXiv nach wissenschaftlichen Publikationen zu einem Thema.

    Args:
        query: Suchbegriff bzw. arXiv-Query (z. B. 'thermal conductivity UO2').
        max_results: Maximale Anzahl zurueckgegebener Publikationen.

    Returns:
        Ein Klartext-Report mit Titel, Autoren, DOI/arXiv-URL und Abstract je
        Treffer. Bei Fehlern ein beschreibender Fehlerstring.
    """
    papers = search_arxiv_papers(query, max_results)

    if isinstance(papers, str):
        # Fehler-/Rohausgabe unveraendert durchreichen.
        return papers
    if not papers:
        return f"Keine arXiv-Treffer fuer '{query}'."

    blocks = []
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.get("authors", [])[:6])
        doi = p.get("doi") or p.get("id") or "n/a"
        summary = " ".join((p.get("summary") or "").split())
        blocks.append(
            f"[{i}] {p.get('title', 'o. T.')}\n"
            f"    Autoren: {authors}\n"
            f"    DOI/URL: {doi}\n"
            f"    arXiv-ID: {p.get('id', 'n/a')} | Kategorie: {p.get('primary_category', 'n/a')}\n"
            f"    Abstract: {summary}"
        )
    return "\n\n".join(blocks)


scholar_prime = Agent(
    model="gemini-3.5-flash",
    name="scholar_prime",
    description=(
        "An academic research agent specialized in querying scientific "
        "databases and extracting material parameters."
    ),
    instruction=(
        "You are Scholar-Prime, a meticulous academic research agent serving "
        "the engineers of the Neo-Simulacrum. Your task is literature "
        "retrieval and parameter extraction for physical simulations. "
        "When asked about a topic, call the search_arxiv tool to query the "
        "arXiv archive. Read the returned abstracts, judge their relevance to "
        "the user's request, and identify the single most relevant paper. "
        "Summarize its abstract concisely and extract any quantitative "
        "material parameters mentioned (e.g. thermal conductivity, melting "
        "point, density). You never fabricate results: every claim is grounded "
        "in a retrieved paper, and you ALWAYS state the DOI or arXiv identifier "
        "of every source you cite. Reason step by step and state which paper "
        "you selected and why before giving your final summary."
    ),
    tools=[search_arxiv],
)

# ADK erwartet ein `root_agent` fuer die Web-UI-Discovery.
root_agent = scholar_prime
