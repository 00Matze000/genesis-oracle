"""Runner fuer die DeepMind science-skills CLI-Skripte hinter einem TLS-Proxy.

Das BTU-Netz terminiert HTTPS ueber einen intercepting Proxy, dessen CA-Zert
die strikte Pruefung von urllib nicht besteht ("Basic Constraints of CA cert
not marked critical"). Die science-skills CLI-Skripte rufen die APIs ueber
`urllib.request.urlopen` mit dem globalen Default-SSL-Kontext auf. Dieser Runner
setzt den globalen Default auf einen unverifizierten Kontext und fuehrt danach
das gewuenschte CLI-Skript unveraendert per runpy aus. Die zurueckgelieferten
Daten stammen weiterhin real von OpenAlex/arXiv; lediglich die TLS-Verifikation
wird umgangen, um den Proxy zu passieren.

Aufruf:
    python sci_runner.py <cli_script.py> <cli-argumente...>
"""

import runpy
import ssl
import sys

# Globalen HTTPS-Kontext auf unverifiziert setzen (Proxy-Workaround).
ssl._create_default_https_context = ssl._create_unverified_context

if len(sys.argv) < 2:
    print("usage: python sci_runner.py <cli_script.py> [args...]", file=sys.stderr)
    sys.exit(2)

target = sys.argv[1]
# argv fuer das Zielskript so aufbauen, als waere es direkt aufgerufen worden.
sys.argv = [target] + sys.argv[2:]
runpy.run_path(target, run_name="__main__")
