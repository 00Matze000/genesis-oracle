"""Wird von Python beim Start automatisch importiert (wenn auf sys.path).

Aktiviert den TLS-Proxy-Fix, damit httpx/google-genai in der ADK-Web-UI die
Gemini-API hinter dem BTU-TLS-Proxy erreichen. Siehe scholar_core/tls_fix.py.
"""

try:
    from scholar_core import tls_fix

    tls_fix.apply()
except Exception:
    pass
