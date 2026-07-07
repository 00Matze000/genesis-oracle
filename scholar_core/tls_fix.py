"""TLS-Fix fuer das BTU-Netz mit TLS-terminierendem Proxy.

Der Proxy praesentiert eine CA, der der Browser/das Windows-Zertifikatsdepot
vertraut, deren Zertifikat aber die RFC-5280-Vorgabe verletzt (BasicConstraints
nicht als critical markiert). Python 3.13+ aktiviert im Default-SSL-Kontext das
Flag ``VERIFY_X509_STRICT`` und weist solche Zertifikate zurueck
("Basic Constraints of CA cert not marked critical").

Dieser Fix patcht ``ssl.create_default_context`` so, dass (1) das exportierte
Windows-CA-Bundle zusaetzlich geladen und (2) nur das ueberstrenge
STRICT-Flag entfernt wird. Die eigentliche Zertifikatspruefung bleibt aktiv;
es wird lediglich die RFC-Formalie ignoriert, an der der Proxy scheitert.

Wird ueber ``sitecustomize.py`` (Repo-Wurzel, via PYTHONPATH) automatisch beim
Python-Start aktiviert, sodass auch httpx/google-genai (ADK-Web-UI) profitieren.
"""

import ssl
from pathlib import Path

_BUNDLE = Path(__file__).resolve().parent / "win_ca_bundle.pem"
_orig_create = ssl.create_default_context


def _patched_create(*args, **kwargs):
    ctx = _orig_create(*args, **kwargs)
    # Ueberstrenge RFC-Pruefung deaktivieren (Ursache der Proxy-Ablehnung).
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
    # Zusaetzlich die vom Windows-Depot exportierten CAs (inkl. Proxy) laden.
    if _BUNDLE.exists():
        try:
            ctx.load_verify_locations(cafile=str(_BUNDLE))
        except ssl.SSLError:
            pass
    return ctx


def apply() -> None:
    """Aktiviert den Patch (idempotent)."""
    ssl.create_default_context = _patched_create
