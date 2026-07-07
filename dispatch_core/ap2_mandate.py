"""AP2-Mandate fuer die sichere Beschaffung (Problem Set 12, Secure Fulfillment).

AP2 (Agent Payments Protocol) sichert Kaeufe, die ein Agent ohne anwesenden
Menschen ausfuehrt, ueber kryptografisch signierte *Mandate* (ECDSA P-256):

* **Intent Mandate** -- der Eigentuemer legt vorab Rahmenbedingungen fest
  (max. Menge, max. Preis, Warenkategorie).
* **Cart Mandate** -- der konkrete Warenkorb wird gegen das Intent-Mandat
  geprueft und vom Eigentuemer signiert freigegeben.

Dieses Modul erzeugt beide Mandate fuer den 2-t-Gurkenkauf, signiert sie mit
einem P-256-Schluessel und verifiziert die Signatur -- so ist der Kauf
zugleich *autorisiert* (Intent) und *kryptografisch nachweisbar* (Signatur).
"""

from __future__ import annotations

import hashlib
import json

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,  # noqa: F401  (dokumentiert das DSS-Format)
)
from cryptography.exceptions import InvalidSignature


def _canonical(payload: dict) -> bytes:
    """Kanonische JSON-Serialisierung (stabile Signatur-Basis)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def build_intent_mandate(owner: str, max_kg: int, max_eur_per_kg: float, category: str) -> dict:
    """Erzeugt das Intent-Mandat (Rahmen-Autorisierung des Eigentuemers)."""
    return {
        "type": "IntentMandate",
        "owner": owner,
        "constraints": {
            "category": category,
            "max_quantity_kg": max_kg,
            "max_price_eur_per_kg": max_eur_per_kg,
        },
    }


def build_cart_mandate(intent: dict, item: str, quantity_kg: int, price_eur_per_kg: float) -> dict:
    """Erzeugt das Cart-Mandat und prueft es gegen die Intent-Constraints.

    Returns:
        Cart-Mandat mit Feld ``authorized`` (True nur, wenn alle Constraints
        des Intent-Mandats eingehalten sind).
    """
    c = intent["constraints"]
    authorized = (
        quantity_kg <= c["max_quantity_kg"]
        and price_eur_per_kg <= c["max_price_eur_per_kg"]
    )
    return {
        "type": "CartMandate",
        "item": item,
        "quantity_kg": quantity_kg,
        "price_eur_per_kg": price_eur_per_kg,
        "total_eur": round(quantity_kg * price_eur_per_kg, 2),
        "intent_hash": hashlib.sha256(_canonical(intent)).hexdigest(),
        "authorized": authorized,
    }


def sign_mandate(mandate: dict, private_key: ec.EllipticCurvePrivateKey) -> str:
    """Signiert ein Mandat mit ECDSA P-256 (SHA-256). Gibt Hex-Signatur zurueck."""
    from cryptography.hazmat.primitives import hashes

    signature = private_key.sign(_canonical(mandate), ec.ECDSA(hashes.SHA256()))
    return signature.hex()


def verify_mandate(mandate: dict, signature_hex: str, public_key: ec.EllipticCurvePublicKey) -> bool:
    """Verifiziert die ECDSA-P256-Signatur eines Mandats."""
    from cryptography.hazmat.primitives import hashes

    try:
        public_key.verify(
            bytes.fromhex(signature_hex), _canonical(mandate), ec.ECDSA(hashes.SHA256())
        )
        return True
    except InvalidSignature:
        return False


def run_secure_fulfillment(quantity_kg: int = 2000, price_eur_per_kg: float = 1.78) -> dict:
    """Kompletter AP2-Ablauf fuer den Gurkenkauf: Mandate, Signatur, Pruefung.

    Returns:
        Dict mit beiden Mandaten, Signatur, Verifikationsergebnis und einem
        Negativtest (verfaelschtes Mandat -> Signatur ungueltig).
    """
    # Der Eigentuemer haelt den privaten Schluessel (P-256).
    owner_key = ec.generate_private_key(ec.SECP256R1())
    owner_pub = owner_key.public_key()

    intent = build_intent_mandate(
        owner="farm-owner@spreeland", max_kg=2500, max_eur_per_kg=1.80, category="vegetables"
    )
    cart = build_cart_mandate(intent, "gherkins", quantity_kg, price_eur_per_kg)
    signature = sign_mandate(cart, owner_key)
    valid = verify_mandate(cart, signature, owner_pub)

    # Negativtest: manipulierter Warenkorb muss die Verifikation brechen.
    tampered = dict(cart, quantity_kg=9999, total_eur=17800.0)
    tampered_valid = verify_mandate(tampered, signature, owner_pub)

    return {
        "intent_mandate": intent,
        "cart_mandate": cart,
        "signature_ecdsa_p256": signature,
        "signature_valid": valid,
        "authorized_by_intent": cart["authorized"],
        "tampered_signature_valid": tampered_valid,
    }


if __name__ == "__main__":
    print(json.dumps(run_secure_fulfillment(), indent=2, ensure_ascii=False))
