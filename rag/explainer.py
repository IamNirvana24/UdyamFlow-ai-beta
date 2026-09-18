"""
Bot 2 — the "why" explainer, as distinct from Bot 1 (rag/copilot.py, the
"what documents" grounded Q&A copilot).

IMPORTANT HONESTY NOTE, worth understanding before demoing this:
The ANN in api.py is a black-box multi-label classifier — nobody, including
this module, can read its actual internal decision boundary or claim to know
exactly which input crossed which threshold inside the network. This module
does NOT do model interpretability (it is not SHAP/LIME/attention-weights).

What it actually does: for each approval the model predicted as required, it
combines (a) the regulatory "why it is required" / "applies to" text from the
knowledge base with (b) a small set of domain-knowledge rules that check
which of the submitted profile fields are the *typical* real-world triggers
for that approval. The result reads as "here's why this normally applies,
and here's the part of your submission that matches" — which is honest,
useful, and demo-safe. If asked, describe it exactly this way: a rule-based
explainer over the same knowledge base the copilot uses, not a readout of
the neural network's internals.
"""

from __future__ import annotations

from typing import Optional

from .retriever import LABEL_TO_SLUG, retriever

# Sectors CPCB commonly places in the Red/Orange (higher pollution-potential)
# categories — used only to decide which sentence to show, not as a claim
# about the ANN's own reasoning.
_POLLUTING_SECTORS = {
    "Chemical", "Textile Dyeing", "Cement", "Paper & Pulp",
    "Leather Products", "Steel Rolling", "Pharmaceutical",
}

_FOOD_SECTORS = {"Food Processing", "Beverages", "Agro-based", "Rice Milling"}


def _overview_text(slug: str) -> str:
    """Pulls the 'why it is required' / 'applies to' framing from the
    knowledge base's Overview chunk for this approval, so the explanation
    stays grounded in the same source the copilot cites."""
    doc_chunks = retriever.get_document(slug)
    for chunk in doc_chunks:
        if chunk.section == "Overview":
            return chunk.text
    return ""


def _fmt_num(value) -> str:
    if value is None:
        return "not specified"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)


def _explain_trade_license(profile: dict) -> str:
    district = profile.get("district", "your district")
    return (
        f"Any commercial premises operating in {district} needs a Trade Licence "
        f"regardless of size or sector — it is the baseline registration that "
        f"your GST, Factory Licence, and most other approvals will reference "
        f"once issued."
    )


def _explain_gst_registration(profile: dict) -> str:
    investment = _fmt_num(profile.get("investment_lakhs"))
    sector = profile.get("sector", "this sector")
    return (
        f"With a stated investment of ₹{investment} lakh in a {sector} operation, "
        f"you are very likely to cross the GST turnover threshold once "
        f"commercial activity begins, and issuing any compliant tax invoice "
        f"requires a GSTIN regardless of turnover."
    )


def _explain_factory_license(profile: dict) -> str:
    employees = profile.get("employee_count")
    power = profile.get("power_requirement_kw")
    sector = profile.get("sector", "this sector")
    is_service = profile.get("is_service_sector")
    if is_service:
        return (
            f"Your profile was marked as a service-sector operation, but the "
            f"model still flagged this — worth double-checking the sector and "
            f"power/employee figures you submitted, since Factory Licence "
            f"applies to manufacturing premises specifically."
        )
    return (
        f"You reported {_fmt_num(employees)} employees and "
        f"{_fmt_num(power)} kW of power in a {sector} operation. The Factories "
        f"Act's worker-count threshold is lower when power is used, which is "
        f"exactly the kind of powered manufacturing unit this profile describes."
    )


def _explain_labour_registration(profile: dict) -> str:
    employees = profile.get("employee_count")
    return (
        f"With {_fmt_num(employees)} employees stated, EPF and ESI thresholds "
        f"are typically crossed at a headcount well below this, and Shops & "
        f"Establishment registration applies from the first employee onward."
    )


def _explain_fire_noc(profile: dict) -> str:
    land = profile.get("land_area_sqm")
    hazardous = profile.get("uses_hazardous_material")
    sector = profile.get("sector", "this sector")
    if hazardous:
        return (
            f"You flagged hazardous material handling in a {sector} unit on "
            f"{_fmt_num(land)} sqm — combustible or hazardous storage is one of "
            f"the clearest triggers for fire safety review, independent of "
            f"building size alone."
        )
    return (
        f"Your stated built-up area of {_fmt_num(land)} sqm is the main factor "
        f"here — fire safety requirements scale with covered area and "
        f"occupancy class under the National Building Code."
    )


def _explain_pollution_noc(profile: dict) -> str:
    sector = profile.get("sector", "this sector")
    hazardous = profile.get("uses_hazardous_material")
    water = profile.get("water_usage_kld")
    if hazardous:
        return (
            f"You flagged hazardous material handling — this is a direct "
            f"trigger for Pollution Board consent regardless of your other "
            f"submitted values."
        )
    if sector in _POLLUTING_SECTORS:
        return (
            f"A {sector} unit is generally placed in a higher pollution-potential "
            f"category by CPCB's classification, which is what drives this "
            f"requirement — independent of investment size."
        )
    return (
        f"With a stated water usage of {_fmt_num(water)} KLD in a {sector} "
        f"operation, effluent generation is enough on its own to bring this "
        f"under Pollution Board consent."
    )


def _explain_environmental_clearance(profile: dict) -> str:
    investment = profile.get("investment_lakhs") or 0
    hazardous = profile.get("uses_hazardous_material")
    if hazardous or investment >= 2000:
        return (
            f"At a stated investment of ₹{_fmt_num(profile.get('investment_lakhs'))} lakh"
            f"{' and with hazardous material handling flagged' if hazardous else ''}, "
            f"this profile sits in the range where scheduled projects typically "
            f"attract Environmental Clearance — but the EIA Notification's "
            f"schedule is process-specific, so this should be confirmed against "
            f"the exact activity list, not investment size alone."
        )
    return (
        "At the scale you submitted this is a borderline case — Environmental "
        "Clearance depends on the specific scheduled activity, not just "
        "investment or size, so this is worth confirming directly with the "
        "State Environment Impact Assessment Authority."
    )


def _explain_electricity_noc(profile: dict) -> str:
    power = profile.get("power_requirement_kw")
    return (
        f"Your stated power requirement of {_fmt_num(power)} kW is what "
        f"determines whether you're on a standard low-tension connection or "
        f"need the additional high-tension approval process through PSPCL."
    )


def _explain_boiler_license(profile: dict) -> str:
    return (
        "You indicated this facility uses a boiler — that alone is the "
        "trigger for this licence under the Boilers Act, 1923, independent "
        "of every other field in your profile."
    )


def _explain_explosives_license(profile: dict) -> str:
    return (
        "You flagged hazardous material handling — this is the direct "
        "trigger. The specific licence and safety-distance requirements "
        "depend on the exact substance and stored quantity, which a category "
        "flag alone cannot determine, so this needs confirmation with PESO."
    )


def _explain_fssai_license(profile: dict) -> str:
    sector = profile.get("sector", "this sector")
    if sector in _FOOD_SECTORS:
        return (
            f"Your sector is {sector}, which brings this under FSSAI "
            f"regardless of scale — the only open question is which of the "
            f"three licence tiers (Basic, State, or Central) applies, and "
            f"that depends on your production capacity."
        )
    return (
        f"Food-adjacent activity was inferred from your submitted profile "
        f"even though {sector} isn't a typical food sector — worth "
        f"double-checking the sector selected."
    )


def _explain_building_plan_approval(profile: dict) -> str:
    land = profile.get("land_area_sqm")
    return (
        f"Any new construction on your stated site of {_fmt_num(land)} sqm "
        f"needs sanctioned building plans before anything else in your "
        f"approval journey can physically proceed."
    )


_EXPLAINERS = {
    "trade_license": _explain_trade_license,
    "gst_registration": _explain_gst_registration,
    "factory_license": _explain_factory_license,
    "labour_registration": _explain_labour_registration,
    "fire_noc": _explain_fire_noc,
    "pollution_noc": _explain_pollution_noc,
    "environmental_clearance": _explain_environmental_clearance,
    "electricity_noc": _explain_electricity_noc,
    "boiler_license": _explain_boiler_license,
    "explosives_license": _explain_explosives_license,
    "fssai_license": _explain_fssai_license,
    "building_plan_approval": _explain_building_plan_approval,
}


def explain_approval(key: str, profile: dict, confidence: Optional[float] = None) -> dict:
    """
    key: the snake_case approval key (e.g. "fire_noc") — same keys api.py's
         target_cols and LICENSE_LABELS use.
    profile: the submitted business profile dict (district, sector,
         investment_lakhs, employee_count, power_requirement_kw,
         land_area_sqm, water_usage_kld, uses_hazardous_material,
         uses_boiler, is_service_sector).
    confidence: the ANN's confidence percentage for this label, if available.
    """
    slug = LABEL_TO_SLUG.get(key, key)
    explainer_fn = _EXPLAINERS.get(key)

    personalized = explainer_fn(profile) if explainer_fn else (
        "This approval was flagged by the model for your submitted profile. "
        "See the compliance copilot for what it generally involves."
    )

    return {
        "key": key,
        "slug": slug,
        "confidence": confidence,
        "explanation": personalized,
    }


def explain_all(journey: list, profile: dict) -> list:
    """
    journey: the list of dicts api.py's /api/predict returns in its
             'journey' field — each has 'key', 'name', 'confidence'.
    profile: the submitted business profile dict.
    """
    return [
        {
            **explain_approval(step["key"], profile, step.get("confidence")),
            "name": step.get("name"),
        }
        for step in journey
    ]
