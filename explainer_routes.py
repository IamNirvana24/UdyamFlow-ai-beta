"""
Flask blueprint for Bot 2 — the "why" explainer.

Distinct from rag_bp (Bot 1, the /api/ask compliance copilot):
  - Bot 1 answers general questions about an approval (documents, process)
  - Bot 2 explains why THIS specific submitted profile triggered it

Wire into api.py alongside the existing rag_bp registration:

    from explainer_routes import explainer_bp
    app.register_blueprint(explainer_bp)
"""

from flask import Blueprint, jsonify, request

import db
from rag import explain_all

explainer_bp = Blueprint("explainer", __name__)


def _profile_from_row(row: "db.Prediction") -> dict:
    return {
        "district": row.district,
        "sector": row.sector,
        "investment_lakhs": row.investment_lakhs,
        "employee_count": row.employee_count,
        "power_requirement_kw": row.power_requirement_kw,
        "land_area_sqm": row.land_area_sqm,
        "water_usage_kld": row.water_usage_kld,
        "uses_hazardous_material": row.uses_hazardous_material,
        "uses_boiler": row.uses_boiler,
        "is_service_sector": row.is_service_sector,
    }


@explainer_bp.route("/api/explain", methods=["POST"])
def explain():
    """
    Request (preferred — looks up a saved prediction):
        { "prediction_id": 17 }

    Request (fallback — no DB lookup needed, e.g. before persistence exists):
        { "profile": {district, sector, investment_lakhs, employee_count,
                       power_requirement_kw, land_area_sqm, water_usage_kld,
                       uses_hazardous_material, uses_boiler, is_service_sector},
          "journey": [{key, name, confidence}, ...] }

    Response:
        { "prediction_id": 17 | null,
          "explanations": [
            { "key": "boiler_license", "name": "Boiler License",
              "confidence": 92.1, "slug": "boiler_license",
              "explanation": "..." }
          ] }
    """
    payload = request.get_json(silent=True) or {}
    prediction_id = payload.get("prediction_id")

    if prediction_id is not None:
        with db.get_db_session() as db_session:
            row = (
                db_session.query(db.Prediction)
                .filter(db.Prediction.id == prediction_id)
                .first()
            )
            if not row:
                return jsonify({"error": "Prediction not found"}), 404

            profile = _profile_from_row(row)
            journey = row.journey_json or []
            explanations = explain_all(journey, profile)

        return jsonify({"prediction_id": prediction_id, "explanations": explanations}), 200

    # Fallback path — explain from an inline profile + journey, no DB lookup
    profile = payload.get("profile")
    journey = payload.get("journey")
    if not profile or not journey:
        return jsonify(
            {"error": "Provide either prediction_id, or both profile and journey."}
        ), 400

    explanations = explain_all(journey, profile)
    return jsonify({"prediction_id": None, "explanations": explanations}), 200
