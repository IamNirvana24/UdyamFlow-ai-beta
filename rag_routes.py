"""
Flask blueprint for the UdyamFlow AI compliance copilot.

Wire into api.py with two lines and nothing else changes:

    from rag_routes import rag_bp
    app.register_blueprint(rag_bp)

Existing endpoints (/api/options, /api/predict, /api/extract) are untouched,
so the frontend developer's current work cannot break.
"""

from flask import Blueprint, jsonify, request

from rag import answer_question, retriever, suggested_questions
import db

rag_bp = Blueprint("rag", __name__)

MAX_QUESTION_LENGTH = 500


@rag_bp.route("/api/ask", methods=["POST"])
def ask():
    """
    Request:
        {
          "question": "What documents do I need for a Fire NOC?",
          "approval":  "Fire NOC",       // optional, when asked from a card
          "session_id": "a1b2c3...",     // optional, links to history
          "prediction_id": 42            // optional, links to a saved run
        }

    Response:
        {
          "answer": "...",
          "citations": [
            {"title": "Fire NOC", "section": "Typical documents required",
             "slug": "fire_noc", "score": 0.37}
          ],
          "mode": "generated" | "extractive" | "no_match",
          "grounded": true
        }
    """
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    approval = (payload.get("approval") or "").strip() or None
    session_id = payload.get("session_id") or "anonymous"
    prediction_id = payload.get("prediction_id")

    if not question:
        return jsonify({"error": "A question is required."}), 400

    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify(
            {"error": f"Question is too long (limit {MAX_QUESTION_LENGTH} characters)."}
        ), 400

    try:
        result = answer_question(question, approval=approval)
    except Exception as exc:  # noqa: BLE001 - never 500 during a live demo
        result = {
            "answer": (
                "The copilot could not process that question. "
                "Please try rephrasing it."
            ),
            "citations": [],
            "mode": "error",
            "grounded": False,
            "detail": str(exc),
        }
        return jsonify(result), 200

    # Save both sides of the exchange — best-effort, a DB hiccup should
    # never break a live answer.
    try:
        with db.get_db_session() as db_session:
            db_session.add(db.CopilotMessage(
                session_id=session_id,
                prediction_id=prediction_id,
                role="user",
                content=question,
            ))
            db_session.add(db.CopilotMessage(
                session_id=session_id,
                prediction_id=prediction_id,
                role="assistant",
                content=result.get("answer", ""),
                mode=result.get("mode"),
            ))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not save copilot message: {exc}")

    return jsonify(result), 200


@rag_bp.route("/api/ask/suggestions", methods=["GET", "POST"])
def suggestions():
    """
    Starter questions for the copilot UI.

    POST {"approvals": ["Fire NOC", "Factory License"]} tailors them to the
    prediction the user just received; GET returns generic ones.
    """
    approvals = None
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        approvals = payload.get("approvals")
        if approvals is not None and not isinstance(approvals, list):
            approvals = None

    return jsonify({"suggestions": suggested_questions(approvals)}), 200


@rag_bp.route("/api/ask/health", methods=["GET"])
def health():
    """Confirms the knowledge base loaded. Useful before a live demo."""
    return jsonify({"status": "ok", **retriever.stats}), 200
