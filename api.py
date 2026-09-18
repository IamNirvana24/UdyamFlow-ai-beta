from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import re
import onnxruntime as ort

from rag_routes import rag_bp
from explainer_routes import explainer_bp
import db

app = Flask(__name__)
CORS(app)
app.register_blueprint(rag_bp)
app.register_blueprint(explainer_bp)
db.init_db()

# ---------------------------------------------------------
# Load trained model + preprocessor (produced by the notebook)
#
# Uses ONNX Runtime instead of full TensorFlow/Keras — the Keras runtime
# alone costs 300-500MB RAM even for a small model, which crashes low-RAM
# hosting tiers. ONNX Runtime costs ~50-80MB for the same model. The .onnx
# file is a numerically-verified conversion of the original .keras model
# (max output difference: 6e-8, floating-point noise only — not a retrain).
# ---------------------------------------------------------
preprocessor = joblib.load('preprocessor.pkl')
target_cols = joblib.load('target_cols.pkl')
category_options = joblib.load('category_options.pkl')

onnx_session = ort.InferenceSession('udyamflow_ann_model.onnx')
ONNX_INPUT_NAME = onnx_session.get_inputs()[0].name


def run_model(X_processed):
    """Runs inference through ONNX Runtime, matching the Keras model's
    .predict(X, verbose=0)[0] return shape (1D array of per-label
    probabilities)."""
    X_processed = np.asarray(X_processed, dtype=np.float32)
    outputs = onnx_session.run(None, {ONNX_INPUT_NAME: X_processed})
    return outputs[0][0]

LICENSE_LABELS = {
    'trade_license': 'Trade License',
    'gst_registration': 'GST Registration',
    'factory_license': 'Factory License',
    'labour_registration': 'Labour / Shops & Establishment Registration',
    'fire_noc': 'Fire NOC',
    'pollution_noc': 'Pollution Control Board NOC',
    'environmental_clearance': 'Environmental Clearance',
    'fssai_license': 'FSSAI License',
    'boiler_license': 'Boiler License',
    'explosives_license': 'Explosives/Hazardous Storage License (PESO)',
    'electricity_noc': 'Electricity Load Sanction / NOC',
    'building_plan_approval': 'Building Plan Approval',
}

# Fixed real-world dependency ordering — this is the sequence a business
# would realistically move through, used to order the "approval journey"
APPROVAL_SEQUENCE = [
    'building_plan_approval',
    'trade_license',
    'gst_registration',
    'labour_registration',
    'factory_license',
    'fire_noc',
    'pollution_noc',
    'environmental_clearance',
    'electricity_noc',
    'boiler_license',
    'explosives_license',
    'fssai_license',
]

# Sector keyword aliases for free-text NLP extraction
SECTOR_KEYWORDS = {
    'Food Processing': ['food processing', 'food factory', 'food unit', 'food product'],
    'Textile': ['textile', 'garment', 'fabric', 'apparel'],
    'Textile Dyeing': ['dyeing', 'dye unit', 'dye factory'],
    'Chemical': ['chemical'],
    'Pharmaceutical': ['pharma', 'pharmaceutical', 'medicine manufacturing', 'drug manufacturing'],
    'Engineering/Metal Fabrication': ['engineering', 'metal fabrication', 'metal works', 'fabrication'],
    'Agro-based': ['agro', 'agriculture based', 'agri-based'],
    'Plastic Products': ['plastic'],
    'Electronics': ['electronics', 'electronic components'],
    'IT/Software Services': ['it services', 'software', 'it company', 'tech company'],
    'Auto Components': ['auto component', 'automobile parts', 'auto parts'],
    'Paper & Pulp': ['paper', 'pulp'],
    'Cement': ['cement'],
    'Beverages': ['beverage', 'drink manufacturing', 'juice', 'soft drink'],
    'Leather Products': ['leather'],
    'Furniture/Wood': ['furniture', 'wood work', 'woodwork', 'carpentry'],
    'Rice Milling': ['rice mill', 'rice milling', 'paddy'],
    'Steel Rolling': ['steel', 'rolling mill'],
}


def extract_business_profile(text):
    """Rule-based NLP extraction: pulls structured fields out of a free-text
    business description. Not a trained NLP model — a deterministic
    keyword + regex extractor, chosen for reliability during a live demo."""
    text_lower = text.lower()
    extracted = {}
    confidence_notes = []

    # --- Investment (₹X crore / ₹X lakh) ---
    crore_match = re.search(r'(\d+(?:\.\d+)?)\s*crore', text_lower)
    lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*lakh', text_lower)
    if crore_match:
        extracted['investment_lakhs'] = float(crore_match.group(1)) * 100
        confidence_notes.append(f"Investment: ₹{crore_match.group(1)} crore detected")
    elif lakh_match:
        extracted['investment_lakhs'] = float(lakh_match.group(1))
        confidence_notes.append(f"Investment: ₹{lakh_match.group(1)} lakh detected")

    # --- Employees ---
    emp_match = re.search(r'(\d+)\s*(?:employees|workers|people|staff)', text_lower)
    if emp_match:
        extracted['employee_count'] = int(emp_match.group(1))
        confidence_notes.append(f"Employees: {emp_match.group(1)} detected")

    # --- District (match against known Punjab districts) ---
    for district in category_options['districts']:
        if district.lower().split(' (')[0] in text_lower:
            extracted['district'] = district
            confidence_notes.append(f"District: {district} detected")
            break

    # --- Sector (keyword match) ---
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            extracted['sector'] = sector
            confidence_notes.append(f"Sector: {sector} detected")
            break

    # --- Hazardous / boiler / service sector flags ---
    if any(kw in text_lower for kw in ['chemical', 'hazardous', 'toxic', 'explosive']):
        extracted['uses_hazardous_material'] = 1
    if any(kw in text_lower for kw in ['boiler', 'steam']):
        extracted['uses_boiler'] = 1
    if any(kw in text_lower for kw in ['it services', 'software', 'consulting', 'service sector']):
        extracted['is_service_sector'] = 1

    return extracted, confidence_notes


@app.route('/api/extract', methods=['POST'])
def extract():
    """NLP intake endpoint — parses a free-text business description into
    structured fields used to auto-fill the form."""
    data = request.get_json()
    text = data.get('description', '')

    extracted, notes = extract_business_profile(text)

    return jsonify({
        'extracted': extracted,
        'notes': notes
    })


@app.route('/api/options', methods=['GET'])
def get_options():
    """Returns dropdown options for the frontend form."""
    return jsonify(category_options)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    session_id = data.get('session_id', 'anonymous')

    input_df = pd.DataFrame([{
        'district': data['district'],
        'sector': data['sector'],
        'investment_lakhs': float(data['investment_lakhs']),
        'employee_count': int(data['employee_count']),
        'power_requirement_kw': float(data['power_requirement_kw']),
        'land_area_sqm': float(data['land_area_sqm']),
        'water_usage_kld': float(data['water_usage_kld']),
        'uses_hazardous_material': int(data.get('uses_hazardous_material', 0)),
        'uses_boiler': int(data.get('uses_boiler', 0)),
        'is_service_sector': int(data.get('is_service_sector', 0)),
    }])

    X_processed = preprocessor.transform(input_df)
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()

    probabilities = run_model(X_processed)

    prob_map = {label: float(prob) for label, prob in zip(target_cols, probabilities)}

    results = []
    for label in target_cols:
        prob = prob_map[label]
        results.append({
            'key': label,
            'name': LICENSE_LABELS[label],
            'required': bool(prob > 0.5),
            'confidence': round(prob * 100, 1)
        })
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Build the dependency-ordered journey — only required approvals,
    # arranged in the realistic sequence they'd need to be pursued
    journey = []
    for label in APPROVAL_SEQUENCE:
        if prob_map[label] > 0.5:
            journey.append({
                'key': label,
                'name': LICENSE_LABELS[label],
                'confidence': round(prob_map[label] * 100, 1)
            })

    required_count = sum(1 for r in results if r['required'])

    # Persist this run so it shows up in the person's history later. This is
    # best-effort — a DB hiccup should never break the live prediction.
    prediction_id = None
    try:
        with db.get_db_session() as db_session:
            row = db.Prediction(
                session_id=session_id,
                district=data['district'],
                sector=data['sector'],
                investment_lakhs=float(data['investment_lakhs']),
                employee_count=int(data['employee_count']),
                power_requirement_kw=float(data['power_requirement_kw']),
                land_area_sqm=float(data['land_area_sqm']),
                water_usage_kld=float(data['water_usage_kld']),
                uses_hazardous_material=bool(data.get('uses_hazardous_material', 0)),
                uses_boiler=bool(data.get('uses_boiler', 0)),
                is_service_sector=bool(data.get('is_service_sector', 0)),
                required_count=required_count,
                results_json=results,
                journey_json=journey,
            )
            db_session.add(row)
            db_session.flush()  # populates row.id before commit
            prediction_id = row.id
    except Exception as exc:  # noqa: BLE001 - never break a live demo over a DB issue
        print(f"[warn] could not save prediction to database: {exc}")

    return jsonify({
        'prediction_id': prediction_id,
        'results': results,
        'journey': journey,
        'required_count': required_count
    })


@app.route('/api/history', methods=['GET'])
def history():
    """Past predictions for this browser's session_id, newest first.
    Summary only — full detail comes from /api/history/<id>."""
    session_id = request.args.get('session_id', 'anonymous')

    with db.get_db_session() as db_session:
        rows = (
            db_session.query(db.Prediction)
            .filter(db.Prediction.session_id == session_id)
            .order_by(db.Prediction.created_at.desc())
            .limit(50)
            .all()
        )
        return jsonify({
            'predictions': [
                {
                    'id': r.id,
                    'district': r.district,
                    'sector': r.sector,
                    'required_count': r.required_count,
                    'created_at': r.created_at.isoformat(),
                }
                for r in rows
            ]
        })


@app.route('/api/history/<int:prediction_id>', methods=['GET'])
def history_detail(prediction_id):
    """Full detail for one past prediction — same shape /api/predict returns,
    so the frontend can reuse its existing render function."""
    with db.get_db_session() as db_session:
        row = db_session.query(db.Prediction).filter(db.Prediction.id == prediction_id).first()
        if not row:
            return jsonify({'error': 'Prediction not found'}), 404

        return jsonify({
            'prediction_id': row.id,
            'district': row.district,
            'sector': row.sector,
            'investment_lakhs': row.investment_lakhs,
            'employee_count': row.employee_count,
            'power_requirement_kw': row.power_requirement_kw,
            'land_area_sqm': row.land_area_sqm,
            'water_usage_kld': row.water_usage_kld,
            'uses_hazardous_material': row.uses_hazardous_material,
            'uses_boiler': row.uses_boiler,
            'is_service_sector': row.is_service_sector,
            'results': row.results_json,
            'journey': row.journey_json,
            'required_count': row.required_count,
            'created_at': row.created_at.isoformat(),
        })


if __name__ == '__main__':
    print("UdyamFlow AI backend running on http://localhost:5000")
    app.run(debug=True, port=5000)
