# ============================================================
# EXACT CHANGES TO MAKE IN api.py
# ============================================================
# This is not a file to run — it's a reference showing the 3 edits.
# Everything else in api.py (routes, extraction, DB, RAG) stays untouched.


# ---- CHANGE 1: imports at the top of the file ----
# REMOVE this line:
#     from tensorflow.keras.models import load_model
# ADD this line instead:
import onnxruntime as ort


# ---- CHANGE 2: model loading section ----
# REMOVE this line:
#     model = load_model('udyamflow_ann_model.keras')
# ADD these two lines instead:
onnx_session = ort.InferenceSession('udyamflow_ann_model.onnx')
onnx_input_name = onnx_session.get_inputs()[0].name


# ---- CHANGE 3: inside the predict() route ----
# REMOVE these lines:
#     X_processed = preprocessor.transform(input_df)
#     if hasattr(X_processed, 'toarray'):
#         X_processed = X_processed.toarray()
#
#     probabilities = model.predict(X_processed, verbose=0)[0]
#
# ADD these lines instead:
X_processed = preprocessor.transform(input_df)
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()
X_processed = X_processed.astype(np.float32)  # ONNX Runtime requires float32

probabilities = onnx_session.run(None, {onnx_input_name: X_processed})[0][0]

# Everything after this line (prob_map, results list, journey building,
# DB save, jsonify) stays EXACTLY the same — probabilities is still a plain
# array of 12 floats, same shape as before.
