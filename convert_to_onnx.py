"""
Run this ONCE on your local machine (where TensorFlow is already installed)
to convert the trained Keras model to ONNX format.

v2: goes through an intermediate SavedModel export, because tf2onnx's
direct from_keras() path has a known compatibility bug with Keras 3's
internal tensor naming (KeyError: 'keras_tensor_N').

Usage:
    pip install tf2onnx
    python convert_to_onnx.py

Produces: udyamflow_ann_model.onnx
"""
import subprocess
import sys

import tensorflow as tf

print("Loading Keras model...")
model = tf.keras.models.load_model('udyamflow_ann_model.keras')
print(f"Model input shape: {model.inputs[0].shape}")

saved_model_dir = 'udyamflow_saved_model_tmp'
print(f"Exporting to intermediate SavedModel format ({saved_model_dir})...")
model.export(saved_model_dir)

print("Converting SavedModel to ONNX via tf2onnx...")
result = subprocess.run(
    [
        sys.executable, '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_dir,
        '--output', 'udyamflow_ann_model.onnx',
        '--opset', '13',
    ],
    capture_output=True, text=True,
)

print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    print("\nConversion FAILED - see error above.")
else:
    print("\nDone. Wrote udyamflow_ann_model.onnx")
    print("You can delete the udyamflow_saved_model_tmp folder now, it was only needed for this conversion.")
