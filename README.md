# UdyamFlow AI — Tekathon Prototype

## Files
- `UdyamFlow_ANN_Training.ipynb` — Jupyter notebook, trains the ANN (already executed, outputs included)
- `punjab_industrial_approval_dataset_large.csv` — synthetic Punjab dataset (8,000 rows)
- `udyamflow_ann_model.keras`, `preprocessor.pkl`, `target_cols.pkl`, `category_options.pkl` — already-trained model + supporting files
- `api.py` — Flask backend that loads the model and serves predictions
- `index.html` — frontend (open this in browser, it calls the API)

## How to run (model already trained — skip straight here)

### 1. Install backend dependencies
```
pip install flask flask-cors tensorflow scikit-learn pandas numpy joblib
```

### 2. Start the backend (keep this terminal running)
```
python api.py
```
This starts the API at http://localhost:5000

### 3. Open the frontend
Just double-click `index.html` to open it in your browser (or right-click → Open with → Chrome/Edge).
Fill in the form, click "Predict required approvals" — it calls the backend and shows results.

## If you want to retrain the model yourself
Open and run `UdyamFlow_ANN_Training.ipynb` top to bottom in Jupyter — it regenerates
`udyamflow_ann_model.keras` and the `.pkl` files. Not required, already done.

## Recording your demo video
1. Run `python api.py` (keep terminal open)
2. Open `index.html` in your browser
3. Fill in a realistic example (Ludhiana, Food Processing, ₹500 lakhs, 25 employees, 
   40kW power, 400 sqm land, 8 KLD water)
4. Click "Predict required approvals" — show the resulting approval list
5. Screen record this, upload to YouTube (unlisted is fine), link on last PPT slide
