"""
Run this in your project folder (same place as api.py and db.py) to confirm
the database is actually saving data.

Usage:
    python check_db.py
"""
import db

with db.get_db_session() as s:
    print("Predictions:", s.query(db.Prediction).count())
    print("Copilot messages:", s.query(db.CopilotMessage).count())
