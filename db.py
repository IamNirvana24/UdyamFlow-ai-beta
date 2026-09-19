"""
Persistence layer for UdyamFlow AI.

No login/auth system — each browser gets an anonymous session_id (a UUID
generated client-side and stored in localStorage), which lets someone see
their own past predictions and copilot conversations without needing an
account. Good enough for a hackathon demo; a real product would swap this
for proper auth without changing the schema much.

DATABASE_URL controls where this points:
  - Not set                          -> local SQLite file (udyamflow.db),
                                         zero setup, works out of the box.
  - postgresql://user:pass@host/db   -> real Postgres (e.g. Render's free
                                         Postgres add-on in production).

Render (and some other hosts) hand out DATABASE_URL as "postgres://", but
SQLAlchemy 1.4+ requires "postgresql://" — handled below so you don't hit a
confusing error on deploy day.
"""

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, JSON, String,
    Text, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///udyamflow.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Prediction(Base):
    """One saved run of the ANN — the business profile submitted plus the
    results it produced. This is what powers the 'past predictions' history."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True, nullable=False)

    district = Column(String(100))
    sector = Column(String(100))
    investment_lakhs = Column(Float)
    employee_count = Column(Integer)
    power_requirement_kw = Column(Float)
    land_area_sqm = Column(Float)
    water_usage_kld = Column(Float)
    uses_hazardous_material = Column(Boolean, default=False)
    uses_boiler = Column(Boolean, default=False)
    is_service_sector = Column(Boolean, default=False)

    required_count = Column(Integer)
    results_json = Column(JSON)   # full per-approval list (name, required, confidence)
    journey_json = Column(JSON)   # dependency-ordered required approvals

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    messages = relationship(
        "CopilotMessage", back_populates="prediction", cascade="all, delete-orphan"
    )


class CopilotMessage(Base):
    """One turn of a compliance-copilot conversation. Linked to a Prediction
    when the person asked from within a results view; nullable otherwise,
    since the copilot is usable before any prediction has been made."""
    __tablename__ = "copilot_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True, nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    role = Column(String(16))       # 'user' or 'assistant'
    content = Column(Text)
    mode = Column(String(32), nullable=True)  # generated / extractive / no_match / scope_notice

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    prediction = relationship("Prediction", back_populates="messages")


class Feedback(Base):
    """A person flagging one specific approval within a prediction as wrong
    — the Accountability feature: never let the model's word be final
    without a way to push back on it. Reviewed manually for now; at scale
    this is what you'd feed back into retraining or a correction list."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(64), index=True, nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    approval_key = Column(String(64), nullable=True)   # which approval was flagged, if specific
    approval_name = Column(String(120), nullable=True)
    issue_type = Column(String(32), nullable=False)     # 'wrongly_required' / 'wrongly_missing' / 'other'
    comment = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    prediction = relationship("Prediction")


def init_db():
    """Creates tables if they don't exist yet. Safe to call every startup."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """One DB session per request — caller is responsible for closing it
    (use the get_db_session() context manager below in route handlers)."""
    return SessionLocal()


from contextlib import contextmanager  # noqa: E402


@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
