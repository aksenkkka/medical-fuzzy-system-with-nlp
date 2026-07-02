from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .db import Base


class Clinic(Base):
    __tablename__ = "clinics"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    reviews = relationship("Review", back_populates="clinic")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    clinic_id = Column(Integer, ForeignKey("clinics.id"))
    original_text = Column(String)
    user_rating = Column(Float, nullable=True)

    # NLP extracted features (0-10)
    wait_time_score = Column(Float)
    doctor_comp_score = Column(Float)
    politeness_score = Column(Float)
    cleanliness_score = Column(Float)
    emotion_score = Column(Float)
    consultation_score = Column(Float)

    # Final fuzzy output (0-10)
    final_fuzzy_rating = Column(Float)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    clinic = relationship("Clinic", back_populates="reviews")
