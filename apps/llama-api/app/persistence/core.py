import json
import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import func

_logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

'''
    inference_time - inference duration in seconds
'''
class LlmInferenceRecord(Base):
    """
    Schema for LLM Inference log
    """
    __tablename__ = "llm_inference_log"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True)
    user_id = Column(String(80), nullable=False)
    datetime_captured = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_id = Column(String(36), nullable=False)
    request = Column(JSON)
    response = Column(JSON)
    inference_time = Column(Integer)
    entities = Column(JSON)
    score = Column(Integer)

    def as_dict(self):
       return {c.name: getattr(self, c.name) for c in self.__table__.columns}

def create_db_engine_from_config(db_uri) -> Engine:
    """The Engine is the starting point for any SQLAlchemy application.
    """

    engine = create_engine(db_uri, json_serializer=lambda x: x)

    _logger.info(f"creating DB conn with URI: {db_uri}")
    return engine


def create_db_session(*, engine: Engine) -> scoped_session:
    """The Session establishes all conversations with the database.
     """
    return scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine,),
    )


def init_database(db_uri, adb_session=None) -> None:
    """Connect to the database and attach DB session to the app."""

    if not adb_session:
        engine = create_db_engine_from_config(db_uri)
        db_session = create_db_session(engine=engine)

    Base.metadata.create_all(engine)
    
    return db_session