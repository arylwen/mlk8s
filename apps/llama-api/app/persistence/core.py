import os
import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

_logger = logging.getLogger(__name__)

#DB_USER = os.getenv("DB_USER","postgres")
#DB_PASSWORD = os.getenv("DB_PASSWORD","")
#DB_HOST = os.getenv("DB_HOST","0.0.0.0")
#DB_NAME = os.getenv("DB_NAME","llama-api")

#SQLALCHEMY_DATABASE_URI = (
#        f"postgresql+psycopg2://{DB_USER}:"
#        f"{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
#    )

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
    user_id = Column(String(36), nullable=False)
    datetime_captured = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_id = Column(String(36), nullable=False)
    request = Column(JSONB)
    response = Column(JSONB)
    inference_time = Column(Integer)
    entities = Column(JSONB)
    score = Column(Integer)


def create_db_engine_from_config(db_uri) -> Engine:
    """The Engine is the starting point for any SQLAlchemy application.
    """

    engine = create_engine(db_uri)

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