
import enum
import logging
import typing as t

from sqlalchemy.orm.session import Session

from ml_api_milvus.api.persistence.models import (
    LiveModelPredictions,
)

_logger = logging.getLogger(__name__)


class ModelType(enum.Enum):
    ClipModel = "clip"
    ModelFeedback = "clip_feedback"

class PredictionPersistence:
    def __init__(self, *, db_session: Session, user_id: str = None) -> None:
        self.db_session = db_session
        if not user_id:
            # in reality, here we would use something like a UUID for anonymous users
            # and if we had user logins, we would record the user ID.
            self.user_id = "007"

    def save_predictions(
        self,
        *,
        inputs: t.List,
        model_version: str,
        predictions: t.List,
        db_model: ModelType,
    ) -> None:
        if db_model == db_model.ClipModel:
            prediction_data = LiveModelPredictions(
                user_id=self.user_id,
                model_version=model_version,
                inputs=inputs,
                outputs=predictions,
            )

        self.db_session.add(prediction_data)
        self.db_session.commit()
        _logger.debug(f"saved data for model: {db_model}")

    
    def save_feedback(
        self,
        *,
        inputs: t.List,
        model_version: str,
        #predictions: t.List,
        db_model: ModelType,
    ) -> None:
        if db_model == db_model.ModelFeedback:
            prediction_data = LiveModelPredictions(
                user_id=self.user_id,
                model_version=model_version,
                inputs=inputs,
                #outputs=predictions,
            )


        self.db_session.add(prediction_data)
        self.db_session.commit()
        _logger.debug(f"saved data for model: {db_model}")