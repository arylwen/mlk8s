
import logging
import typing as t

from sqlalchemy.orm.session import Session

from app.persistence.core import LlmInferenceRecord

_logger = logging.getLogger(__name__)

class LlmInferencePersistence:
    def __init__(self, *, db_session: Session, user_id: str = None) -> None:
        self.db_session = db_session
        if not user_id:
            # in reality, here we would use something like a UUID for anonymous users
            # and if we had user logins, we would record the user ID.
            self.user_id = "007"
        else:
            self.user_id = user_id

    def save_predictions(
        self,
        *,
        model_id: str,
        request: t.List,
        response: t.List,
        inference_time: int 
    ) -> None:
        prediction_data = LlmInferenceRecord(
                user_id=self.user_id,
                model_id=model_id,
                request=request,
                response=response,
                inference_time=inference_time
            )

        self.db_session.add(prediction_data)
        self.db_session.commit()
        _logger.debug(f"saved data for model: {model_id}")
