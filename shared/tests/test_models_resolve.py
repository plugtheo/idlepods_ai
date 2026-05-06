from shared.contracts.models import BackendEntry


_URL = "http://localhost:8000"


def test_resolve_training_model_id_uses_override():
    backend = BackendEntry(served_url=_URL, model_id="base/Model", training_model_id="train/Model")
    assert backend.resolve_training_model_id() == "train/Model"


def test_resolve_training_model_id_falls_back_to_model_id():
    backend = BackendEntry(served_url=_URL, model_id="base/Model", training_model_id=None)
    assert backend.resolve_training_model_id() == "base/Model"
