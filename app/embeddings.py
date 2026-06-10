from sentence_transformers import SentenceTransformer
from app.config import settings

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def get_embedding(text: str) -> list:
    return get_model().encode(text).tolist()


def get_embeddings(texts: list) -> list:
    return get_model().encode(texts).tolist()
