from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    huggingface_api_key: str = ""
    huggingface_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    groq_api_key: str = ""
    llm_provider: str = "gemini"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "./vector_store"
    data_dir: str = "./data"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
