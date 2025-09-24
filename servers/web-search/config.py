from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Optional
from os import getenv

load_dotenv()

class Settings(BaseSettings):
    brave_search_api_key: str
    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_model: str = "embeddinggemma"
    num_results: int = 3
    max_len_chars: int = 400
    max_len_words: int = 50
    top_chunks_count: int = 6
    request_timeout: int = 10

    class Config:
        env_file = ".env"

settings = Settings(brave_search_api_key=getenv("BRAVE_SEARCH_API_KEY", ""))
