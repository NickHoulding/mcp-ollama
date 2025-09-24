from typing import Optional, Any
from config import settings
import logging

logger = logging.getLogger(__name__)

class MCPSearchError(Exception):
    """Base exception for MCP search operations"""
    def __init__(self, message: str, error_code: Optional[int] = None, details: Optional[Any] = None) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(self.message)

class SearchError(MCPSearchError):
    """Custom exception for search-related errors"""
    pass

class ScrapingError(MCPSearchError):
    """Custom exception for web scraping errors"""
    pass

class EmbeddingError(MCPSearchError):
    """Custom exception for embedding-related errors"""
    pass

class ValidationError(MCPSearchError):
    """Custom exception for input validation errors"""
    pass
