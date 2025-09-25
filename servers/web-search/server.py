from error import (
    SearchError, ScrapingError, EmbeddingError, 
    ValidationError
)
from utils import (
    SearchValidator, logger, cleanup, do_search, 
    scrape_urls, get_top_relevant_chunks
)
from mcp.server.fastmcp import FastMCP
from typing import List
import asyncio

mcp = FastMCP("test_mcp")

@mcp.tool()
async def web_search(search_query: str) -> List[dict]:
    """
    Performs an internet search with the specified search query.

    Args:
        search_query (str): The search query to search the web.
    Returns:
        top_chunks List[dict]: A list of dictionaries containing search results with title, url, and description.
    """
    try:
        sanitized_query = SearchValidator.validate_query(search_query)
        search_results = await do_search(sanitized_query)

        if not search_results:
            logger.warning(f"No search results found for your query: {sanitized_query}")
            return [{"content": "No search results found for your query."}]
        
        urls = [result["url"] for result in search_results]
        logger.info(f"Found {len(urls)} URLs to scrape for query: {sanitized_query}")

        scraped_data = await scrape_urls(urls)
        top_chunks = await get_top_relevant_chunks(sanitized_query, scraped_data)

        return top_chunks
    
    except ValidationError as e:
        logger.error(f"Validation error: {e.message}")
        return [{"content": f"Invalid search query: {e.message}"}]
    
    except SearchError as e:
        logger.error(f"Search error: {e.message}")
        return [{"content": f"Search failed: {e.message}"}]
    
    except ScrapingError as e:
        logger.error(f"Scraping error: {e.message}")
        return [{"content": f"Failed to retrieve content: {e.message}"}]
    
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e.message}")
        return [{"content": f"Failed to process search results: {e.message}"}]
    
    except Exception as e:
        logger.exception(f"Unexpected error in web_search: {str(e)}")
        return [{"content": f"An unexpected error occurred: {str(e)}"}]

if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        asyncio.run(cleanup())
