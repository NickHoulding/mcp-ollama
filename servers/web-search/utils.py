from error import (
    SearchError, ScrapingError, EmbeddingError, 
    ValidationError
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from bs4 import BeautifulSoup
from config import settings
from ollama import embed
from typing import List
import numpy as np
import warnings
import logging
import aiohttp
import asyncio
import atexit

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
logger = logging.getLogger(__name__)
_http_session = None

class SearchValidator:
    @staticmethod
    def validate_query(search_query: str) -> str:
        """Validate search query parameters"""
        if not search_query or len(search_query.strip()) == 0:
            raise ValidationError("Search query cannot be empty")
        
        sanitized_query = search_query.strip()
        if len(sanitized_query) > settings.max_len_chars:
            raise ValidationError(f"Search too long (> {settings.max_len_chars} characters)")
        if len(sanitized_query.split()) > settings.max_len_words:
            raise ValidationError(f"Search too long (> {settings.max_len_words}) words")
        
        return sanitized_query
    
async def get_http_session():
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=settings.request_timeout)
        )
    return _http_session

async def cleanup():
    """Clean up http session resources"""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()

async def do_search(search_query: str) -> List[dict]:
    """Perform search using Brave search API"""
    try:
        session = await get_http_session()

        async with session.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": settings.brave_search_api_key,
                "Accept": "application/json",
                "Accept-Encoding": "gzip"
            },
            params={
                "q": search_query,
                "count": settings.num_results,
                "country": "US",
                "search_lang": "en"
            }
        ) as response:
            if response.status == 401:
                raise SearchError("Invalid API key or unauthorized access")
            elif response.status == 429:
                raise SearchError("Rate limit exceeded. Please try again later")
            elif response.status != 200:
                raise SearchError(f"Search API returned status {response.status}")
        
            data = await response.json()
            results = data.get('web', {}).get('results', [])

            return results
    
    except asyncio.TimeoutError:
        raise SearchError("Search request timed out")
    except aiohttp.ClientError as e:
        raise SearchError(f"Search request failed: {str(e)}")

async def chunk_data(scraped_data: List[dict]) -> List[Document]:
    """
    Splits the scraped data from each url into chunks.

    Args:
        scraped_data (List[dict]): The scraped data from each url.
    Returns:
        chunks (List[dict]): The chunks of scraped data
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

        documents = []
        for page in scraped_data:
            if page["content"] and len(page["content"].strip()) > 0:
                doc = Document(
                    page_content=page["content"],
                    metadata={
                        "url": page.get("url", ""),
                        "title": page.get("title", "")
                    }
                )
                documents.append(doc)

        if not documents:
            raise EmbeddingError("No valid documents found for chunking")

        chunks = text_splitter.split_documents(documents)
        return chunks

    except Exception as e:
        raise EmbeddingError(f"Failed to chunk document data: {str(e)}")

async def get_top_relevant_chunks(search_query: str, scraped_data: List[dict]) -> List[dict]:
    """
    Calculates the top relevant chunks of text data from the scraped data.

    Args:
        search_query (str): The user's internet search query.
        scraped_data (List[dict]): The scraped text data from each url.
    Returns:
        top_chunks (List[dict]): The chunks of data most relevant to the search query from the scraped data.
    """
    try:
        chunks = await chunk_data(scraped_data)
        if not chunks:
            raise EmbeddingError("No content chunks generated from scraped data")

        chunk_texts = [chunk.page_content for chunk in chunks]
        
        try:
            all_inputs = [search_query] + chunk_texts
            all_embeddings = embed(
                model=settings.embedding_model,
                input=all_inputs,
            )["embeddings"]

            embedded_query = all_embeddings[0]
            embedded_chunks = all_embeddings[1:]

        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

        embedded_query_np = np.array(embedded_query)
        embedded_chunks_np = np.array(embedded_chunks)

        query_norm = np.linalg.norm(embedded_query_np)
        chunks_norms = np.linalg.norm(embedded_chunks_np, axis=1)

        valid_indices = (chunks_norms > 0) & (query_norm > 0)
        similarities = np.zeros(len(embedded_chunks_np))

        if query_norm > 0:
            similarities[valid_indices] = np.dot(
                embedded_chunks_np[valid_indices],
                embedded_query_np
            ) / (chunks_norms[valid_indices] * query_norm)

        top_indices = np.argsort(similarities)[::-1][:settings.top_chunks_count]
        top_chunks = [{
            "content": chunks[i].page_content,
            "similarity": float(similarities[i]),
            "url": chunks[i].metadata.get("url", ""),
            "title": chunks[i].metadata.get("title", "")
        } for i in top_indices]

        return top_chunks

    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Error processing content chunks: {str(e)}")

async def scrape_urls(urls: List[str]) -> List[dict]:
    """
    Scrapes text data from each web url.

    Args:
        urls (List[str]): The web urls to scrape.
    Returns:
        scraped_data List[dict]: The scraped title, url, and content from each url.
    """
    semaphore = asyncio.Semaphore(3)

    async def scrape_with_semaphore(url):
        async with semaphore:
            try:
                return await scrape_single_url(url)
            except ScrapingError as e:
                logger.warning(f"failed to scrape {url}: {e.message}")
                return {
                    "title": "Content unavailable",
                    "url": url,
                    "content": f"Could not retrieve content: {e.message}"
                }
            
    results = await asyncio.gather(*[scrape_with_semaphore(url) for url in urls])
    successful_scrapes = sum(1 for r in results if "Could not retrieve" not in r.get("content", ""))

    if successful_scrapes == 0:
        raise ScrapingError("Failed to scrape any content from search results")
    
    logger.info(f"Successfully scraped {successful_scrapes}/{len(urls)} URLs")
    return results

async def scrape_single_url(url: str) -> dict:
    """Scrape a single URL with detailed handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        session = await get_http_session()
        async with session.get(url, headers=headers) as response:
            if response.status == 403:
                raise ScrapingError(f"Access forbidden for {url}")
            elif response.status == 404:
                raise ScrapingError(f"Page not found {url}")
            elif response.status != 200:
                raise ScrapingError(f"HTTP {response.status} for {url}")
            
            content = await response.text()
            soup = BeautifulSoup(content, 'html.parser')

            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title found"

            content_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'article'])

            text_content = []
            for element in content_elements:
                text = element.get_text(separator=' ', strip=True)
                if text and len(text) > 10:
                    cleaned_text = ' '.join(text.split())
                    text_content.append(cleaned_text)

            if not text_content:
                raise ScrapingError("No meaningful content found")
            
            full_content = '\n\n'.join(text_content)
            return {
                "title": title_text,
                "url": url,
                "content": full_content
            }
    
    except asyncio.TimeoutError:
        raise ScrapingError(f"Timeout while scraping {url}")
    except aiohttp.ClientError as e:
        raise ScrapingError(f"Request failed for {url}: {str(e)}")
    except Exception as e:
        raise ScrapingError(f"Unexpected error scraping {url}: {str(e)}")

def sync_cleanup():
    """Synchronous cleanup wrapper for atexit"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(cleanup())
        else:
            loop.run_until_complete(cleanup())
    except RuntimeError:
        asyncio.run(cleanup())

atexit.register(sync_cleanup)
