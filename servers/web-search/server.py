from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from typing import List
import numpy as np
import warnings
import requests
import ollama
import heapq
import os

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
mcp = FastMCP("test_mcp")
load_dotenv()

async def chunk_data(scraped_data: List[dict]) -> List[Document]:
    """
    Splits the scraped data from each url into chunks.

    Args:
        scraped_data (List[dict]): The scraped data from each url.
    Returns:
        chunks (List[dict]): The chunks of scraped data
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False
    )
    documents = [
        Document(page_content=page["content"]) 
        for page in scraped_data
    ]
    return text_splitter.split_documents(documents)

async def cosine_similarity(emb_A, emb_B) -> float:
    """
    Calculates the cosine similarity between two embeddings.

    Args:
        emb_A: Embedding A
        emb_B: Embedding B
    Returns:
        similarity (float): The cosine similarity between emb_A and emb_b.
    """
    emb_A = np.array(emb_A)
    emb_B = np.array(emb_B)

    norm_A = np.linalg.norm(emb_A)
    norm_B = np.linalg.norm(emb_B)

    if norm_A == 0 or norm_B == 0:
        return 0.0

    similarity = np.dot(emb_A, emb_B) / (norm_A * norm_B)
    return float(similarity)

async def get_top_relevant_chunks(search_query: str, scraped_data: List[dict]) -> List[dict]:
    """
    Calculates the top relevant chunks of text data from the scraped data.

    Args:
        search_query (str): The internet search query.
        scraped_data (List[dict]): The scraped text data from each url.
    Returns:
        top_chunks (List[dict]): The top relevant chunks of data from the scraped data.
    """
    chunks = await chunk_data(scraped_data)

    embedded_query = ollama.embed(
        model='embeddinggemma',
        input=search_query,
    )["embeddings"]
    embeddings = ollama.embed(
        model='embeddinggemma',
        input=[chunk.page_content for chunk in chunks],
    )["embeddings"]

    priority_queue = []
    heapq.heapify(priority_queue)

    for i in range(len(embeddings)):
        chunks[i].metadata["embeddings"] = embeddings[i]
        similarity = await cosine_similarity(
            embedded_query, 
            chunks[i].metadata["embeddings"]
        )
        heapq.heappush(priority_queue, (-similarity, chunks[i]))

    top_chunk_tuples = heapq.nsmallest(6, priority_queue)
    top_chunks = [{
        "content": chunk.page_content,
        "similarity": -similarity
    } for similarity, chunk in top_chunk_tuples]

    return top_chunks

async def scrape_urls(urls: List[str]) -> List[dict]:
    """
    Scrapes text data from each web url.

    Args:
        urls (List[str]): The web urls to scrape.
    Returns:
        scraped_data List[dict]: A list of dictionaries containing the scraped text data from each url.
    """
    scraped_data = []
    
    for url in urls:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(
                url=url, 
                headers=headers, 
                timeout=10
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "No title found"
            content_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
            
            text_content = []
            for element in content_elements:
                text = element.get_text(separator=' ', strip=True)

                if text and len(text) > 10:
                    cleaned_text = ' '.join(text.split())
                    text_content.append(cleaned_text)
            
            full_content = '\n\n'.join(text_content)
            scraped_data.append({
                "title": title_text,
                "url": url,
                "content": full_content
            })
            
        except Exception as e:
            scraped_data.append({
                "title": "Error", 
                "url": url,
                "content": f"Parsing error: {str(e)}"
            })
    
    return scraped_data

@mcp.tool()
async def web_search(search_query: str) -> List[dict]:
    """
    Performs an internet search with the specified search query.

    Args:
        search_query (str): The search query to search the web.
    Returns:
        List[dict]: A list of dictionaries containing search results with title, url, and description.
    """
    if len(search_query) == 0:
        return [{"content": "Search failed: Search query cannot be empty"}]
    elif len(search_query) > 400:
        return [{"content": "Search failed: Search query too long (> 400 characters)"}]
    
    if len(search_query.split(" ")) > 50:
        return [{"content": "Search query too long (> 50 words)"}]

    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "X-Subscription-Token": str(os.getenv("BRAVE_SEARCH_API_KEY")),
                "Accept": "application/json",
                "Accept-Encoding": "gzip"
            },
            params={
                "q": str(search_query),
                "count": 3,
                "country": "US",
                "search_lang": "en"
            }
        )
        response.raise_for_status()
        
        if response.status_code != 200:
            return [{"content": f"Search failed with status code: {response.status_code}"}]
        
        data = response.json()
        results = data.get('web', {}).get('results', [])
        
        if not results:
            return [{"content": "No search results found."}]

        scraped_data = await scrape_urls([result["url"] for result in results])
        top_relevant_chunks = await get_top_relevant_chunks(search_query, scraped_data)
        return top_relevant_chunks
        
    except Exception as e:
        return [{"content": f"Error performing search: {str(e)}"}]

if __name__ == "__main__":
    mcp.run()
