from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from typing import List
import requests
import os

mcp = FastMCP("test_mcp")
load_dotenv()

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
        return [{"error": "Search failed: Search query cannot be empty"}]
    elif len(search_query) > 400:
        return [{"error": "Search failed: Search query too long (> 400 characters)"}]
    
    if len(search_query.split(" ")) > 50:
        return [{"error": "Search query too long (> 50 words)"}]

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
        
        if response.status_code != 200:
            return [{"error": f"Search failed with status code: {response.status_code}"}]
        
        data = response.json()
        results = data.get('web', {}).get('results', [])
        
        if not results:
            return [{"message": "No search results found."}]
        
        search_results = []
        for result in results:
            search_results.append({
                "title": result.get('title', 'No title'),
                "url": result.get('url', 'No URL'),
                "description": result.get('description', 'No description')
            })
        
        return search_results
        
    except Exception as e:
        return [{"error": f"Error performing search: {str(e)}"}]

if __name__ == "__main__":
    mcp.run()
