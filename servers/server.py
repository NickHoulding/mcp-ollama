from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test_mcp")

@mcp.tool()
async def add(a: float, b: float) -> float:
    """
    Adds two numbers together.

    Args:
        a (float): The first number.
        b (float): The second number.
    Returns:
        The sum of a and b.
    """
    return a + b

if __name__ == "__main__":
    mcp.run()
