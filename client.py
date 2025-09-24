from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from mcp.types import TextContent
import asyncio
import ollama
import json
import sys

class MCPClient():
    def __init__(self) -> None:
        self.session: ClientSession
        self.exit_stack = AsyncExitStack()
        self.model = "llama3.2:3b"

    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to a local MCP server

        Args:
            server_script_path: Path to the server (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')

        if not (is_python or is_js):
            raise ValueError("Server script must be .py or .js file")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to the server with tools: ", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Ollama model and available tools"""
        if not query.strip():
            return "Error: Query must not be empty."

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        response = ollama.chat(
            model=self.model,
            messages=messages,
            stream=False,
            tools=available_tools
        )

        final_text = []
        assistant_message = response['message']
        
        if assistant_message.get('content'):
            final_text.append(assistant_message['content'])
        
        if 'tool_calls' in assistant_message:
            messages.append({
                "role": "assistant", 
                "content": assistant_message.get('content', ''),
                "tool_calls": assistant_message['tool_calls']
            })
            
            tool_results = []
            for tool_call in assistant_message['tool_calls']:
                tool_name = tool_call['function']['name']
                tool_args = tool_call['function']['arguments']
                
                result = await self.session.call_tool(tool_name, tool_args)
                result_content = ""
                
                if result.content:
                    for item in result.content:
                        if isinstance(item, TextContent):
                            try:
                                text = json.loads(item.text)
                                result_content += text["content"]
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}")
                                result_content += item.text
                    result_content = result_content.strip()

                tool_results.append({
                    "role": "tool",
                    "content": result_content,
                    "tool_call_id": tool_call.get('id', f"call_{tool_name}")
                })
            
            messages.extend(tool_results)
            messages.append({
                "role": "user",
                "content": f"Please provide a clear, natural language answer based on these tool results: {[x['content'] for x in tool_results]}"
            })
            
            follow_up_response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=False
            )
            
            if follow_up_response['message'].get('content'):
                final_text.append(follow_up_response['message']['content'])
            elif tool_results:
                    result_summary = f"The result is: {tool_results[0]['content']}"
                    final_text.append(result_summary)

        return "\n".join(final_text)
    
    async def chat_loop(self) -> None:
        """Run an iterative chat loop"""
        print("\nMCP Client started!")
        print("Type your queries or 'quit' to exit")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
            
                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main() -> None:
    """Main function to run the Ollama MCP server"""
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        print("Example: python client.py ./server.py")
        return

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
