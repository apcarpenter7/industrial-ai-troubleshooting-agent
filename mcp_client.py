# https://github.com/kirillsaidov/ollama-mcp-example/blob/main/mcp_client.py

# system
import os, sys
import json
import asyncio
import traceback

# libs
import ollama
from fastmcp import Client
from fastmcp.client.transports import SSETransport


async def main(model: str):
    print(f'Running with model={model}. Type "/quit" to exit.')
    
    # Connect to server via transport=sse.
    # The server exposes an HTTP endpoint that our client can talk to.
    transport = SSETransport(url="http://127.0.0.1:8050/sse")
    async with Client(transport) as client:
        history = [{
            'role': 'system',
            'content': 'You are a helpfull assistant. Use tools if possible to get the latest data.'
        }]
        mcp_tools = await client.list_tools()
        ollama_tools = []
        
        # List available tools provided by the MCP server.
        print('Available tools:')
        for tool in mcp_tools:
            print(f'\t- {tool.name}')
            tool_desc = (tool.description or '').split('\n')[0]
            print(f'\t  {tool_desc}')

        # Convert MCP tools to Ollama function calling format.
        # If you want to create MCP client from scratch for another LLM provider,
        # you will have to convert tools to the format it accepts.
        for tool in mcp_tools:
            ollama_tools.append({
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.inputSchema
                }
            })

        # Chat
        while True:
            try:
                # Get user input
                user_input = input('>> ').strip()
                if user_input.startswith('/'):
                    break
                elif not user_input:
                    continue
                
                # Append history
                history.append({'role': 'user', 'content': user_input})

                # Generate response
                response = ollama.chat(model, messages=history, tools=ollama_tools)

                # Check if we need to make any functions calls to MCP server.
                if response.message.tool_calls:
                    # Record the assistant's tool-call turn *before* the tool
                    # results, so the history keeps the required
                    # assistant(tool_calls) -> tool(result) ordering. Without
                    # this, the model gets a malformed transcript on the next
                    # call and can silently return empty content.
                    history.append({
                        'role': 'assistant',
                        'content': response.message.content,
                        'tool_calls': response.message.tool_calls,
                    })

                    for tool in response.message.tool_calls:
                        print(f'Calling {tool.function.name}')
                        print(f'\tWith params: {tool.function.arguments}')

                        # call function via MCP server
                        result = await client.call_tool(tool.function.name, tool.function.arguments)

                        # Extract the actual text/JSON payload from the
                        # fastmcp CallToolResult instead of stringifying the
                        # whole object (which produces an unparseable repr
                        # blob for the model).
                        result_text = '\n'.join(
                            getattr(block, 'text', str(block)) for block in result.content
                        )

                        # append the results to history as role:tool
                        history.append({
                            'role': 'tool',
                            'content': result_text,
                            'name': tool.function.name,
                        })

                    # run ollama again to get the final coherent response
                    response = ollama.chat(model, messages=history, tools=ollama_tools)

                # append ollama response
                history.append({
                    'role': 'assistant',
                    'content': response.message.content,
                })
                print(response.message.content)
            except:
                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main(
        model='gemma4'
    ))