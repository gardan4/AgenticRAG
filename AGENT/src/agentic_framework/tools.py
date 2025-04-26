# src/agentic_framework/tools.py

import math
import requests

class ToolInterface:
    """
    All tools should inherit from ToolInterface and implement 'run(input_str)'.
    """
    name = "base_tool"
    description = "An abstract tool interface."

    def run(self, input_str: str) -> str:
        raise NotImplementedError


class CalculatorTool(ToolInterface):
    name = "calculator"
    description = "Evaluates a simple math expression using Python's eval or math."

    def run(self, expression: str) -> str:
        try:
            # WARNING: Direct eval can be unsafe in production. This is a demo only.
            result = eval(expression, {"__builtins__": None}, {"math": math})
            return str(result)
        except Exception as e:
            return f"Calculator Error: {str(e)}"


class SearchTool(ToolInterface):
    name = "search"
    description = "Performs a web search and returns a short text result."

    def run(self, query: str) -> str:
        try:
            # Simple demonstration using a public search API or placeholder.
            # For example: query a free search API or a local knowledge base
            # This is just a placeholder to illustrate usage.
            # Example with a fake endpoint:
            response = requests.get(
                "https://api.publicsearch.example.com/search",
                params={"q": query, "limit": 1},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                # Return the snippet from the first search result
                if "results" in data and len(data["results"]) > 0:
                    snippet = data["results"][0].get("snippet", "No snippet found.")
                    return snippet
                else:
                    return "No search results."
            else:
                return f"Search Error: HTTP {response.status_code}"
        except Exception as e:
            return f"Search Error: {str(e)}"
