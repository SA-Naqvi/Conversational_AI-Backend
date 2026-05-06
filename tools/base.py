"""
Base class for all callable tools.
Each tool defines its name, description, JSON schema, and async execute() method.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Abstract base for all tools callable by the LLM."""

    name: str = ""
    description: str = ""
    parameters: Dict = {}   # JSON Schema for the arguments

    def to_openai_schema(self) -> Dict:
        """Return the OpenAI function-calling schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with parsed arguments. Return a serializable result."""
        ...
