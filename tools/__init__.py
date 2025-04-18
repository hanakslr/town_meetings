from abc import ABC, abstractmethod
from typing import Any, Dict

from anthropic.types import ToolParam


class Tool(ABC):
    @classmethod
    def is_structured_output(cls) -> bool:
        return False
    
    @classmethod
    @abstractmethod
    def get_tool_definition(cls) -> ToolParam:
        raise Exception("Subclass must implement.")

    
    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise Exception("Subclass must implement")
