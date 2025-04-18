from abc import ABC, abstractmethod
from typing import Any, Dict

from anthropic.types import ToolParam


class Tool(ABC):
    @abstractmethod
    def get_tool_definition(self) -> ToolParam:
        raise Exception("Subclass must implement.")

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        raise Exception("Subclass must implement")
