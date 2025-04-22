
from abc import ABC, abstractmethod
from typing import Any, Type
from dataclasses import dataclass

@dataclass
class Meeting:
    date: str
    agenda: str

@dataclass 
class AgendaFetchResult:
    name: str
    meetings: list[Meeting]



class FetchingStrategy(ABC):
    registry: dict[str, Type["FetchingStrategy"]] = {}
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        key = getattr(cls, "name", cls.__name__)
        FetchingStrategy.registry[key] = cls

    @abstractmethod
    def fetch(self, input_params) -> AgendaFetchResult:
        pass

    @classmethod
    def get_agendas(cls, for_strategy: str, input_params: dict[str, Any]) -> AgendaFetchResult:
        return FetchingStrategy.registry[for_strategy]().fetch(input_params=input_params)

class Test(FetchingStrategy):
    name = "test"

    def fetch(self, input_params):
        return "gotcha"