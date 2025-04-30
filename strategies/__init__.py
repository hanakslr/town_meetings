from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type
import os
import importlib
import pkgutil


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

    @classmethod
    def _import_all_strategies(cls):
        """Dynamically import all strategy modules in the current package."""
        package_dir = os.path.dirname(__file__)
        for _, module_name, is_pkg in pkgutil.iter_modules([package_dir]):
            if not is_pkg and module_name != "__init__":
                importlib.import_module(f".{module_name}", package=__package__)

    @abstractmethod
    def fetch(self, input_params) -> list[Meeting]:
        pass

    @classmethod
    def get_agendas(
        cls, for_strategy: str, input_params: dict[str, Any]
    ) -> AgendaFetchResult:
        return FetchingStrategy.registry[for_strategy]().fetch(**input_params)


class Test(FetchingStrategy):
    name = "fake1"

    def fetch(self, *, input_field_1, expected_faker_vals):
        # This is a fake strategy - we are passing in the output in the input for the
        # test harness
        return expected_faker_vals

# Import all strategies when the package is imported
FetchingStrategy._import_all_strategies()
