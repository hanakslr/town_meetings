import pytest

from strategies import FetchingStrategy

def pytest_generate_tests(metafunc):
    if "strategy_name" in metafunc.fixturenames:
        metafunc.parametric

def test_strategy_type(strategy_name, input_params, expected_output):
    result = FetchingStrategy.get_agendas(strategy_name, input_params=input_params)

    assert result == expected_output