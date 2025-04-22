import json
from pathlib import Path

import pytest

from strategies import FetchingStrategy


def pytest_addoption(parser):
    parser.addoption(
        "--strategy",
        action="store",
        default=None,
        help="Only run tests for the specified strategy",
    )


def pytest_generate_tests(metafunc):
    # Find all strategy directories
    strategies_dir = Path("strategies/fixtures")
    if not strategies_dir.exists():
        pytest.skip("No strategies directory found")

    # Get all strategy names from the directory structure
    strategy_names = [d.name for d in strategies_dir.iterdir() if d.is_dir()]

    print(f"Found strategy names: {strategy_names}")

    # Filter by command line argument if specified
    requested_strategy = metafunc.config.getoption("--strategy", None)
    if requested_strategy:
        if requested_strategy not in strategy_names:
            pytest.skip(f"Requested strategy '{requested_strategy}' not found")
        strategy_names = [requested_strategy]

    # For each strategy, load its test data
    test_cases = []
    test_ids = []

    for strategy_name in strategy_names:
        strategy_dir = Path(strategies_dir / strategy_name)
        group_names = [d.name for d in strategy_dir.iterdir() if d.is_dir()]

        for group in group_names:
            expected_path = strategy_dir / group / "expected.json"
            params_path = strategy_dir / group / "params.json"
            if expected_path.exists() and params_path.exists():
                with open(expected_path) as f:
                    expected_output = json.load(f)

                with open(params_path) as f:
                    input_params = json.load(f)
                test_cases.append((strategy_name, input_params, expected_output))
                # Add a human-friendly test ID
                test_ids.append(f"{strategy_name}-{group}")
            else:
                raise Exception(
                    f"Did not find expected files for {group} in {strategy_name} strategy."
                )

    if not test_cases:
        pytest.skip("No valid strategy test cases found")

    # Parametrize the test with all strategy test cases
    metafunc.parametrize(
        "strategy_name,input_params,expected_output", test_cases, ids=test_ids
    )


def test_strategy_type(strategy_name, input_params, expected_output):
    result = FetchingStrategy.get_agendas(strategy_name, input_params=input_params)
    assert result == expected_output
