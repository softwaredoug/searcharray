import pytest
from typing import Dict, Any


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        [key for key in scenarios.values()][0].keys(),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )
