"""
MAESTRO — strategies package
"""

from maestro.strategies.base import BaseStrategy
from maestro.strategies.single import SingleAgentStrategy
from maestro.strategies.sop import SOPStrategy

# Stub imports for future strategies — uncomment as implemented
# from maestro.strategies.crew import CrewAIStrategy
# from maestro.strategies.langgraph import LangGraphStrategy

__all__ = [
    "BaseStrategy",
    "SingleAgentStrategy",
    "SOPStrategy",
]