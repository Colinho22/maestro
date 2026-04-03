"""
MAESTRO — strategies package
"""

from maestro.strategies.base import BaseStrategy
from maestro.strategies.single import SingleAgentStrategy

__all__ = ["BaseStrategy", "SingleAgentStrategy"]
from maestro.strategies.single import SingleAgentStrategy

# Stub imports for future strategies — uncomment as implemented
# from maestro.strategies.sop import SOPStrategy
# from maestro.strategies.crew import CrewAIStrategy
# from maestro.strategies.langgraph import LangGraphStrategy

__all__ = [
    "BaseStrategy",
    "SingleAgentStrategy",
]