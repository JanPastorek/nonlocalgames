"""NonLocalGames - Framework for nonlocal games with support for multiple players."""

from nonlocalgames.game import NonLocalGame
from nonlocalgames.deterministic import DeterministicSolver
from nonlocalgames.genetic import GeneticAlgorithm
from nonlocalgames.quantum import QuantumState, QuantumOperation, GateOptimizer

__all__ = [
    "NonLocalGame",
    "DeterministicSolver",
    "GeneticAlgorithm",
    "QuantumState",
    "QuantumOperation",
    "GateOptimizer",
]
