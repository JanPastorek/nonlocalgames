"""Quantum state and operation primitives for nonlocal game strategies.

Provides quantum state representation, unitary operations (rotation gates),
and a gate optimizer using genetic algorithms for N-player games.

No external quantum computing library is required; gates are implemented
directly with numpy.
"""

import itertools
import math

import numpy as np

from nonlocalgames.game import NonLocalGame
from nonlocalgames.genetic import GeneticAlgorithm


class QuantumState:
    """A quantum state represented as a complex unit vector.

    Parameters
    ----------
    vector : array-like
        The state vector (must have unit norm).
    """

    def __init__(self, vector):
        self.vector = np.array(vector, dtype=np.complex128)
        norm = np.linalg.norm(self.vector)
        if not np.isclose(norm, 1.0, atol=1e-6):
            raise ValueError(
                f"Quantum state must have unit norm, got {norm}"
            )

    def probabilities(self):
        """Return the measurement probability for each basis state."""
        return np.abs(self.vector) ** 2

    def num_qubits(self):
        """Return the number of qubits in this state."""
        return int(math.log2(len(self.vector)))

    def __repr__(self):
        return f"QuantumState({self.vector})"


class QuantumOperation:
    """A quantum operation represented as a unitary matrix.

    Parameters
    ----------
    matrix : array-like
        A unitary matrix.
    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.complex128)
        identity = np.eye(len(self.matrix))
        product = self.matrix.conj().T @ self.matrix
        if not np.allclose(identity, product, atol=1e-6):
            raise ValueError("Quantum operation must be unitary")

    def apply(self, state):
        """Apply this operation to a quantum state."""
        new_vector = self.matrix @ state.vector
        return QuantumState(new_vector)

    def tensor(self, other):
        """Return the tensor product of this operation with another."""
        new_matrix = np.kron(self.matrix, other.matrix)
        return QuantumOperation(new_matrix)

    def __repr__(self):
        return f"QuantumOperation({self.matrix})"


# Standard single-qubit rotation gates

def ry_gate(angle):
    """RY rotation gate.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    """
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rx_gate(angle):
    """RX rotation gate.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    """
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def rz_gate(angle):
    """RZ rotation gate.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    """
    return np.array(
        [[np.exp(-1j * angle / 2), 0], [0, np.exp(1j * angle / 2)]],
        dtype=np.complex128,
    )


def identity_gate():
    """2x2 identity gate."""
    return np.eye(2, dtype=np.complex128)


GATE_MAP = {
    "ry": ry_gate,
    "rx": rx_gate,
    "rz": rz_gate,
}


def build_player_gate(gate_matrix, player_index, num_players):
    """Build a multi-qubit gate that applies a single-qubit gate to one player's qubit.

    Assumes each player has exactly 1 qubit.

    Parameters
    ----------
    gate_matrix : ndarray
        2x2 gate matrix.
    player_index : int
        Which player's qubit to act on (0-indexed).
    num_players : int
        Total number of players (= total qubits for 1-qubit-per-player games).

    Returns
    -------
    ndarray
        The full (2^num_players x 2^num_players) gate matrix.
    """
    matrices = []
    for i in range(num_players):
        if i == player_index:
            matrices.append(gate_matrix)
        else:
            matrices.append(identity_gate())
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


class GateOptimizer(GeneticAlgorithm):
    """Genetic algorithm optimizer for quantum gate angles in nonlocal games.

    Optimizes rotation gate angles to maximize (or minimize) the winning
    probability for N players.

    Parameters
    ----------
    game : NonLocalGame
        The nonlocal game.
    initial_state : array-like
        The initial shared quantum state vector.
    gate_type : str
        Type of rotation gate ("ry", "rx", or "rz").
    population_size : int
        GA population size.
    n_crossover : int
        Number of crossover points.
    mutation_prob : float
        Probability of mutation for each gene.
    maximize : bool
        If True, maximize winning probability. If False, minimize.
    """

    def __init__(
        self,
        game,
        initial_state=None,
        gate_type="ry",
        population_size=30,
        n_crossover=3,
        mutation_prob=0.1,
        maximize=True,
    ):
        self.game = game
        self.gate_type = gate_type
        self.gate_fn = GATE_MAP[gate_type]
        self.maximize = maximize

        if initial_state is None:
            # Default: Bell state for 2 qubits, or GHZ-like for N qubits
            dim = 2 ** game.num_players
            state = np.zeros(dim, dtype=np.complex128)
            state[0] = 1 / math.sqrt(2)
            state[-1] = 1 / math.sqrt(2)
            self.initial_state = state
        else:
            self.initial_state = np.array(initial_state, dtype=np.complex128)

        # Each player has a gate angle for each question: num_players * n_questions genes
        self.n_genes = game.num_players * game.n_questions

        super().__init__(
            population_size=population_size,
            n_crossover=min(n_crossover, self.n_genes - 1) if self.n_genes > 1 else 0,
            mutation_prob=mutation_prob,
            maximize=maximize,
        )

    def generate_individual(self):
        """Generate an individual as a list of gate angles in degrees."""
        return [float(np.random.uniform(-180, 180)) for _ in range(self.n_genes)]

    def fitness(self, individual):
        """Evaluate fitness by computing winning probability for the given gate angles."""
        game = self.game
        probabilities = []

        for q_combo in game.question_combos:
            state = self.initial_state.copy()

            # Apply each player's gate based on their question
            for player in range(game.num_players):
                question = q_combo[player]
                # Gene index: player * n_questions + question
                gene_idx = player * game.n_questions + question
                angle_deg = individual[gene_idx]
                angle_rad = angle_deg * math.pi / 180.0

                gate = self.gate_fn(angle_rad)
                full_gate = build_player_gate(gate, player, game.num_players)
                state = full_gate @ state

            probs = np.abs(state) ** 2
            probabilities.append(probs.tolist())

        return game.calc_win_probability(probabilities)

    def mutation(self, individual, prob):
        """Mutate gate angles with Gaussian perturbation."""
        result = individual.copy()
        for i in range(len(result)):
            if np.random.random() <= prob:
                # Gaussian perturbation scaled by spread of current angles
                angles = [abs(a) for a in result]
                std = max(np.std(angles) / 10.0, 1.0)
                result[i] += np.random.normal(0, std)
        return result

    def solve(self, max_generations, goal_fitness=1.0):
        """Run the genetic algorithm and return best individual with fitness.

        Returns
        -------
        tuple of (list, float)
            (best_gate_angles, best_fitness)
        """
        best = super().solve(max_generations, goal_fitness)
        best_fitness = self.fitness(best)
        return best, best_fitness

    def decode_strategy(self, individual):
        """Decode an individual into a human-readable strategy description.

        Returns
        -------
        dict
            Mapping from (player, question) to gate angle in degrees.
        """
        strategy = {}
        for player in range(self.game.num_players):
            for question in range(self.game.n_questions):
                gene_idx = player * self.game.n_questions + question
                strategy[(player, question)] = individual[gene_idx]
        return strategy
