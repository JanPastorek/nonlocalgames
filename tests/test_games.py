"""Tests for the nonlocalgames package."""

import math
import unittest

import numpy as np

from nonlocalgames.game import NonLocalGame, generate_interesting_games
from nonlocalgames.deterministic import DeterministicSolver
from nonlocalgames.quantum import (
    QuantumState,
    QuantumOperation,
    ry_gate,
    rx_gate,
    rz_gate,
    identity_gate,
    build_player_gate,
    GateOptimizer,
)
from nonlocalgames.genetic import GeneticAlgorithm


# ---- CHSH game definition (standard 2-player, 2-question, 2-answer) ----
CHSH_MATRIX = [
    [1, 0, 0, 1],  # q=(0,0): win if a=b
    [1, 0, 0, 1],  # q=(0,1): win if a=b
    [1, 0, 0, 1],  # q=(1,0): win if a=b
    [0, 1, 1, 0],  # q=(1,1): win if a!=b
]


class TestNonLocalGame(unittest.TestCase):
    """Tests for the NonLocalGame class."""

    def test_chsh_creation(self):
        game = NonLocalGame(CHSH_MATRIX, num_players=2, n_questions=2, n_answers=2)
        self.assertEqual(game.num_players, 2)
        self.assertEqual(len(game.question_combos), 4)
        self.assertEqual(len(game.answer_combos), 4)

    def test_chsh_wins(self):
        game = NonLocalGame(CHSH_MATRIX)
        # q=(0,0): win if a=b
        self.assertTrue(game.wins((0, 0), (0, 0)))
        self.assertFalse(game.wins((0, 0), (0, 1)))
        self.assertFalse(game.wins((0, 0), (1, 0)))
        self.assertTrue(game.wins((0, 0), (1, 1)))
        # q=(1,1): win if a!=b
        self.assertFalse(game.wins((1, 1), (0, 0)))
        self.assertTrue(game.wins((1, 1), (0, 1)))
        self.assertTrue(game.wins((1, 1), (1, 0)))
        self.assertFalse(game.wins((1, 1), (1, 1)))

    def test_calc_win_probability_uniform(self):
        game = NonLocalGame(CHSH_MATRIX)
        # All equal probabilities => 0.25 each for 4 outcomes
        probs = [[0.25, 0.25, 0.25, 0.25]] * 4
        wp = game.calc_win_probability(probs)
        self.assertAlmostEqual(wp, 0.5, places=5)

    def test_calc_win_probability_all_win(self):
        """A game where everything wins should give probability 1."""
        game = NonLocalGame([[1, 1, 1, 1]] * 4)
        probs = [[0.25, 0.25, 0.25, 0.25]] * 4
        wp = game.calc_win_probability(probs)
        self.assertAlmostEqual(wp, 1.0, places=5)

    def test_calc_win_probability_no_win(self):
        """A game where nothing wins should give probability 0."""
        game = NonLocalGame([[0, 0, 0, 0]] * 4)
        probs = [[0.25, 0.25, 0.25, 0.25]] * 4
        wp = game.calc_win_probability(probs)
        self.assertAlmostEqual(wp, 0.0, places=5)

    def test_invalid_matrix_rows(self):
        with self.assertRaises(ValueError):
            NonLocalGame([[1, 0, 0, 1]], num_players=2)  # Only 1 row

    def test_invalid_matrix_cols(self):
        with self.assertRaises(ValueError):
            NonLocalGame([[1, 0], [1, 0], [1, 0], [1, 0]], num_players=2)

    def test_difficulty(self):
        game = NonLocalGame(CHSH_MATRIX)
        self.assertEqual(game.difficulty(), 8)

    def test_3player_game_creation(self):
        """Test that a 3-player game can be created with correct dimensions."""
        # 3 players, 2 questions each, 2 answers each
        # Matrix: 2^3=8 rows (question combos), 2^3=8 cols (answer combos)
        matrix = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
        game = NonLocalGame(matrix, num_players=3, n_questions=2, n_answers=2)
        self.assertEqual(len(game.question_combos), 8)
        self.assertEqual(len(game.answer_combos), 8)

    def test_3player_game_wins(self):
        """Test win condition for a simple 3-player game."""
        # Identity-like game: win if answer combo matches question combo
        matrix = [[1 if i == j else 0 for j in range(8)] for i in range(8)]
        game = NonLocalGame(matrix, num_players=3, n_questions=2, n_answers=2)
        # q=(0,0,0) -> win only for a=(0,0,0)
        self.assertTrue(game.wins((0, 0, 0), (0, 0, 0)))
        self.assertFalse(game.wins((0, 0, 0), (0, 0, 1)))
        # q=(1,1,1) -> win only for a=(1,1,1)
        self.assertTrue(game.wins((1, 1, 1), (1, 1, 1)))
        self.assertFalse(game.wins((1, 1, 1), (0, 0, 0)))


class TestDeterministicSolver(unittest.TestCase):
    """Tests for deterministic (classical) strategy evaluation."""

    def test_chsh_classical_optimum(self):
        """The best classical CHSH strategy achieves 0.75."""
        game = NonLocalGame(CHSH_MATRIX)
        solver = DeterministicSolver(game)
        best, worst = solver.find_optimal()
        self.assertAlmostEqual(best, 0.75, places=5)
        self.assertAlmostEqual(worst, 0.25, places=5)

    def test_all_win_game(self):
        """A trivially winnable game should have classical optimum of 1.0."""
        game = NonLocalGame([[1, 1, 1, 1]] * 4)
        solver = DeterministicSolver(game)
        best, worst = solver.find_optimal()
        self.assertAlmostEqual(best, 1.0, places=5)

    def test_evaluate_single_strategy(self):
        """Test evaluating a specific strategy."""
        game = NonLocalGame(CHSH_MATRIX)
        solver = DeterministicSolver(game)
        # Strategy: both players always answer 0
        # a=0,b=0 wins for q=(0,0),(0,1),(1,0) but not (1,1) -> 3/4 = 0.75
        wp = solver.evaluate_strategy(((0, 0), (0, 0)))
        self.assertAlmostEqual(wp, 0.75, places=5)

    def test_find_best_returns_strategy(self):
        game = NonLocalGame(CHSH_MATRIX)
        solver = DeterministicSolver(game)
        best_wp, best_combo = solver.find_best()
        self.assertAlmostEqual(best_wp, 0.75, places=5)
        self.assertIsNotNone(best_combo)
        self.assertEqual(len(best_combo), 2)

    def test_3player_deterministic(self):
        """Test deterministic solver for a simple 3-player game.

        3 players, 2 questions each, 2 answers each.
        Win condition: players win if all answers are the same (all 0 or all 1),
        regardless of the questions.
        """
        matrix = []
        for q in range(8):
            row = [0] * 8
            row[0] = 1  # all answer 0 always wins
            row[7] = 1  # all answer 1 always wins
            matrix.append(row)
        game = NonLocalGame(matrix, num_players=3, n_questions=2, n_answers=2)
        solver = DeterministicSolver(game)
        best, worst = solver.find_optimal()
        # Best: all players use same constant strategy -> always all match -> 1.0
        self.assertAlmostEqual(best, 1.0, places=5)
        self.assertGreaterEqual(worst, 0.0)

    def test_symmetric_game(self):
        """Test a symmetric game where rows are all the same."""
        matrix = [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
        ]
        game = NonLocalGame(matrix)
        solver = DeterministicSolver(game)
        best, worst = solver.find_optimal()
        self.assertAlmostEqual(best, 1.0, places=5)
        self.assertAlmostEqual(worst, 0.0, places=5)


class TestQuantumState(unittest.TestCase):
    """Tests for quantum state operations."""

    def test_create_valid_state(self):
        state = QuantumState([1, 0])
        np.testing.assert_array_almost_equal(state.vector, [1, 0])

    def test_create_bell_state(self):
        state = QuantumState([1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
        probs = state.probabilities()
        np.testing.assert_array_almost_equal(probs, [0.5, 0, 0, 0.5])

    def test_invalid_state_norm(self):
        with self.assertRaises(ValueError):
            QuantumState([1, 1])

    def test_num_qubits(self):
        state = QuantumState([1, 0, 0, 0])
        self.assertEqual(state.num_qubits(), 2)


class TestQuantumOperation(unittest.TestCase):
    """Tests for quantum operations."""

    def test_identity(self):
        op = QuantumOperation(identity_gate())
        state = QuantumState([1, 0])
        new_state = op.apply(state)
        np.testing.assert_array_almost_equal(new_state.vector, [1, 0])

    def test_ry_gate_zero_angle(self):
        gate = ry_gate(0)
        np.testing.assert_array_almost_equal(gate, np.eye(2))

    def test_ry_gate_pi(self):
        """RY(pi) should swap |0> and |1>."""
        gate = ry_gate(math.pi)
        state = np.array([1, 0], dtype=np.complex128)
        new_state = gate @ state
        np.testing.assert_array_almost_equal(np.abs(new_state), [0, 1], decimal=5)

    def test_rx_gate_zero_angle(self):
        gate = rx_gate(0)
        np.testing.assert_array_almost_equal(gate, np.eye(2))

    def test_rz_gate_zero_angle(self):
        gate = rz_gate(0)
        np.testing.assert_array_almost_equal(gate, np.eye(2))

    def test_tensor_product(self):
        op1 = QuantumOperation(np.eye(2))
        op2 = QuantumOperation(np.eye(2))
        combined = op1.tensor(op2)
        np.testing.assert_array_almost_equal(combined.matrix, np.eye(4))

    def test_non_unitary_raises(self):
        with self.assertRaises(ValueError):
            QuantumOperation([[1, 1], [0, 0]])


class TestBuildPlayerGate(unittest.TestCase):
    """Tests for building multi-player gates."""

    def test_2player_first_player(self):
        """Gate on player 0 in a 2-player game."""
        gate = ry_gate(math.pi / 4)
        full = build_player_gate(gate, 0, 2)
        expected = np.kron(gate, identity_gate())
        np.testing.assert_array_almost_equal(full, expected)

    def test_2player_second_player(self):
        """Gate on player 1 in a 2-player game."""
        gate = ry_gate(math.pi / 4)
        full = build_player_gate(gate, 1, 2)
        expected = np.kron(identity_gate(), gate)
        np.testing.assert_array_almost_equal(full, expected)

    def test_3player_middle_player(self):
        """Gate on player 1 in a 3-player game."""
        gate = ry_gate(math.pi / 3)
        full = build_player_gate(gate, 1, 3)
        expected = np.kron(np.kron(identity_gate(), gate), identity_gate())
        np.testing.assert_array_almost_equal(full, expected)


class TestGateOptimizer(unittest.TestCase):
    """Tests for the quantum gate optimizer."""

    def test_chsh_quantum_beats_classical(self):
        """Quantum strategy for CHSH should exceed 0.75 classical bound."""
        np.random.seed(42)
        game = NonLocalGame(CHSH_MATRIX)
        state = np.array(
            [0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0], dtype=np.complex128
        )
        optimizer = GateOptimizer(
            game,
            initial_state=state,
            gate_type="ry",
            population_size=30,
            n_crossover=3,
            mutation_prob=0.1,
        )
        best_individual, best_fitness = optimizer.solve(25)
        # Quantum CHSH optimal is cos^2(pi/8) ≈ 0.854
        self.assertGreater(best_fitness, 0.80)

    def test_optimizer_decode_strategy(self):
        game = NonLocalGame(CHSH_MATRIX)
        optimizer = GateOptimizer(game)
        individual = [45.0, -30.0, 90.0, 0.0]
        strategy = optimizer.decode_strategy(individual)
        self.assertEqual(strategy[(0, 0)], 45.0)
        self.assertEqual(strategy[(0, 1)], -30.0)
        self.assertEqual(strategy[(1, 0)], 90.0)
        self.assertEqual(strategy[(1, 1)], 0.0)

    def test_3player_optimizer(self):
        """Test that optimizer works for 3-player games."""
        np.random.seed(123)
        # Simple 3-player game: win if all answers agree
        matrix = []
        for q in range(8):
            row = [0] * 8
            row[0] = 1
            row[7] = 1
            matrix.append(row)
        game = NonLocalGame(matrix, num_players=3, n_questions=2, n_answers=2)
        optimizer = GateOptimizer(
            game,
            gate_type="ry",
            population_size=20,
            mutation_prob=0.1,
        )
        best_individual, best_fitness = optimizer.solve(15)
        # Should achieve at least some positive fitness
        self.assertGreater(best_fitness, 0.0)
        self.assertEqual(len(best_individual), 6)  # 3 players * 2 questions

    def test_optimizer_fitness_between_0_and_1(self):
        game = NonLocalGame(CHSH_MATRIX)
        optimizer = GateOptimizer(game, population_size=5)
        individual = optimizer.generate_individual()
        f = optimizer.fitness(individual)
        self.assertGreaterEqual(f, 0.0)
        self.assertLessEqual(f, 1.0)


class TestGeneticAlgorithm(unittest.TestCase):
    """Tests for the abstract genetic algorithm."""

    def test_simple_optimization(self):
        """Test GA on a simple function: maximize sum of bits."""

        class BitSumGA(GeneticAlgorithm):
            def generate_individual(self):
                return [np.random.randint(0, 2) for _ in range(5)]

            def fitness(self, x):
                return sum(x)

            def mutation(self, x, prob):
                result = x.copy()
                for i in range(len(result)):
                    if np.random.random() < prob:
                        result[i] = 1 - result[i]
                return result

        np.random.seed(42)
        ga = BitSumGA(population_size=20, n_crossover=2, mutation_prob=0.1)
        best = ga.solve(30, goal_fitness=5)
        self.assertEqual(sum(best), 5)

    def test_crossover(self):
        """Test that crossover produces valid offspring."""

        class DummyGA(GeneticAlgorithm):
            def generate_individual(self):
                return [0] * 4

            def fitness(self, x):
                return 0

            def mutation(self, x, prob):
                return x

        ga = DummyGA()
        parent_a = [1, 1, 1, 1]
        parent_b = [0, 0, 0, 0]
        child_a, child_b = ga.crossover(parent_a, parent_b, 1)
        # Children should contain genes from both parents
        self.assertEqual(len(child_a), 4)
        self.assertEqual(len(child_b), 4)


class TestGenerateInterestingGames(unittest.TestCase):
    """Tests for game generation."""

    def test_generates_games(self):
        games = generate_interesting_games(num_players=2, n_questions=2, n_answers=2)
        self.assertGreater(len(games), 0)
        # Each game should be a 4x4 matrix
        for game in games:
            self.assertEqual(len(game), 4)
            for row in game:
                self.assertEqual(len(row), 4)

    def test_no_trivial_rows(self):
        """Generated games should not have all-zero or all-one rows."""
        games = generate_interesting_games(num_players=2, n_questions=2, n_answers=2)
        for game in games:
            for row in game:
                self.assertFalse(all(v == 0 for v in row))
                self.assertFalse(all(v == 1 for v in row))


if __name__ == "__main__":
    unittest.main()
