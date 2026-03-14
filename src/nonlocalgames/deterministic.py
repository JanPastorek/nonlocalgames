"""Deterministic (classical) strategy evaluation for N-player nonlocal games.

For N players, each with n_questions possible questions and n_answers possible
answers, a deterministic strategy for player i is a mapping from questions to
answers, i.e. a function f_i: {0,...,n_questions-1} -> {0,...,n_answers-1}.

Each player has n_answers^n_questions possible strategies. We evaluate all
combinations of strategies across all players.
"""

import itertools

from nonlocalgames.game import NonLocalGame


class DeterministicSolver:
    """Evaluates all classical deterministic strategies for a nonlocal game.

    Parameters
    ----------
    game : NonLocalGame
        The nonlocal game to solve.
    """

    def __init__(self, game):
        self.game = game

    def _all_strategies(self):
        """Generate all possible deterministic strategies for one player.

        A strategy maps each question to an answer.

        Returns
        -------
        list of tuple
            Each tuple is a strategy: strategy[q] = answer for question q.
        """
        return list(
            itertools.product(
                range(self.game.n_answers), repeat=self.game.n_questions
            )
        )

    def evaluate_strategy(self, strategies):
        """Evaluate a combination of strategies (one per player).

        Parameters
        ----------
        strategies : tuple of tuple
            strategies[i][q] = answer that player i gives for question q.

        Returns
        -------
        float
            The winning probability for uniform question distribution.
        """
        game = self.game
        total_win = 0
        n_question_combos = len(game.question_combos)

        for q_combo in game.question_combos:
            # Each player answers based on their strategy and their question
            answers = tuple(
                strategies[player][q_combo[player]]
                for player in range(game.num_players)
            )
            if game.wins(q_combo, answers):
                total_win += 1

        return total_win / n_question_combos

    def find_optimal(self):
        """Find the best and worst classical strategies.

        Returns
        -------
        tuple of (float, float)
            (best_win_probability, worst_win_probability)
        """
        all_strats = self._all_strategies()
        # All combinations of strategies across N players
        strategy_combos = itertools.product(all_strats, repeat=self.game.num_players)

        best = 0.0
        worst = 1.0

        for combo in strategy_combos:
            wp = self.evaluate_strategy(combo)
            if wp > best:
                best = wp
            if wp < worst:
                worst = wp

        return best, worst

    def find_best(self):
        """Find the best classical strategy and its winning probability.

        Returns
        -------
        tuple of (float, tuple)
            (best_win_probability, best_strategy_combination)
        """
        all_strats = self._all_strategies()
        strategy_combos = itertools.product(all_strats, repeat=self.game.num_players)

        best_wp = 0.0
        best_combo = None

        for combo in strategy_combos:
            wp = self.evaluate_strategy(combo)
            if wp > best_wp:
                best_wp = wp
                best_combo = combo

        return best_wp, best_combo
