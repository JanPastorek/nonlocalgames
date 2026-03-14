"""Core nonlocal game framework generalized for N players.

A nonlocal game involves:
- A referee who sends questions to players
- N players who cannot communicate with each other
- Each player receives a question and must provide an answer
- A winning condition (game matrix) determines if the players win

For N players, each with n_questions possible questions and n_answers possible
answers, the game matrix has shape (n_questions^N, n_answers^N) where entry
[q_combo, a_combo] is 1 if answer combination a_combo wins for question
combination q_combo.
"""

import itertools
import math
from abc import ABC, abstractmethod

import numpy as np


class NonLocalGame:
    """Represents a nonlocal game for an arbitrary number of players.

    Parameters
    ----------
    game_matrix : list of list of int
        The game (winning condition) matrix. Rows correspond to question
        combinations and columns to answer combinations.
        Entry [i][j] == 1 means answer combo j wins for question combo i.
    num_players : int
        Number of players (default 2).
    n_questions : int
        Number of possible questions per player (default 2).
    n_answers : int
        Number of possible answers per player (default 2).
    """

    def __init__(self, game_matrix, num_players=2, n_questions=2, n_answers=2):
        self.game_matrix = [list(row) for row in game_matrix]
        self.num_players = num_players
        self.n_questions = n_questions
        self.n_answers = n_answers

        expected_rows = n_questions ** num_players
        expected_cols = n_answers ** num_players
        if len(self.game_matrix) != expected_rows:
            raise ValueError(
                f"Game matrix should have {expected_rows} rows "
                f"(n_questions^num_players), got {len(self.game_matrix)}"
            )
        for i, row in enumerate(self.game_matrix):
            if len(row) != expected_cols:
                raise ValueError(
                    f"Game matrix row {i} should have {expected_cols} columns "
                    f"(n_answers^num_players), got {len(row)}"
                )

        self.question_combos = list(
            itertools.product(range(n_questions), repeat=num_players)
        )
        self.answer_combos = list(
            itertools.product(range(n_answers), repeat=num_players)
        )

    def wins(self, questions, answers):
        """Check if the given answers win for the given questions.

        Parameters
        ----------
        questions : tuple of int
            Question for each player, length num_players.
        answers : tuple of int
            Answer from each player, length num_players.

        Returns
        -------
        bool
            True if the answer combination wins for the question combination.
        """
        q_idx = self._combo_index(questions, self.n_questions)
        a_idx = self._combo_index(answers, self.n_answers)
        return self.game_matrix[q_idx][a_idx] == 1

    def _combo_index(self, combo, base):
        """Convert a combination tuple to a linear index.

        For example, with base=2: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3.
        """
        idx = 0
        for val in combo:
            idx = idx * base + val
        return idx

    def calc_win_probability(self, probabilities):
        """Calculate the winning probability from outcome probabilities.

        Parameters
        ----------
        probabilities : list of list of float
            For each question combination, the probability distribution over
            answer combinations. Shape: (n_question_combos, n_answer_combos).

        Returns
        -------
        float
            The overall winning probability (uniform distribution over
            questions).
        """
        n_q = len(self.game_matrix)
        win_rate = 0.0
        for q_idx, row in enumerate(self.game_matrix):
            for a_idx, wins in enumerate(row):
                win_rate += wins * probabilities[q_idx][a_idx]
        win_rate /= n_q
        return win_rate

    def difficulty(self):
        """Calculate game difficulty as the count of winning entries."""
        return sum(cell for row in self.game_matrix for cell in row)


def generate_interesting_games(num_players=2, n_questions=2, n_answers=2):
    """Generate interesting game matrices by filtering out trivial/symmetric ones.

    Parameters
    ----------
    num_players : int
        Number of players.
    n_questions : int
        Number of questions per player.
    n_answers : int
        Number of answers per player.

    Returns
    -------
    list
        List of game matrices (as lists of lists).
    """
    n_rows = n_questions ** num_players
    n_cols = n_answers ** num_players

    # Generate all possible rows
    possible_rows = list(itertools.product(range(n_answers), repeat=n_cols))
    # Generate all possible game matrices
    all_games = itertools.product(possible_rows, repeat=n_rows)

    interesting = {}
    for game in all_games:
        game_list = [list(row) for row in game]

        # Skip games with all-zero or all-one rows (trivial)
        skip = False
        for row in game_list:
            if all(v == 0 for v in row) or all(v == 1 for v in row):
                skip = True
                break
        if skip:
            continue

        # For 2-player games with 2 questions, apply symmetry reduction
        if num_players == 2 and n_questions == 2 and n_answers == 2:
            g = tuple(tuple(r) for r in game_list)
            # Swap players (swap rows for question combos, swap columns for answer combos)
            swapped_q = (g[0], g[2], g[1], g[3]) if len(g) == 4 else g
            swapped_a = tuple(
                tuple(row[i] for i in [0, 2, 1, 3]) if len(row) == 4 else row
                for row in g
            )
            # Skip if a symmetric version was already seen
            if swapped_q in interesting or swapped_a in interesting:
                continue
            interesting[g] = game_list
        else:
            interesting[tuple(tuple(r) for r in game_list)] = game_list

    return list(interesting.values())
