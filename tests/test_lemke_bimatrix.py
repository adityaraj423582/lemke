"""
Tests for the bimatrix game solver (lemke.bimatrix): load game file,
Lemke-Howson (LH) for labels 1 and 2, and a known 2x2 Nash equilibrium.
Uses fractions.Fraction throughout; asserts equilibrium length, probability sums, and non-negativity.
"""
import fractions

import numpy as np
import pytest

from lemke.bimatrix import bimatrix, getequil, supports


# -----------------------------------------------------------------------------
# Helpers: equilibrium checks
# -----------------------------------------------------------------------------

def _equilibrium_length_is_m_plus_n(eq, m, n):
    """getequil returns solution[1:tabl.n-1]; for game LCP tabl.n = m+n+2, so len = m+n."""
    return len(eq) == m + n


def _row_player_sum(eq, m, n):
    """Sum of row player's mixed strategy (first m entries)."""
    return sum(eq[i] for i in range(m))


def _col_player_sum(eq, m, n):
    """Sum of column player's mixed strategy (next n entries)."""
    return sum(eq[m + j] for j in range(n))


def _all_nonnegative(eq):
    """All entries of equilibrium tuple are >= 0."""
    return all(x >= 0 for x in eq)


# -----------------------------------------------------------------------------
# Tests: examples/game file, LH for labels 1 and 2
# -----------------------------------------------------------------------------

def test_bimatrix_load_example_game(example_game_path):
    """Load bimatrix from examples/game; assert dimensions and that A, B are set."""
    G = bimatrix(example_game_path)
    assert G.A.numrows == 2 and G.A.numcolumns == 2
    assert G.B.numrows == 2 and G.B.numcolumns == 2


def test_lh_label_1_returns_equilibrium(example_game_path):
    """
    Run Lemke-Howson with dropped label 1 on examples/game.
    Assert returned tuple has length m+n, row and column sums are 1, and all entries >= 0.
    """
    G = bimatrix(example_game_path)
    m, n = G.A.numrows, G.A.numcolumns
    eq = G.runLH(1)
    assert eq is not None
    assert _equilibrium_length_is_m_plus_n(eq, m, n), f"expected len={m+n}, got {len(eq)}"
    assert _row_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _col_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _all_nonnegative(eq)


def test_lh_label_2_returns_equilibrium(example_game_path):
    """
    Run Lemke-Howson with dropped label 2 on examples/game.
    Same assertions: length m+n, probability sums 1, non-negative.
    """
    G = bimatrix(example_game_path)
    m, n = G.A.numrows, G.A.numcolumns
    eq = G.runLH(2)
    assert eq is not None
    assert _equilibrium_length_is_m_plus_n(eq, m, n)
    assert _row_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _col_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _all_nonnegative(eq)


# -----------------------------------------------------------------------------
# Tests: known 2x2 game with known Nash equilibrium
# -----------------------------------------------------------------------------

def test_known_2x2_game_nash_equilibrium(temp_game_file):
    """
    Game: A = [[3,0],[0,2]], B = [[2,0],[0,7]] (examples/game).
    Known equilibria: pure (1,0),(1,0); pure (0,1),(0,1); mixed (7/9, 2/9), (2/5, 3/5).
    Run LH for labels 1 and 2; at least one run should return a valid equilibrium.
    We check that runLH(1) and runLH(2) each return a valid mixed strategy and
    that one of the pure equilibria or the mixed equilibrium appears.
    """
    content = """2 2
3 0
0 2
2 0
0 7
"""
    path = temp_game_file(content)
    G = bimatrix(path)
    m, n = 2, 2

    eq1 = G.runLH(1)
    eq2 = G.runLH(2)

    for eq in (eq1, eq2):
        assert len(eq) == m + n
        assert _row_player_sum(eq, m, n) == fractions.Fraction(1)
        assert _col_player_sum(eq, m, n) == fractions.Fraction(1)
        assert _all_nonnegative(eq)

    # Known mixed equilibrium: row (7/9, 2/9), col (2/5, 3/5)
    mixed_row = (fractions.Fraction(7, 9), fractions.Fraction(2, 9))
    mixed_col = (fractions.Fraction(2, 5), fractions.Fraction(3, 5))
    # One of the two runs should find either a pure or the mixed equilibrium
    found = False
    for eq in (eq1, eq2):
        row_part = tuple(eq[0:m])
        col_part = tuple(eq[m : m + n])
        if row_part == mixed_row and col_part == mixed_col:
            found = True
            break
        # Pure (1,0),(1,0)
        if row_part == (fractions.Fraction(1), fractions.Fraction(0)) and col_part == (fractions.Fraction(1), fractions.Fraction(0)):
            found = True
            break
        # Pure (0,1),(0,1)
        if row_part == (fractions.Fraction(0), fractions.Fraction(1)) and col_part == (fractions.Fraction(0), fractions.Fraction(1)):
            found = True
            break
    assert found, f"Expected one of the known equilibria; got eq1={eq1}, eq2={eq2}"


def test_supports_and_equilibrium_structure(example_game_path):
    """Run LH(1), check supports() returns correct row/col indices for positive entries."""
    G = bimatrix(example_game_path)
    m, n = G.A.numrows, G.A.numcolumns
    eq = G.runLH(1)
    rowset, colset = supports(eq, m, n)
    for i in range(m):
        assert (eq[i] > 0) == (i in rowset)
    for j in range(n):
        assert (eq[m + j] > 0) == (j in colset)
