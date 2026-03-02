# bimatrix / LH tests
import fractions

import numpy as np
import pytest

from lemke.bimatrix import bimatrix, getequil, supports


def _equilibrium_length_is_m_plus_n(eq, m, n):
    return len(eq) == m + n


def _row_player_sum(eq, m, n):
    return sum(eq[i] for i in range(m))


def _col_player_sum(eq, m, n):
    return sum(eq[m + j] for j in range(n))


def _all_nonnegative(eq):
    return all(x >= 0 for x in eq)


def test_bimatrix_load_example_game(example_game_path):
    G = bimatrix(example_game_path)
    assert G.A.numrows == 2 and G.A.numcolumns == 2
    assert G.B.numrows == 2 and G.B.numcolumns == 2


def test_lh_label_1_returns_equilibrium(example_game_path):
    # LH with label 1 should give something that looks like an eq (len, sums, nonneg)
    G = bimatrix(example_game_path)
    m, n = G.A.numrows, G.A.numcolumns
    eq = G.runLH(1)
    assert eq is not None
    assert _equilibrium_length_is_m_plus_n(eq, m, n), f"expected len={m+n}, got {len(eq)}"
    assert _row_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _col_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _all_nonnegative(eq)


def test_lh_label_2_returns_equilibrium(example_game_path):
    # same as label 1 but for label 2
    G = bimatrix(example_game_path)
    m, n = G.A.numrows, G.A.numcolumns
    eq = G.runLH(2)
    assert eq is not None
    assert _equilibrium_length_is_m_plus_n(eq, m, n)
    assert _row_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _col_player_sum(eq, m, n) == fractions.Fraction(1)
    assert _all_nonnegative(eq)


def test_known_2x2_game_nash_equilibrium(temp_game_file):
    # the 2x2 from examples/game - we know the equilibria so check we get one of them
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

    mixed_row = (fractions.Fraction(7, 9), fractions.Fraction(2, 9))
    mixed_col = (fractions.Fraction(2, 5), fractions.Fraction(3, 5))
    found = False
    for eq in (eq1, eq2):
        row_part = tuple(eq[0:m])
        col_part = tuple(eq[m : m + n])
        if row_part == mixed_row and col_part == mixed_col:
            found = True
            break
        if row_part == (fractions.Fraction(1), fractions.Fraction(0)) and col_part == (fractions.Fraction(1), fractions.Fraction(0)):
            found = True
            break
        if row_part == (fractions.Fraction(0), fractions.Fraction(1)) and col_part == (fractions.Fraction(0), fractions.Fraction(1)):
            found = True
            break
    assert found, f"Expected one of the known equilibria; got eq1={eq1}, eq2={eq2}"


def test_supports_and_equilibrium_structure(example_game_path):
    # checking if supports() matches where the probs are > 0
    G = bimatrix(example_game_path)
    m, n = G.A.numrows, G.A.numcolumns
    eq = G.runLH(1)
    rowset, colset = supports(eq, m, n)
    for i in range(m):
        assert (eq[i] > 0) == (i in rowset)
    for j in range(n):
        assert (eq[m + j] > 0) == (j in colset)
