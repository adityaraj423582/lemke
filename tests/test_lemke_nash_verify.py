# use pygambit to check lemke's equilibria are actually nash (max_regret = 0)
# if pygambit not installed the tests just skip
import fractions

import numpy as np
import pytest

pytest.importorskip("pygambit")
import pygambit as gbt

from lemke.bimatrix import bimatrix


def lemke_eq_to_profile(game_gbt, eq, m, n, rational=True):
    # lemke gives (row probs, col probs), gambit wants [[row],[col]]
    row_probs = [eq[i] for i in range(m)]
    col_probs = [eq[m + j] for j in range(n)]
    if rational:
        data = [
            [fractions.Fraction(p) for p in row_probs],
            [fractions.Fraction(p) for p in col_probs],
        ]
    else:
        data = [
            [float(p) for p in row_probs],
            [float(p) for p in col_probs],
        ]
    return game_gbt.mixed_strategy_profile(data=data, rational=rational)


def check_max_regret_zero(profile, rational=True, tol=1e-10):
    # nash eq means nobody wants to deviate so regret should be 0
    mr = profile.max_regret()
    if rational:
        assert mr == 0, f"Expected max_regret=0, got {mr}"
    else:
        assert abs(float(mr)) <= tol, f"Expected |max_regret| <= {tol}, got {mr}"


# game 1: has pure nash, easy to check
GAME1_A = np.array([[3, 0], [0, 2]], dtype=float)
GAME1_B = np.array([[2, 0], [0, 7]], dtype=float)


@pytest.fixture
def temp_game1_file(tmp_path):
    content = """2 2
3 0
0 2
2 0
0 7
"""
    path = tmp_path / "game1.game"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_nash_verify_game1_pure_or_mixed(temp_game1_file):
    # run lemke get eq, put it in pygambit, check max_regret is 0
    G_lemke = bimatrix(temp_game1_file)
    m, n = G_lemke.A.numrows, G_lemke.A.numcolumns
    eq = G_lemke.runLH(1)
    assert len(eq) == m + n

    game_gbt = gbt.Game.from_arrays(GAME1_A, GAME1_B)
    profile = lemke_eq_to_profile(game_gbt, eq, m, n, rational=True)

    check_max_regret_zero(profile, rational=True)


# game 2: matching pennies, only mixed nash (1/2, 1/2)
GAME2_A = np.array([[1, -1], [-1, 1]], dtype=float)
GAME2_B = np.array([[-1, 1], [1, -1]], dtype=float)


@pytest.fixture
def temp_game2_file(tmp_path):
    content = """2 2
1 -1
-1 1
-1 1
1 -1
"""
    path = tmp_path / "game2.game"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_nash_verify_game2_mixed(temp_game2_file):
    # matching pennies - only one eq so we know what to expect
    G_lemke = bimatrix(temp_game2_file)
    m, n = G_lemke.A.numrows, G_lemke.A.numcolumns
    eq = G_lemke.runLH(1)
    assert len(eq) == m + n

    game_gbt = gbt.Game.from_arrays(GAME2_A, GAME2_B)
    profile = lemke_eq_to_profile(game_gbt, eq, m, n, rational=True)

    check_max_regret_zero(profile, rational=True)

    half = fractions.Fraction(1, 2)
    assert eq[0] == half and eq[1] == half
    assert eq[2] == half and eq[3] == half


def test_nash_verify_game1_float_profile(temp_game1_file):
    # same as game1 but with floats, use a tolerance
    G_lemke = bimatrix(temp_game1_file)
    m, n = G_lemke.A.numrows, G_lemke.A.numcolumns
    eq = G_lemke.runLH(1)

    game_gbt = gbt.Game.from_arrays(GAME1_A, GAME1_B)
    profile = lemke_eq_to_profile(game_gbt, eq, m, n, rational=False)

    check_max_regret_zero(profile, rational=False, tol=1e-9)
