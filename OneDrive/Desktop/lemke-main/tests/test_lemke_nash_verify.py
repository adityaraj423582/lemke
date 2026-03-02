"""
Bridge test: check that equilibria computed by lemke's Lemke-Howson (LH) algorithm
are valid Nash equilibria by verifying them with pygambit.

This is the core test that connects lemke's output to pygambit verification.

What we do:
  1. Run lemke's LH on a bimatrix game to get an equilibrium (mixed strategy tuple).
  2. Create the same game in pygambit using gbt.Game.from_arrays(A, B).
  3. Convert the lemke equilibrium into a pygambit MixedStrategyProfile.
  4. Call profile.max_regret() and assert it is 0 (or very close for floats).

Why max_regret = 0 means we have a Nash equilibrium:
  In game theory, a Nash equilibrium is a strategy profile where no player can
  improve their payoff by unilaterally changing strategy. "Regret" for a player
  is how much they could gain by switching to their best response. So if the
  maximum regret across all players is 0, nobody has an incentive to deviate —
  the profile is a Nash equilibrium. We use pygambit's max_regret() as an
  independent check that lemke's solution is correct.
"""
import fractions

import numpy as np
import pytest

# Skip the entire module if pygambit is not installed (optional dependency for this test)
pytest.importorskip("pygambit")
import pygambit as gbt

from lemke.bimatrix import bimatrix


# -----------------------------------------------------------------------------
# Helper: convert lemke equilibrium to pygambit MixedStrategyProfile
# -----------------------------------------------------------------------------

def lemke_eq_to_profile(game_gbt, eq, m, n, rational=True):
    """
    Build a pygambit MixedStrategyProfile from a lemke equilibrium tuple.

    lemke's getequil returns (x_0, ..., x_{m-1}, y_0, ..., y_{n-1}):
    row player probabilities x_i, then column player probabilities y_j.
    pygambit's mixed_strategy_profile(data=...) expects a nested list:
    [ [row_probs], [col_probs] ], with same ordering (row0, row1, ...), (col0, col1, ...).
    """
    row_probs = [eq[i] for i in range(m)]
    col_probs = [eq[m + j] for j in range(n)]
    if rational:
        # Pass as Python fractions for exact rational arithmetic in Gambit
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
    """
    Assert max_regret is 0 (exact for rational, within tol for float).
    If this passes, the profile is a Nash equilibrium.
    """
    mr = profile.max_regret()
    if rational:
        # With rationals, Gambit can return exact 0
        assert mr == 0, f"Expected max_regret=0, got {mr}"
    else:
        assert abs(float(mr)) <= tol, f"Expected |max_regret| <= {tol}, got {mr}"


# =============================================================================
# Game 1: Simple 2x2 with a known PURE Nash equilibrium
# =============================================================================
#
# Why this game? It has a single pure Nash equilibrium (row 0, col 0) that is
# very easy to verify by hand. If lemke finds it, we can check that pygambit
# agrees it has zero regret.
#
# Payoffs: row player A, column player B
#   A[0,0]=3, A[0,1]=0, A[1,0]=0, A[1,1]=2  (row prefers (0,0) or (1,1))
#   B[0,0]=2, B[0,1]=0, B[1,0]=0, B[1,1]=7  (col prefers (0,0) or (1,1))
# Pure NE: (row 0, col 0) and (row 1, col 1). LH with different starting labels
# can return either; we run LH and verify whatever equilibrium we get has max_regret 0.
#

# Same as examples/game: 2x2 with two pure and one mixed equilibrium
GAME1_A = np.array([[3, 0], [0, 2]], dtype=float)
GAME1_B = np.array([[2, 0], [0, 7]], dtype=float)


@pytest.fixture
def temp_game1_file(tmp_path):
    """Write the first game (pure + mixed equilibria) to a temp file for lemke."""
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
    """
    Game 1: 2x2 with known pure Nash equilibria (e.g. (1,0),(1,0) and (0,1),(0,1))
    and one mixed equilibrium. We run lemke's LH for label 1, get an equilibrium,
    build the same game in pygambit, convert the equilibrium to a MixedStrategyProfile,
    and assert max_regret is 0. This confirms lemke's output is a valid Nash equilibrium.
    """
    # 1) Run lemke's Lemke-Howson to get one equilibrium
    G_lemke = bimatrix(temp_game1_file)
    m, n = G_lemke.A.numrows, G_lemke.A.numcolumns
    eq = G_lemke.runLH(1)
    assert len(eq) == m + n

    # 2) Create the same game in pygambit (row player = first array, column = second)
    game_gbt = gbt.Game.from_arrays(GAME1_A, GAME1_B)

    # 3) Convert lemke equilibrium to a pygambit mixed strategy profile (rational)
    profile = lemke_eq_to_profile(game_gbt, eq, m, n, rational=True)

    # 4) At a Nash equilibrium, no player can gain by deviating, so max_regret must be 0
    check_max_regret_zero(profile, rational=True)


# =============================================================================
# Game 2: 2x2 with a MIXED Nash equilibrium (Matching Pennies style)
# =============================================================================
#
# Why this game? Matching Pennies has no pure Nash equilibrium; the only NE is
# mixed (1/2, 1/2) for both players. So we force the test to check a genuinely
# mixed profile. We use a small variant so the mixed NE is still simple to verify.
#
# Standard Matching Pennies: A = [[1,-1],[-1,1]], B = [[-1,1],[1,-1]].
# Unique NE: both play (1/2, 1/2). We run LH and verify the returned profile
# has max_regret 0.
#

GAME2_A = np.array([[1, -1], [-1, 1]], dtype=float)
GAME2_B = np.array([[-1, 1], [1, -1]], dtype=float)


@pytest.fixture
def temp_game2_file(tmp_path):
    """Write Matching Pennies to a temp file for lemke."""
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
    """
    Game 2: Matching Pennies — only Nash equilibrium is mixed (1/2, 1/2) for both.
    We run lemke's LH (e.g. label 1 or 2), get the equilibrium, build the game in
    pygambit, convert to MixedStrategyProfile, and assert max_regret is 0.
    How we know the expected output is correct: for this symmetric zero-sum game,
    the unique NE is the fully mixed strategy; any LH run that finds an equilibrium
    must find this one, and pygambit's max_regret confirms it is Nash.
    """
    G_lemke = bimatrix(temp_game2_file)
    m, n = G_lemke.A.numrows, G_lemke.A.numcolumns
    eq = G_lemke.runLH(1)
    assert len(eq) == m + n

    game_gbt = gbt.Game.from_arrays(GAME2_A, GAME2_B)
    profile = lemke_eq_to_profile(game_gbt, eq, m, n, rational=True)

    check_max_regret_zero(profile, rational=True)

    # Optional: assert we got the known mixed equilibrium (1/2, 1/2) for both
    half = fractions.Fraction(1, 2)
    assert eq[0] == half and eq[1] == half
    assert eq[2] == half and eq[3] == half


# =============================================================================
# Float profile (optional): same checks with float profile and tolerance
# =============================================================================

def test_nash_verify_game1_float_profile(temp_game1_file):
    """
    Same as test_nash_verify_game1_pure_or_mixed but we build the profile with
    rational=False (floats). We then assert max_regret is 0 within a small
    tolerance, since floating point can introduce tiny errors.
    """
    G_lemke = bimatrix(temp_game1_file)
    m, n = G_lemke.A.numrows, G_lemke.A.numcolumns
    eq = G_lemke.runLH(1)

    game_gbt = gbt.Game.from_arrays(GAME1_A, GAME1_B)
    profile = lemke_eq_to_profile(game_gbt, eq, m, n, rational=False)

    check_max_regret_zero(profile, rational=False, tol=1e-9)
