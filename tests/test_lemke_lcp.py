"""
Tests for the LCP solver (lemke.lemke): lcp, tableau, runlemke.
Verifies complementarity, feasibility (w = M*z + q), and non-negativity.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

import lemke.lemke as lemke_module
from lemke.lemke import lcp, tableau

import fractions


# -----------------------------------------------------------------------------
# Test case dataclasses (PR #724 style): describe LCP and expected outcome
# -----------------------------------------------------------------------------

@dataclass
class LCPTestCase:
    """Single LCP test case: optional file path or (M, q, d), and expected solution if known."""
    name: str
    # Either path to LCP file (content as in examples/lcp), or None for programmatic
    path: Optional[str] = None
    # For programmatic: n and optional M, q, d (as lists; use Fraction or int)
    n: Optional[int] = None
    M: Optional[List[List]] = None
    q: Optional[List] = None
    d: Optional[List] = None
    # If True, runlemke is expected to raise SystemExit (e.g. ray termination)
    expect_exit: bool = False


def _solution_z_w(tabl):
    """Return (z, w) from a solved tableau. z = z1..zn, w = w1..wn."""
    n = tabl.n
    z = [tabl.solution[i] for i in range(1, n + 1)]
    w = [tabl.solution[i] for i in range(n + 1, 2 * n + 1)]
    return z, w


def _check_complementarity(z, w):
    """Verify z_i * w_i == 0 for all i."""
    for i in range(len(z)):
        prod = z[i] * w[i]
        assert prod == 0, f"Complementarity failed at i={i}: z_i*w_i = {prod}"


def _check_feasibility(M, q, z, w):
    """Verify w = M*z + q (using Fraction/numpy as appropriate)."""
    n = len(q)
    M_arr = np.array([[float(M[i][j]) for j in range(n)] for i in range(n)])
    z_arr = np.array([float(z[i]) for i in range(n)])
    q_arr = np.array([float(q[i]) for i in range(n)])
    w_expected = M_arr @ z_arr + q_arr
    w_actual = np.array([float(w[i]) for i in range(n)])
    np.testing.assert_allclose(w_actual, w_expected, rtol=1e-9, atol=1e-9)


def _check_nonnegativity(z, w):
    """Verify z >= 0 and w >= 0 (elementwise)."""
    for i, val in enumerate(z):
        assert val >= 0, f"z[{i}] = {val} < 0"
    for i, val in enumerate(w):
        assert val >= 0, f"w[{i}] = {val} < 0"


# -----------------------------------------------------------------------------
# Fixtures: build LCP from test case or from example file
# -----------------------------------------------------------------------------

@pytest.fixture
def lcp_from_example_file(example_lcp_path):
    """Load LCP from the bundled examples/lcp file."""
    m = lcp(example_lcp_path)
    return m


@pytest.fixture
def hand_crafted_2x2_lcp():
    """
    Hand-crafted 2x2 LCP with known solution: M = I, q = (-1,-1), d = (1,1).
    Solution: z = (1, 1), w = (0, 0). Satisfies w = z + q, complementarity, and non-negativity.
    """
    n = 2
    m = lcp(n)
    m.M[0][0] = fractions.Fraction(1)
    m.M[0][1] = fractions.Fraction(0)
    m.M[1][0] = fractions.Fraction(0)
    m.M[1][1] = fractions.Fraction(1)
    m.q[0] = fractions.Fraction(-1)
    m.q[1] = fractions.Fraction(-1)
    m.d[0] = fractions.Fraction(1)
    m.d[1] = fractions.Fraction(1)
    return m


@pytest.fixture
def ray_termination_lcp(temp_lcp_file):
    """
    LCP with no solution: M = 0, q = (-1,-1). Leads to ray termination in Lemke.
    runlemke() calls exit(1), so we expect SystemExit.
    """
    content = """n= 2
M= 0 0 0 0
q= -1 -1
d= 1 1
"""
    return temp_lcp_file(content)


# -----------------------------------------------------------------------------
# Tests: examples/lcp file
# -----------------------------------------------------------------------------

def test_lcp_from_example_file_solves(lcp_from_example_file, capsys):
    """
    Load LCP from examples/lcp, run Lemke, and verify solution satisfies
    complementarity (z_i*w_i=0), feasibility (w = M*z + q), and z,w >= 0.
    """
    m = lcp_from_example_file
    tabl = tableau(m)
    # Do not use silent=True to avoid writing to global outfile; capture stdout instead
    tabl.runlemke(verbose=False, z0=False, silent=False)
    capsys.readouterr()  # consume stdout

    tabl.createsol()
    z, w = _solution_z_w(tabl)
    n = m.n
    assert len(z) == n and len(w) == n

    _check_complementarity(z, w)
    _check_feasibility(m.M, m.q, z, w)
    _check_nonnegativity(z, w)


# -----------------------------------------------------------------------------
# Tests: hand-crafted 2x2 LCP with known solution
# -----------------------------------------------------------------------------

def test_hand_crafted_2x2_lcp_known_solution(hand_crafted_2x2_lcp, capsys):
    """
    Solve 2x2 LCP with M=I, q=(-1,-1). Known solution: z=(1,1), w=(0,0).
    Assert solution matches and satisfies all three conditions.
    """
    m = hand_crafted_2x2_lcp
    tabl = tableau(m)
    tabl.runlemke(verbose=False, z0=False, silent=False)
    capsys.readouterr()

    tabl.createsol()
    z, w = _solution_z_w(tabl)

    _check_complementarity(z, w)
    _check_feasibility(m.M, m.q, z, w)
    _check_nonnegativity(z, w)

    # Known solution
    assert z[0] == fractions.Fraction(1) and z[1] == fractions.Fraction(1)
    assert w[0] == fractions.Fraction(0) and w[1] == fractions.Fraction(0)


# -----------------------------------------------------------------------------
# Tests: ray termination raises SystemExit
# -----------------------------------------------------------------------------

def test_runlemke_ray_termination_raises_system_exit(ray_termination_lcp):
    """
    LCP with M=0, q negative has no solution; Lemke hits ray termination
    and calls exit(1). We expect SystemExit when runlemke is invoked.
    """
    m = lcp(ray_termination_lcp)
    tabl = tableau(m)
    with pytest.raises(SystemExit) as exc_info:
        tabl.runlemke(verbose=False, silent=False)
    assert exc_info.value.code == 1


# -----------------------------------------------------------------------------
# Tests: lcp(n) programmatic and invalid file
# -----------------------------------------------------------------------------

def test_lcp_construct_from_integer():
    """lcp(n) creates an n-dimensional LCP with zero M, q, d."""
    n = 3
    m = lcp(n)
    assert m.n == n
    assert len(m.M) == n and len(m.M[0]) == n
    assert len(m.q) == n and len(m.d) == n


# -----------------------------------------------------------------------------
# Parameterized test using LCPTestCase dataclass (PR #724 style)
# -----------------------------------------------------------------------------

def test_lcp_solution_satisfies_conditions_cases(lcp_from_example_file, hand_crafted_2x2_lcp, capsys):
    """
    Parameterized by fixtures: for each LCP that has a solution, run Lemke and
    assert complementarity, feasibility, and non-negativity. Uses the same
    check helpers as above; exercises the dataclass-style "test case" idea.
    """
    cases = [
        ("example_file", lcp_from_example_file),
        ("hand_crafted_2x2", hand_crafted_2x2_lcp),
    ]
    for name, m in cases:
        tabl = tableau(m)
        tabl.runlemke(verbose=False, z0=False, silent=False)
        capsys.readouterr()
        tabl.createsol()
        z, w = _solution_z_w(tabl)
        _check_complementarity(z, w)
        _check_feasibility(m.M, m.q, z, w)
        _check_nonnegativity(z, w)


def test_lcp_invalid_file_raises_system_exit(temp_lcp_file):
    """Loading an LCP file with wrong format (e.g. missing tokens) calls exit(1)."""
    # Too few words: n= 2, then M= and only 2 numbers instead of 4
    content = "n= 2\nM= 1 2\nq= -1 -1\nd= 1 1\n"
    path = temp_lcp_file(content)
    with pytest.raises(SystemExit) as exc_info:
        lcp(path)
    assert exc_info.value.code == 1
