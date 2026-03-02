# LCP solver tests - checking complementarity, w = Mz + q, and z,w >= 0
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pytest

import lemke.lemke as lemke_module
from lemke.lemke import lcp, tableau

import fractions


@dataclass
class LCPTestCase:
    name: str
    path: Optional[str] = None
    n: Optional[int] = None
    M: Optional[List[List]] = None
    q: Optional[List] = None
    d: Optional[List] = None
    expect_exit: bool = False


def _solution_z_w(tabl):
    n = tabl.n
    z = [tabl.solution[i] for i in range(1, n + 1)]
    w = [tabl.solution[i] for i in range(n + 1, 2 * n + 1)]
    return z, w


def _check_complementarity(z, w):
    for i in range(len(z)):
        prod = z[i] * w[i]
        assert prod == 0, f"Complementarity failed at i={i}: z_i*w_i = {prod}"


def _check_feasibility(M, q, z, w):
    n = len(q)
    M_arr = np.array([[float(M[i][j]) for j in range(n)] for i in range(n)])
    z_arr = np.array([float(z[i]) for i in range(n)])
    q_arr = np.array([float(q[i]) for i in range(n)])
    w_expected = M_arr @ z_arr + q_arr
    w_actual = np.array([float(w[i]) for i in range(n)])
    np.testing.assert_allclose(w_actual, w_expected, rtol=1e-9, atol=1e-9)


def _check_nonnegativity(z, w):
    for i, val in enumerate(z):
        assert val >= 0, f"z[{i}] = {val} < 0"
    for i, val in enumerate(w):
        assert val >= 0, f"w[{i}] = {val} < 0"


@pytest.fixture
def lcp_from_example_file(example_lcp_path):
    m = lcp(example_lcp_path)
    return m


@pytest.fixture
def hand_crafted_2x2_lcp():
    # simple 2x2: M = I, q = (-1,-1). solution should be z=(1,1) w=(0,0)
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
    # no solution so lemke exits with 1
    content = """n= 2
M= 0 0 0 0
q= -1 -1
d= 1 1
"""
    return temp_lcp_file(content)


def test_lcp_from_example_file_solves(lcp_from_example_file, capsys):
    # run lemke on the example file and check the 3 conditions
    m = lcp_from_example_file
    tabl = tableau(m)
    tabl.runlemke(verbose=False, z0=False, silent=False)
    capsys.readouterr()  # consume stdout

    tabl.createsol()
    z, w = _solution_z_w(tabl)
    n = m.n
    assert len(z) == n and len(w) == n

    _check_complementarity(z, w)
    _check_feasibility(m.M, m.q, z, w)
    _check_nonnegativity(z, w)


def test_hand_crafted_2x2_lcp_known_solution(hand_crafted_2x2_lcp, capsys):
    # we know the answer so just check it matches
    m = hand_crafted_2x2_lcp
    tabl = tableau(m)
    tabl.runlemke(verbose=False, z0=False, silent=False)
    capsys.readouterr()

    tabl.createsol()
    z, w = _solution_z_w(tabl)

    _check_complementarity(z, w)
    _check_feasibility(m.M, m.q, z, w)
    _check_nonnegativity(z, w)

    assert z[0] == fractions.Fraction(1) and z[1] == fractions.Fraction(1)
    assert w[0] == fractions.Fraction(0) and w[1] == fractions.Fraction(0)


def test_runlemke_ray_termination_raises_system_exit(ray_termination_lcp):
    # when there's no solution it calls exit(1)
    m = lcp(ray_termination_lcp)
    tabl = tableau(m)
    with pytest.raises(SystemExit) as exc_info:
        tabl.runlemke(verbose=False, silent=False)
    assert exc_info.value.code == 1


def test_lcp_construct_from_integer():
    # basic test for lcp(n) making empty LCP
    n = 3
    m = lcp(n)
    assert m.n == n
    assert len(m.M) == n and len(m.M[0]) == n
    assert len(m.q) == n and len(m.d) == n


def test_lcp_solution_satisfies_conditions_cases(lcp_from_example_file, hand_crafted_2x2_lcp, capsys):
    # same checks on both the example and the hand made one
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
    # wrong format = too few numbers after M=
    content = "n= 2\nM= 1 2\nq= -1 -1\nd= 1 1\n"
    path = temp_lcp_file(content)
    with pytest.raises(SystemExit) as exc_info:
        lcp(path)
    assert exc_info.value.code == 1
