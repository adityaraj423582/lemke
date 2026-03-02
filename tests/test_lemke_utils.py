# utils tests - stripcomments, tofraction, tovector, tomatrix
import fractions

import numpy as np
import pytest

from lemke import utils


def test_stripcomments_ignores_blank_lines_and_comments(temp_lcp_file):
    content = """
# comment line
n= 2
% also comment
M= 1 0 0 1
q= -1 -1
d= 1 1
"""
    path = temp_lcp_file(content)
    lines = utils.stripcomments(path)
    assert "# comment line" not in lines
    assert "% also comment" not in lines
    assert "" not in lines
    assert "n= 2" in lines
    assert "M= 1 0 0 1" in lines
    assert "q= -1 -1" in lines
    assert "d= 1 1" in lines


def test_stripcomments_strips_leading_trailing_blanks(temp_lcp_file):
    content = "  n= 1  \n  M= 0  \n  q= 0  \n  d= 1  \n"
    path = temp_lcp_file(content)
    lines = utils.stripcomments(path)
    assert all(line == line.strip() for line in lines)
    assert "  n= 1  " not in lines
    assert any("n=" in line for line in lines)


def test_stripcomments_commentchars(temp_lcp_file):
    # # % * all count as comment
    path = temp_lcp_file("# a\n% b\n* c\n1 2\n")
    lines = utils.stripcomments(path)
    assert lines == ["1 2"]


def test_tofraction_integer():
    assert utils.tofraction(3) == fractions.Fraction(3)
    assert utils.tofraction(-2) == fractions.Fraction(-2)


def test_tofraction_string_integer():
    assert utils.tofraction("5") == fractions.Fraction(5)


def test_tofraction_decimal():
    # uses decimals global so 1.5 -> 3/2
    assert utils.tofraction(1.5) == fractions.Fraction(3, 2)
    assert utils.tofraction("1.5") == fractions.Fraction(3, 2)


def test_tofraction_fraction_string():
    assert utils.tofraction("80/9") == fractions.Fraction(80, 9)
    assert utils.tofraction("57/8") == fractions.Fraction(57, 8)


def test_tovector():
    words = ["10", "20", "30", "40"]
    v = utils.tovector(2, words, 1)
    assert v.shape == (2,) or len(v) == 2
    assert v[0] == fractions.Fraction(20)
    assert v[1] == fractions.Fraction(30)


def test_tovector_fractions():
    words = ["1/2", "0.25", "3"]
    v = utils.tovector(3, words, 0)
    assert v[0] == fractions.Fraction(1, 2)
    assert v[1] == fractions.Fraction(1, 4)
    assert v[2] == fractions.Fraction(3)


def test_tomatrix():
    words = ["1", "0", "0", "1", "0", "0"]
    C = utils.tomatrix(2, 3, words, 0)
    assert C.shape == (2, 3)
    assert C[0][0] == fractions.Fraction(1) and C[0][1] == fractions.Fraction(0)
    assert C[1][2] == fractions.Fraction(0)


def test_tomatrix_fraction_entries():
    words = ["80/9", "1.0", "0", "0"]
    C = utils.tomatrix(2, 2, words, 0)
    assert C[0][0] == fractions.Fraction(80, 9)
    assert C[0][1] == fractions.Fraction(1)
    assert C[1][0] == fractions.Fraction(0) and C[1][1] == fractions.Fraction(0)


def test_towords():
    lines = ["n= 2", "M= 1 0", "0 1"]
    words = utils.towords(lines)
    assert words == ["n=", "2", "M=", "1", "0", "0", "1"]
