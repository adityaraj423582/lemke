# Lemke Package — Improvement Proposal (GSoC)

**Current state.** The package provides a pure-Python implementation of Lemke’s algorithm for linear complementarity problems (LCP) and Nash equilibria in bimatrix games. It is functional and algorithmically sound but has no tests, ad-hoc CLI parsing, broken constructors in two classes, and no README or API documentation. The codebase would benefit from testing, a clear CLI, a fixed public API, and light modernization.

**Proposed improvements** (in priority order):

---

### 1. Add test suite *(highest priority)*

**What:** Introduce pytest and a `tests/` directory. Add tests for: (a) LCP solving using the existing example file and at least one small hand-checked case; (b) bimatrix/Lemke–Howson using the example game and a known equilibrium; (c) file parsing and fraction conversion in `utils`. Add a minimal CI workflow (e.g. GitHub Actions) that runs the suite.

**Why:** There are no automated tests today. Any refactor (including the fixes below) risks regressions. Tests are a prerequisite for confident changes and for contributors.

**Estimate:** 12–16 hours.

---

### 2. Replace CLI with Click

**What:** Replace the current `sys.argv` parsing in `lemke.py` and `bimatrix.py` with Click. Define clear commands and options (e.g. `--verbose`, `--silent`, `--output`) and wire them to the existing solvers. Pass options into the library as arguments or a small config object instead of global variables so the core logic stays testable.

**Why:** Hand-rolled parsing is brittle and inconsistent. Click gives standard `--help`, validation, and a maintainable single place for CLI behavior. Removing CLI-related globals also simplifies testing and reuse.

**Estimate:** 6–8 hours.

---

### 3. Fix broken API (duplicate `__init__`)

**What:** In `bimatrix.py`, both `payoffmatrix` and `bimatrix` define two `__init__` methods; the second overwrites the first, so constructors like `payoffmatrix(m, n)` and `bimatrix(m, n)` do not work. Refactor to a single `__init__` per class that supports both signatures (e.g. from file/path vs. dimensions or matrix) via type checks or optional arguments.

**Why:** The current API is misleading and prevents valid use cases (e.g. building a game programmatically). Fixing it is a small, high-impact change that unblocks library users and keeps the public interface coherent.

**Estimate:** 4–6 hours.

---

### 4. Add README and docstrings

**What:** Add a README with a short description, install instructions, and usage examples for both entry points. Document the LCP and game file formats (with reference to the example files). Add brief docstrings to all public classes and main functions (purpose, arguments, return value, and main exceptions where relevant).

**Why:** The project references a README in pyproject.toml but the file is missing; new users have no guidance. Docstrings make the package usable as a library and reduce repeated questions about behavior and file formats.

**Estimate:** 5–7 hours.

---

### 5. Add type hints and modernize code

**What:** Add type annotations to public functions and classes. Apply a single style/formatting standard (e.g. Ruff or Black) and add a minimal config to the repo. Optionally introduce a module-level logger instead of ad-hoc `print`/`printout` for progress and diagnostics. Loosen strict dependency upper bounds where safe (e.g. `numpy>=2.2` instead of `<2.3`) and add a dev extra for pytest and Click.

**Why:** Type hints and consistent style improve readability and catch simple bugs; they also signal that the project is maintained. Small modernization steps make future contributions easier.

**Estimate:** 8–10 hours.

---

### Total time estimate

**35–47 hours** (roughly one full GSoC coding period, depending on scope and familiarity with the codebase).

---

### Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Refactors break existing behavior | Implement the test suite first and run it after each change; add at least one LCP and one game test before touching CLI or API. |
| Scope creep (e.g. full Sphinx docs) | Limit scope to README + docstrings; defer generated API docs to a follow-up. |
| Dependency or Python-version issues | Pin minimum versions only where necessary; run CI on 2–3 Python versions (e.g. 3.9, 3.11) to catch compatibility problems early. |
| Duplicate `__init__` fix changes behavior | Use type checks and tests to preserve current “from file” and “from matrix” behavior; add tests for new “from dimensions” paths. |

---

*Proposal for GSoC mentor review. Lemke package: LCP solver and bimatrix Nash equilibria (Lemke–Howson).*
