import pytest
from rkstiff.util.solver_type import SolverType


# Create minimal dummy solver classes that mimic BaseSolverAS and BaseSolverCS
# without importing the full solver stack.

class DummyAS:
    """Mimic an adaptive-step solver (inherits BaseSolverAS)."""
    pass


class DummyCS:
    """Mimic a constant-step solver (inherits BaseSolverCS)."""
    pass


# Monkeypatch the BaseSolverAS / BaseSolverCS types inside solver_type.from_solver
@pytest.fixture
def patch_solver_bases(monkeypatch):
    from rkstiff import solveras, solvercs

    # Replace BaseSolverAS / BaseSolverCS with our dummy classes
    monkeypatch.setattr(solveras, "BaseSolverAS", DummyAS)
    monkeypatch.setattr(solvercs, "BaseSolverCS", DummyCS)


# ======================================================================
# Enum Member Tests
# ======================================================================

def test_solver_type_members():
    assert SolverType.CONSTANT_STEP.name == "CONSTANT_STEP"
    assert SolverType.ADAPTIVE_STEP.name == "ADAPTIVE_STEP"


def test_solver_type_aliases():
    assert SolverType.CS is SolverType.CONSTANT_STEP
    assert SolverType.AS is SolverType.ADAPTIVE_STEP


def test_solver_type_str():
    assert str(SolverType.CONSTANT_STEP) == "Constant Step"
    assert str(SolverType.ADAPTIVE_STEP) == "Adaptive Step"


# ======================================================================
# from_solver() Auto-detection Tests
# ======================================================================

def test_from_solver_adaptive_step(patch_solver_bases):
    solver = DummyAS()
    assert SolverType.from_solver(solver) == SolverType.ADAPTIVE_STEP


def test_from_solver_constant_step(patch_solver_bases):
    solver = DummyCS()
    assert SolverType.from_solver(solver) == SolverType.CONSTANT_STEP


def test_from_solver_invalid_type(patch_solver_bases):
    class NotASolver:
        pass

    with pytest.raises(TypeError):
        SolverType.from_solver(NotASolver())