"""
Solver type enumeration and improved testing utilities.

This module defines SolverType enum to distinguish between constant-step (CS)
and adaptive-step (AS) solvers, providing type-safe solver categorization.

Compatible with Python 3.9+.
"""

from __future__ import annotations
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rkstiff.solver import BaseSolver


# ======================================================================
# Solver Type Enumeration
# ======================================================================
class SolverType(Enum):
    """
    Enumeration of solver stepping strategies.
    
    Attributes
    ----------
    CONSTANT_STEP : enum member
        Constant-step (fixed time step) solvers.
        Examples: ETD4, ETD5, IF4
    ADAPTIVE_STEP : enum member
        Adaptive-step (variable time step) solvers.
        Examples: ETD35, ETD45
    """
    CONSTANT_STEP = auto()
    ADAPTIVE_STEP = auto()
    
    # Convenient aliases
    CS = CONSTANT_STEP
    AS = ADAPTIVE_STEP
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        return self.name.replace('_', ' ').title()
    
    @classmethod
    def from_solver(cls, solver: BaseSolver) -> SolverType:
        """
        Automatically detect solver type from a solver instance.
        
        Parameters
        ----------
        solver : BaseSolver
            A solver instance (either BaseSolverCS or BaseSolverAS subclass).
            
        Returns
        -------
        SolverType
            The detected solver type.
            
        Raises
        ------
        TypeError
            If solver type cannot be determined.
            
        Examples
        --------
        >>> from rkstiff.etd4 import ETD4
        >>> solver = ETD4(lin_op, nl_func)
        >>> SolverType.from_solver(solver)
        <SolverType.CONSTANT_STEP: 1>
        """
        # Import here to avoid circular dependencies
        from rkstiff.solvercs import BaseSolverCS
        from rkstiff.solveras import BaseSolverAS
        
        if isinstance(solver, BaseSolverAS):
            return cls.ADAPTIVE_STEP
        elif isinstance(solver, BaseSolverCS):
            return cls.CONSTANT_STEP
        else:
            raise TypeError(
                f"Cannot determine solver type for {type(solver).__name__}. "
                "Solver must inherit from BaseSolverCS or BaseSolverAS."
            )
