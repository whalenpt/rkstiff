API Reference
=============

This page provides a complete reference for the rkstiff package, organized by functionality.
   
Core Solver
-----------

The main solver interfaces for solving stiff differential equations.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
   rkstiff.solver
   rkstiff.solveras
   rkstiff.solvercs

Exponential Time Differencing Methods
--------------------------------------

ETD (Exponential Time Differencing) methods for stiff differential equations.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
   rkstiff.etd
   rkstiff.etd34
   rkstiff.etd35
   rkstiff.etd4
   rkstiff.etd5

Integrating Factor Methods
---------------------------

IF (Integrating Factor) methods for stiff differential equations.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
   rkstiff.if34
   rkstiff.if4
   rkstiff.if45dp

Supporting Modules
------------------

Utilities and helper modules for numerical computations.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
   rkstiff.derivatives
   rkstiff.grids
   rkstiff.models
   rkstiff.transforms

Utilities
---------

Additional utility functions and helpers.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:
   
   rkstiff.util.loghelper
