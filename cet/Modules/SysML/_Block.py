"""
SysML Block
===========

This file implements a basic SysML Block.

References
----------
SysML Documentation:
    https://sysml.org/.res/docs/specs/OMGSysML-v1.4-15-06-03.pdf
"""

from __future__ import annotations

from typing import List

from cet.Modules.Utilities.Labelling import name_2_abbreviation, name_2_display
from cet.Modules.SysML import ValueProperty
from cet.Modules.Solver import Solver


class Block:
    """SysML Block element."""

    __slots__ = ['_resetting', '_bool_parent_reset',  'determination_tests',
                 'name', 'abbreviation', 'parent', '_tolerance',
                 'parts', 'ports', 'requirements', 'solvers']

    def __init__(self, name: str, abbreviation: str = None,
                 parent: Block = None, tolerance: float = 0) -> None:
        # region Solver Flags
        self._resetting = False
        self._bool_parent_reset = True
        self.determination_tests = {}
        # endregion

        # region Attributes
        self.name = name
        if abbreviation is None:
            abbreviation = name_2_abbreviation(name)
        self.abbreviation = abbreviation
        self.parent = parent
        if parent is not None and self not in parent.parts:
            parent.parts += [self]
        # endregion

        # region References
        self.parts: List[Block] = []
        self.ports = []
        self.requirements = []
        self.solvers: List[Solver] = []
        # endregion

        # region Tolerance
        self._tolerance = 0
        self.tolerance = tolerance
        # endregion

        # region Instance Passthrough to Value Properties
        for vp, n in [(getattr(type(self), n), n) for n in self.__dir__()
                      if isinstance(getattr(type(self), n), ValueProperty)]:
            vp.__set_name__(self, n)
        # endregion

    # region Solver Functions
    def reset(self, parent_reset: bool = None) -> None:
        """Tell the instance and sub-blocks to resolve before the next
        value output."""
        if not self._resetting:
            self._resetting = True
            # reset all sub solvers while not calling the instances own reset
            for s in self.solvers:
                s.reset(parent_reset=False)

            # Reset parent instance if desired:
            if parent_reset is None:
                parent_reset = self._bool_parent_reset
            if parent_reset and self.parent is not None:
                self.parent.reset()

            self._resetting = False

    @property
    def solved_self(self) -> bool:
        """Return bool if the block and its solvers are solved."""
        return True and (len(self.solvers) == 0
                         or all([s.solved for s in self.solvers]))

    @property
    def solved(self) -> bool:
        """Return bool if the block, its solvers, and its contained blocks
        are solved."""
        return self.solved_self and (len(self.parts) == 0
                                     or all([p.solved for p in self.parts]))

    def solve_self(self) -> None:
        """Solve the block and its solvers."""
        [s.solve() for s in self.solvers]

    def solve(self) -> None:
        """Solve the block, its solvers, and its parts."""
        self.solve_self()
        [p.solve() for p in self.parts]
    # endregion

    # region Input Properties
    @property
    def tolerance(self) -> float:
        """Block solver tolerances."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val: float) -> None:
        if val < self._tolerance:
            self.reset()
        self._tolerance = val
        for p in [p for p in self.parts if p.tolerance > val]:
            p.tolerance = val
        for s in [s for s in self.parts if s.tolerance > val]:
            s.tolerance = val
    # endregion

    # region User Output
    @property
    def name_display(self) -> str:
        """Return a display formatted version of the block name."""
        return name_2_display(self.name)
    # endregion
