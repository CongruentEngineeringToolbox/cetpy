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

from cet.Modules.Utilities.Labelling import name_2_abbreviation
from cet.Modules.Solver import Solver


class Block:
    """SysML Block element."""

    __slots__ = ['_resetting', '_bool_parent_reset',
                 'name', 'abbreviation', 'parent', 'tolerance',
                 'parts', 'ports', 'requirements', 'solvers']

    def __init__(self, name: str, abbreviation: str = None,
                 parent: Block = None, tolerance: float = None) -> None:
        # region Solver Flags
        self._resetting = False
        self._bool_parent_reset = False
        # endregion

        # region Attributes
        self.name = name
        if abbreviation is None:
            abbreviation = name_2_abbreviation(name)
        self.abbreviation = abbreviation
        self.parent = parent
        self.tolerance = tolerance
        # endregion

        # region References
        self.parts: List[Block] = []
        self.ports = []
        self.requirements = []
        self.solvers: List[Solver] = []
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
            if parent_reset:
                self.parent.reset()

            self._resetting = False
    # endregion
