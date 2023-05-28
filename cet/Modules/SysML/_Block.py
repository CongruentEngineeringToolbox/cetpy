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

import re
from typing import List


def name_2_abbreviation(name: str) -> str:
    """Return a sensible abbreviation for a component name."""
    name_split = name.split('_')
    if len(name_split) == 1:
        abbreviation = name_split[0][:3].upper()
    else:
        abbreviation = "".join([n[0].upper() for n in name_split[:3]])
    suffix = re.findall(r'^[A-Za-z0-9]*?([0-9]+)$', name)
    if len(suffix) > 0:
        abbreviation += suffix[0]
    return abbreviation


class Block:
    """SysML Block element."""

    __slots__ = ['_resetting', '_bool_parent_reset',
                 'name', 'abbreviation', 'parent', 'tolerance',
                 'parts', 'ports', 'requirements']

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
        # endregion

    # region Solver Functions
    def reset(self) -> None:
        """Tell the instance to resolve before the next output."""
        pass
    # endregion
