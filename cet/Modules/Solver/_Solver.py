"""
Generic Decentralised Solver
============================

This file specifies the base Solver class which defines the base structure
of the decentralised solver architecture.
"""

from typing import List


class Solver:
    """Decentralised Solver of the Congruent Engineering Toolbox."""
    __slots__ = ['_recalculate', '_calculating', '_hold', '_resetting',
                 '_name', 'parent', 'convergence_keys']

    def __init__(self):
        self._recalculate = False
        self._calculating = False
        self._resetting = False
        self._hold = 0
        self.convergence_keys: List[str] = []

    # region Interface Functions
    def __set_name__(self, instance, name):
        self._name = name
        self.parent = instance

    def reset(self, parent_reset: bool = True) -> None:
        """Tell the solver to resolve before the next value output."""
        if not self._resetting:
            self._resetting = True
            self._recalculate = True
            # Reset parent instance if desired
            if parent_reset:
                self.parent.reset()
            self._resetting = False

    def hard_reset(self, convergence_reset: bool = False) -> None:
        """Reset the in progress solver flags and call a normal reset."""
        self._resetting = False
        self._calculating = False
        self.reset()
        if convergence_reset:
            for key in self.convergence_keys:
                self.__setattr__(key, None)
    # endregion

    # region Solver Flags
    @property
    def solved(self) -> bool:
        """Return bool if the solver is solved."""
        return not self._recalculate

    @property
    def calculating(self) -> bool:
        """Return bool if the solver is currently running."""
        return self._calculating

    @property
    def solved_calculating(self) -> bool:
        """Return bool if the solver is solved or currently running."""
        return self.solved or self.calculating

    def force_solve(self) -> None:
        """Run the solver, regardless of the current solver state."""
        self._solve()
        # ToDo: Add timing and logging.

    def solve(self) -> None:
        """Run the solver if necessary."""
        if not self.solved_calculating:
            self.force_solve()
    # endregion

    # region Solver Functions
    def _pre_solve(self) -> None:
        """Conduct standardised pre-run of the solver."""
        self._calculating = True

    def _solve_function(self) -> None:
        """This is the actual solve function of the solver."""
        raise NotImplementedError

    def _post_solve(self) -> None:
        """Conduct standardised post-run of the solver."""
        self._calculating = False
        self._recalculate = False

    def _solve(self) -> None:
        """Private solve function combining the pre-, core-, and post-solve
        functions."""
        self._pre_solve()
        self._solve_function()
        self._post_solve()
    # endregion
