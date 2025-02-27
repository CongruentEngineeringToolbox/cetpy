"""
Generic Decentralised Solver
============================

This file specifies the base Solver class which defines the base structure of the decentralised solver architecture.
"""

from __future__ import annotations

from typing import List, Any, Dict
from copy import deepcopy
from time import perf_counter
from os import mkdir
from os.path import join, isdir
import uuid

import cetpy
from cetpy.Modules.SysML import ValuePrinter
from cetpy.Modules.Report import ReportSolver
from cetpy.Modules.Utilities.Labelling import name_2_display


class Solver:
    """Decentralised Solver of the Congruent Engineering Toolbox."""

    __slots__ = ['_recalculate', '_calculating', '_hold', '_resetting', '_parent', '_tolerance', 'report',
                 '_last_input', '__uuid__']

    print = ValuePrinter()

    input_keys: List[str] = []
    convergence_keys: List[str] = []
    _reset_dict: Dict[str, Any] = {}
    __parent_solve_priority__: bool = False

    def __init__(self, parent, tolerance: float = None):
        self.__uuid__ = uuid.uuid4()
        self._recalculate = True
        self._calculating = False
        self._resetting = False
        self._last_input = []
        self._hold = 0
        self._parent = None
        self.parent = parent
        self._tolerance = 0
        if tolerance is None and self.parent is not None:
            tolerance = self.parent.tolerance
        self.tolerance = tolerance
        self.report = ReportSolver(parent=self)

    # region System References
    @property
    def name(self) -> str:
        """Solver name."""
        return type(self).__name__

    @property
    def long_name(self) -> str:
        """Solver name with owning block name."""
        return self.parent.name_display + ' ' + self.name

    @property
    def name_display(self) -> str:
        """Return a display formatted version of the block name."""
        return name_2_display(self.name)

    @property
    def parent(self):
        """Solver owner block."""
        return self._parent

    @parent.setter
    def parent(self, val) -> None:
        val_initial = self._parent
        if val_initial == val:
            return
        self._parent = val
        if val_initial is not None:
            val_initial.solvers.remove(self)
            val_initial.reset()
        if val is not None and self not in val.solvers:
            val.solvers += [self]
        self.reset()

    def replace(self, val: Solver) -> None:
        """Replace this solver with a new solver for the parent."""
        parent = self.parent
        self.parent = None
        val.parent = parent

    def copy(self) -> Solver:
        """Return a copy of this solver without block references."""
        solver_copy = deepcopy(self)
        solver_copy.parent = None
        return solver_copy

    @property
    def directory(self) -> str | None:
        """Return the save directory of the block. If a parent block is defined, the folder is defined within the
        folder of the parent block. If no parent block is defined, the folder is defined within the session
        directory. If the session directory is not defined, no folder will be defined and saving is disabled. If a
        directory is defined but does not yet exist, a directory will be created."""
        if self.parent is not None:
            directory = join(self.parent.directory, self.name)
        elif cetpy.active_session.directory is not None:
            directory = join(cetpy.active_session.directory, self.name)
        else:
            return None
        if not isdir(directory):
            mkdir(directory)
        return directory
    # endregion

    # region Interface Functions
    def reset(self, parent_reset: bool = True) -> None:
        """Tell the solver to resolve before the next value output."""
        if not self._resetting:
            self._resetting = True
            self.reset_self()
            # Reset parent instance if desired
            if parent_reset and self.parent is not None:
                self.parent.reset()
            self._resetting = False

    def reset_self(self) -> None:
        """Reset just the solver parameters."""
        self._recalculate = True
        # Reset all local attributes to the desired reset value
        for key, val in self._reset_dict.items():
            self.__setattr__(key, val)

    def hard_reset(self, convergence_reset: bool = False) -> None:
        """Reset the in progress solver flags and call a normal reset."""
        self._resetting = False
        self._calculating = False
        self.reset()
        self._last_input = []
        if convergence_reset:
            for key in self.convergence_keys:
                self.__deep_setattr__(key, None)

    def __deep_getattr__(self, name: str) -> Any:
        """Get value from block or its parts, solvers, and ports."""
        if '.' not in name:
            return self.__getattribute__(name)
        else:
            name_split = name.split('.')
            return self.__getattribute__(name_split[0]).__deep_getattr__('.'.join(name_split[1:]))

    def __deep_setattr__(self, name: str, val: Any) -> None:
        """Set value on block or its parts, solvers, and ports."""
        if '.' not in name:
            return self.__setattr__(name, val)
        else:
            name_split = name.split('.')
            self.__getattribute__(name_split[0]).__deep_setattr__('.'.join(name_split[1:]), val)

    def __deep_get_vp__(self, name: str) -> Any:
        """Get value property from block or its parts, solvers, and ports."""
        if '.' not in name:
            return getattr(type(self), name)
        else:
            name_split = name.split('.')
            return self.__getattribute__(name_split[0]).__deep_get_vp__('.'.join(name_split[1:]))
    # endregion

    # region Solver Flags
    @property
    def solved(self) -> bool:
        """Return bool if the solver is solved.

        If the recalculate flag is True and input keys are specified the function further calls Solver.necessary to
        perform an additional check if a resolve is actually necessary using the last stored solve input parameters.
        """
        return not self._recalculate or not self.necessary

    # region Necessity Test
    @property
    def necessary(self) -> bool:
        """Return bool if a rerun in necessary. If no input keys are specified, the function always returns True as
        it cannot verify the inputs are still the same. The solved check still prioritises the recalculate flag set
        by the post_solver."""
        input_keys = self.input_keys
        last_input = self._last_input
        if len(input_keys) == 0 or len(last_input) == 0:
            return True
        for val_last, key in zip(last_input, input_keys):
            vp = self.__deep_get_vp__(key)
            val_new = self.__deep_getattr__(key)
            if vp.__reset_necessary(self, val_new, val_last):
                # break on first indication that a resolve is necessary
                return True
        if not self.calculating:  # Shortcut for next call
            self._recalculate = False
        return False

    def __get_input__(self) -> list:
        """Pull and return all current input properties."""
        return [self.__deep_getattr__(k) for k in self.input_keys]

    def __get_input_sensitivity__(self) -> List[float]:
        """Pull and return all current input properties."""
        return [self.__deep_get_vp__(k).necessity_test for k in self.input_keys]

    def _write_last_input(self) -> None:
        """Write current input properties to _last_input attribute.

        This function is called as part of the pre-run. It's recommended that if the solver is recursive,
        the function is also called once at the start of every solver loop.
        """
        self._last_input = self.__get_input__()
    # endregion

    @property
    def calculating(self) -> bool:
        """Return bool if the solver is currently running."""
        return self._calculating

    @property
    def solved_calculating(self) -> bool:
        """Return bool if the solver is solved or currently running."""
        return self.solved or self.calculating

    @property
    def hold(self) -> bool:
        """Current hold status. Hold enables an additional user input to enable and disable the automatic run of
        individual solvers. The hold is stored as an integer enabling nested holds throughout the system.

        See Also
        --------
        Solver.raise_hold
        Solver.lower_hold
        """
        return bool(self._hold)

    @hold.setter
    def hold(self, val: (bool, int)) -> None:
        self._hold = int(val)

    def raise_hold(self) -> None:
        """Raise the hold level by one."""
        self._hold += 1

    def lower_hold(self) -> None:
        """Lower the hold level by one."""
        self._hold += 1

    def force_solve(self) -> None:
        """Run the solver, regardless of the current solver state."""
        self._recalculate = True  # Ensure even with parent priority the solver is rerun regardless of checks in
        # the _solve function.
        if self.parent.__detailed_debug__:
            name = self.long_name
            logger = self.parent.__logger__
            logger.debug(f"Starting {name}.")
            t1 = perf_counter()

            self._solve()
            logger.debug(f"Finished {name} in {perf_counter() - t1} s.")
        else:
            self._solve()

    def solve(self) -> None:
        """Run the solver if necessary and allowed.

        The solver will run if the solver is not already solved, the solver is not currently already running,
        and the solver does not currently have an enabled hold condition.
        """
        if not self.solved_calculating and not self.hold:
            self.force_solve()
    # endregion

    # region Solver Functions
    def _pre_solve(self) -> None:
        """Conduct standardised pre-run of the solver."""
        if self.__parent_solve_priority__ and self.parent.parent is not None:
            # Ensure that this solver is only run as part of the larger solver.
            self.parent.parent.solve_self()
        self._calculating = True
        self._write_last_input()

    def _solve_function(self) -> None:
        """This is the actual solve function of the solver."""
        raise NotImplementedError

    def _post_solve(self) -> None:
        """Conduct standardised post-run of the solver."""
        self._calculating = False
        self._recalculate = False

    def _solve(self) -> None:
        """Private solve function combining the pre-, core-, and post-solve functions."""
        self._pre_solve()
        if self.__parent_solve_priority__ and self.solved:
            # This solver was solved as part of a solver of a larger system, started by the call in the _pre_solve
            # function of this solver. Checks for the parent priority first since the bool check is much less costly
            # than the necessity evaluation of the solved function.
            return
        self._solve_function()
        self._post_solve()
    # endregion

    # region Input Properties
    @property
    def tolerance(self) -> float:
        """Solver tolerance."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val: float) -> None:
        if val < self._tolerance:
            self.reset()
        self._tolerance = val
    # endregion
