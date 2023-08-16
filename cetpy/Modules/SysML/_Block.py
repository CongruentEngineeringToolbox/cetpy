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

from typing import List, Any

import cetpy
from cetpy.Modules.Utilities.Labelling import name_2_abbreviation, \
    name_2_display
from cetpy.Modules.SysML import ValuePrinter
from cetpy.Modules.Solver import Solver
from cetpy.Modules.Report import Report


class Block:
    """SysML Block element."""

    __slots__ = ['_resetting', 'name', 'abbreviation', '_parent',
                 '_tolerance', '_logger',
                 'parts', 'ports', 'requirements', 'solvers',
                 '_get_init_parameters', '__init_kwargs__', '__dict__']

    __init_parameters__ = []
    __init_parts__ = []
    _reset_dict = {}
    _hard_reset_dict = {}
    _bool_parent_reset = True

    print = ValuePrinter()

    def __init__(self, name: str, abbreviation: str = None,
                 parent: Block = None, tolerance: float = None, **kwargs
                 ) -> None:
        # region Logging
        if cetpy.active_session is not None:
            self._logger = cetpy.active_session.logger
        else:
            self._logger = None
        # endregion

        # region Solver Flags
        self._resetting = False
        self._parent = None
        # endregion

        # region Attributes
        self.name = name
        if abbreviation is None:
            abbreviation = name_2_abbreviation(name)
        self.abbreviation = abbreviation
        self.parent = parent

        # region Config Loading
        session = cetpy.active_session

        def get_parent_name_list(block: Block) -> List[str]:
            """Return list of names of parent objects."""
            if block.parent is None:
                return [block.name]
            else:
                return get_parent_name_list(block.parent) + [block.name]
        parent_names = get_parent_name_list(self)

        def get_parent_class_list(block: Block) -> List[str]:
            """Return list of class names of parent objects."""
            if block.parent is None:
                return [block.__class__.__name__.lower()]
            else:
                return get_parent_name_list(block.parent) + [
                    block.__class__.__name__.lower()]
        class_names = get_parent_class_list(self)
        type_name = str(type(self))

        parameter_strings = [
            '.'.join(parent_names[i:]) for i in range(-len(parent_names), 0)
            ] + [
            '.'.join(class_names[i:]) for i in range(-len(class_names), 0)
            ] + [
            '.'.join(type_name[
                     type_name.find('Modules') + 8:-2].split('.')[:-2]).lower()
        ]

        config_keys = cetpy.active_session.config_manager.config_keys

        def parameter(key: str):
            """Return parameter setting for a given key with prioritisation.

            Source Prioritisation:
            1. kwargs
            2. session config
            3. default config

            Key Name Prioritisation:
            1. Chained block names from the highest parent to the block
            2. Block name
            3. Chained block class names from the highest parent to the block
            4. Block class name
            5. Key

            Example:
                Scenario: Setting the radius of the Moon Concordia of the
                Planet Mandalore:

                1. mandalore.concordia.radius
                2. concordia.radius
                3. Planet.Moon.radius
                4. Moon.radius
                5. radius

            Please note, for comparison to kwargs, all '.' separators are
            replaced by '_'.
            """
            key_strings = [ps + '.' + key for ps in parameter_strings] + [key]
            in_kwargs = [k.replace('.', '_') for k in key_strings
                         if k.replace('.', '_') in kwargs.keys()]
            in_config = [k for k in key_strings if k in config_keys]
            if len(in_kwargs) > 0:
                key_load = in_kwargs[0]
                source = 'kwargs'
                val = kwargs.get(key_load)
            elif len(in_config) > 0:
                key_load = in_config[0]
                source = 'config'
                val = session.parameter(key_load)
            else:
                return None
            if isinstance(val, str | float | int | None):
                self._logger.log(
                    15,
                    f"{key} loaded as {key_load} from {source}: {str(val)}")
            else:
                self._logger.log(
                    15, f"{key} loaded as {key_load} from {source}")
            return val
        self._get_init_parameters = parameter
        self.__init_kwargs__ = kwargs

        for key in self.__init_parameters__:
            self._resetting = True  # Avoid unnecessary resets.
            getattr(type(self), key).__set__(self, parameter(key))
            self._resetting = False
        # endregion
        # endregion

        # region References
        self.parts: List[Block] = []
        self.ports = []
        self.requirements = []
        self.solvers: List[Solver] = []
        self.report = Report(parent=self)

        # region Part Initialisation
        for key in self.__init_parts__:
            try:
                self.__setattr__(key, parameter(key + '_model'))
            except KeyError:
                self.__setattr__(key, None)
        # endregion
        # endregion

        # region Tolerance
        self._tolerance = 0
        if tolerance is None:
            tolerance = cetpy.active_session.parameter('tolerance')
        self.tolerance = tolerance
        # endregion

    # region Block References
    @property
    def parent(self) -> (Block, None):
        return self._parent

    @parent.setter
    def parent(self, val: Block) -> None:
        if val is None and self._parent is not None:
            try:
                self._parent.parts.remove(self)
            except ValueError:
                pass
        elif val is not None and self not in val.parts:
            val.parts += [self]
        self._parent = val

    def __deep_getattr__(self, name: str) -> Any:
        """Get value from block or its parts, solvers, and ports."""
        if '.' not in name:
            return self.__getattribute__(name)
        else:
            name_split = name.split('.')
            return self.__getattribute__(name_split[0]).__deep_getattr__(
                '.'.join(name_split[1:]))

    def __deep_setattr__(self, name: str, val: Any) -> None:
        """Set value on block or its parts, solvers, and ports."""
        if '.' not in name:
            return self.__setattr__(name, val)
        else:
            name_split = name.split('.')
            self.__getattribute__(name_split[0]).__deep_setattr__(
                '.'.join(name_split[1:]), val)

    def __call__(self, *args, **kwargs) -> None:
        return self.report()
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

            # Reset all local attributes to the desired reset value
            for key, val in self._reset_dict.items():
                self.__setattr__(key, val)

            # Reset parent instance if desired:
            if parent_reset is None:
                parent_reset = self._bool_parent_reset
            if parent_reset and self.parent is not None:
                self.parent.reset()

            self._resetting = False

    def hard_reset(self, convergence_reset: bool = False) -> None:
        """Reset all blocks including solver flags and intermediate values.

        This should only be used to get the program unstuck as it undermines
        recursion stops and deletes any progress made.
        """
        self._resetting = False
        for key, val in self._hard_reset_dict.items():
            self.__setattr__(key, val)
        for solver in self.solvers:
            solver.hard_reset(convergence_reset)
        self.reset()

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
        for s in [s for s in self.solvers if s.tolerance > val]:
            s.tolerance = val
        for p in [p for p in self.ports if p.tolerance > val]:
            p.tolerance = val
    # endregion

    # region User Output
    @property
    def name_display(self) -> str:
        """Return a display formatted version of the block name."""
        return name_2_display(self.name)
    # endregion
