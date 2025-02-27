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

import logging
from typing import List, Any, Dict, Callable

import networkx as nx
import uuid

import cetpy
from cetpy.Modules.Utilities.Labelling import name_2_abbreviation, name_2_display
from cetpy.Modules.Utilities.ModelStructureGraph import get_model_structure_graph
from cetpy.Modules.SysML import ValuePrinter
from cetpy.Modules.Solver import Solver
from cetpy.Modules.Report import ReportBlock
from cetpy.Modules.plotting import PlotDescriptor


class Block:
    """SysML Block element.

    This is the basic system block of SysML. It can contain parts, themselves Blocks. It can contain ports,
    which handle interfaces to other elements. The block can also contain references to solvers, which are a product
    of cetpy, not SysML, that handle a decentralised solver architecture for any slow computations.

    This base class contains functions that handle initialisation, loading from config files and keyword arguments,
    attach a session logger, and create sub parts according to model definitions.
    """

    __slots__ = ['_resetting', 'name', 'abbreviation', '_parent', '_tolerance', '_structure_graph',
                 '_composition_structure_graph', '_logger', 'parts', 'ports', 'requirements', 'solvers',
                 '_get_init_parameters', '__init_kwargs__', '__dict__']

    __init_parameters__: List[str] = []
    __init_parts__: List[str] = []
    _reset_dict: Dict[str, Any] = {}
    _hard_reset_dict: Dict[str, Any] = {'_structure_graph': None, '_composition_structure_graph': None}
    __fixed_parameters__: List[str] = []
    __default_parameters__: Dict[str, Any] = {}
    __plot_functions__: Dict[str, Callable] = {}
    __default_plot_x_axis__ = None
    _bool_parent_reset = True
    __detailed_debug__ = False

    print = ValuePrinter()
    plot = PlotDescriptor()
    directory = Solver.directory

    def __init__(self, name: str, abbreviation: str = None, parent: Block = None, tolerance: float = None, **kwargs
                 ) -> None:
        """Initialise a Block instance.

        Parameters
        ----------
        name
            A name for the block, used for visualizations, reports, and other user output.
        abbreviation: optional
            A shortened abbreviation of the name that can also be used for output functions where a long-name would
            be a prohibitive identifier. If none is given, the block predicts a sensible abbreviation.
        parent: optional
            Another block, which is to contain this block as a part. The bidirectional connection is made automatically.
        tolerance: optional
            A general tolerance for the block, its solvers, ports, and parts. If None is provided, takes the
            tolerance of the parent if available or from the active session configuration.
        """
        # region Logging
        if cetpy.active_session is not None:
            self._logger = cetpy.active_session.logger
        else:
            self._logger = None
        self.__uuid__ = uuid.uuid4()
        # endregion

        # region Solver Flags
        self._resetting = False
        self._parent = None
        self._structure_graph: nx.DiGraph | None = None
        self._composition_structure_graph: nx.DiGraph | None = None
        self._tolerance = 0  # Full initialisation at the end.
        # endregion

        # region Attributes
        self.name = name
        if abbreviation is None:
            abbreviation = name_2_abbreviation(name)
        self.abbreviation = abbreviation

        # region References
        self.parts: List[Block] = []
        self.ports = []
        self.requirements = []
        self.solvers: List[Solver] = []
        self.report = ReportBlock(parent=self)
        self.parent = parent
        # endregion

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
            '.'.join(type_name[type_name.find('Modules') + 8:-2].split('.')[:-2]).lower()
        ]

        config_keys = cetpy.active_session.config_manager.config_keys

        def get_key_strings(key_in: str) -> List[str]:
            """Return joined parameter strings with the key attached.

            Strings are sorted in order of complexity.
            """
            return [ps + '.' + key_in for ps in parameter_strings] + [key_in]

        def get_key_in_kwargs(key_in: str) -> List[str]:
            """Return joined key strings which are present in the keyword arguments."""
            key_strings = get_key_strings(key_in)
            return [k.replace('.', '_') for k in key_strings if k.replace('.', '_') in kwargs.keys()]

        def get_key_in_config(key_in: str) -> List[str]:
            """Return joined key strings which are present in the configs."""
            key_strings = get_key_strings(key_in)
            return [k.replace('.', '_') for k in key_strings if k.replace('.', '_') in config_keys]

        def parameter(key_in: str):
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

            Please note, for comparison all '.' separators are replaced by '_'.
            """
            in_kwargs = get_key_in_kwargs(key_in)
            in_config = get_key_in_config(key_in)
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
                self._logger.log(15, f"{key_in} loaded as {key_load} from {source}: {str(val)}")
            else:
                self._logger.log(15, f"{key_in} loaded as {key_load} from {source}")
            return val
        self._get_init_parameters = parameter
        self.__init_kwargs__ = kwargs
        cls = type(self)

        self._resetting = True  # Avoid unnecessary resets.
        for key in self.__init_parameters__:
            getattr(cls, key).__set__(self, parameter(key))
        self._resetting = False

        # Check default parameters before running the determination tests.
        for key, value in self.__default_parameters__.items():
            vp = getattr(cls, key)
            d_test = vp.determination_test
            if (d_test is not None and d_test.under_determined(self)) or (d_test is None and not vp.fixed(self)):
                vp.__set__(self, value)

        # Determination test is disabled on initial initialisation as the parameters are initialised in sequence,
        # and it would fail regardless on the first parameters.
        d_tests = []
        for key in [k for k in self.__init_parameters__ if getattr(type(self), k).determination_test is not None]:
            d_test = getattr(cls, key).determination_test
            if d_test not in d_tests:
                d_tests += [d_test]
                d_test.test(self)
        # endregion

        # region Part Initialisation
        for key in self.__init_parts__:
            try:
                self.__setattr__(key, parameter(key + '_model'))
            except KeyError:
                self.__setattr__(key, None)
        # endregion
        # endregion

        # region Tolerance
        if tolerance is None:
            if self.parent is not None:
                try:
                    tolerance = self.parent.tolerance
                except AttributeError:
                    tolerance = cetpy.active_session.parameter('tolerance')
            else:
                tolerance = cetpy.active_session.parameter('tolerance')
        self.tolerance = tolerance
        # endregion

    # region Block References
    @property
    def parent(self) -> (Block, None):
        return self._parent

    @parent.setter
    def parent(self, val: Block) -> None:
        if val is not self._parent and self._parent is not None:
            try:
                self._parent.parts.remove(self)
                self._parent.reset()
            except ValueError:
                pass
        if val is not None and self not in val.parts:
            val.parts += [self]
        self._parent = val
        self.reset()
        self.__reset_structure__()

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

    def __call__(self, *args, **kwargs) -> None:
        return self.report()

    @property
    def logger(self) -> logging.Logger:
        """Return the appropriate logger for the block."""
        return self._logger

    @property
    def structure_graph(self) -> nx.DiGraph:
        """Entire model structure graph."""
        if self.parent is not None:
            return self.parent.structure_graph
        else:
            if self._structure_graph is None:
                self._structure_graph = get_model_structure_graph(
                    self, include_solvers=True, include_ports=True, include_parts=True, include_references=True,
                    include_value_properties=False, include_connected_blocks=False, include_instances=True)
            return self._structure_graph

    @property
    def composition_structure_graph(self) -> nx.DiGraph:
        """Entire model structure graph."""
        if self.parent is not None:
            return self.parent.composition_structure_graph
        else:
            if self._composition_structure_graph is None:
                self._composition_structure_graph = get_model_structure_graph(
                    self, include_solvers=True, include_ports=True, include_parts=True, include_references=False,
                    include_value_properties=False, include_connected_blocks=False, include_instances=True)

                # Clear all others in internal structure. Not called in structure reset to avoid unnecessary generation.
                parts_full_depth = [node for node in self._composition_structure_graph.nodes.values()
                                    if node['type'] == 'block' and node['instance'] != self]
                for p in parts_full_depth:
                    # noinspection PyProtectedMember
                    p['instance']._structure_graph = None
                    # noinspection PyProtectedMember
                    p['instance']._composition_structure_graph = None
            return self._composition_structure_graph
    # endregion

    # region Solver Functions
    # region Reset
    def reset(self, parent_reset: bool = None) -> None:
        """Tell the instance and sub-blocks to resolve before the next value output."""
        if not self._resetting:
            self._resetting = True
            # reset all subcomponents while not calling the instances own reset
            for s in self.solvers:
                s.reset(parent_reset=False)
            for p in self.parts:
                p.reset(parent_reset=False)
            for p in self.ports:
                p.reset()

            self.reset_self()

            # Reset parent instance if desired:
            if parent_reset is None:
                parent_reset = self._bool_parent_reset
            if parent_reset and self.parent is not None:
                self.parent.reset()

            self._resetting = False

    def reset_self(self) -> None:
        """Reset own reset parameters without calling any changes on attached blocks, ports, solvers."""
        # Reset all local attributes to the desired reset value
        for key, val in self._reset_dict.items():
            self.__setattr__(key, val)

    def hard_reset(self, convergence_reset: bool = False) -> None:
        """Reset all blocks including solver flags and intermediate values.

        This should only be used to get the program unstuck as it undermines recursion stops and deletes any progress
        made.
        """
        self._resetting = True  # Set True while resetting parts
        for key, val in self._hard_reset_dict.items():
            self.__setattr__(key, val)
        for solver in self.solvers:
            solver.hard_reset(convergence_reset)
        for p in self.parts:
            p.hard_reset(convergence_reset)
        self._resetting = False
        self.reset()

    def __reset_structure__(self) -> None:
        """Reset the structure graphs to be re-build on next request. Propagates through the full system."""
        if self.parent is not None:
            self.parent.__reset_structure__()
            return

        self._structure_graph = None
        self._composition_structure_graph = None
    # endregion

    # region Solve
    @property
    def solved_self(self) -> bool:
        """Return bool if the block and its solvers are solved."""
        return True and (len(self.solvers) == 0 or all([s.solved for s in self.solvers]))

    @property
    def solved(self) -> bool:
        """Return bool if the block, its solvers, and its contained blocks
        are solved."""
        return self.solved_self and (len(self.parts) == 0 or all([p.solved for p in self.parts]))

    def solve_self(self) -> None:
        """Solve the block and its solvers."""
        [s.solve() for s in self.solvers]

    def solve(self) -> None:
        """Solve the block, its solvers, and its parts."""
        self.solve_self()
        [p.solve() for p in self.parts]
    # endregion

    # region Fixed
    @property
    def fixed_self(self) -> bool:
        """Bool if this block, regardless of its parts is fixed."""
        return all([getattr(type(self), vp).fixed(self) for vp in self.__fixed_parameters__])

    @fixed_self.setter
    def fixed_self(self, val: bool) -> None:
        if val:
            vp_not_fixed = [getattr(type(self), vp) for vp in self.__fixed_parameters__
                            if not getattr(type(self), vp).fixed(self)]

            # First get all values to not re-trigger a solve after every setting
            values = [vp.__get__(self) for vp in vp_not_fixed]
            [vp.__set_converging_value__(self, value) for vp, value in zip(vp_not_fixed, values)]
        else:
            val_dict = {}
            for vp in self.__fixed_parameters__:
                if getattr(type(self), vp).determination_test is not None:
                    vp_free = getattr(type(self), vp).determination_test.vp_free(self)
                    if len(vp_free) == 1 and vp_free[0] not in self.__fixed_parameters__:
                        val_dict.update({vp_free[0]: self.__getattribute__(vp_free[0])})

            for key, val in val_dict.items():
                self.__setattr__(key, val)

    @property
    def fixed(self) -> bool:
        """Bool if block and parts are fixed."""
        return self.fixed_self and all([p.fixed for p in self.parts])

    @fixed.setter
    def fixed(self, val: bool):
        self.fixed_self = val
        for p in self.parts:
            p.fixed = val
    # endregion
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
        for p in [p for p in self.parts]:
            p.tolerance = val
        for s in [s for s in self.solvers]:
            s.tolerance = val
        for p in [p for p in self.ports]:
            p.tolerance = val
    # endregion

    # region User Output
    @property
    def name_display(self) -> str:
        """Return a display formatted version of the block name."""
        return name_2_display(self.name)
    # endregion
