"""
ContinuousFlowSolver
====================

This file implements an extension of the basic Solver to solve continuous
flow systems such as fluid systems.

See Also
--------
cetpy.Modules.SysML.SML.ContinuousPort
"""
from __future__ import annotations

from typing import List
import numpy as np

import cetpy.Modules.SysML as SML

from cetpy.Modules.Solver import Solver


def apply_transfer(block: SML.Block,
                   flow_property: SML.FlowProperty,
                   inlet_port_name: str,
                   outlet_port_name: str,
                   simple_only: bool = False) -> None:
    name = flow_property.__getattribute__('_name')
    inlet: SML.ContinuousPort = getattr(block, inlet_port_name)
    outlet: SML.ContinuousPort = getattr(block, outlet_port_name)
    direction = flow_property.get_direction(inlet)

    if direction == 'downstream':
        source = inlet
        target = outlet
        factor = 1
    else:
        source = outlet
        target = inlet
        factor = -1

    try:
        # Use stored because no computational time cost
        delta_last = block.__getattribute__(f'_d{name}_stored')
    except AttributeError:
        # No delta function, just pass the value through to the next port
        flow_property.__set_converging_value__(
            target, flow_property.__get__(source))
        return

    flow_property.__set_converging_value__(
        target, (flow_property.__get__(source) + factor * delta_last))
    if not simple_only:
        flow_property.__set_converging_value__(
            target, (flow_property.__get__(source)
                     + factor * block.__getattribute__('d' + name)))


class ContinuousFlowSolver(Solver):
    """Specification of the Solver class to solve a continuous flow system."""

    __slots__ = ['_parents', '_flow_properties', '_parent_solver', '_ref_port',
                 'dict_name', 'inlet_port_name', 'outlet_port_name',
                 'boundary_pull_function_name', 'boundary_push_function_name']

    def __init__(self, parent: SML.Block, tolerance: float = None,
                 parent_solver: ContinuousFlowSolver = None,
                 dict_name: str = 'solver',
                 inlet_port_name: str = '_inlet',
                 outlet_port_name: str = '_outlet',
                 boundary_pull_function_name: str = 'pull_boundary_conditions',
                 boundary_push_function_name: str = 'push_boundary_conditions'
                 ) -> None:
        self._flow_properties = None
        self._parent_solver = None
        self.dict_name = dict_name
        self.inlet_port_name = inlet_port_name
        self.outlet_port_name = outlet_port_name
        self.boundary_pull_function_name = boundary_pull_function_name
        self.boundary_push_function_name = boundary_push_function_name
        super().__init__(parent, tolerance)
        self.parents = [parent]
        self._parent_solver = parent_solver

    # region System References
    @property
    def parent(self):
        return super().parent

    @parent.setter
    def parent(self, val) -> None:
        Solver.parent.fset(self, val)
        if val is None:
            self._ref_port = None
        else:
            try:
                self._ref_port: SML.ContinuousPort = getattr(
                    self.parent, self.outlet_port_name)
            except AttributeError:
                self._ref_port: SML.ContinuousPort = getattr(
                    self.parent, self.inlet_port_name)
            self.parents = self._ref_port.flow_system_list

    @property
    def parents(self):
        """All associated parent blocks which share this solver."""
        return self._parents

    @parents.setter
    def parents(self, val) -> None:
        self._parents = val
        for b in self._parents:
            try:
                val_initial = getattr(b, self.dict_name)
                if val_initial is not self:
                    b.solvers.remove(val_initial)
                    try:
                        val_initial.parents.remove(b)
                    except ValueError:
                        pass
            except AttributeError:
                pass
            setattr(b, self.dict_name, self)
            if self not in b.solvers:
                b.solvers += [self]
    # endregion

    # region Interface Functions
    def reset(self, parent_reset: bool = True) -> None:
        if not self._resetting:
            self._resetting = True
            self._recalculate = True
            if self._parent_solver is not None:
                self._parent_solver.reset(parent_reset)
            # Reset parent instance if desired
            if parent_reset and self.parent is not None:
                self.parent.reset()
            self._resetting = False
    # endregion

    # region Input Properties
    @property
    def flow_properties(self) -> List[SML.FlowProperty]:
        if self._flow_properties is None:
            self._flow_properties = self._ref_port.__flow_properties__
        return self._flow_properties

    @property
    def ordered_solver_lists(self) -> List[List[SML.Block]]:
        """Lists of connected flow blocks for each flow property in the
        direction of the solver."""
        blocks = self._ref_port.flow_system_list

        # Drop first and last block from solver list if they don't have to
        # solve a transfer.
        try:
            getattr(blocks[0], self.inlet_port_name)
        except AttributeError:
            blocks = blocks[1:]

        try:
            getattr(blocks[-1], self.outlet_port_name)
        except AttributeError:
            blocks = blocks[:-1]

        flow_properties = self.flow_properties
        ordered_lists = []
        for fp in flow_properties:
            fp_direction = fp.get_direction(
                getattr(blocks[-1], self.inlet_port_name))
            if fp_direction == 'upstream':
                ordered_lists += [blocks.copy()]
                ordered_lists[-1].reverse()
            else:
                ordered_lists += [blocks.copy()]
        return ordered_lists

    @property
    def parent_solver(self) -> ContinuousFlowSolver | None:
        """Return overarching flow solver."""
        return self._parent_solver

    @parent_solver.setter
    def parent_solver(self, val: ContinuousFlowSolver | None) -> None:
        self._parent_solver = val
        self.reset()
    # endregion

    # region Solver Functions
    def _pre_solve(self) -> None:
        # Solve top-down -> better performance and ensures the lower solvers
        # get the correct boundary conditions to start with. Do this before
        # the calculating flag is set so this solver is run on each parent
        # solver loop.
        if self._parent_solver is not None:
            self._parent_solver.solve()
        super()._pre_solve()
        # Push any new results to end blocks. Do this after setting
        # calculating True so as not to trigger this solver.
        blocks = self._ref_port.flow_system_list
        end_blocks = [blocks[0], blocks[-1]]
        for block in end_blocks:
            try:
                getattr(block, self.boundary_pull_function_name)()
            except AttributeError:
                pass

    def __solve_simple_step__(self) -> None:
        """Propagate boundary conditions through flow system without
        running any element solvers."""
        flow_properties = self.flow_properties
        ordered_lists = self.ordered_solver_lists
        inlet_name, outlet_name = self.inlet_port_name, self.outlet_port_name

        calculating = self.calculating
        self._calculating = True
        [[apply_transfer(b, fp, inlet_name, outlet_name,
                         simple_only=True) for b in l]
         for fp, l in zip(flow_properties, ordered_lists)]
        self._calculating = calculating

    def _solve_function(self) -> None:
        """Solve the flow system transfer functions until convergence."""
        self.__solve_simple_step__()

        flow_properties = self.flow_properties
        ordered_lists = self.ordered_solver_lists
        inlet_name, outlet_name = self.inlet_port_name, self.outlet_port_name

        ports = self._ref_port.flow_system_port_list

        values_last = np.asarray(
            [[fp.__get__(p) for p in ports] for fp in flow_properties])

        iteration = 0
        tolerance = self.tolerance
        residuals = np.ones((len(flow_properties), len(ports)))

        while np.any(residuals > tolerance):
            iteration += 1
            if iteration > 25:
                self._calculating = False
                raise ValueError(f"Flow System solver did not converge in "
                                 f"{iteration - 1} iterations.")

            [[apply_transfer(b, fp, inlet_name, outlet_name) for b in l]
             for fp, l in zip(flow_properties, ordered_lists)]

            values = np.asarray(
                [[fp.__get__(p) for p in ports] for fp in flow_properties])

            idx_zero = values == 0
            # Avoid division by zero -> split in zero and not zero
            # Zero must be an exact match
            residuals[~idx_zero] = np.abs(
                values - values_last)[~idx_zero] / values[~idx_zero]
            residuals[idx_zero] = values_last[idx_zero] != 0

            values_last = values

    def _post_solve(self) -> None:
        super()._post_solve()
        # Push any new results to end blocks. Do this after setting
        # recalculate True so if calculations in the end blocks change the
        # boundary conditions, this solver is reset again.
        blocks = self._ref_port.flow_system_list
        end_blocks = [blocks[0], blocks[-1]]
        for block in end_blocks:
            try:
                getattr(block, self.boundary_push_function_name)()
            except AttributeError:
                pass
    # endregion
