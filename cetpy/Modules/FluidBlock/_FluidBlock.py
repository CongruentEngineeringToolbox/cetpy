"""
Fluid Block
===========

This file implements an extension of the  basic SysML Block to specify
interactions along a continuous fluid system.

The model is intended for steady-state incompressible flow and is quite rough.
"""
from __future__ import annotations

from typing import List
import numpy as np

import cetpy.Modules.SysML as SML
from cetpy.Modules.SysML import value_property, DeterminationTest
from cetpy.Modules.Fluid import FluidSkeleton
from cetpy.Modules.FluidBlock import FluidPort

from cetpy.Modules.Solver import Solver


def apply_transfer(block: FluidBlock,
                   flow_property: SML.FlowProperty,
                   simple_only: bool = False) -> None:
    name = flow_property.__getattribute__('_name')
    inlet = block.inlet_no_solve
    outlet = block.outlet_no_solve
    direction = flow_property.get_direction(block.inlet_no_solve)

    if direction == 'downstream':
        source = inlet
        target = outlet
        factor = 1
    else:
        source = outlet
        target = inlet
        factor = -1

    flow_property.__set_converging_value__(
        target, (flow_property.__get__(source)
                 + factor * block.__getattribute__(f'_d{name}_stored')))
    if not simple_only:
        flow_property.__set_converging_value__(
            target, (flow_property.__get__(source)
                     + factor * block.__getattribute__('d' + name)))


class FluidSolver(Solver):
    """Specification of the Solver class to solve a continuous fluid system."""

    __slots__ = ['parents', '_flow_properties']

    def __init__(self, parent: FluidBlock, tolerance: float = None):
        self._flow_properties = None
        super().__init__(parent, tolerance)
        self.parents = [parent]

    # region Input Properties
    @property
    def flow_properties(self) -> List[SML.FlowProperty]:
        if self._flow_properties is None:
            try:
                port = self.parent.outlet_no_solve
            except AttributeError:
                port = self.parent.inlet_no_solve
            self._flow_properties = port.__flow_properties__
        return self._flow_properties

    @property
    def ordered_solver_lists(self) -> List[List[SML.Block]]:
        """Lists of connected fluid blocks for each flow property in the
        direction of the solver."""
        try:
            blocks = self.parent.outlet_no_solve.flow_system_list
        except AttributeError:
            blocks = self.parent.inlet_no_solve.flow_system_list

        flow_properties = self.flow_properties
        ordered_lists = []
        for fp in flow_properties:
            fp_direction = fp.get_direction(blocks[-1].inlet_no_solve)
            if fp_direction == 'upstream':
                ordered_lists += [blocks.copy()]
                ordered_lists[-1].reverse()
            else:
                ordered_lists += [blocks.copy()]
        return ordered_lists
    # endregion

    def __solve_simple_step__(self) -> None:
        """Propagate boundary conditions through fluid system without
        running any element solvers."""
        flow_properties = self.flow_properties
        ordered_lists = self.ordered_solver_lists

        calculating = self.calculating
        self._calculating = True
        [[apply_transfer(b, fp, simple_only=True) for b in l]
         for fp, l in zip(flow_properties, ordered_lists)]
        self._calculating = calculating

    def _solve_function(self) -> None:
        """Solve the fluid system transfer functions until convergence."""
        self.__solve_simple_step__()

        flow_properties = self.flow_properties
        ordered_lists = self.ordered_solver_lists

        try:
            ports = self.parent.outlet_no_solve.flow_system_port_list
        except AttributeError:
            ports = self.parent.inlet_no_solve.flow_system_port_list

        values_last = np.asarray(
            [[fp.__get__(p) for p in ports] for fp in flow_properties])

        iteration = 0
        tolerance = self.tolerance
        residuals = np.ones((len(flow_properties), len(ports)))

        while np.any(residuals > tolerance):
            iteration += 1
            if iteration > 25:
                raise ValueError(f"Fluid System solver did not converge in"
                                 f"{iteration - 1} iterations.")

            [[apply_transfer(b, fp) for b in l]
             for fp, l in zip(flow_properties, ordered_lists)]

            values = np.asarray(
                [[fp.__get__(p) for p in ports] for fp in flow_properties])

            residuals = np.abs(values - values_last) / values
            values_last = values


class FluidBlock(SML.Block):
    """Fluid Block element."""

    _reset_dict = SML.Block._reset_dict
    _reset_dict.update({'_dp_recalculate': True, '_dt_recalculate': True,
                        '_dmdot_recalculate': True})
    _hard_reset_dict = SML.Block._hard_reset_dict
    _hard_reset_dict.update({'_dp_stored': True, '_dt_stored': True,
                             '_dmdot_stored': True})
    dp_fixed = SML.ValueProperty()
    dt_fixed = SML.ValueProperty()
    dmdot_fixed = SML.ValueProperty()

    __init_parameters__ = SML.Block.__init_parameters__ + [
        'dp_fixed', 'dt_fixed', 'dmdot_fixed', 'area', 'hydraulic_diameter'
    ]

    def __init__(self, name: str, abbreviation: str = None,
                 parent: SML.Block = None,
                 tolerance: float = None,
                 upstream: FluidBlock = None,
                 downstream: FluidBlock = None,
                 fluid: FluidSkeleton | str = None,
                 dp_fixed: float = 0,
                 dt_fixed: float = 0,
                 dmdot_fixed: float = 0,
                 inlet_pressure: float = None,
                 outlet_pressure: float = None,
                 inlet_temperature: float = None,
                 outlet_temperature: float = None,
                 inlet_mass_flow: float = None,
                 outlet_mass_flow: float = None,
                 pressure_solver_direction: str = None,
                 temperature_solver_direction: str = None,
                 mass_flow_solver_direction: str = None,
                 **kwargs) -> None:
        self._dp_recalculate = True
        self._dt_recalculate = True
        self._dmdot_recalculate = True
        self._dp_stored = 0
        self._dt_stored = 0
        self._dmdot_stored = 0

        super().__init__(name=name, abbreviation=abbreviation, parent=parent,
                         tolerance=tolerance,
                         dp_fixed=dp_fixed, dt_fixed=dt_fixed,
                         dmdot_fixed=dmdot_fixed,
                         **kwargs)

        # region Port References
        # To enable detection of inlet as inlet, otherwise the direction is
        # misidentified.
        self._outlet = FluidPort()
        self._inlet = FluidPort(
            downstream=self, flow_item=fluid, tolerance=self.tolerance,
            pressure=inlet_pressure, temperature=inlet_temperature,
            mass_flow=inlet_mass_flow)
        self._outlet = FluidPort(
            upstream=self, flow_item=fluid, tolerance=self.tolerance,
            pressure=outlet_pressure, temperature=outlet_temperature,
            mass_flow=outlet_mass_flow)
        self.ports += [self._inlet, self._outlet]
        self.fluid_solver = FluidSolver(parent=self)
        # endregion

        # region Verify Input Validity
        if self._inlet.upstream is not None and (
                inlet_mass_flow is not None or
                inlet_pressure is not None or
                inlet_temperature is not None):
            raise ValueError("Incompatible input, an upstream element is set "
                             "but custom inlet conditions are defined.")
        if self._outlet.downstream is not None and (
                outlet_mass_flow is not None or
                outlet_pressure is not None or
                outlet_temperature is not None):
            raise ValueError("Incompatible input, a downstream element is set "
                             "but custom outlet conditions are defined.")
        for prop, inlet, outlet, direction in zip(
                ['Pressure', 'Temperature', 'Mass flow'],
                [inlet_pressure, inlet_temperature, inlet_mass_flow],
                [outlet_pressure, outlet_temperature, outlet_mass_flow],
                [pressure_solver_direction, temperature_solver_direction,
                 mass_flow_solver_direction]):
            if direction == 'downstream' and outlet is not None:
                raise ValueError(f"{prop} input mismatch, the property is "
                                 f"solved from the inlet, but a custom "
                                 f"outlet value is defined.")
            if direction == 'upstream' and inlet is not None:
                raise ValueError(f"{prop} input mismatch, the property is "
                                 f"solved from the outlet, but a custom "
                                 f"inlet value is defined.")
        # endregion

        # region Fluid References

        # Do this after verifying the input is valid so as not to corrupt
        # the up- or downstream references.
        self.upstream = upstream
        self.downstream = downstream
        # endregion

    # region Transfer Functions
    def _dp_solve(self) -> float:
        """Pressure difference across the fluid element [Pa].

        This is the private calculate function.

        See Also
        --------
        FluidBlock.dp: public function with necessity check
        """
        return self.dp_fixed

    def _dt_solve(self) -> float:
        """Temperature difference across the fluid element [K].

        This is the private calculate function.

        See Also
        --------
        FluidBlock.dt: public function with necessity check
        """
        return self.dt_fixed

    def _dmdot_solve(self) -> float:
        """Mass flow difference across the fluid element [kg/s].

        This is the private calculate function.

        See Also
        --------
        FluidBlock.dmdot: public function with necessity check
        """
        return self.dmdot_fixed

    @value_property()
    def dp(self) -> float:
        r"""Difference in pressure from inlet to outlet [Pa].

        ..math::
            \Delta p = p_{out} - p_{in}
        """
        if self._dp_recalculate:
            self._dp_stored = self._dp_solve()
            self._dp_recalculate = False
        return self._dp_stored

    @value_property()
    def dt(self) -> float:
        r"""Difference in temperature from inlet to outlet [K].

        ..math::
            \Delta T = T_{out} - T_{in}
        """
        if self._dt_recalculate:
            self._dt_stored = self._dt_solve()
            self._dt_recalculate = False
        return self._dt_stored

    @value_property()
    def dmdot(self) -> float:
        r"""Difference in mass flow from inlet to outlet [kg/s].

        ..math::
            \Delta \dot{m} = \dot{m}_{out} - \dot{m}_{in}
        """
        if self._dmdot_recalculate:
            self._dmdot_stored = self._dmdot_solve()
            self._dmdot_recalculate = False
        return self._dmdot_stored
    # endregion

    # region System References and Ports
    @property
    def upstream(self) -> SML.Block | None:
        """Fluid block upstream element."""
        return self._inlet.upstream

    @upstream.setter
    def upstream(self, val: SML.Block | None):
        self._inlet.upstream = val

    @property
    def downstream(self) -> SML.Block | None:
        """Fluid block downstream element."""
        return self._outlet.downstream

    @downstream.setter
    def downstream(self, val: SML.Block | None):
        self._outlet.downstream = val

    @property
    def inlet(self) -> FluidPort:
        """Fluid block inlet port with solved properties.

        See Also
        --------
        FluidBlock.inlet_no_solve: Useful alternative to set new properties
                                   without first calling the solver.
        """
        self.fluid_solver.solve()
        return self._inlet

    @property
    def inlet_no_solve(self) -> FluidPort:
        """Fluid block inlet port without solver properties.

        This should only be used for setting new parameters without calling
        the solver. Do not trust port value properties when accessing through
        this function.

        See Also
        --------
        FluidBlock.inlet
        """
        return self._inlet

    @property
    def outlet(self) -> FluidPort:
        """Fluid block outlet port with solved properties.

        See Also
        --------
        FluidBlock.outlet_no_solve: Useful alternative to set new properties
                                    without first calling the solver.
        """
        self.fluid_solver.solve()
        return self._outlet

    @property
    def outlet_no_solve(self) -> FluidPort:
        """Fluid block outlet port without solver properties.

        This should only be used for setting new parameters without calling
        the solver. Do not trust port value properties when accessing through
        this function.

        See Also
        --------
        FluidBlock.inlet
        """
        return self._outlet
    # endregion

    # region Input Properties
    @value_property(determination_test=DeterminationTest())
    def area(self) -> float:
        """Fluid element characteristic flow area [m^2]."""
        return np.pi * (self.hydraulic_diameter / 2) ** 2

    @value_property(determination_test=area.determination_test)
    def hydraulic_diameter(self) -> float:
        """Fluid block characteristic hydraulic diameter [m]."""
        return np.sqrt(self.area / np.pi) * 2
    # endregion
