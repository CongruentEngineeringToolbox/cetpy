"""
Fluid Block
===========

This file implements an extension of the basic SysML Block to specify
interactions along a continuous fluid system.

The model is intended for steady-state incompressible flow and is quite rough.
"""
from __future__ import annotations

import numpy as np

import cetpy.Modules.SysML as SML
from cetpy.Modules.SysML import value_property, DeterminationTest
from cetpy.Modules.Fluid import FluidSkeleton
from cetpy.Modules.FluidBlock import FluidPort

from cetpy.Modules.Solver import ContinuousFlowSolver


class FluidSolver(ContinuousFlowSolver):
    """Specification of the Solver class to solve a continuous fluid system."""

    __slots__ = ['parents', '_flow_properties', '_parent_solver',
                 '_sub_solvers']

    def __init__(self, parent: FluidBlock, tolerance: float = None,
                 parent_solver: FluidSolver = None) -> None:
        super().__init__(
            parent, tolerance, parent_solver,
            inlet_port_name='_inlet', outlet_port_name='_outlet',
            boundary_pull_function_name='pull_fluid_boundary_conditions',
            boundary_push_function_name='push_fluid_boundary_conditions')


class FluidBlock(SML.Block):
    """Fluid Block element."""

    _reset_dict = SML.Block._reset_dict.copy()
    _reset_dict.update({'_dp_recalculate': True, '_dt_recalculate': True,
                        '_dmdot_recalculate': True})
    _hard_reset_dict = SML.Block._hard_reset_dict.copy()
    _hard_reset_dict.update({'_dp_stored': 0, '_dt_stored': 0,
                             '_dmdot_stored': 0})
    dp_fixed = SML.ValueProperty()
    dt_fixed = SML.ValueProperty()
    dmdot_fixed = SML.ValueProperty()

    __init_parameters__ = SML.Block.__init_parameters__.copy() + [
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
                         inlet_pressure=inlet_pressure,
                         outlet_pressure=outlet_pressure,
                         inlet_temperature=inlet_temperature,
                         outlet_temperature=outlet_temperature,
                         inlet_mass_flow=inlet_mass_flow,
                         outlet_mass_flow=outlet_mass_flow,
                         **kwargs)

        # region Port References
        # To enable detection of inlet as inlet, otherwise the direction is
        # misidentified.
        # region Associate config
        # Only associate config if not an internal point.
        if upstream is not None:
            inlet_pressure = self._get_init_parameters('inlet_pressure')
            inlet_temperature = self._get_init_parameters('inlet_temperature')
            inlet_mass_flow = self._get_init_parameters('inlet_mass_flow')
        if downstream is not None:
            outlet_pressure = self._get_init_parameters('outlet_pressure')
            outlet_temperature = \
                self._get_init_parameters('outlet_temperature')
            outlet_mass_flow = self._get_init_parameters('outlet_mass_flow')
        # endregion

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
        fluid = self._get_init_parameters('fluid')
        if fluid is not None:
            self.fluid = fluid
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

    @dp.setter
    def dp(self, val: float | None) -> None:
        self.dp_fixed = val

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

    @dt.setter
    def dt(self, val: float | None) -> None:
        self.dt_fixed = val

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

    @dmdot.setter
    def dmdot(self, val: float | None) -> None:
        self.dmdot_fixed = val
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
        FluidBlock.outlet
        """
        return self._outlet
    # endregion

    # region Input Properties
    @value_property()
    def fluid(self) -> FluidSkeleton:
        """Fluid Behavioural Model. References the outlet port."""
        return self._outlet.flow_item

    @fluid.setter
    def fluid(self, val: str | FluidSkeleton) -> None:
        self._outlet.flow_item = val

    @value_property(determination_test=DeterminationTest())
    def area(self) -> float:
        """Fluid element characteristic flow area [m^2]."""
        return np.pi * (self.hydraulic_diameter / 2) ** 2

    @value_property(determination_test=area.determination_test)
    def hydraulic_diameter(self) -> float:
        """Fluid block characteristic hydraulic diameter [m]."""
        return np.sqrt(self.area / np.pi) * 2
    # endregion
