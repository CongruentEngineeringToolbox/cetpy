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

    def __init__(self, parent: FluidBlock, tolerance: float = None,
                 parent_solver: FluidSolver = None) -> None:
        super().__init__(
            parent, tolerance, parent_solver, dict_name='fluid_solver',
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
    __default_parameters__ = SML.Block.__default_parameters__.copy()
    __default_parameters__.update({
        'dp_fixed': 0, 'dt_fixed': 0, 'dmdot_fixed': 0
    })

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
                 dp_fixed: float = None,
                 dt_fixed: float = None,
                 dmdot_fixed: float = None,
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
        """Specification of the generic Block class to add a solver for
        fluid flow.

        Parameters
        ----------
        name
            A name for the block, used for visualizations, reports,
            and other user output.
        abbreviation: optional
            A shortened abbreviation of the name that can also be used for
            output functions where a long-name would be a prohibitive
            identifier. If none is given, the block predicts a sensible
            abbreviation.
        parent: optional
            Another block, which is to contain this block as a part. The
            bidirectional connection is made automatically.
        tolerance: optional
            A general tolerance for the block, its solvers, ports,
            and parts. If None is provided, takes the tolerance of the
            parent if available or from the active session configuration.
        upstream: optional
            An upstream fluid element.
        downstream: optional
            A downstream fluid element.
        fluid: optional
            A flow item, can be specified as a cetpy Fluid instance or a
            CoolProp compatible string. If up- or downstream elements are
            supplied the element can pul the fluid from those, or push its
            own to them if they do not yet have a specified fluid instance.
        dp_fixed: optional, default = 0 Pa
            A fixed pressure difference from inlet to outlet [Pa]. Note a
            pressure loss should be specified as a negative.
        dt_fixed: optional, default = 0 K
            A fixed temperature rise across the fluid element from inlet to
            outlet [K]. A temperature rise is positive.
        dmdot_fixed: optional, default = 0 kg/s
            A fixed mass-flow difference across the fluid element from inlet to
            outlet [kg/s]. A leakage is negative.
        inlet_pressure: optional, default = 1 MPa
            Pressure at the inlet of the element, only permissible if an
            upstream element is not specified. Sets a downstream pressure
            solver.
        outlet_pressure: optional, default = 1 MPa
            Pressure at the outlet of the element, only permissible if a
            downstream element is not specified. Sets an upstream pressure
            solver.
        inlet_temperature: optional, default = 298.15 K
            Temperature at the inlet of the element, only permissible if an
            upstream element is not specified. Sets a downstream temperature
            solver.
        outlet_temperature: optional, default = 298.15 K
            Temperature at the outlet of the element, only permissible if a
            downstream element is not specified. Sets an upstream
            temperature solver.
        inlet_mass_flow: optional, default = 0 kg/s
            Mass flow at the inlet of the element, only permissible if an
            upstream element is not specified. Sets a downstream mass flow
            solver.
        outlet_mass_flow: optional, default = 0 kg/s
            Mass flow at the outlet of the element, only permissible if a
            downstream element is not specified. Sets an upstream
            mass flow solver.
        pressure_solver_direction: optional, default = 'downstream'
            Direction of the pressure solver. 'downstream', solve the
            pressure from an upstream pressure boundary condition,
            or 'upstream', solve the pressure from a downstream pressure
            boundary condition.
        temperature_solver_direction: optional, default = 'downstream'
            Direction of the temperature solver. 'downstream', solve the
            temperature from an upstream temperature boundary condition,
            or 'upstream', solve the temperature from a downstream temperature
            boundary condition.
        mass_flow_solver_direction: optional, default = 'downstream'
            Direction of the mass flow solver. 'downstream', solve the
            mass flow from an upstream mass flow boundary condition,
            or 'upstream', solve the mass flow from a downstream mass flow
            boundary condition.

        Examples
        --------
        A fluid block can be initialised with just a name or with all
        desired properties. To request any inlet or outlet properties,
        at minimum a fluid must be specified as well:
        >>> fe = FluidBlock('Test', fluid='H2O')
        >>> fe.inlet.p
        1000000.0
        >>> fe.outlet.t
        298.15
        >>> fe.outlet.mdot
        0

        Per default the block has no impact on pressure, temperature,
        or mass-flow and is initialised at 1 MPa, 298.15 K, and 0 kg/s.
        Property differences or different boundary conditions can be
        specified directly on initialisation or interactively afterward.
        Here wer will demonstrate modifying a block.

        We will first add a mass-flow of 1 kg/s to the inlet. We will use
        the 'inlet_no_solve' property to not unintentionally trigger a fluid
        system solve as we are setting a new boundary condition:
        >>> fe.inlet_no_solve.mdot = 1.0
        >>> fe.outlet.mdot
        1.0

        Next we will add a pressure loss, temperature increase, and leakage:
        >>> fe.dp_fixed = -1e5
        >>> fe.dt_fixed = 5
        >>> fe.dmdot_fixed = -0.01

        We can see that these differences are applied by looking at the
        block outlet:
        >>> fe.outlet.p
        900000.0
        >>> fe.outlet.t
        303.15
        >>> fe.outlet.mdot
        0.99

        For just a calcuation like this, the block is a little overkill.
        It's purpose comes as the architecture for more elaborate fluid
        models and combinations of fluid blocks. Consider looking at the
        cetpy.Modules.Fluid.GenericFluidBlock for a specification that adds
        flow coefficients, loss coefficients, and discharge coefficients.

        Notably, not just the base solver parameters can be
        requested at the ports, but also many more, such as the viscosity.
        Get a full list of available properties with: help(fe.outlet)
        >>> np.round(fe.outlet.mu, 6)
        0.000797

        You can also directly format these values in a more managable way
        with their units by appending 'print' before the value:
        >>> fe.outlet.print.mu
        '797.21 µPas'

        If we specify an area, also velocity dependent values can be
        requested, such as the Reynold's Number:
        >>> fe.area = np.pi * 0.04 ** 2
        >>> np.round(fe.outlet.re)
        19764.0

        So far we have been using upstream boundary conditions, but we can
        also move things around. E.g. if an experiment should have 7 bar at
        the outlet of the fluid block we can evaluate the necessary inlet
        pressure.
        >>> fe.outlet_no_solve.p = 7e5
        >>> fe.inlet.p
        800000.0

        As said above the real power comes with the combination of multiple
        blocks. We will create a new block with just a name:
        >>> fe2 = FluidBlock('Test2')

        And append it to the fluid system:
        >>> fe2.upstream = fe

        For this call you will get a warning in the console and the log
        file that the solver direciton could not be automatically solved. We
        can resolve this easily:
        >>> fe2.outlet_no_solve.p = fe2.outlet_no_solve.p

        You will note that the new fluid element has automatically adopted
        the fluid and solver directions from the first block as these were
        still unspecified.
        >>> fe2.fluid.name
        'H2O'
        >>> fe2.fluid is fe.fluid
        True
        >>> fe2.outlet.p
        1000000.0
        >>> fe.inlet.p
        1100000.0

        The elements also share the same fluid solver, so making a change to
        one lets the other know it needs ot be resolved.
        >>> fe2.fluid_solver is fe.fluid_solver
        True
        >>> fe.solved
        True
        >>> fe2.dp_fixed = -0.5e5
        >>> fe.solved
        False

        Requesting a solved inlet or outlet from any of the blocks in the
        fluid system solves the system as a whole:
        >>> fe.inlet.p
        1150000.0
        >>> fe2.solved
        True

        Both blocks also share the same port between them. So there is no
        possibility of model disconnect.
        >>> fe.outlet is fe2.inlet
        True

        We can also add a thrid fluid block, specifying a range of
        parameters on initialisation. we will directly specify the next
        element in line, a new outlet pressure boundary condition,
        a new outlet mass flow boundary condition, which will
        automatically change the solver direction across the fluid system,
        and a new solver direction for the temperature.
        >>> fe3 = FluidBlock('Test3', upstream=fe2, outlet_pressure=20e5,
        ...                  outlet_mass_flow=0.5,
        ...                  temperature_solver_direction='upstream')
        >>> np.round(fe.inlet.p)
        2150000.0
        >>> np.round(fe.inlet.mdot, 2)
        0.51
        >>> type(fe.inlet).t.get_direction(fe.inlet)
        'upstream'

        Of course, we can also add a fluid element on the upstream side:
        >>> fe4 = FluidBlock('Test4', downstream=fe)
        >>> type(fe4.inlet).t.get_direction(fe4.inlet)
        'upstream'
        """
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
        if upstream is None:
            inlet_pressure = self._get_init_parameters('inlet_pressure')
            inlet_temperature = self._get_init_parameters('inlet_temperature')
            inlet_mass_flow = self._get_init_parameters('inlet_mass_flow')
        if downstream is None:
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

        # region Solver Direction
        for fp, direction in zip(['p', 't', 'mdot'],
                                 [pressure_solver_direction,
                                  temperature_solver_direction,
                                  mass_flow_solver_direction]):
            fp = getattr(type(self._inlet), fp)
            if direction != fp.get_direction(self._inlet):
                if direction == 'upstream':
                    port = self._outlet
                elif direction == 'downstream':
                    port = self._inlet
                else:
                    continue
                fp.__set__(port, fp.__get__(port))
        # endregion

        # region Fluid References

        # Do this after verifying the input is valid so as not to corrupt
        # the up- or downstream references. For each first clear any
        # conflicting inputs.
        if upstream is not None:
            upstream_outlet = upstream.outlet_no_solve
            end_port = upstream_outlet.upstream_list[0]
            for fp in self._inlet.__flow_properties__:
                if (fp.get_direction(self._inlet)
                        != fp.get_direction(upstream_outlet)):
                    setattr(end_port, getattr(fp, '_name_instance_input'),
                            None)
            self.upstream = upstream
        if downstream is not None:
            downstream_inlet = downstream.inlet_no_solve
            end_port = downstream_inlet.downstream_list[-1]
            for fp in self._outlet.__flow_properties__:
                if (fp.get_direction(self._outlet)
                        != fp.get_direction(downstream_inlet)):
                    setattr(end_port, getattr(fp, '_name_instance_input'),
                            None)
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
