"""
Fluid Port
==========

This file implements an extension of the Continuous Port which specialises
for a fluid port. This adds an area property, and properties for pressure,
temperature, and mass-flow which together define the state of the fluid at
the port.
"""

from __future__ import annotations
import numpy as np
from copy import copy

import cetpy
import cetpy.Modules.SysML as SML
from cetpy.Modules.SysML import value_property
from cetpy.Modules.Fluid import FluidSkeleton


class FluidPort(SML.ContinuousPort):
    """SysML Port element adding functions for continuous flow systems
    containing specific flow properties and output value properties
    tailored for fluid flow."""

    p = SML.FlowProperty()
    t = SML.FlowProperty()
    mdot = SML.FlowProperty()
    area_mode = SML.ValueProperty(
        '', permissible_list=['fixed', 'upstream', 'downstream'],
        permissible_types_list=str)

    def __init__(self,
                 upstream: SML.Block | None = None,
                 downstream: SML.Block | None = None,
                 name: str = None,
                 tolerance: float = None,
                 flow_item: FluidSkeleton | None = None,
                 area_mode: str = None,
                 area: float = None,
                 pressure: float = None,
                 temperature: float = None,
                 mass_flow: float = None) -> None:
        super().__init__(upstream=upstream,
                         downstream=downstream,
                         name=name,
                         flow_item=flow_item,
                         upstream_dict_name='_outlet',
                         downstream_dict_name='_inlet',
                         tolerance=tolerance)
        self._area_mode = 'fixed'
        self._area = None
        if area_mode is None and area is None:
            area_mode = 'upstream'
        elif area_mode is None:
            area_mode = 'fixed'
        self.area = area
        self.area_mode = area_mode  # After area to preserve user mode choice
        if pressure is not None:
            self.p = pressure
        else:
            self.__class__.p.__set_converging_value__(self, 10e5)
        if temperature is not None:
            self.t = temperature
        else:
            self.__class__.t.__set_converging_value__(self, 298.15)
        if mass_flow is not None:
            self.mdot = mass_flow
        else:
            self.__class__.mdot.__set_converging_value__(self, 0)

    # region System References
    def _update_reference(self, reference: str,
                          val: SML.Block | None):
        val_initial = self.__getattribute__(reference)
        SML.ContinuousPort._update_reference(self, reference, val)

        if val is val_initial:
            return
        if val_initial is not None:
            # noinspection PyUnresolvedReferences
            fluid_solver_copy = copy(val.fluid_solver)
            fluid_blocks_initial = val_initial.inlet_no_solve.flow_system_list
            fluid_solver_copy.parent = val_initial
            fluid_solver_copy.parents = fluid_blocks_initial
            for fb in fluid_blocks_initial:
                fb.fluid_solver = fluid_solver_copy
                # noinspection PyUnresolvedReferences
                fb.solvers.remove(val.fluid_solver)
                fb.solvers += [fluid_solver_copy]

        if val is not None:
            # noinspection PyUnresolvedReferences
            fluid_blocks = val.inlet_no_solve.flow_system_list
            # noinspection PyUnresolvedReferences
            fluid_solver = val.fluid_solver
            fluid_solver.parent = val
            fluid_solver.parents = fluid_blocks
            for fb in fluid_blocks:
                fb.fluid_solver = fluid_solver
                if fluid_solver not in fb.solvers:
                    fb.solvers += [fluid_solver]

    def __copy_unlinked__(self) -> FluidPort:
        # noinspection PyPropertyAccess
        port = FluidPort(name=self.name,
                         flow_item=self.flow_item,
                         area_mode=self.area_mode,
                         area=self.area)
        for fp in self.__flow_properties__:
            # noinspection PyProtectedMember
            for prop in [fp._name_instance, fp._name_instance_input,
                         fp._name_instance_reset, fp._name_instance_direction]:
                port.__setattr__(prop, self.__getattribute__(prop))
        return port
    # endregion

    # region Input Properties
    @SML.ContinuousPort.flow_item.setter
    def flow_item(self, val: FluidSkeleton | str | None) -> None:
        if isinstance(val, str):
            val = cetpy.Modules.Fluid.Fluid(val)
        SML.ContinuousPort.flow_item.fset(self, val)

    @value_property()
    def area(self) -> float:
        """Port flow area [m^2]."""
        if self.area_mode == 'fixed':
            return self._area
        elif self.area_mode == 'upstream' and self._upstream is not None:
            return self._upstream.__getattribute__('area')
        elif self.area_mode == 'upstream' and self._downstream is not None:
            return self._downstream.__getattribute__('area')
        elif self.area_mode == 'downstream' and self._downstream is not None:
            return self._downstream.__getattribute__('area')

    @area.setter
    def area(self, val: float) -> None:
        self._area = val
        if val is not None:
            self.area_mode = 'fixed'

    # noinspection PyPropertyAccess
    @value_property()
    def hydraulic_diameter(self) -> float:
        """Port hydraulic diameter [m]."""
        return np.sqrt(self.area / np.pi) * 2

    @hydraulic_diameter.setter
    def hydraulic_diameter(self, val: float) -> None:
        self.area = np.pi * (val / 2) ** 2
    # endregion

    # region Fluid Output Properties
    # noinspection PyPropertyAccess
    @value_property()
    def q(self) -> float:
        """Fluid flow velocity [m/s]."""
        return self.vdot / self.area

    @value_property()
    def p_s(self) -> float:
        """Fluid static pressure [Pa]."""
        return self.p - 0.5 * self.q ** 2 * self.rho

    @value_property()
    def t_s(self) -> float:
        """Fluid static temperature [K]."""
        return self.t - 0.5 * self.q ** 2 / self.c

    @value_property()
    def rho(self) -> float:
        """Fluid density [kg/m^3].

        This value is based on total temperature and pressure, due to model
        accuracy and complexity. You can verify the error with Port.rho_s.

        See Also
        --------
        FluidPort.rho_s: Density based on static temperature and pressure.
        """
        return self.flow_item.rho(self.t, self.p)

    @value_property()
    def rho_s(self) -> float:
        """Fluid density based on static properties [kg/m^3]."""
        return self.flow_item.rho(self.t_s, self.p_s)

    @value_property()
    def vdot(self) -> float:
        """Port volume flow [m^3/s]."""
        return self.mdot / self.rho

    @vdot.setter
    def vdot(self, val: float) -> None:
        self.mdot = val * self.rho

    # noinspection PyPropertyAccess
    @value_property()
    def re(self) -> float:
        """Fluid Reynold's Number [-]."""
        return self.q * self.hydraulic_diameter / self.nu

    @value_property()
    def h(self) -> float:
        """Fluid total enthalpy [J/kg]."""
        return self.flow_item.h(self.t, self.p)

    @value_property()
    def h_s(self) -> float:
        """Fluid static enthalpy [J/kg]."""
        return self.flow_item.h(self.t_s, self.p_s)

    @value_property()
    def phase(self) -> str:
        """Fluid phase name."""
        return self.flow_item.phase(self.t, self.p)

    @value_property()
    def mu(self) -> float:
        """Fluid dynamic viscosity [Pas]."""
        return self.flow_item.mu(self.t, self.p)

    @value_property()
    def nu(self) -> float:
        """Fluid kinematic viscosity [m^2/s]."""
        return self.flow_item.nu(self.t, self.p)

    @value_property()
    def c(self) -> float:
        """Fluid specific heat capacity [J/(kgK)]."""
        return self.flow_item.c(self.t, self.p)

    @value_property()
    def a(self) -> float:
        """Fluid speed of sound [m/s]."""
        return self.flow_item.a(self.t, self.p)

    @value_property()
    def pr(self) -> float:
        """Fluid Prandtl number [-]."""
        return self.flow_item.pr(self.t, self.p)

    @value_property()
    def k(self) -> float:
        """Fluid thermal conductivity [W/(mK)]."""
        return self.flow_item.k(self.t, self.p)

    @value_property()
    def kappa(self) -> float:
        """Fluid isentropic expansion coefficient [-]."""
        return self.flow_item.kappa(self.t, self.p)

    @value_property()
    def m(self) -> float:
        """Fluid molecular mass [J/kg]."""
        return self.flow_item.m()
    # endregion
