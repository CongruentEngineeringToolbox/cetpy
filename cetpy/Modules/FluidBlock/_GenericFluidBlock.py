"""
Generic Fluid Block
===================

This file specialises the FluidBlock to add generic properties common for
most fluid elements.

The model is intended for steady-state incompressible flow and is quite rough.
"""
from __future__ import annotations

import numpy as np

from cetpy.Modules.SysML import value_property, DeterminationTest
from cetpy.Modules.FluidBlock import FluidBlock


class GenericFluidBlock(FluidBlock):
    """Generic Fluid Block element."""

    dp_fixed = FluidBlock.dp_fixed
    dp_fixed.determination_test = DeterminationTest()

    __init_parameters__ = FluidBlock.__init_parameters__ + [
        'kv', 'cd',
    ]

    # region Transfer Functions
    def _dp_solve(self) -> float:
        """Pressure difference across the fluid element [Pa].

        This is the private calculate function.

        See Also
        --------
        FluidBlock.dp: public function with necessity check
        """
        if self.__class__.kv.fixed(self):
            return 1e5 * self.inlet.rho / 1000 / (
                    self.kv / (self.inlet.vdot * 3600)) ** 2
        elif self.__class__.cd.fixed(self) and self.area is not None:
            return (self.inlet.mdot / self.cda) ** 2 / (2 * self.inlet.rho)
        elif self.__class__.loss_factor.fixed(self):
            return self.loss_factor * self.inlet.rho / 2 * self.inlet.q ** 2
        else:
            return self.dp_fixed
    # endregion

    # region Input Properties
    @value_property(
        determination_test=FluidBlock.hydraulic_diameter.determination_test)
    def area(self) -> float:
        """Fluid element characteristic flow area [m^2]."""
        if self.__class__.hydraulic_diameter.fixed(self):
            return np.pi * (self.hydraulic_diameter / 2) ** 2
        elif self.__class__.cd.fixed(self):
            return self.cda / self.cd
        else:
            raise ValueError(
                "Neither hydraulic diameter, area, or discharge coefficient "
                "are set. An area cannot be calculated.")

    @value_property(determination_test=dp_fixed.determination_test)
    def kv(self) -> float:
        """Fluid element flow coefficient [m^3/h]."""
        return self.inlet.vdot * 3600 * np.sqrt(
            1e5 / self.dp * self.inlet.rho / 1000)

    @value_property(determination_test=dp_fixed.determination_test)
    def cd(self) -> float:
        """Fluid element discharge coefficient [-]."""
        return self.cda / self.area

    @value_property()
    def cda(self) -> float:
        """Fluid element effective flow area [m^2]."""
        if self.__class__.cd.fixed(self) and self.area is not None:
            return self.cd * self.area
        else:
            return self.inlet.mdot / np.sqrt(self.dp * self.inlet.rho * 2)

    @cda.setter
    def cda(self, val: float | None) -> None:
        if val is None:
            self.cd = None
        else:
            self.cd = val / self.area

    @value_property(determination_test=dp_fixed.determination_test)
    def loss_factor(self) -> float:
        """Fluid element loss coefficient [-]."""
        return 2 * self.dp / self.inlet.rho / self.inlet.q ** 2
    # endregion
