"""
Compressible Fluid Block
========================

This file specialises the FluidBlock to add a more specific temperature
delta function to calculate temperature changes in gases with expansion or
compression.

The model is intended for steady-state flow and is quite rough.
"""

from cetpy.Modules.FluidBlock import FluidBlock


class CompressibleFluidBlock(FluidBlock):
    """Compressible Fluid Block element."""

    # region Transfer Functions
    def _dt_solve(self) -> float:
        """Temperature difference across the fluid element [K].

        This is the private calculate function.

        The function is expanded from the base FluidBlock to calculate the
        temperature rise from compression or expansion of a gas.

        See Also
        --------
        FluidBlock.dt: public function with necessity check
        """
        if self.dt_fixed == 0:
            return self.dt_fixed
        else:
            # Calculate temperature change with expansion / compression.
            t_out = self.inlet.flow_item.t_kappa(self.inlet.t, self.inlet.p,
                                                 self.outlet.p)
            return t_out - self.inlet.t
    # endregion
