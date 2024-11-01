"""
Plot Descriptor
=====================

This file defines a plotting descriptor for the cetpy Block class. The descriptor provides a convenient command line
bases user interface to plotting Block vector value properties.
"""
from typing import Sized, Iterable
import warnings

from cetpy.Modules.SysML import ValueProperty
from cetpy.Modules.plotting.line import get_plot_2d_value_properties


class PlotDescriptor:
    """Command line user interface to conveniently plot block value properties."""
    __slots__ = ['_instance']

    line_function = staticmethod(get_plot_2d_value_properties)

    def __get__(self, instance, owner=None):
        self._instance = instance
        return self

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            block = self._instance
            if name in block.__plot_functions__.keys():
                block.__plot_functions__[name](block)
            else:
                vp = getattr(type(block), name)
                if not isinstance(vp, ValueProperty):
                    raise AttributeError("The plotting interface is only supported for cetpy ValueProperty properties.")

                val = vp.value_raw(block)
                if not isinstance(val, Sized):
                    raise ValueError("The plotting interface can only plot sized properties.")

                known_x_vp = False
                x_vp = None
                if isinstance(block.__default_plot_x_axis__, str):
                    x_vp = block.__deep_get_vp__(block.__default_plot_x_axis__)
                    x_val = x_vp.value_raw(block)
                    if len(x_val) == len(val):
                        known_x_vp = True
                elif isinstance(block.__default_plot_x_axis__, Iterable):
                    for x_vp in [block.__deep_get_vp__(x_option) for x_option in block.__default_plot_x_axis__]:
                        x_val = x_vp.value_raw(block)
                        if len(x_val) == len(val):
                            known_x_vp = True
                            break
                if not known_x_vp:
                    while True:
                        x_vp_name = input("An x-axis property could not be automatically assigned. Please enter the "
                                          "name of the x-axis property for this plot.")
                        x_vp = block.__deep_get_vp__(x_vp_name)
                        if not isinstance(x_vp, ValueProperty):
                            warnings.warn("The x_axis property must be a cetpy ValueProperty.")

                fig, ax, ax2 = self.line_function(block, x_vp, vp)
                fig.show()
            return None
