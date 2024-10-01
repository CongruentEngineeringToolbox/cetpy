"""
Plot Descriptor
=====================

This file defines a plotting descriptor for the cetpy Block class. The descriptor provides a convenient command line
bases user interface to plotting Block vector value properties.
"""
from cetpy.Modules.SysML import ValueProperty


class PlotDescriptor:
    """Command line user interface to conveniently plot block value properties."""
    __slots__ = ['_instance']

    def __get__(self, instance, owner=None):
        self._instance = instance
        return self

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            block = self._instance
            if name in block.__plot_functions__.keys():
                block.__plot_functions__[name]()
            else:
                vp = getattr(type(block), name)
                if isinstance(vp, ValueProperty):
                    # ToDo: insert plot function call
                    pass
                else:
                    raise AttributeError("The plotting interface is only supported for cetpy ValueProperty properties.")
            return None
