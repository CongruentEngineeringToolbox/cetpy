"""
SysML ValueProperty
===================

This file implements a basic SysML Value Property.

References
----------
SysML Documentation:
    https://sysml.org/.res/docs/specs/OMGSysML-v1.4-15-06-03.pdf
"""

from __future__ import annotations

from typing import Any, Tuple, Callable

from cet.Modules.Utilities.Labelling import name_2_unit, name_2_axis_label


def value_property(func: Callable) -> ValueProperty:
    """Decorator to create ValueProperties from getter functions."""

    prop = ValueProperty(fget=func)
    return prop


class ValueProperty:
    """SysML ValueProperty which adds units of measure and error
    calculation."""
    __slots__ = ['_name', '_instance', '_value', '_unit', '_axis_label',
                 'fget', 'fset']

    def __init__(self,
                 unit: str = None, axis_label: str = None,
                 fget: Callable = None, fset: Callable = None) -> None:
        self._name = ''
        self._instance = None
        self._value = None
        self._unit = unit
        self._axis_label = axis_label

        self.fget = fget
        self.fset = fset

    # region Getters and Setters
    def __set_name__(self, instance, name):
        self._name = name
        self._instance = instance
        if self._unit is None:
            self._unit = name_2_unit(name)
        if self._axis_label is None:
            self._axis_label = name_2_axis_label(name)

    def __str__(self) -> str:
        return str(self.value) + ' ' + self.unit

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def value(self) -> Any:
        """Return the value property value with significant figure rounding."""
        value = self.value_raw
        if isinstance(value, float):
            return float('{:.{p}g}'.format(value, p=5))
        else:
            return value

    @property
    def value_raw(self) -> Any:
        """Return the value property value without further modification."""
        if self._value is not None or self.fget is None:
            return self._value
        else:
            return self.fget(self._instance)

    def __get__(self, instance, owner):
        return self

    def __set__(self, instance, value) -> None:
        if self.fset is None:
            self._value = value
        else:
            self.fset(instance, value)
        instance.reset()
    # endregion

    # region Labelling
    @property
    def unit(self) -> str:
        """Unit of measure."""
        return self._unit

    @unit.setter
    def unit(self, val: str) -> None:
        self._unit = val

    @property
    def axis_label(self) -> str:
        """Axis label for visualisation."""
        return self._axis_label

    @axis_label.setter
    def axis_label(self, val: str) -> None:
        self._axis_label = val
    # endregion

    # region Pickling
    def __getstate__(self) -> Tuple:
        return self._value, self._unit, self._axis_label

    def __setstate__(self, state: Tuple) -> None:
        self._value, self._unit, self._axis_label = state

    def __hash__(self):
        return super().__hash__()
    # endregion

    # region Math
    def __invert__(self) -> float:
        return ~self.value_raw

    def __add__(self, x: float) -> float:
        return self.value_raw + x

    def __sub__(self, x: float) -> float:
        return self.value_raw - x

    def __mul__(self, x: float) -> float:
        return self.value_raw * x

    def __floordiv__(self, x: float) -> float:
        return self.value_raw // x

    def __truediv__(self, x: float) -> float:
        return self.value_raw / x

    def __mod__(self, x: float) -> float:
        return self.value_raw % x

    def __divmod__(self, x: float) -> Tuple[float, float]:
        return divmod(self.value_raw, x)

    def __pow__(self, x: float, mod=None) -> float:
        return pow(self.value_raw, x, mod)

    def __radd__(self, x: float) -> float:
        return x + self.value_raw

    def __rsub__(self, x: float) -> float:
        return x - self.value_raw

    def __rmul__(self, x: float) -> float:
        return x * self.value_raw

    def __rfloordiv__(self, x: float) -> float:
        return x // self.value_raw

    def __rtruediv__(self, x: float) -> float:
        return x / self.value_raw

    def __rmod__(self, x: float) -> float:
        return x % self.value_raw

    def __rdivmod__(self, x: float) -> Tuple[float, float]:
        return divmod(x, self.value_raw)

    def __rpow__(self, x: float, mod=None) -> float:
        return pow(x, self.value_raw, mod)

    def __eq__(self, x):
        return self.value_raw == x

    def __ge__(self, x):
        return self.value_raw >= x

    def __gt__(self, x):
        return self.value_raw > x

    def __le__(self, x):
        return self.value_raw <= x

    def __lt__(self, x):
        return self.value_raw < x

    def __pos__(self) -> float:
        return +self.value_raw

    def __neg__(self) -> float:
        return -self.value_raw
    # endregion

    # region Decorators

    # endregion
