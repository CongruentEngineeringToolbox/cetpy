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

from typing import Any, Tuple, Callable, List
import logging

from cet.Modules.Utilities.Labelling import name_2_unit, name_2_axis_label, \
    scale_value


def value_property(func: Callable) -> ValueProperty:
    """Decorator to create ValueProperties from getter functions."""

    prop = ValueProperty(fget=func)
    return prop


class UnitFloat(float):
    """Extension fo the python float to add a unit representation."""

    __slots__ = ['unit']

    def __new__(cls, value: float, unit: str = None):
        return float.__new__(cls, value)

    def __init__(self, value: float, unit: str = None):
        super().__init__()
        self.unit = unit

    def __str__(self) -> str:
        value, prefix = scale_value(self * 1.0)
        return str(value) + ' ' + prefix + self.unit

    def __repr__(self) -> str:
        return self.__str__()


class DeterminationTest:
    """Class to manage over- and under-determination of model inputs."""

    def __init__(self, instance: object, properties: List[str] = None,
                 num: int = 1, auto_fix: bool = True) -> None:
        self._instance = instance
        self._num = num
        self.auto_fix = auto_fix
        if properties is None:
            properties: List[str] = []
        self.properties = properties

    @property
    def num(self) -> int:
        """Number of properties allowed to be fixed."""
        return self._num

    @num.setter
    def num(self, val: int) -> None:
        self._num = val

    def test(self, new: str = None) -> None:
        """Test the determination of the instance. If auto fix is on,
        the test attempts to automatically fix it."""
        n_actual = sum([self._instance.__getattribute__(n)._value is not None
                        for n in self.properties])
        n_target = self._num

        direction = ''
        amendment = ''
        if n_actual == n_target:
            return
        elif n_actual > n_target:
            direction = 'over-'
            if self.auto_fix and self._num == 1:
                [self._instance.__setattr__(n, None) for n in self.properties
                 if n == new]
                # noinspection PyUnresolvedReferences
                logging.info(f"Autocorrected {self._instance.name_display} "
                             f"over-determination. {new} is the new input.")
            elif self.auto_fix:
                amendment = ' Autofix failed.'

        elif n_actual < n_target:
            direction = 'under-'

        # noinspection PyUnresolvedReferences
        logging.warning(
            f"{self._instance.name_display} is {direction}determined. Of "
            f"{', '.join(self.properties)} {n_target} must be set. "
            f"Currently {n_actual} are set.{amendment}")


class ValueProperty:
    """SysML ValueProperty which adds units of measure and error
    calculation."""
    __slots__ = ['_name', '_name_instance', 'determination_test',
                 '_unit', '_axis_label', 'fget', 'fset']

    def __init__(self,
                 unit: str = None, axis_label: str = None,
                 fget: Callable = None, fset: Callable = None) -> None:
        self._name = ''
        self.determination_test = None
        self._unit = unit
        self._axis_label = axis_label

        self.fget = fget
        self.fset = fset

    # region Getters and Setters
    def __set_name__(self, instance, name):
        self._name = name
        self._name_instance = '_' + name
        if self._unit is None:
            self._unit = name_2_unit(name)
        if self._axis_label is None:
            self._axis_label = name_2_axis_label(name)

    def value(self, instance) -> Any:
        """Return the value property value with significant figure rounding."""
        value = self.value_raw(instance)
        if isinstance(value, float):
            return float('{:.{p}g}'.format(value, p=5))
        else:
            return value

    def value_raw(self, instance) -> Any:
        """Return the value property value without further modification."""
        return self.__get__(instance, None)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        name = self._name_instance
        if name in instance.__slots__:
            value = instance.__getattribute__(name)
        else:
            value = instance.__dict__.get(name, None)
        if value is not None or self.fget is None:
            return value
        else:
            return self.fget(instance)

    def __set__(self, instance, value) -> None:
        if self.fset is None:
            instance.__setattr__(self._name_instance, value)
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
        return self._name, self._name_instance, self._unit, self._axis_label

    def __setstate__(self, state: Tuple) -> None:
        self._name, self._name_instance, self._unit, self._axis_label = state

    def __hash__(self):
        return super().__hash__()
    # endregion
