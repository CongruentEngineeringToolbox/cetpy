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
    scale_value, name_2_display


def value_property(equation: str = None,
                   determination_test: DeterminationTest = None
                   ) -> Callable[[Callable], ValueProperty]:
    """Decorator Factory to create ValueProperties from getter functions."""

    def decorator(func: Callable) -> ValueProperty:
        """Decorator to create a ValueProperty with metadata from getter."""
        prop = ValueProperty(fget=func, equation=equation,
                             determination_test=determination_test)
        return prop

    return decorator


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

    __slots__ = ['_num', 'auto_fix', 'properties']

    def __init__(self, properties: List[str] = None,
                 num: int = 1, auto_fix: bool = True) -> None:
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
        self._num = int(val)

    def test(self, instance, new: str = None) -> None:
        """Test the determination of the instance. If auto fix is on,
        the test attempts to automatically fix it."""
        n_actual = sum([getattr(type(instance), n).fixed(instance)
                        for n in self.properties])
        n_target = self._num

        direction = ''
        amendment = ''
        if n_actual == n_target:
            return
        elif n_actual > n_target:
            direction = 'over-'
            if self.auto_fix and self._num == 1:
                [instance.__setattr__(n, None) for n in self.properties
                 if n != new]
                # noinspection PyUnresolvedReferences
                logging.info(f"Autocorrected {instance.name_display} "
                             f"over-determination. {new} is the new input.")
                return
            elif self.auto_fix:
                amendment = ' Autofix failed.'

        elif n_actual < n_target:
            direction = 'under-'

        # noinspection PyUnresolvedReferences
        logging.warning(
            f"{instance.name_display} is {direction}determined. Of "
            f"{', '.join(self.properties)} {n_target} must be set. "
            f"Currently {n_actual} are set.{amendment}")


class ValuePropertyDoc:
    """Simple Descriptor to pass a more detailed doc string."""

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        elif instance.fget is not None:
            return instance.fget.__doc__
        else:
            return instance.name_display + f"[{instance.unit}]."

    def __set__(self, instance, value: str) -> None:
        if instance is not None and instance.fget is not None:
            instance.fget.__doc__ = value


class ValueProperty:
    """SysML ValueProperty which adds units of measure and error
    calculation."""
    __slots__ = ['_name', '_name_instance', '_determination_test', 'equation',
                 '_unit', '_axis_label', 'fget', 'fset']

    __doc__ = ValuePropertyDoc()

    def __init__(self,
                 unit: str = None, axis_label: str = None,
                 fget: Callable = None, fset: Callable = None,
                 equation: str = None,
                 determination_test: DeterminationTest = None) -> None:
        self._determination_test = None
        self._name = ''
        self._name_instance = ''
        self.equation = equation
        self.determination_test = determination_test
        if (unit is None and fget is not None
                and fget.__doc__ is not None and '].' in fget.__doc__):
            doc = fget.__doc__
            unit = doc[doc.find('[') + 1: doc.find('].')]
        self._unit = unit
        if (axis_label is None and fget is not None
                and fget.__doc__ is not None
                and '..cet:axis_label:' in fget.__doc__):
            doc = fget.__doc__
            axis_label = doc[
                doc.find('..cet:axis_label:') + 17: doc.find('\n')].strip()
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
        # Rerun determination test setter when the name is known
        self.determination_test = self._determination_test

    def str(self, instance) -> str:
        """Return formatted string of value."""
        value = self.value(instance)
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            value, prefix = scale_value(self.value(instance))
            return str(value) + ' ' + prefix + self.unit
        else:
            return str(value)

    def value(self, instance) -> Any:
        """Return the value property value with significant figure rounding."""
        value = self.value_raw(instance)
        if isinstance(value, float):
            return float('{:.{p}g}'.format(value, p=5))
        else:
            return value

    def value_raw(self, instance) -> Any:
        """Return the value property value without further modification."""
        return self.__get__(instance)

    def __get__(self, instance, owner=None):
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
        if self._determination_test is not None:
            self._determination_test.test(instance, self._name)
        instance.reset()

    def fixed(self, instance) -> bool:
        """Return bool if value property is a fixed parameter for the given
        instance."""
        name = self._name_instance
        if instance is None:
            return False
        elif name in instance.__slots__:
            return instance.__getattribute__(name) is not None
        else:
            return instance.__dict__.get(name, None) is not None

    @property
    def determination_test(self) -> (DeterminationTest, None):
        """Over- and Under-Determination Test that applies to this value
        property."""
        return self._determination_test

    @determination_test.setter
    def determination_test(self, val: (DeterminationTest, None)) -> None:
        name = self._name
        if name != '':  # init call, rerun this function in __set_name__
            # Remove self from past test
            if self._determination_test is not None:
                try:
                    self._determination_test.properties.remove(name)
                except ValueError:  # Not in list.
                    pass
            # Add self to new test
            if val is not None and name not in val.properties:
                val.properties += [name]
        self._determination_test = val
    # endregion

    # region Labelling
    @property
    def name_display(self) -> str:
        """Return the display formatted name of the value property."""
        return name_2_display(self._name)

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


class ValuePrinter:
    """Formatter for block Value Properties to display the property in the
    truncated value form with unit of measure."""
    __slots__ = ['_instance']

    def __get__(self, instance, owner=None):
        self._instance = instance
        return self

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            vp = getattr(type(self._instance), name)
            if isinstance(vp, ValueProperty):
                return vp.str(self._instance)
            else:
                return str(vp.fget(self._instance))


class AggregateValueProperty(ValueProperty):
    """An extension of the SysML ValueProperty class to return the sum of
    the value property on the current block and all its contained parts."""

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = super().__get__(instance, owner)

        if value is None:  # Make compatible for addition with parts' values
            value = 0.

        for p in instance.parts:
            try:  # Try except should be faster than dir lookup.
                part_value = p.__getattribute__(self._name)
                if part_value is not None:
                    value = value + part_value  # add non-mutable
            except AttributeError:
                pass

        return value

    def __get_self__(self, instance, owner=None):
        return super().__get__(instance, owner)
