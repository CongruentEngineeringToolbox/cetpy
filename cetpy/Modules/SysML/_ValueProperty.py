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

from typing import Any, Tuple, Callable, List, Iterable, Sized
import numpy as np

import cetpy
from cetpy.Modules.Utilities.Labelling import name_2_unit, name_2_axis_label, \
    scale_value, name_2_display
from cetpy.Modules.Utilities.InputValidation import validate_input


def value_property(equation: str = None,
                   determination_test: DeterminationTest = None,
                   necessity_test: float = 0.1,
                   permissible_list: None | List | Tuple = None,
                   permissible_types_list: None | type | List = None
                   ) -> Callable[[Callable], ValueProperty]:
    """Decorator Factory to create ValueProperties from getter functions."""

    def decorator(func: Callable) -> ValueProperty:
        """Decorator to create a ValueProperty with metadata from getter."""
        prop = ValueProperty(fget=func, equation=equation,
                             determination_test=determination_test,
                             necessity_test=necessity_test,
                             permissible_list=permissible_list,
                             permissible_types_list=permissible_types_list)
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
                cetpy.active_session.logger.info(
                    f"Autocorrected {instance.name_display} "
                    f"over-determination. {new} is the new input.")
                return
            elif self.auto_fix:
                amendment = ' Autofix failed.'

        elif n_actual < n_target:
            direction = 'under-'

        # noinspection PyUnresolvedReferences
        cetpy.active_session.logger.warning(
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
    __slots__ = ['_name', '_name_instance', '_name_instance_reset',
                 '_determination_test', '_necessity_test', 'equation',
                 '_unit', '_axis_label', 'fget', 'fset', 'fdel',
                 '_permissible_list', '_permissible_types_list']

    __doc__ = ValuePropertyDoc()

    def __init__(self,
                 unit: str = None, axis_label: str = None,
                 fget: Callable = None, fset: Callable = None,
                 fdel: Callable = None, equation: str = None,
                 determination_test: DeterminationTest = None,
                 necessity_test: float = 0.1,
                 permissible_list: None | List | Tuple = None,
                 permissible_types_list: None | type | List = None) -> None:
        self._determination_test = None
        self._necessity_test = None
        self._permissible_list = None
        self._permissible_types_list = None
        self._name = ''
        self._name_instance = ''
        self._name_instance_reset = ''
        self.equation = equation
        self.determination_test = determination_test
        self.necessity_test = necessity_test
        self.permissible_list = permissible_list
        self.permissible_types_list = permissible_types_list
        if (unit is None and fget is not None
                and fget.__doc__ is not None and '].' in fget.__doc__):
            doc = fget.__doc__
            unit = doc[doc.find('[') + 1: doc.find('].')]
        self._unit = unit
        if (axis_label is None and fget is not None
                and fget.__doc__ is not None
                and '..cetpy:axis_label:' in fget.__doc__):
            doc = fget.__doc__
            axis_label = doc[
                doc.find('..cetpy:axis_label:') + 17: doc.find('\n')].strip()
        self._axis_label = axis_label

        self.fget = fget
        self.fset = fset
        self.fdel = fdel

    # region Decorators
    def getter(self, fget: Callable) -> ValueProperty:
        prop = self
        prop.fget = fget
        return prop

    # ToDo: Figure out type hinting for read access
    def setter(self, fset: Callable) -> ValueProperty:
        prop = self
        prop.fset = fset
        return prop

    def deleter(self, fdel: Callable) -> ValueProperty:
        prop = self
        prop.fdel = fdel
        return prop
    # endregion

    # region Getters and Setters
    def __set_name__(self, cls, name):
        self._name = name
        self._name_instance = '_' + name
        self._name_instance_reset = self._name_instance + '_reset'
        if self._unit is None:
            self._unit = name_2_unit(name)
        if self._axis_label is None:
            self._axis_label = name_2_axis_label(name)
        # Rerun determination test setter when the name is known
        self.determination_test = self._determination_test

    def name(self) -> str:
        """Value Property Name"""
        return self._name

    def str(self, instance) -> str:
        """Return formatted string of value."""
        value = self.value(instance)
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            value, prefix = scale_value(self.value(instance))
            unit = self.unit
            if any([str(i) in unit for i in range(10)]):
                unit = prefix + '(' + unit + ')'
            else:
                unit = prefix + unit
            return str(value) + ' ' + unit
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

        try:
            value = instance.__getattribute__(name)
        except AttributeError:
            value = None

        if value is not None or self.fget is None:
            return value
        else:
            return self.fget(instance)

    def __set__(self, instance, value) -> None:
        try:
            val_initial = instance.__getattribute__(self._name_instance_reset)
        except AttributeError:
            val_initial = None
        value = validate_input(value, self.permissible_types_list,
                               self.permissible_list, self._name)
        if self.fset is None:
            instance.__setattr__(self._name_instance, value)
        else:
            self.fset(instance, value)

        if self._determination_test is not None:
            self._determination_test.test(instance, self._name)

        if self.__reset_necessary__(instance, value, val_initial):
            instance.__setattr__(self._name_instance_reset, value)
            instance.reset()

    def __reset_necessary__(self, instance, value, val_initial=None) -> bool:
        """Return bool if a reset should be performed on an instance if the
        initial value is changed to value.
        """
        necessity_test = self.necessity_test
        reset = True
        if necessity_test >= 0:
            same = (value == val_initial)
            if isinstance(same, Iterable):
                same = all(same)
            if same:
                reset = False
            elif type(value) == type(val_initial):
                if isinstance(value, int | float):
                    reset = not np.isclose(
                        value, val_initial,
                        rtol=instance.tolerance * necessity_test, atol=0)
                if isinstance(value, Iterable | Sized) and (
                        len(value) == len(val_initial)):
                    reset = not all(np.isclose(
                        value, val_initial,
                        rtol=instance.tolerance * necessity_test, atol=0))
        return reset

    def __delete__(self, instance):
        if self.fdel is not None:
            self.fdel(instance)
        else:
            delattr(instance, self._name_instance)

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

    @property
    def necessity_test(self) -> float:
        """Reset necessity test multiplier. If this value property of an
        instance is changed by a larger relative factor than this multiplier
        times the instance tolerance, a reset is called. This multiplier is
        intended to counteract the error sensitivity of an input value.

        The default is 0.1, which is moderately conservative for a large number
        of models. In effect this means, that if a block is set to a
        tolerance of 1e-3, a 1e-4 tolerance is applied to the inputs.

        Evaluating the error sensitivity and calibrating this value can
        significantly speed up models.

        This setter does not automatically call a reset. The user has to
        manually reset all relevant models after changing this value.
        """
        return self._necessity_test

    @necessity_test.setter
    def necessity_test(self, val: float) -> None:
        if not isinstance(val, float | int) and val >= 0:
            raise ValueError("The necessity test multiplier must be a "
                             "positive float. Set to 0 to disable.")
        self._necessity_test = val

    @property
    def permissible_list(self) -> None | List | Tuple:
        """List of permissible values for the value property. Set to None to
        disable. Set to a list of approved values, particularly useful for
        string switches. Or set to a tuple of length two with integer or
        float lower and upper limits. None represents positive or negative
        infinity.
        """
        return self._permissible_list

    @permissible_list.setter
    def permissible_list(self, val: None | List | Tuple) -> None:
        if not (val is None or isinstance(val, List)
                or (isinstance(val, Tuple) and len(val) == 2
                    and all([isinstance(v, None | float | int) for v in val]))
                ):
            raise ValueError("The permissible list must be None or a "
                             "List or a Tuple of length 2 with float / None.")
        if isinstance(val, Tuple):
            if val[0] is None:
                val = (-np.inf, val[1])
            if val[1] is None:
                val = (val[0], np.inf)
            if val[1] < val[0]:
                raise ValueError("The allowed minimum limit is above the "
                                 "upper limit. The requirement cannot be "
                                 "fulfilled.")
        self._permissible_list = val

    @property
    def permissible_types_list(self) -> None | type | List[type]:
        """List of permissible types for input values. New inputs are tested
        against this list and an error is raised if it cannot be satisfied.
        Minor conversions are attempted automatically. These include float |
        int to bool and vice-versa or list to np array.
        """
        return self._permissible_types_list

    @permissible_types_list.setter
    def permissible_types_list(self, val: None | type | List[type]) -> None:
        if not isinstance(val, None | type | List):
            raise ValueError("The permissible types list must be None, "
                             "a type or a List of allowed types.")
        self._permissible_types_list = val
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
