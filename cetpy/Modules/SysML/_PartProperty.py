"""
SysML PartProperty
==================

This file implements a basic SysML Part Property. Moreover it handles the
setting and replacing of part properties, ensuring the references are
correctly broken and created and providing a more convenient interface to
selecting the desired model.

References
----------
SysML Documentation:
    https://sysml.org/.res/docs/specs/OMGSysML-v1.4-15-06-03.pdf
"""

from __future__ import annotations

from typing import Any, List, Tuple

from cetpy.Modules.Utilities.Labelling import name_2_display
import cetpy.Configuration


class PartPropertyDoc:
    """Simple Descriptor to pass a more detailed doc string to a part
    property."""

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            # noinspection PyProtectedMember
            return instance._super_class.__doc__

    def __set__(self, instance, value: str) -> None:
        if instance is not None and instance.fget is not None:
            instance.fget.__doc__ = value


class PartProperty:
    """SysML ValueProperty which adds units of measure and error
    calculation."""
    __slots__ = ['_name', '_name_instance', '_super_class', '_allowed_list']

    __doc__ = PartPropertyDoc()

    def __init__(self, super_class: type(object),
                 allowed_list: List[str | type(object) | None] = None) -> None:
        self._name = ''
        self._name_instance = ''

        self._super_class = super_class
        if allowed_list is None:
            allowed_list = [super_class]
        if any([isinstance(a, str) for a in allowed_list]):
            for i_a, a in allowed_list:
                if isinstance(a, str):
                    allowed_list[i_a] = cetpy.Configuration.get_module(a)
        self._allowed_list = allowed_list

    # region Getters and Setters
    def __set_name__(self, instance, name):
        self._name = name
        self._name_instance = '_' + name

    def str(self, instance) -> str:
        """Return formatted display name of the part."""
        return self.value(instance).name_display

    def value(self, instance) -> Any:
        """Return the part property value."""
        return self.__get__(instance)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        name = self._name_instance
        if name in instance.__slots__:
            return instance.__getattribute__(name)
        else:
            return instance.__dict__.get(name, None)

    def __set__(self, instance, value) -> None:
        if value is None:
            if None in self._allowed_list:
                instance.__setattr__(self._name_instance, None)
                instance.reset()
                return
            value = self._super_class
        if isinstance(value, str):
            match_list = [
                cls for cls in self._allowed_list
                if value.lower().replace('_', '') == cls.__name__.lower()]
            if len(match_list) > 0:
                value = match_list[0]
            else:
                allowed_string = ', '.join([
                    cls.__name__.lower() for cls in self._allowed_list])
                raise ValueError(
                    f"The desired class could not be evaluated from the "
                    f"input {value}. Allowed classes are: {allowed_string}")
        if self._super_class in value.mro():
            initial_part = self.__get__(instance)
            part = value(name=self._name, parent=instance,
                         **instance.__init_kwargs__)
            instance.__setattr__(self._name_instance, part)
            if initial_part is not None:
                # noinspection PyUnresolvedReferences, PyDunderSlots
                initial_part.parent = None
                self._super_class.__set_state__(
                    part, self._super_class.__get_state__(initial_part))
            instance.reset()
        else:
            raise ValueError(f"The part property input is not an instance of "
                             f"the allowed super class {self._super_class}, "
                             f"instead it is {type(value)}.")
    # endregion

    # region Labelling
    @property
    def name_display(self) -> str:
        """Return the display formatted name of the value property."""
        return name_2_display(self._name)
    # endregion


class PartsProperty(PartProperty):
    """Extension of the cetpy SysML PartProperty class additionally
    incorporating support for multiplicity. Value returns are aggregate
    values of the individual parts."""
    __slots__ = ['_multiplicity']

    def __init__(self, super_class: type(object),
                 allowed_list: List[str | type(object)] = None,
                 multiplicity: int | List[int | None]
                 | Tuple[int | None] | str | None = None
                 ) -> None:
        super().__init__(super_class, allowed_list)
        self.multiplicity = multiplicity

    # region Getters and Setters
    def __set__(self, instance, value) -> None:
        if value is None:
            instance.__setattr__(self._name_instance, [])
            instance.reset()
            return
        if not isinstance(value, list):
            value = [value]
        for i_v, val in enumerate(value):
            if isinstance(val, str):
                match_list = [
                    cls for cls in self._allowed_list
                    if val.lower().replace('_', '') == cls.__name__.lower()]
                if len(match_list) > 0:
                    value[i_v] = match_list[0]
                else:
                    allowed_string = ', '.join([
                        cls.__name__.lower() for cls in self._allowed_list])
                    raise ValueError(
                        f"The desired class could not be evaluated from the "
                        f"input {val}. Allowed classes are: {allowed_string}")

        if not all([self._super_class not in val.mro() for val in value]):
            raise ValueError(f"The parts property input are not instances of "
                             f"the allowed super class {self._super_class}, "
                             f"instead they are "
                             f"{[type(val) for val in value]}.")

        initial_parts = self.__get__(instance)
        parts = []
        for val in value:
            parts += [val(name=self._name, parent=instance,
                          **instance.__init_kwargs__)]
        instance.__setattr__(self._name_instance, parts)
        for i_p, i_part in enumerate(initial_parts):
            # noinspection PyUnresolvedReferences, PyDunderSlots
            i_part.parent = None
            if len(parts) > i_part:
                self._super_class.__set_state__(
                    parts[i_p], self._super_class.__get_state__(i_part))
        instance.reset()
    # endregion

    # region Input Properties
    @property
    def multiplicity(self) -> int | List[int | None] | Tuple[int | None]:
        """Allowed number of instances of a given part property."""
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, val: int | List[int | None]
                     | Tuple[int | None] | str | None) -> None:
        if isinstance(val, str):
            if '.' not in val and ',' not in val:
                if val == '*':
                    val = None
                else:
                    val = int(val)
            elif ',' in val:
                val = [int(v) if v != '*' else None for v in val.split(',')]
            elif '..' in val:
                val = tuple([
                    int(v) if v != '*' else None for v in val.split('..')])
            else:
                raise ValueError(
                    "Input string could not be parsed. allowed are 0, "
                    "positive integers, '*' for infinity, and the separators "
                    "',' for a list of possibilities, and '..' for lower and "
                    "upper bounds.")
        if val is None:
            self._multiplicity = 1
        elif isinstance(val, int):
            self._multiplicity = val
        elif isinstance(val, tuple) and len(val) == 2 and (
                isinstance(val[0], int | None)
                and isinstance(val[1], int | None)):
            self._multiplicity = val
        elif isinstance(val, list) and all([
                isinstance(val[i], int | None) for i in range(len(val))]):
            self._multiplicity = val
        else:
            raise ValueError("Invalid multiplicity input. Input must be "
                             "either None (== 1), an integer, list of "
                             "integers, or a string e.g. '1', '0,4', '0..*'.")
    # endregion
