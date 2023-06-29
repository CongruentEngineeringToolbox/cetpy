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

from typing import Any, List

from cetpy.Modules.Utilities.Labelling import name_2_display


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
                 allowed_list: List[str | type(object)] = None) -> None:
        self._name = ''
        self._name_instance = ''

        self._super_class = super_class
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
