"""
ProxyProperty
=============

This file implements a Proxy Property, this property simply passes a
ValueProperty of a part through to its parent without further modification.
The purpose is accessibility. Making the specific part property accessible
to the user at a higher system structure level, while minimizing developer
effort and chance for error.
"""

from __future__ import annotations

from cetpy.Modules.Utilities.Labelling import name_2_display


class ProxyPropertyDoc:
    """Simple Descriptor to pass a more detailed doc string to a proxy
    property."""

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            # noinspection PyProtectedMember
            return f"{getattr(instance, '_name')} property of the " \
                   f"{instance.part_name} part."

    def __set__(self, instance, value: str) -> None:
        pass


class ProxyProperty:
    """ProxyProperty which makes a part's property accessible on the
    parent level."""
    __slots__ = ['_name', 'part_name']

    __doc__ = ProxyPropertyDoc()

    def __init__(self, part_name: str,
                 part_property_name: str | None = None) -> None:
        self._name = ''

        self.part_name = part_name
        self._name = part_property_name

    # region Getters and Setters
    def __set_name__(self, instance, name):
        if self._name in ['', None]:
            self._name = name

    def str(self, instance) -> str:
        """Return formatted display name of the property value."""
        return str(getattr(getattr(instance, self.part_name), self._name))

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(getattr(instance, self.part_name), self._name)

    def __set__(self, instance, value) -> None:
        setattr(getattr(instance, self.part_name), self._name, value)
    # endregion

    # region Labelling
    @property
    def name_display(self) -> str:
        """Return the display formatted name of the value property."""
        return name_2_display(self._name)
    # endregion
