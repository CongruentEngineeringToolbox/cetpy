"""
SysML Port
==========

This file implements a basic SysML Port, an interface between multiple SysML
Blocks for Flow Items.
"""

from __future__ import annotations
from typing import Any

import cetpy.Modules.SysML
from cetpy.Modules.Report import ReportPort


class Port:
    """SysML Port element."""
    __slots__ = ['_upstream', '_downstream', '_upstream_dict_name',
                 '_downstream_dict_name', 'name', 'parent', '_flow_item',
                 '_resetting', '_tolerance', '__dict__']

    __flow_properties__ = []

    def __init__(self,
                 upstream: cetpy.Modules.SysML.Block | None = None,
                 downstream: cetpy.Modules.SysML.Block | None = None,
                 name: str = None, flow_item: object | None = None,
                 upstream_dict_name: str = None,
                 downstream_dict_name: str = None,
                 tolerance: float = None):
        self._resetting = False
        self._flow_item = None
        self._upstream = None
        self._downstream = None
        self.name = name
        self._upstream = upstream
        self._downstream = downstream
        self._upstream_dict_name = upstream_dict_name
        self._downstream_dict_name = downstream_dict_name
        self.flow_item = flow_item
        self._tolerance = 0
        if tolerance is None:
            tolerance = 0
        self.tolerance = tolerance

        for fp in self.__flow_properties__:
            fp.__init_private_attributes__(self)

        self.report = ReportPort(parent=self)

    # region System References
    @property
    def upstream(self) -> cetpy.Modules.SysML.Block | None:
        """Port upstream element."""
        return self._upstream

    @upstream.setter
    def upstream(self, val: cetpy.Modules.SysML.Block | None):
        self._update_reference('_upstream', val)

    @property
    def downstream(self) -> cetpy.Modules.SysML.Block | None:
        """Port downstream element."""
        return self._downstream

    @downstream.setter
    def downstream(self, val: cetpy.Modules.SysML.Block | None):
        self._update_reference('_downstream', val)

    def _update_reference(self, reference: str,
                          val: cetpy.Modules.SysML.Block | None):
        """Update a specified system reference."""
        val_initial = self.__getattribute__(reference)
        dict_name = self.__getattribute__(reference + '_dict_name')
        if val is val_initial:
            return
        elif val_initial is not None:
            val_initial.ports.remove(self)
            port_copy = self.__copy_unlinked__()
            val_initial.ports += [port_copy]
            val_initial.__setattr__(dict_name, port_copy)
            if reference == '_upstream':
                opposite_name = '_downstream'
            else:
                opposite_name = '_upstream'
            port_copy.__setattr__(opposite_name, val_initial)
            val_initial.reset()
        if val is not None:
            val.ports += [self]
            val.__setattr__(dict_name, self)
            val.reset()
        self.__setattr__(reference, val)

    @property
    def is_endpoint(self) -> bool:
        """Return bool if the port is either a starting or ending node."""
        return True

    @property
    def is_upstream_endpoint(self) -> bool:
        """Return bool if the port is a starting node."""
        return True

    @property
    def is_downstream_endpoint(self) -> bool:
        """Return bool if the port is an ending node."""
        return True

    def __copy_unlinked__(self) -> Port:
        """Return a copy of the port without up- or downstream references."""
        port = type(self)(name=self.name,
                          flow_item=self.flow_item,
                          upstream_dict_name=self._upstream_dict_name,
                          downstream_dict_name=self._downstream_dict_name,
                          tolerance=self.tolerance)
        for fp in self.__flow_properties__:
            # noinspection PyProtectedMember
            for prop in [fp._name_instance, fp._name_instance_input,
                         fp._name_instance_reset, fp._name_instance_direction]:
                port.__setattr__(prop, self.__getattribute__(prop))
        return port

    def __deep_getattr__(self, name: str) -> Any:
        """Get value from block or its parts, solvers, and ports."""
        if '.' not in name:
            return self.__getattribute__(name)
        else:
            name_split = name.split('.')
            return self.__getattribute__(name_split[0]).__deep_getattr__(
                '.'.join(name_split[1:]))

    def __deep_setattr__(self, name: str, val: Any) -> None:
        """Set value on block or its parts, solvers, and ports."""
        if '.' not in name:
            return self.__setattr__(name, val)
        else:
            name_split = name.split('.')
            self.__getattribute__(name_split[0]).__deep_setattr__(
                '.'.join(name_split[1:]), val)
    # endregion

    # region Resetting
    def reset(self) -> None:
        """Reset port, up- and downstream elements."""
        if not self._resetting:
            self._resetting = True
            self.reset_upstream()
            self.reset_downstream()
            # reset self is called by both up- and down-stream resets
            self._resetting = False

    def reset_upstream(self) -> None:
        """Reset port and upstream elements."""
        self.reset_self()
        if self._upstream is not None:
            self._upstream.reset()

    def reset_downstream(self) -> None:
        """Reset port and downstream elements."""
        self.reset_self()
        if self._downstream is not None:
            self._downstream.reset()

    def reset_self(self) -> None:
        """Reset port stored intermediate values."""
        pass
    # endregion

    # region Input Properties
    @property
    def flow_item(self) -> object | None:
        """Port flow item. This object describes the behaviour of the flow."""
        return self._flow_item

    @flow_item.setter
    def flow_item(self, val: object | None) -> None:
        initial_val = self._flow_item
        if initial_val is val:
            return
        else:
            self._flow_item = val
            self.reset()

    @property
    def tolerance(self) -> float:
        """Port solver tolerances."""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, val: float) -> None:
        if val < self._tolerance:
            self.reset()
        self._tolerance = val
    # endregion
