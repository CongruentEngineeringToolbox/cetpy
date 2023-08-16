"""
SysML Continuous Port
=====================

This file implements a SysML Port, an interface between multiple SysML
Blocks for Flow Items with relevant functions for continuous flow, such as
fluid flow.
"""

from __future__ import annotations

from typing import List

import cetpy.Modules.SysML as SML


class ContinuousPort(SML.Port):
    """SysML Port element adding functions for continuous flow systems,
    such as fluid flow."""

    __slots__ = ['_upstream_list', '_upstream_port_list', '_downstream_list',
                 '_downstream_port_list']

    def __init__(self,
                 upstream: SML.Block | None = None,
                 downstream: SML.Block | None = None,
                 name: str = None, flow_item: object | None = None,
                 upstream_dict_name: str = None,
                 downstream_dict_name: str = None,
                 tolerance: float = None):
        self._upstream_list = None
        self._upstream_port_list = None
        self._downstream_list = None
        self._downstream_port_list = None
        super().__init__(upstream=upstream,
                         downstream=downstream,
                         name=name,
                         flow_item=flow_item,
                         upstream_dict_name=upstream_dict_name,
                         downstream_dict_name=downstream_dict_name,
                         tolerance=tolerance)

    # region System References
    @SML.Port.upstream.setter
    def upstream(self, val: SML.Block | None) -> None:
        SML.Port.upstream.fset(self, val)
        self.__generate_upstream_list__()
        for port in self.downstream_port_list:
            port.__generate_upstream_list__()

    @SML.Port.downstream.setter
    def downstream(self, val: SML.Block | None) -> None:
        SML.Port.downstream.fset(self, val)
        self.__generate_downstream_list__()
        for port in self.upstream_port_list:
            port.__generate_downstream_list__()

    def _update_reference(self, reference: str,
                          val: SML.Block | None):
        val_initial = self.__getattribute__(reference)
        SML.Port._update_reference(self, reference, val)

        if val is val_initial:
            return
        # Update all flow system lists
        self.__generate_upstream_list__()
        self.__generate_downstream_list__()

        for port in self.flow_system_port_list:
            port.__generate_downstream_list__()
            port.__generate_upstream_list__()

        # Update flow item from possible opposite port. This must be done
        # after all new references are established.

        # Step 1: Identify opposite port:
        if reference == '_upstream':
            opposite_name = '_downstream'
        else:
            opposite_name = '_upstream'
        try:
            opposite = val.__getattribute__(self.__getattribute__(
                opposite_name + '_dict_name'))
        except AttributeError:
            opposite = None

        if opposite is None:
            return  # Do nothing
        elif self._flow_item is None and opposite.flow_item is not None:
            # Step 2: Attempt to update own and previously attached elements'
            # flow items using new flow item.
            self.flow_item = opposite.flow_item
        elif self._flow_item is not None and opposite.flow_item is None:
            # Step 3: Attempt to update opposite and its previously attached
            # elements' flow items using own flow item.
            opposite.flow_item = self._flow_item

        # create transfer functions, only if opposite exists (not an end point)
        for fp in self.__flow_properties__:
            fp.__create_block_delta_attribute__(val)

    @property
    def upstream_list(self) -> List[SML.Block]:
        """Return list of upstream blocks with this flow from upstream to
        downstream."""
        if self._upstream_list is None:
            self.__generate_upstream_list__()
        return self._upstream_list

    @property
    def downstream_list(self) -> List[SML.Block]:
        """Return list of downstream blocks with this flow from upstream to
        downstream."""
        if self._downstream_list is None:
            self.__generate_downstream_list__()
        return self._downstream_list

    @property
    def flow_system_list(self) -> List[SML.Block]:
        """Return list of blocks with this flow from upstream to downstream."""
        return self.upstream_list + self.downstream_list

    @property
    def upstream_port_list(self) -> List[ContinuousPort]:
        """Return list of upstream ports from upstream to downstream."""
        if self._upstream_port_list is None:
            self.__generate_upstream_list__()
        return self._upstream_port_list

    @property
    def downstream_port_list(self) -> List[ContinuousPort]:
        """Return list of downstream ports from upstream to downstream."""
        if self._downstream_port_list is None:
            self.__generate_downstream_list__()
        return self._downstream_port_list

    @property
    def flow_system_port_list(self) -> List[ContinuousPort]:
        """Return list of ports from upstream to downstream."""
        return self.upstream_port_list + [self] + self.downstream_port_list

    def __generate_upstream_list__(self) -> None:
        """Update the upstream port list."""
        if self._upstream is None:
            self._upstream_list = []
            self._upstream_port_list = []
        else:
            try:
                next_port = self._upstream.__getattribute__(
                    self._downstream_dict_name)
                self._upstream_list = \
                    next_port.upstream_list + [self._upstream]
                self._upstream_port_list = \
                    next_port.upstream_port_list + [next_port]
            except AttributeError:
                self._upstream_list = [self._upstream]
                self._upstream_port_list = []

    def __generate_downstream_list__(self) -> None:
        """Update the upstream port list."""
        if self._downstream is None:
            self._downstream_list = []
            self._downstream_port_list = []
        else:
            try:
                next_port = self._downstream.__getattribute__(
                    self._upstream_dict_name)
                self._downstream_list = \
                    [self._downstream] + next_port.downstream_list
                self._downstream_port_list = \
                    [next_port] + next_port.downstream_port_list
            except AttributeError:
                self._downstream_list = [self._downstream]
                self._downstream_port_list = []

    @property
    def is_endpoint(self) -> bool:
        port_list = self.flow_system_port_list
        return self is port_list[0] or self is port_list[-1]

    @property
    def is_upstream_endpoint(self) -> bool:
        return self is self.flow_system_port_list[0]

    @property
    def is_downstream_endpoint(self) -> bool:
        return self is self.flow_system_port_list[-1]
    # endregion

    # region Input Properties
    @SML.Port.flow_item.setter
    def flow_item(self, val: object | None) -> None:
        SML.Port.flow_item.fset(self, val)
        # Additionally see if adjacent ports should be updated. This must be
        # done after the own flow item is updated.
        for port in self.upstream_port_list[::-1]:
            if port.flow_item is not None:  # Does not update any ports that
                # have a flow item set and breaks loop.
                break
            # Reset on all ports already called in super() setter method.
            port.__setattr__('_flow_item', self._flow_item)
        for port in self.downstream_port_list:
            if port.flow_item is not None:
                break
            port.__setattr__('_flow_item', self._flow_item)
    # endregion
