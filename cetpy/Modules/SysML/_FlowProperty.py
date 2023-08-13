"""
FlowProperty
============

This file implements a basic Flow Property. A flow property acts like a
value property on a continuous flow port adding relevant solve functions to
solve the changes in the property along the continuous flow as well as deal
with over- and under determination, solver directions, and discontinuities.

An example of the use of the FlowProperty may be the pressure, temperature,
and mass-flow which could define the fluid condition in a fluid system. Note
the fluid itself, which describes the behaviour of the flow, should be a
flow item, shared by all fluid elements, not a FlowProperty. The chemical
composition of a reacting gas on the other hand, should be a FlowProperty.
"""

from __future__ import annotations

from typing import Any, Tuple, Callable, List

from cetpy.Modules.SysML._ValueProperty import ValueProperty, ValuePropertyDoc
from cetpy.Modules.Utilities.InputValidation import validate_input


class FlowProperty(ValueProperty):
    """FlowProperty which extends the ValueProperty to add solver functions
    along a continuous flow. Such as for pressure, temperature,
    and mass-flow in a fluid system."""
    __slots__ = ['_name_instance_input', '_name_instance_direction',
                 '_name_instance_recalculate']

    def __init__(self,
                 unit: str = None, axis_label: str = None,
                 necessity_test: float = 0.1) -> None:
        self._name_instance_input = ''
        self._name_instance_direction = ''
        self._name_instance_recalculate = ''
        super().__init__(unit, axis_label, necessity_test=necessity_test)

    # region Decorators
    def getter(self, fget: Callable) -> ValueProperty:
        raise AttributeError("Flow properties do not have getter methods.")

    def setter(self, fset: Callable) -> ValueProperty:
        raise AttributeError("Flow properties do not have setter methods.")

    def deleter(self, fdel: Callable) -> ValueProperty:
        raise AttributeError("Flow properties do not have deleter methods.")
    # endregion

    # region Getters and Setters
    def __set_name__(self, cls, name):
        ValueProperty.__set_name__(self, cls, name)
        self._name_instance_input = '_' + name + '_input'
        self._name_instance_direction = '_' + name + '_direction'
        self._name_instance_recalculate = '_d' + name + '_recalculate'
        cls.__flow_properties__ += [self]

    def __init_private_attributes__(self, instance):
        """Create relevant private attributes on the port."""
        instance.__setattr__(self._name_instance, 0)
        instance.__setattr__(self._name_instance_reset, 0)
        instance.__setattr__(self._name_instance_input, None)
        instance.__setattr__(self._name_instance_direction, 'downstream')

    def __create_block_delta_attribute__(self, instance) -> None:
        """Create an inlet and outlet delta attribute for port parent block."""
        name = self._name
        name_delta = 'd' + name
        name_update = '_d' + name + '_solve'
        name_recalculate = self._name_instance_recalculate
        name_stored = '_d' + name + '_stored'

        # Try if it exists, if yes, exit
        try:
            instance.__getattribute__(name_recalculate)
            return
        except AttributeError:
            pass  # Create attributes

        instance.__setattr__(name_recalculate, False)
        instance.__setattr__(name_stored, 0)
        reset_dict = type(instance).__getattribute__('_reset_dict')
        reset_dict.update({self._name_instance_recalculate: True})
        hard_reset_dict = type(instance).__getattribute__('_hard_reset_dict')
        hard_reset_dict.update({name_stored: 0})

        def delta() -> float:
            fr"""Difference in {name} from inlet to outlet [{self.unit}].
            
            ..math::
                \Delta {name} = {name}_o_u_t - {name}_i_n
            """
            if not instance.__getattribute__(name_recalculate):
                instance.__setattr__(name_recalculate, True)
                try:
                    instance.__setattr__(
                        name_stored, instance.__getattribute__(name_update))
                except AttributeError:
                    pass
                instance.__setattr__(name_recalculate, False)
            return instance.__getattribute__(name_stored)

        instance.__setattr__(name_delta, property(delta))

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return instance.__getattribute__(self._name_instance)

    def __set_converging_value__(self, instance, value) -> None:
        instance.__setattr__(self._name_instance, value)

    def __set__(self, instance, value) -> None:
        if not instance.is_endpoint:  # Verify instance is a valid input point
            raise ValueError("A boundary condition cannot be set on an "
                             "internal point of a flow system.")

        val_initial_inp = instance.__getattribute__(self._name_instance_input)
        val_initial = instance.__getattribute__(self._name_instance_reset)

        value = validate_input(value, self.permissible_types_list,
                               self.permissible_list, self._name)
        self.__set_converging_value__(instance, value)
        instance.__setattr__(self._name_instance_input, value)

        if val_initial_inp is None and value is not None:
            # Port was not previously a boundary condition -> update solver
            # direction.
            port_list = instance.flow_system_port_list

            # Evaluate direction and remove opposite boundary condition.
            if instance.is_downstream_endpoint:
                direction = 'upstream'
                port_list[0].__setattr__(self._name_instance_input, None)
            else:
                direction = 'downstream'
                port_list[-1].__setattr__(self._name_instance_input, None)
            for p in port_list:
                self.set_direction(p, direction)

        if self._determination_test is not None:
            self._determination_test.test(instance, self._name)

        if self.__reset_necessary__(instance, value, val_initial):
            instance.__setattr__(self._name_instance_reset, value)
            instance.reset()

    def __delete__(self, instance):
        delattr(instance, self._name_instance)
    # endregion

    # region Input properties
    def get_direction(self, instance) -> str:
        """Solve direction of the flow property at the instance. 'downstream'
        means that the property is set upstream and solved downstream.
        'upstream' means the opposite."""
        return instance.__getattribute__(self._name_instance_direction)

    def set_direction(self, instance, val: str) -> None:
        """Set the solve direction at the instance."""
        if val not in ['upstream', 'downstream']:
            raise ValueError("Direction must be either up- or downstream.")
        for p in instance.flow_system_port_list:
            p.__setattr__(self._name_instance_direction, val)
    # endregion
