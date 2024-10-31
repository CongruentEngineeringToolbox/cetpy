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
from cetpy.Modules.Utilities.Labelling import name_2_unit, name_2_axis_label, scale_value_unit, name_2_display, \
    unit_2_latex
from cetpy.Modules.Utilities.InputValidation import validate_input


def value_property(equation: str = None,
                   determination_test: DeterminationTest | bool = None,
                   necessity_test: float = 0.1,
                   permissible_list: None | List | Tuple = None,
                   permissible_types_list: None | type | List = None,
                   input_permissible: bool = None,
                   no_reset: bool = False
                   ) -> Callable[[Callable], ValueProperty]:
    """Decorator Factory to create ValueProperties from getter functions.

    Use the doc string of the fget function to set a unit for the property.
    The specification must follow the pattern '[unit].'

    See Also
    --------
    ValueProperty
    DeterminationTest
    """

    if isinstance(determination_test, bool):
        if determination_test:
            determination_test = DeterminationTest()
        else:
            determination_test = None

    if input_permissible is None:
        if (permissible_list is not None or permissible_types_list is not None
                or determination_test is not None):
            input_permissible = True
        else:
            input_permissible = False

    def decorator(func: Callable) -> ValueProperty:
        """Decorator to create a ValueProperty with metadata from getter."""
        prop = ValueProperty(fget=func, equation=equation,
                             determination_test=determination_test,
                             necessity_test=necessity_test,
                             permissible_list=permissible_list,
                             permissible_types_list=permissible_types_list,
                             input_permissible=input_permissible,
                             no_reset=no_reset)
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
        value, unit = scale_value_unit(self * 1.0, self.unit)
        return str(value) + ' ' + unit

    def __repr__(self) -> str:
        return self.__str__()


class DeterminationTest:
    """Class to manage over- and under-determination of model inputs.

    See Also
    --------
    ValueProperty
    """

    __slots__ = ['_num', 'auto_fix', 'properties', 'deep_properties',
                 'optional']

    def __init__(self, properties: List[str] = None,
                 num: int = 1, auto_fix: bool = True,
                 deep_properties: List[str] = None,
                 optional: bool = False) -> None:
        """Initialise a determination test.

        Parameters
        ----------
        properties: optional, default = None
            A list of strings for the ValueProperties that are overlapping.
            When adding a determination test to a ValueProperty,
            the ValueProperty is automatically added to the
            DeterminationTest's properties list.
        num: optional, default = 1
            The amount of properties that should be defined to be properly
            determined.
        auto_fix: optional, default = True
            Bool flag whether the determination test should try to
            automatically fix an over- or under-determination. Only possible if
            the desired number of defined properties is 1.
        deep_properties: optional, default = None
            A string list of properties in parts or ports that can count
            towards the determination. If a deep property has a
            determination test, a properly determined determination test
            counts as one determined property. Navigate through the levels
            of the system by separating each level with a '.' in the string.
        optional: optional, default = False
            Whether the property is optional, e.g. if parts of the model can be
            used without this property set, and no warning for
            under-determination should be raised.
        """
        self._num = num
        self.auto_fix = auto_fix
        if properties is None:
            properties: List[str] = []
        self.properties = properties
        if deep_properties is None:
            deep_properties: List[str] = []
        self.deep_properties = deep_properties
        self.optional = optional

    @property
    def num(self) -> int:
        """Number of properties allowed to be fixed."""
        return self._num

    @num.setter
    def num(self, val: int) -> None:
        self._num = int(val)

    def n_actual_deep(self, instance) -> int:
        """Return number of properties that are either fixed or determined
        on an attached node of the instance."""
        n_deep = 0
        for key in self.deep_properties:
            try:
                vp = instance.__deep_get_vp__(key)
                ref = instance.__deep_getattr__(".".join(key.split(".")[:-1]))
                if vp.determination_test is not None:
                    n_deep += vp.determination_test.n_determination(ref) + 1
                else:
                    n_deep += int(vp.fixed(ref))
            except AttributeError:
                pass
        return n_deep

    def vp_fixed(self, instance) -> List[str]:
        """Return value-properties that are currently fixed."""
        return [n for n in self.properties
                if getattr(type(instance), n).fixed(instance)]

    def vp_free(self, instance) -> List[str]:
        """Return value-properties that are currently calculated."""
        return [n for n in self.properties
                if not getattr(type(instance), n).fixed(instance)]

    def actual(self, instance) -> List[str]:
        """Inputs currently defined. Does not include deep attributes."""
        return [n for n in self.properties
                if getattr(type(instance), n).fixed(instance)]

    def n_actual(self, instance) -> int:
        """Number of inputs currently defined."""
        return sum([getattr(type(instance), n).fixed(instance)
                    for n in self.properties]) + self.n_actual_deep(instance)

    def n_determination(self, instance) -> int:
        """Return number of free variables."""
        return self._num - self.n_actual(instance)

    def determined(self, instance) -> bool:
        """Return bool if value property set are correctly determined."""
        n = self.n_determination(instance)
        return n == 0 or (self.optional and n == 1)

    def under_determined(self, instance) -> bool:
        """Return bool if value property set is under-determined."""
        return self.n_determination(instance) > int(self.optional)

    def over_determined(self, instance) -> bool:
        """Return bool if value property set is over-determined."""
        return self.n_determination(instance) < 0

    def test(self, instance, new: str = None) -> None:
        """Test the determination of the instance. If auto fix is on,
        the test attempts to automatically fix it."""
        n_actual = self.n_actual(instance)
        n_target = self._num

        direction = ''
        amendment = ''
        if n_actual == n_target:
            return
        elif self.optional and n_actual == 0:
            return
        elif n_actual > n_target:
            direction = 'over-'
            if self.auto_fix and self._num == 1:
                cls = type(instance)
                actual = self.actual(instance)

                # Take first element in list as higher priority if new isn't
                # set.
                if new is None:
                    new = actual[0]

                [getattr(cls, n).__set_converging_value__(instance, None)
                 for n in self.properties if n != new and n in actual]
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
            f"{', '.join(self.properties + self.deep_properties)} {n_target} "
            f"must be set. Currently {n_actual} are set.{amendment}")


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
    calculation.

    The ValueProperty is a python data descriptor, similar to the pure python
    'property' data descriptor. Like the property's '@property' function
    decorator, the ValueProperty also has a '@value_property()' function
    decorator to generate a ValueProperty from a getter method. As a data
    descriptor, the ValueProperty does not itself contain the value, instead it
    acts as a formatter around an attribute of the owning class. Per default a
    ValueProperties attribute is stored with a preceding '_' to mark it as a
    private attribute. A single ValueProperty instance is shared across all
    instances of a class, while each instance of the class may have its own
    individual private attribute storing the instance's value.

    The ValueProperty's tasks are:
    - protect the private attribute from the user
    - only return the value to the user when the value is solved and accurate
    - inform the model of relevant resolves, if the user updates the value
    - run determination tests to ensure the model is not over- or under-defined
    - conduct type and range checking on new user inputs
    - perform necessity tests on new inputs to determine whether the model
      must be rerun
    - provide more informative output of values to the user, truncating
      based on tolerance and adding units to string output of values
    - provide axis labels and display strings for plotting
    - determine whether a value property is an input or output.

    Property functions can be accessed by calling:
    type(instance).value_property_name.value_property_function_name

    Some may require passing the instance, as the Value Property per default
    does not know from which instance it is being called:
    type(instance).value_property_name.fixed(instance)

    ValueProperty names can be added to various lists of the Block class to
    extend their capabilities.
    - __init_parameters__:      Any properties in this list can be set on
                                initialisation from keyword arguments or from
                                cetpy config files.
    - _reset_dict:              Any attributes in this dictionary (as keys)
                                will be reset to the specified value when an
                                instance reset is called. Useful for simple
                                but slow calculations:

                                @value_property()
                                def slow_property(self) -> float:
                                    if self._slow_property is None:
                                        self._slow_property = slow_function(..)
                                    return self._slow_property
    - _hard_reset_dict:         Any attributes in this dictionary (as keys)
                                will be reset to the specified value when a
                                hard_reset is called on the instance. The
                                hard_reset can be used to reset all solver
                                flags or intermediate results, incase the
                                solvers raised errors or got stuck
                                somewhere. A factory reset so to say.
    - __fixed_parameters__:     Any property names in this list will be set
                                to their value at that time, when a
                                'instance.fixed = True' is called. With a
                                determination test, other over-defining
                                properties are automatically corrected to be
                                outputs. Useful for example if you are
                                designing a pipe for a given flow velocity,
                                but then want to know the flow velocity in
                                alternate load-points.
    - __default_parameters__:   Any properties in this dictionary (as keys)
                                will receive their dictionary value if no
                                other value could be determined in
                                initialisation from either keyword arguments or
                                config files.

    When modifying the lists or dictionaries, make copies of the
    list/dictionary of the inherited class so as not to modify the original.

    See Also
    --------
    property: Pure python simple data descriptor.
    value_property: function decorator to generate a ValueProperty from a
                    getter function.
    DeterminationTest: Helper class to solve model under- and
                       over-determination.

    Examples
    --------
    class Cylinder(Block):

        length = ValueProperty()

        __init_parameters__ = Block.__init_parameters.copy() + [
            'length', 'radius', 'diameter'
        ]
        _reset_dict = Block._reset_dict.copy()
        _reset_dict.update({'_volume': None})

        @value_property(determination_test=True)
        def radius(self) -> float:
            return self.diameter / 2

        @value_property(determination_test=radius.determination_test)
        def diameter(self) -> float:
            return self.radius * 2

        @value_property()
        def area(self) -> float:
            return np.pi * self.radius ** 2

        @value_property()
        def volume(self) -> float:
            if self._volume is None:  # lets pretend this is slow
                self._volume = self.area * self.length
            return self._volume
    """
    __slots__ = ['_name', '_name_instance', '_name_instance_reset',
                 '_determination_test', '_necessity_test', 'equation',
                 '_unit', '_axis_label', 'fget', 'fset', 'fdel', 'ffixed',
                 '_permissible_list', '_permissible_types_list',
                 'input_permissible', 'no_reset']

    __doc__ = ValuePropertyDoc()

    def __init__(self,
                 unit: str = None, axis_label: str = None,
                 fget: Callable = None, fset: Callable = None,
                 fdel: Callable = None, ffixed: Callable = None,
                 equation: str = None,
                 determination_test: DeterminationTest | bool = None,
                 necessity_test: float = 0.1,
                 permissible_list: None | List | Tuple = None,
                 permissible_types_list: None | type | List = None,
                 input_permissible: bool = True,
                 no_reset: bool = False) -> None:
        """Initialise a ValueProperty for a cetpy Block class.

        Properties
        ----------
        unit: optional, default = None
            Unit of measure. If None is specified, the property attempts to
            auto-detect from getter function, if unit is bracketed by '[]'.
            Else, the ValueProperty attempts to auto-generate it from the
            property name. Defaults to base SI units.
        axis_label: optional, default = None
            Axis label used for plotting. If None is specified, the property
            attempts to auto-detect it from the doc-string, looking for
            '..cetpy:axis-label:'. If None is found, the property attempts
            to auto-generate it from the property name.
        fget: optional, default = None
            Custom getter function. If None is supplied, the property will
            return the Block's private attribute ('_' prefix + property
            name). When a getter function is supplied, it will also
            prioritise the private attribute, allowing engineers to
            overwrite sections of the code for testing purposes. Per default
            the input_permissible property is False when setting via the
            decorator, in which case the property always uses the getter
            function. The doc string of the fget function can also be used
            to specify a unit for the property. This must follow the pattern
            '[unit].'
        fset: optional, default = None
            Custom setter function. If None is supplied, the input value is
            directly written to the private attribute of the instance ('_'
            prefix + property name). The Value property performs the
            necessity test, reset, determination test, type and range check,
            regardless if a custom evaluation function is set or not.
        fdel: optional, default = None
            Custom deleter function. If None is set, the ValueProperty
            deletes the private attribute of the instance.
        ffixed: optional, default = None
            Custom function to determine whether the ValueProperty is fixed
            or not. If None is set, it tests whether the private attribute
            is None (not fixed) or not None (fixed).
        equation: optional, default = None
            For error calculations of simple functions. Still a placeholder
            for future error calculations as part of the ValueProperty. Then
            the ValueProperty will automatically for single line equations
            trace the error through the model based on the equation used.
        determination_test: optional, default = None
            A DeterminationTest instance to trace model under- and
            over-determination. Write True to simply add a basic
            Determination Test. Link the determination test in any
            overlapping properties to complete the initialisation.

            class Circle(Block):
                @value_property(determination_test=True)
                def radius(self) -> float:
                    return self.diameter / 2

                @value_property(determination_test=radius.determination_test)
                def diameter(self) -> float:
                    return self.radius * 2
        necessity_test: optional, default = 0.1
            A float multiplier for the tolerance when testing input changes. A
            reset is called on the setting instance if the input changes
            relatively larger than the instance tolerance times the multiplier.
            The default is 0.1, a very conservative estimate. It maintains
            model output tolerance integrity up to y = x^10. Most physical
            systems are rather dampening (multiplier values greater than 1).
            Something like a radius to area transition would require a
            multiplier of 0.5. The higher the multiplier, the fewer resets
            are called as the model converges and the faster and more
            efficient the model is. Consider running a sensitivity study on
            a parameter using the cetpy.CaseTools.CaseRunner to evaluate
            necessary multipliers.
        permissible_list: optional, default = None
            A list of values that are allowed to be set, if a tuple of
            length two is passed, float, int, and vector inputs are tested
            to the min (first value) and max (last value) of the tuple. An
            error is raised if the ranges are violated. Set either value to
            None for negative and positive infinity respectively.
        permissible_types_list: optional, default = None
            A specific type that should be input or a list of permissible
            types.
        input_permissible: optional
            Bool flag, whether the value property should permit the user to
            enter values which overwrite the getter function (if available).
            Default is True if created directly as a class (likely a
            configuration parameter) and False if created via the function
            decorator around a getter function (if no determination test is
            provided) (like a simple output function). it can be overwritten by
            the user after initialisation, but remember it applies to all
            instances of a class.
        no_reset: optional, default = False
            Bool flag whether the property should not trigger resets on new
            inputs. This is useful when it is used to pass values through
            the system and reset necessity checks are conducted elsewhere.
            Consider also using ProxyProperties for negligible overhead for
            such operations.
        """
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
        self.input_permissible = input_permissible
        self.no_reset = no_reset
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
        self.ffixed = ffixed

    # region Decorators
    def getter(self, fget: Callable) -> ValueProperty:
        prop = self
        prop.fget = fget
        return prop

    # ToDo: Figure out type hinting for read access
    def setter(self, fset: Callable[[Any], None]) -> ValueProperty:
        prop = self
        prop.fset = fset
        prop.input_permissible = True
        return prop

    def deleter(self, fdel: Callable) -> ValueProperty:
        prop = self
        prop.fdel = fdel
        return prop

    def fixer(self, ffixed: Callable) -> ValueProperty:
        prop = self
        prop.ffixed = ffixed
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

    def str(self, instance) -> str:
        """Return formatted string of value."""
        value = self.value(instance)
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            value, unit = scale_value_unit(self.value(instance), self.unit)
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

        if self.input_permissible:
            try:
                value = instance.__getattribute__(name)
            except AttributeError:
                value = None
        else:
            value = None

        if value is not None or self.fget is None:
            return value
        else:
            return self.fget(instance)

    def __set_converging_value__(self, instance, value) -> None:
        if not self.input_permissible:
            raise AttributeError(f"{self.name} does not allow inputs. Set "
                                 f"input_permissible to True before input.")

        value = validate_input(value, self.permissible_types_list,
                               self.permissible_list, self._name)
        if self.fset is None:
            instance.__setattr__(self._name_instance, value)
        else:
            self.fset(instance, value)

        if (self._determination_test is not None
                and not getattr(instance, '_resetting')):
            # Disable for initialisation, test separately afterward when
            # the flag is false.
            self._determination_test.test(instance, self._name)

    def __set__(self, instance, value) -> None:
        try:
            val_initial = instance.__getattribute__(self._name_instance_reset)
        except AttributeError:
            val_initial = None
        self.__set_converging_value__(instance, value)
        if self.__reset_necessary__(instance, value, val_initial):
            instance.__setattr__(self._name_instance_reset, value)
            instance.reset()

    def __reset_necessary__(self, instance, value, val_initial=None) -> bool:
        """Return bool if a reset should be performed on an instance if the
        initial value is changed to value.
        """
        if self.no_reset:
            return False
        necessity_test = self.necessity_test
        reset = True
        if necessity_test >= 0:
            try:
                same = (value == val_initial)
            except ValueError as err:
                if "broadcast together with shapes" in err.args[0]:
                    return True  # Capture numpy length difference
                else:
                    raise err
            if isinstance(same, Iterable):
                same = all(same)
            if same:
                return False
            elif type(value) == type(val_initial) or all(
                    [isinstance(v, int | float | bool) for v in [value, val_initial]]):
                if isinstance(value, str):
                    return True
                elif isinstance(value, int | float):
                    return not np.isclose(
                        value, val_initial, rtol=instance.tolerance * necessity_test, atol=0)
                elif isinstance(value, dict) and (len(value) == len(val_initial)) and all([
                        k in val_initial.keys() for k in value.keys()]):
                    return not all([np.isclose(value[k], val_initial[k])
                                    if isinstance(value[k], float | int)
                                    else value[k] == val_initial[k]
                                    for k in value.keys()])
                elif isinstance(value, Iterable | Sized) and (len(value) == len(val_initial)):
                    # noinspection PyUnresolvedReferences
                    if not isinstance(value[0], float | int | bool):
                        return True  # no proximity check necessary
                    else:
                        try:
                            return not all(np.isclose(
                                value, val_initial, atol=0, rtol=instance.tolerance * necessity_test))
                        except TypeError:
                            return True  # Occurs if None in list.
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
        elif self.ffixed is not None:
            return self.ffixed(instance)
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
    def name(self) -> str:
        """Value Property Name"""
        return self._name

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
    def unit_latex(self) -> str:
        """Unit of measure formatted for LaTeX."""
        return unit_2_latex(self.unit)

    @property
    def axis_label(self) -> str:
        """Axis label for visualisation."""
        return self._axis_label

    @axis_label.setter
    def axis_label(self, val: str) -> None:
        self._axis_label = val
    # endregion


class ValuePrinter:
    """Formatter for Block Value Properties to display the property in the
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
            except AttributeError as err:
                # Verify the error occurred on the first request not deeper in
                # the calculation thereof.
                if self._name in err.args[0]:
                    pass
                else:
                    raise err

        return value

    def __get_self__(self, instance, owner=None):
        return super().__get__(instance, owner)
