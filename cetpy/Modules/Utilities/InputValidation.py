"""
Input Validation
================

Provides functions related to verifying types, values, and ranges of input
values.
"""
from typing import Any, Tuple, List, Iterable
import numpy as np


def validate_input(value: Any, permissible_types: None | type | List = None,
                   permissible: None | List | Tuple = None,
                   property_name: str = None):
    """Test the input value against the permissible type and value lists
    and correct any minor deviations.

    Properties
    ----------
    value
        The input value which should be verified against the provided criteria.
    permissible_types: optional, default = None
        List of permissible python types which the value should have. Set to
        None to disable. A single type is a permissible input. small
        deviations are corrected automatically. Notably, bool, int,
        and float, values are converted to target types if they are any one of
        bool, int, float, str. Additionally, single values can be converted
        to 1d np arrays, or general iterable types to numpy arrays.
    permissible: optional, default = None
        Description of range testing of the value or individual values. Pass a
        list to describe a set of permissible specific values, or a tuple of
        length 2 to specify a minium and maximum value. The limits are
        inclusive. None is equivalent to -np.inf, and np.inf when specified
        in the first or second index. Pass None to disable.
    property_name: optional, default = None
        An optional property name of the input property, this allows
        provision of clearer error messages.

    Returns
    -------
    Any
        The input value, converted to a permissible type and within a
        permissible range.

    Raises
    ------
    ValueError
        If either the type cannot be matched or automatically corrected,
        or the value is outside the permissible limits or not in the
        permissible set.

    Examples
    --------
    >>> validate_input(1, float, (0, 1))
    1.0
    >>> validate_input(3., bool)
    True
    >>> validate_input(True, float, (0, None))
    1.0
    >>> validate_input(8.756, str)
    '8.756'
    >>> validate_input(42.24, [float, np.ndarray])
    42.24
    >>> validate_input('option1', str, ['option1', 'option2', 'option3'])
    'option1'
    >>> validate_input(-3.6, None, (None, 42))
    -3.6
    >>> validate_input(2.7, np.ndarray)
    array([2.7])
    >>> validate_input([1, 2, 3, 4.5], np.ndarray, [1, 2, 3, 4, 4.5, 5, 5.5])
    array([1. , 2. , 3. , 4.5])
    >>> validate_input((4.5, 6, 8), tuple, (0, 10))
    (4.5, 6, 8)
    >>> try:
    ...     validate_input([3, 2, 8], bool)
    ... except ValueError as err:
    ...     err.args[0]
    'The input value must be one of bool. It is list.'
    >>> try:
    ...     validate_input(72.1, float, (0, 13.2), 'asimov')
    ... except ValueError as err:
    ...     err.args[0]
    'The input value for asimov must be between 0 and 13.2. It is 72.1.'
    >>> try:
    ...     validate_input('option6', str, ['option1', 'option2', 'option3'])
    ... except ValueError as err:
    ...     err.args[0]
    'The input value must be one of option1, option2, option3. It is option6.'
    >>> try:
    ...     validate_input('option6', str, ['option1', 'option2', 'option3'])
    ... except ValueError as err:
    ...     err.args[0]
    'The input value must be one of option1, option2, option3. It is option6.'
    >>> try:
    ...     validate_input(2, int, [0, 1, 3, 9])
    ... except ValueError as err:
    ...     err.args[0]
    'The input value must be one of 0, 1, 3, 9. It is 2.'
    >>> try:
    ...     validate_input(np.array([5, -3, 7]), np.ndarray, (None, 3))
    ... except ValueError as err:
    ...     err.args[0]
    'The input value must be between -inf and 3. It is between -3 and 7.'
    >>> try:
    ...     validate_input(np.array([2, 9, 3]), np.ndarray, [0, 1, 3, 9])
    ... except ValueError as err:
    ...     err.args[0]
    'The input value must be one of 0, 1, 3, 9.'
    """
    if property_name is None:
        name = ""
    else:
        name = " for " + property_name
    if permissible_types is None:
        pass  # Disabled
    elif isinstance(permissible_types, List) and (
            isinstance(value, tuple(permissible_types))):
        pass  # Already correct
    elif not isinstance(permissible_types, List) and (
            isinstance(value, permissible_types)):
        pass  # Already correct
    elif (permissible_types in [bool, int, float, str]
          and isinstance(value, bool | int | float)):
        # Easy conversion
        value = permissible_types(value)
    elif permissible_types is np.ndarray and isinstance(
            value, bool | int | float):
        # Create array of length 1 with value
        value = np.atleast_1d(value)
    elif permissible_types is np.ndarray and isinstance(value, Iterable):
        # Convert List, Tuple, etc. to numpy array
        value = np.asarray(value)
    else:
        if not isinstance(permissible_types, List):
            permissible_types = [permissible_types]  # For unified error
        raise ValueError(
            f"The input value{name} must be one of "
            f"{','.join([p.__name__ for p in permissible_types])}. "
            f"It is {type(value).__name__}.")

    if permissible is None:
        return value
    elif isinstance(permissible, Tuple):
        if permissible[0] is None:
            permissible = (-np.inf, permissible[1])
        if permissible[1] is None:
            permissible = (permissible[0], np.inf)
    if isinstance(value, float | int):
        if isinstance(permissible, List):
            if value not in permissible:
                raise ValueError(
                    f"The input value{name} must be one of "
                    f"{', '.join([str(p) for p in permissible])}. "
                    f"It is {str(value)}.")
        elif isinstance(permissible, Tuple):
            if not (permissible[0] <= value <= permissible[1]):
                raise ValueError(
                    f"The input value{name} must be between "
                    f"{str(permissible[0])} and "
                    f"{str(permissible[1])}. It is {str(value)}.")
        return value
    elif isinstance(value, str):
        if value not in permissible:
            raise ValueError(
                f"The input value{name} must be one of "
                f"{', '.join([str(p) for p in permissible])}. "
                f"It is {value}.")
        else:
            return value
    elif isinstance(value, Iterable):
        if isinstance(permissible, List):
            if not all([v in permissible for v in value]):
                raise ValueError(
                    f"The input value{name} must be one of "
                    f"{', '.join([str(p) for p in permissible])}.")
        elif isinstance(permissible, Tuple):
            if isinstance(value, np.ndarray):  # Much faster for large arrays
                error = not (np.all(value >= permissible[0])
                             and np.all(value <= permissible[1]))
            else:
                error = not all([permissible[0] <= v <= permissible[1]
                                 for v in value])
            if error:
                raise ValueError(
                    f"The input value{name} must be between "
                    f"{str(permissible[0])} and {str(permissible[1])}. "
                    f"It is between {str(np.min(value))} and "
                    f"{str(np.max(value))}.")
        return value
