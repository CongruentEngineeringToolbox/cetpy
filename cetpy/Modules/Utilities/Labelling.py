"""
Labelling Functions
===================

Provides a number of functions for predicting abbreviations, units, labels,
name formatting based on block or value property names.
"""

import re
import numpy as np


def name_2_abbreviation(name: str) -> str:
    """Return a sensible abbreviation for a component name."""
    name_split = name.strip('_').split('_')
    if len(name_split) == 1:
        abbreviation = name_split[0][:3].upper()
    else:
        abbreviation = "".join([n[0].upper() for n in name_split[:3]])
    suffix = re.findall(r'^[A-Za-z0-9]*?([0-9]+)$', name)
    if len(suffix) > 0:
        abbreviation += suffix[0]
    return abbreviation


def name_2_display(name: str) -> str:
    """Return a display formatted version of a block name."""
    return ' '.join([w.title() if w.islower() else w for w in name.split('_')])


# standard units, keys are the trigger and values the unit string
standard_units = {
    'eta': '-',
    'error': '-',
    'ratio': '-',
    'factor of safety': '-',
    'dp': 'Pa',
    'prandtl': '-',
    'dt': 'K',
    'dh': 'J/kg',
    'dmdot': 'kg/s',
    'angle': '-',
    'gamma': '-',
    'kappa': '-',
    'mach': '-',
    'radius': 'm',
    'diameter': 'm',
    'length': 'm',
    'area': 'm^2',
    'volume': 'm^3',
    'thickness': 'm',
    'height': 'm',
    'width': 'm',
    'pressure': 'Pa',
    'stress': 'Pa',
    'temperature': 'K',
    'density': 'kg/m^3',
    'rho': 'kg/m^3',
    'dynamic viscosity': 'Pa*s',
    'kinematic viscosity': 'm^2/s',
    'specific heat capacity': 'J/kgK',
    'thermal conductivity': 'W/mK',
    'enthalpy': 'J/kg',
    'entropy': 'J/kg',
    'heat flux': 'W/m^2',
    'heat transfer coefficient': 'W/m^2K',
    'velocity': 'm/s',
    'mass fraction': '-',
    'mass flow': 'kg/s',
    'mass': 'kg',
    'volume flow': 'm^3/s',
    'gas constant': 'J/kgK',
    'force': 'N',
    'torque': 'Nm',
    'power': 'W',
    'energy': 'J',
    'voltage': 'V',
    'charge': 'C',
    'magnetic flux': 'Wb',
    'capacitance': 'F',
    'resistance': 'ohm',
    'magnetic induction': 'T',
    'frequency': 'Hz',
    'time': 's',
    'rpm': '1/min',
    'index': '-',
    'kv': 'm^3/h',
    'mw': 'kg/kmol',
    'mu': 'Pa*s',
    'nu': 'm^2/s',
    'cp': 'J/kgK',
    'cv': 'J/kgK',
    'x': 'm',
    'y': 'm',
    'z': 'm'
}


def name_2_unit(name: str) -> str:
    """Return unit from property name."""
    keys = [k for k in standard_units.keys()
            if k in name.lower().replace('_', ' ')]
    if len(keys) > 0:
        return standard_units[keys[0]]
    else:
        return '-'


def unit_2_latex(unit: str) -> str:
    """Return Latex formatted unit."""
    # Ensure numbers are raised to the power
    unit_copy = ''
    for char in unit:
        if char in [str(n) for n in range(10)]:
            if (len(unit_copy) == 0
                    or unit_copy[-1] in [
                        str(n) for n in range(10)] + ['^']):
                unit_copy += char
            else:
                unit_copy += '^' + char
        else:
            unit_copy += char
    unit = unit_copy
    # Detect division
    if '/' in unit:
        unit = unit.split('/')
        unit = r'\frac{' + unit[0] + '}{' + unit[1] + '}'

    # Enable math mode
    return '$' + unit + '$'


# standard axis labels, keys are the trigger and values the display string
standard_labels = ['ratio', 'error', 'radius', 'diameter', 'length', 'area',
                   'volume', 'thickness', 'height', 'width', 'stress',
                   'pressure', 'temperature', 'density', 'mass fraction',
                   'mass', 'specific heat capacity', 'thermal conductivity',
                   'enthalpy', 'entropy', 'heat flux',
                   'heat transfer coefficient', 'velocity', 'gas constant',
                   'prandtl', 'force', 'torque', 'power', 'energy', 'voltage',
                   'charge', 'magnetic flux', 'capacitance', 'resistance',
                   'magnetic induction', 'frequency', 'index',
                   'factor of safety']
standard_labels = dict(zip(standard_labels,
                           [label.title() for label in standard_labels]))
standard_labels.update({
    'rpm': 'RPM',
    'dp': 'Pressure Delta',
    'dt': 'Temperature Delta',
    'dh': 'Enthalpy Delta',
    'eta': r'$\eta$',
    'angle': r'$\theta$',
    'gamma': r'$\gamma$',
    'kappa': r'$\kappa$',
    'rho': r'$\rho$',
    'mu': r'$\mu$',
    'nu': r'$\nu$',
    'cp': r'$c_P$',
    'cv': r'$c_V$',
    'mw': r'$M_W$',
    'dynamic viscosity': r'$\mu$',
    'kinematic viscosity': r'$\nu$',
    'mach': 'Mach Number',
    'x': 'x-Coordinate',
    'y': 'y-Coordinate',
    'z': 'z-Coordinate'
})


def name_2_axis_label(name: str) -> str:
    """Return axis label from property name."""
    if 'eta' in name:  # Prioritise efficiency
        return standard_labels['eta']
    elif 'ratio' in name:  # Prioritise ratio
        return standard_labels['ratio']
    else:
        keys = [k for k in standard_labels.keys()
                if k in name.lower().replace('_', ' ')]
        if len(keys) > 0:
            return standard_labels[keys[0]]
        else:
            return name.title().replace('_', ' ')


unit_abbreviations = {
    'm': 'meters',
    'kg': 'kilogram',
    'J': 'Joules',
    'K': 'Kelvin',
    'Pa': 'Pascal',
    'mol': 'Mole',
    's': 'Seconds',
    'A': 'Ampere',
    'Hz': 'Hertz',
    'C': 'Coulomb',
    'cd': 'Candela',
    'V': 'Volt',
    'W': 'Watt',
    'N': 'Newton',
    'rad': 'Radian',
    'F': 'Farad',
    'ohm': 'Ohm',
    'Wb': 'Weber',
    'T': 'Tesla'
}
si_base_units = ['s', 'm', 'kg', 'A', 'K', 'mol', 'cd']
si_units = ['m', 'kg', 'J', 'K', 'Pa', 'mol', 's', 'A', 'cd']
si_units_signs = si_units + ['/', '^', '(', ')'] + [str(n) for n in range(10)]

unit_base_equivalences = {
    'J': 'kgm^2s^-3',
    'Pa': 'kgm^-1s^-2',
    'Hz': 's^-1',
    'C': 'sA',
    'V': 'kgm^2s^-3A^-1',
    'W': 'kgm^2s^-3',
    'N': 'kgms^-2',
    'rad': '',
    'F': 'kg^-1m^-2s^4A^2',
    'ohm': 'kgm^2s^4A^2',
    'Wb': 'kgm^2s^-2A^-1',
    'T': 'kgs^-2A^-1'
}


def unit_2_baseSI(unit: str) -> str:
    """Convert a unit to base SI units."""
    unit_out = unit
    for key, value in unit_base_equivalences:
        unit_out.replace(key, value)
    return unit_out


standard_unit_magnitude_abbreviations = [
    'E', 'P', 'T', 'G', 'M', 'k', 'h', 'da', '', 'd', 'c', 'm', 'µ', 'n',
    'p', 'f'
]
standard_unit_magnitude_long_form = [
    'Exo', 'Peta', 'Tera', 'Giga', 'Mega', 'kilo', 'hecta', 'deka', '', 'deci',
    'centi', 'mili', 'micro', 'nano', 'pico', 'femto'
]
standard_unit_magnitudes = np.array([
    1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-6, 1e-9,
    1e-12, 1e-15
])


def scale_value(value: float) -> (float, str):
    """Apply base 3 scaling prefixes to a value and return new value/prefix."""
    if value == 0 or np.isnan(value) or np.isinf(value):
        index_min = 8
    else:
        index_min = np.where((abs(value) > standard_unit_magnitudes))[0][0]
    if index_min in [6, 7, 9, 10]:  # restrict to base 3
        index_min = 8
    value = value / standard_unit_magnitudes[index_min]
    return value, standard_unit_magnitude_abbreviations[index_min]


def scale_value_unit(value: float, unit: str) -> (float, str):
    """Apply base 3 scaling prefixes to a value and unit pair incorporating
    handling of units with exponents."""
    val_scaled, prefix = scale_value(value)

    if not any([str(i) in unit for i in range(10)]) or prefix == '':
        # No higher order unit exponents, direct output is correct.
        return val_scaled, prefix + unit
    else:
        if 'm^' in unit:
            # Try special handling of areas and volumes, as these quickly lead
            # to vastly scaled properties
            if '/' not in unit or unit.find('m^') < unit.find('/'):
                # No division or area in numerator
                u = 'm^2'
                if u in unit:
                    match prefix:
                        case 'E':
                            return val_scaled, unit.replace(u, 'T' + u)
                        case 'T':
                            return val_scaled, unit.replace(u, 'M' + u)
                        case 'M':
                            return val_scaled, unit.replace(u, 'k' + u)
                        case 'µ':
                            return val_scaled, unit.replace(u, 'm' + u)
                        case 'p':
                            return val_scaled, unit.replace(u, 'µ' + u)
                        case _:
                            return value, unit
                u = 'm^3'
                if u in unit:
                    match prefix:
                        case 'E':
                            return val_scaled, unit.replace(u, 'M' + u)
                        case 'G':
                            return val_scaled, unit.replace(u, 'k' + u)
                        case 'n':
                            return val_scaled, unit.replace(u, 'm' + u)
                        case _:
                            return value, unit
                return value, unit
            else:
                # Denominator
                u = 'm^2'
                if u in unit:
                    match prefix:
                        case 'T':
                            return val_scaled, unit.replace(u, 'µ' + u)
                        case 'M':
                            return val_scaled, unit.replace(u, 'm' + u)
                        case 'µ':
                            return val_scaled, unit.replace(u, 'k' + u)
                        case 'p':
                            return val_scaled, unit.replace(u, 'M' + u)
                        case _:
                            return value, unit
                u = 'm^3'
                if u in unit:
                    match prefix:
                        case 'G':
                            return val_scaled, unit.replace(u, 'm' + u)
                        case 'm':
                            return val_scaled, unit.replace(u, 'L')
                        case 'n':
                            return val_scaled, unit.replace(u, 'k' + u)
                        case _:
                            return value, unit
                return value, unit
        else:
            return value, prefix + '(' + unit + ')'


def round_sig_figs(value: float, sig_figs: int) -> float:
    """Round a float to a given number of significant figures.

    Parameters
    ----------
    value
        A float value to be rounded.

    sig_figs
        An integer of the number of significant figures to round to. E.g.
        1.23 is three significant figures, as is 478 000 000 or 0.00875.

    Returns
    -------
    float
        A significant figure rounded float value. Note additional zeros are
        not attached. Neither is the .0 removed if the value could be
        expressed by an integer.

    Notes
    -----
    - The function uses a float -> str format -> float conversion, so should
    be considered to be slow.

    - Standard machine number rounding errors still apply. See round(0.075)

    Examples
    --------
    >>> round_sig_figs(1267, 2)
    1300.0

    >>> round_sig_figs(576000, 4)
    576000.0

    >>> round_sig_figs(354.387290, 5)
    354.39

    >>> round_sig_figs(24.2450, 3)
    24.2

    >>> round_sig_figs(1.289, 1)
    1.0

    Machine number rounding errors still apply. Here 0.075 in floating point
    numbers is represented as 0.074999 ... 999 so the round evalues to 0.07
    >>> round_sig_figs(0.075, 1)
    0.07
    """
    return float('{:.{p}g}'.format(value, p=sig_figs))


def ceil_sig_figs(value: float, sig_figs: int) -> float:
    """Round up a float to a given number of significant figures.

    Parameters
    ----------
    value
        A float value to be rounded.

    sig_figs
        An integer of the number of significant figures to round to. E.g.
        1.23 is three significant figures, as is 478 000 000 or 0.00875.

    Returns
    -------
    float
        A significant figure rounded float value. Note additional zeros are
        not attached. Neither is the .0 removed if the value could be
        expressed by an integer.

    Notes
    -----
    - The function uses a series of float -> str format -> float conversion,
    so should be considered to be very slow.

    - Standard machine number rounding errors still apply. See round(0.075)

    Examples
    --------
    >>> ceil_sig_figs(1267, 2)
    1300.0

    >>> ceil_sig_figs(576000, 4)
    576000.0

    >>> ceil_sig_figs(354.387290, 5)
    354.39

    >>> ceil_sig_figs(24.2450, 3)
    24.3

    >>> ceil_sig_figs(1.289, 1)
    2.0
    """
    value_new = float('{:.{p}g}'.format(value, p=sig_figs))
    if value_new >= value:
        return value_new
    else:
        offset = 10. ** np.arange(-10, 10)
        value_new = [float('{:.{p}g}'.format(value + o, p=sig_figs))
                     for o in offset]
        return min([v for v in value_new if v > value])


def floor_sig_figs(value: float, sig_figs: int) -> float:
    """Round up a float to a given number of significant figures.

    Parameters
    ----------
    value
        A float value to be rounded.

    sig_figs
        An integer of the number of significant figures to round to. E.g.
        1.23 is three significant figures, as is 478 000 000 or 0.00875.

    Returns
    -------
    float
        A significant figure rounded float value. Note additional zeros are
        not attached. Neither is the .0 removed if the value could be
        expressed by an integer.

    Notes
    -----
    - The function uses a series of float -> str format -> float conversion,
    so should be considered to be very slow.

    - Standard machine number rounding errors still apply. See round(0.075)

    Examples
    --------
    >>> floor_sig_figs(1267, 2)
    1200.0

    >>> floor_sig_figs(576000, 4)
    576000.0

    >>> floor_sig_figs(354.387290, 5)
    354.38

    >>> floor_sig_figs(24.2450, 3)
    24.2

    >>> floor_sig_figs(1.289, 1)
    1.0
    """
    value_new = float('{:.{p}g}'.format(value, p=sig_figs))
    if value_new <= value:
        return value_new
    else:
        offset = 10. ** np.arange(-10, 10)
        value_new = [float('{:.{p}g}'.format(value - o, p=sig_figs))
                     for o in offset]
        return max([v for v in value_new if v < value])
