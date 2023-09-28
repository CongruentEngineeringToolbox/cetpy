"""
FilterFunctions.py
==================

This file specifies functions for creating, splitting, and processing pandas
DataFrame queries.
"""

from typing import List
import re

import pandas as pd


def join_and_filter(filter_list: List[str]) -> str:
    """Join a list of filters into a single string compatible with the
    pandas query function. Filters are joined with an 'and' condition."""
    return "(" + ") and (".join(filter_list) + ")"


def join_or_filter(filter_list: List[str]) -> str:
    """Join a list of filters into a single string compatible with the
    pandas query function. Filters are joined with an 'or' condition."""
    return "(" + ") or (".join(filter_list) + ")"


def split_filter_str(filter_str: str) -> List[str | float]:
    """Break filter string into sub_components, convert floats.

    Examples
    --------
    >>> split_filter_str('scalar < 42')
    ['scalar', '<', 42.0]
    >>> split_filter_str('planet == "Mandalore"')
    ['planet', '==', '"Mandalore"']
    >>> split_filter_str('3.50 <= scalar_2 < 420')
    [3.5, '<=', 'scalar_2', '<', 420.0]
    >>> split_filter_str('1962 > scalar_3 >= 0815')
    [1962.0, '>', 'scalar_3', '>=', 815.0]
    >>> split_filter_str('string != "other_string"')
    ['string', '!=', '"other_string"']
    """
    split_filter = re.split('(==|!=|<=|>=|<|>)', filter_str)
    split_filter = [e.strip() for e in split_filter]
    for i_e, e in enumerate(split_filter):
        try:
            # noinspection PyTypeChecker
            split_filter[i_e] = float(e)
        except ValueError:
            pass
    return split_filter


def interpret_filter(filter_str: str) -> dict:
    """Break filter into sub elements and associate to logical properties.

    Examples
    --------
    >>> interpret_filter('scalar < 42')
    {'key': 'scalar', 'lim_low': None, 'lim_high': 42.0, 'include_low': None, 'include_high': False, 'equals': None, 'not_equals': None}
    >>> interpret_filter('planet == "Mandalore"')
    {'key': 'planet', 'lim_low': None, 'lim_high': None, 'include_low': None, 'include_high': None, 'equals': '"Mandalore"', 'not_equals': None}
    >>> interpret_filter('3.50 <= scalar_2 < 420.0')
    {'key': 'scalar_2', 'lim_low': 3.5, 'lim_high': 420.0, 'include_low': True, 'include_high': False, 'equals': None, 'not_equals': None}
    >>> interpret_filter('1962 > scalar_3 >= 0815')
    {'key': 'scalar_3', 'lim_low': 815.0, 'lim_high': 1962.0, 'include_low': True, 'include_high': False, 'equals': None, 'not_equals': None}
    >>> interpret_filter('string != "other_string"')
    {'key': 'string', 'lim_low': None, 'lim_high': None, 'include_low': None, 'include_high': None, 'equals': None, 'not_equals': '"other_string"'}
    """
    out = {k: None for k in ['key', 'lim_low', 'lim_high', 'include_low',
                             'include_high', 'equals', 'not_equals']}
    split_filter = split_filter_str(filter_str)
    i_k = None
    for i_e, e in enumerate(split_filter):
        if isinstance(e, str) and e not in ['<', '<=', '>=', '>']:
            out['key'] = e
            i_k = i_e
            break
    if out['key'] is None:
        raise ValueError("The filter key could not be identified.")

    if i_k != 0:
        value, operator = split_filter[:i_k]
        if operator == '==':
            out['equals'] = value
        elif operator == '!=':
            out['not_equals'] = value
        else:
            if '<' in operator:
                out['lim_low'] = value
                out['include_low'] = '=' in operator
            elif '>' in operator:
                out['lim_high'] = value
                out['include_high'] = '=' in operator

    if i_k != len(split_filter):
        operator, value = split_filter[i_k + 1:]
        if operator == '==':
            out['equals'] = value
        elif operator == '!=':
            out['not_equals'] = value
        else:
            if '>' in operator:
                out['lim_low'] = value
                out['include_low'] = '=' in operator
            elif '<' in operator:
                out['lim_high'] = value
                out['include_high'] = '=' in operator

    return out


def clean_filter_list(filter_list: List[str]) -> List[str]:
    """Return filter list after combining overlapping filters."""
    filter_list = filter_list.copy()  # Ensure original is not mutated
    new_filter_list = [f for f in filter_list if '==' in f or '!=' in f]
    filter_list = [f for f in filter_list if f not in new_filter_list]
    explored_filters = []
    for f_active in filter_list:
        if f_active in explored_filters: continue
        # Split into elements, then identify the key, search for secondary
        # occurrences of key, test limits
        active = interpret_filter(f_active)
        # Throw out keys that also match with appended _ to ensure exact match
        supplementary_filters = [f for f in filter_list
                                 if active['key'] in f
                                 and not active['key'] + '_' in f
                                 and not '_' + active['key'] in f]
        if len(supplementary_filters) == 0:
            continue

        # Always replace with narrower definition
        for f_sup in supplementary_filters:
            sup = interpret_filter(f_sup)
            if sup['lim_low'] is not None:
                if active['lim_low'] is None or (
                        sup['lim_low'] > active['lim_low']):
                    active['lim_low'] = sup['lim_low']
                    active['include_low'] = sup['include_low']
                elif active['lim_low'] is None or (
                        sup['lim_low'] == active['lim_low']):
                    active['include_low'] = sup[
                        'include_low'] and active['include_low']
            if sup['lim_high'] is not None:
                if active['lim_high'] is None or (
                        sup['lim_high'] < active['lim_high']):
                    active['lim_high'] = sup['lim_high']
                    active['include_high'] = sup['include_high']
                elif active['lim_high'] is None or (
                        sup['lim_high'] == active['lim_high']):
                    active['include_high'] = sup[
                        'include_high'] and active['include_high']
            explored_filters += [f_sup]

        if None not in [active['lim_low'], active['lim_high']]:
            new_str = (
                str(active['lim_low']) + ' <' + '=' * active['include_low']
                + ' ' + active['key'] + ' <' +
                '=' * active['include_high'] + ' ' + str(active['lim_high']))
        else:
            new_str = (
                active['key'] + ' '
                + ('>' + '=' * active['include_low'] + ' ' +
                   str(active['lim_low'])) * (active['lim_low'] is not None)
                + ('<' + '=' * active['include_high'] + ' ' +
                   str(active['lim_high'])) * (active['lim_high'] is not None))
        new_filter_list += [new_str]

    return new_filter_list


def apply_filter(df: pd.DataFrame, filter_list: str | List[str],
                 join_and: bool = True) -> pd.DataFrame:
    """Return a dataframe with a str or list of string filter applied.

    Parameters
    ----------
    df
        Original dataframe. This is not modified.
    filter_list
        A string or list of string conditions compatible with the pandas query
        command. A list of string is automatically joined into one string
        and applied at the same time. Use the subsequent selector to choose
        between 'or' and 'and' default joints.
    join_and: optional, default = True
        Bool switch if the individual filters should be joined with an 'and'
        condition.

    See also
    --------
    pd.Dataframe.query
    """
    if filter_list is None or filter_list == []:
        return df.copy()
    elif isinstance(filter_list, str):
        if filter_list in ['', '()']:
            return df.copy()
        else:
            return df.query(filter_list)
    elif isinstance(filter_list, list):
        if join_and:
            filter_str = join_and_filter(filter_list)
        else:
            filter_str = join_or_filter(filter_list)
        return df.query(filter_str)


def drop_no_variance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return input dataframe without any columns that do not have any
    variance."""
    return df.drop(columns=df.columns[df.nunique() == 1])


def apply_one_hot_encoding(df: pd.DataFrame, input_keys: List[str]
                           ) -> pd.DataFrame:
    """Return one-hot-encoded version of the input dataframe. The
    dataframe remains split into input and output columns and the new
    columns are added at the end of each section. Utilises the
    pandas get_dummies function.

    See Also
    --------
    pd.get_dummies
    """
    last_input_key = [col for col in df.columns if col in input_keys][-1]
    idx_split = df.columns.get_loc(last_input_key) + 1
    df_input = pd.get_dummies(df.iloc[:, :idx_split])
    df_output = pd.get_dummies(df.iloc[:, idx_split:])
    return pd.concat((df_input, df_output), axis=1)
