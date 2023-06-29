"""
Configuration Manager
=====================

This file defines a Configuration Manager for CET. The manager handles the
interface to the .toml config files, layering of config files,
and traceability.
"""

from os import listdir
from os.path import join, isdir, isfile, abspath, dirname
from typing import List
import pandas as pd
import numpy as np
import pickle
import tomli

import cetpy
import cetpy.Configuration


def get_absolute_path(file_path: str, additional_locations: List[str] = None
                      ) -> str:
    """Return absolute path of a file within the CET config structure.

    A valid absolute path is prioritised above all else. Second the
    additional locations in order of occurrence, and lastly the CET config
    locations from cetpy.Configurations in order of occurrence.
    """
    if isfile(abspath(file_path)):
        return abspath(file_path)
    else:

        for location in (additional_locations +
                         cetpy.Configuration.config_locations):
            abs_path = join(location, file_path)
            if isfile(abs_path):
                return abs_path


def load_config(file_path: str) -> dict:
    """Return dictionary fitting to a TOML file at the specified location."""
    with open(file_path, "rb") as f:
        return tomli.load(f)


def interpret_config(config: dict, additional_locations: List[str] = None
                     ) -> dict:
    """Parse a config dictionary to load numpy, pickle, or csv files."""
    if additional_locations is None:
        additional_locations = []
    out = config.copy()
    for key in out.keys():
        if isinstance(out[key], dict) and 'file' in out[key].keys():
            match out[key]['file'][-3:]:
                case 'csv':
                    out[key] = pd.read_csv(get_absolute_path(
                        out[key]['file'], additional_locations))
                case '.pkl':
                    with open(out[key]['file'], "rb") as f:
                        out[key] = pickle.load(f)
                case '.txt':
                    out[key] = np.loadtxt(out[key]['file'])
                case _:
                    pass
        elif isinstance(out[key], dict):
            out[key] = interpret_config(out[key], additional_locations)
        elif isinstance(out[key], str) and out[key] in ['none', 'None']:
            out[key] = None
    return out


def get_parameter_dict() -> dict:
    """Return parameter dict incorporating all possible parameters
    across all model elements."""
    out = load_config(
        join(dirname(__file__), 'default_config_parameters.toml'))

    def load_parameters_from_dir(file_path: str) -> dict:
        target_dict = {}
        for item in listdir(file_path):
            if isdir(join(file_path, item)):
                sub_dict = load_parameters_from_dir(
                    join(file_path, item))
                if len(sub_dict) > 0:
                    target_dict[item.lower()] = sub_dict
            if isfile(join(file_path, item)) and item[-5:] == '.toml':
                target_dict.update(load_config(join(file_path, item)))
        return target_dict

    for loc in cetpy.Configuration.module_locations:
        out.update(load_parameters_from_dir(loc))
    return out


def get_default_dict(parameter_dict: dict) -> dict:
    """Return a default parameter dict based on a parameter dict."""
    out = {}
    for key, value in parameter_dict.items():
        if isinstance(value, dict) and 'default' in value.keys():
            out[key] = value['default']
        elif isinstance(value, dict):
            out[key] = get_default_dict(value)
    return out


def get_config_keys(config: dict) -> List[str]:
    """Return list of all value keys in the config, split by a dot,
    as would be the case in a config file."""
    keys = list(config.keys())
    for key, val in config.items():
        if isinstance(val, dict):
            keys += [key + '.' + k for k in get_config_keys(val)]
    return keys


def get_parameter(config: dict, key: str | List[str]):
    """Return the parameter value from a config dictionary using either the
    chained name (table headers split by a '.') or a list of keys."""
    if isinstance(key, str):
        key = key.split('.')
    val = config[key[0]]
    if len(key) > 1:
        return get_parameter(val, key[1:])
    else:
        return val


def format_config_header(header: str) -> List[str]:
    """Format the Title header of a config file."""
    lines = ["=" * 80 + "\n"]
    lines += [header.center(80) + "\n"]
    lines += ["=" * 80 + '\n\n']
    return lines


def format_config_table_header(table_level: List[str]) -> List[str]:
    """Format the table header of a config file"""
    return ["[" + ".".join(table_level) + "]"]


def format_value_to_toml(value) -> str:
    """Convert a config value to a print string for a .toml config file."""
    if isinstance(value, float | int | bool):
        return str(value)
    elif isinstance(value, str):
        return "'" + value + "'"
    elif isinstance(value, list):
        return "[" + ",".join([format_value_to_toml(v) for v in value]) + "]"
    elif isinstance(value, dict):
        return "{" + ",".join([f"{k} = {v}" for k, v in value.items()]) + "}"
    else:
        raise ValueError(
            f"Value type could not be converted to toml: {type(value)}")


def format_config_key(key: str, value, parameter_entry: dict) -> List[str]:
    """Format the table header of a config file"""
    value = format_value_to_toml(value)
    lines = [key + ' = ' + value]
    match parameter_entry.get('type', None), parameter_entry.get('unit', None):
        case None, None:
            pass
        case _, None:
            lines += ["(" + parameter_entry['type'] + ")"]
        case None, _:
            lines += ["[" + parameter_entry['unit'] + "]"]
        case _, _:
            lines += ["(" + parameter_entry['type'] + ") ["
                      + parameter_entry['unit'] + "]"]
        case _:
            pass

    lines += [parameter_entry['description']]
    return lines


def format_toml_body_text(default_dict: dict, parameter_dict: dict,
                          table_master_level: list = None) -> List[str]:
    """Format the body text of a TOML config file."""
    if table_master_level is None:
        table_master_level = []
    lines = []

    keys_values = [key for key, value in default_dict.items()
                   if not isinstance(value, dict)]
    keys_sub_keys = [key for key in default_dict.keys() if key not in
                     keys_values]
    for key in keys_values:
        lines += format_config_key(
            key, default_dict[key], parameter_dict[key])
    lines += ["\n\n"]

    for key in keys_sub_keys:
        table_sub_level = table_master_level + [key]
        lines += format_config_table_header(table_sub_level)
        parameter_sub_dict = parameter_dict[key]
        default_sub_dict = default_dict[key]
        lines += format_toml_body_text(default_sub_dict, parameter_sub_dict,
                                       table_sub_level)
    return lines


class ConfigurationManager:
    """CET Config Manger for loading, layering, and tracing configs."""

    __slots__ = ['_parameter_dict', '_config', '_configs', '_config_keys',
                 '_directory', '_session_config_name']

    def __init__(self, directory: str | None = None,
                 session_config: str | None = None) -> None:
        self._parameter_dict = {}
        self._config = {}
        self._configs = {}
        self._config_keys = []
        self.session_config_name = session_config
        self.directory = directory

    def reset(self) -> None:
        """Clear all loaded configs."""
        self._parameter_dict = {}
        self._config = {}
        self._configs = {}

    @property
    def config(self) -> dict:
        """Session config dictionary defining system and program parameters."""
        return self._config

    @property
    def directory(self) -> str | None:
        """Active directory for the config manager."""
        return self._directory

    @directory.setter
    def directory(self, val: str) -> None:
        self._directory = val
        self.load()

    @property
    def session_config_name(self) -> str:
        """Name of the session config file. If none is specified,
        the default 'session_config.toml' is used."""
        if self._session_config_name is None:
            return "session_config.toml"
        else:
            return self._session_config_name

    @session_config_name.setter
    def session_config_name(self, val: str | None) -> None:
        self._session_config_name = val

    @property
    def parameter_dict(self) -> dict:
        """Return dictionary of parameter properties, e.g. unit, data type,
        and description."""
        if len(self._parameter_dict) == 0:
            self._parameter_dict = get_parameter_dict()
        return self._parameter_dict

    @property
    def default_dict_raw(self) -> dict:
        """Return uninterpreted version of the default dictionary."""
        return get_default_dict(self.parameter_dict)

    @property
    def config_keys(self) -> List[str]:
        """Return list of all value keys in the config, split by a dot,
        as would be the case in a config file."""
        return self._config_keys

    def parameter(self, key: str | List[str]):
        """Return config parameter based on chained key string split by '.'
        or list of keys."""
        return get_parameter(self._config, key)

    def __get_abs_path__(self, file_path: str) -> str:
        """Return absolute path of a file within the CET config structure."""
        return get_absolute_path(file_path, [self.directory])

    def __get_config__(self, file_path: str) -> dict:
        """Return dictionary fitting to a TOML file at the specified location
        while finding a relevant config in the CET config structure."""
        return load_config(self.__get_abs_path__(file_path))

    def __load_config_amendment__(self, config: dict,
                                  config_dict: dict = None,
                                  allow_user_directory: bool = True
                                  ) -> dict:
        """Overload a config with its stated amendments."""
        if 'config_amendment' in config.keys():
            config_in = config.copy()
            config_amendment = config_in['config_amendment']
            if not isinstance(config_amendment, list):
                config_amendment = [config_amendment]
            for cam in config_amendment:
                if allow_user_directory:
                    config_add = self.__get_config__(cam)
                else:
                    config_add = load_config(get_absolute_path(cam))
                # Add config to list first, to preserve initial state
                # and prioritisation.
                if config_dict is not None:
                    config_dict[cam] = config_add
                config_add = self.__load_config_amendment__(
                    config_add, config_dict, allow_user_directory)
                config_in.update(config_add)
        return config

    def load(self) -> None:
        """Load the configuration at the specified directory."""
        # Always start from zero for a clean load.
        # region Default Config
        config = self.default_dict_raw
        self._configs = {'default': config.copy()}
        self._config = self.__load_config_amendment__(
            config, self._configs, False)
        # endregion

        # region Session Config
        if self.directory is not None:
            config = self.__get_config__(self.session_config_name)
            self._configs['session'] = config.copy()
            self._config.update(self.__load_config_amendment__(
                config, self._configs, True))

        self._config = interpret_config(self._config, [self.directory])
        # endregion

        self._config_keys = get_config_keys(self.config)

    def write_session_config_template(self, file_path: str = None) -> None:
        """Write template for the session config. The session config
        documents the available options with all installed modules, units,
        acceptable inputs, and parameter descriptions."""
        directory = self.directory
        if directory is None and file_path is None:
            raise ValueError("The config manager needs a defined directory "
                             "to write the session config template.")
        if file_path is None and isfile(
                join(directory, 'session_config.toml')):
            raise FileExistsError("Session config already exists. Delete "
                                  "this file first before writing again.")
        if file_path is None:
            file_path = join(directory, 'session_config.toml'),

        # region Write Text
        lines = format_config_header(
            "Congruent Engineering Toolbox Session Config")
        lines += format_toml_body_text(
            self.default_dict_raw, self.parameter_dict)
        # endregion

        with open(file_path, 'w') as f:
            [f.write("#" + line + '\n') for line in lines]
