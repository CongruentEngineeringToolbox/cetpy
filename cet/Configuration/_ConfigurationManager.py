"""
Configuration Manager
=====================

This file defines a Configuration Manager for CET. The manager handles the
interface to the .toml config files, layering of config files,
and traceability.
"""

from os.path import join, isdir, isfile, abspath
from typing import List
import pandas as pd
import numpy as np
import pickle
import tomli

import cet
import cet.Configuration


def get_absolute_path(file_path: str, additional_locations: List[str] = None
                      ) -> str:
    """Return absolute path of a file within the CET config structure.

    A valid absolute path is prioritised above all else. Second the
    additional locations in order of occurrence, and lastly the CET config
    locations from cet.Configurations in order of occurrence.
    """
    if isfile(abspath(file_path)):
        return abspath(file_path)
    else:

        for location in (additional_locations +
                         cet.Configuration.config_locations):
            abs_path = join(location, file_path)
            if isfile(abs_path):
                return abs_path


def load_config(file_path: str) -> dict:
    """Return dictionary fitting to a TOML file at the specified location."""
    with open(file_path, "rb") as f:
        return tomli.load(f)


def interpret_config(config: dict, additional_locations: List[str]) -> dict:
    """Parse a config dictionary to load numpy, pickle, or csv files."""
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
    return out


class ConfigurationManager:
    """CET Config Manger for loading, layering, and tracing configs."""

    __slots__ = ['_parameter_dict', '_config', '_configs', 'directory']

    def __init__(self, directory: str = None) -> None:
        self._parameter_dict = {}
        self._config = {}
        self._configs = {}
        self.directory = directory

    def __get_abs_path__(self, file_path: str) -> str:
        """Return absolute path of a file within the CET config structure."""
        if self.directory is not None:
            directory = self.directory
        elif cet.active_session is not None:
            directory = cet.active_session.directory
        else:
            directory = None
        return get_absolute_path(file_path, [directory])

    def __get_config__(self, file_path: str) -> dict:
        """Return dictionary fitting to a TOML file at the specified location
        while finding a relevant config in the CET config structure."""
        return load_config(self.__get_abs_path__(file_path))

    def load(self) -> None:
        """Load the configuration at the specified directory."""
        # Always start from zero for a clean load.
        self._config = {}
        self._configs = {}

