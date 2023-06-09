"""
Congruent Engineering Toolbox (CET) Session
===========================================

This file defines a CET program session, which defines the configuration,
program parameters, and working directory.
"""

from cet.Configuration import ConfigurationManager


class Session:
    """Congruent Engineering Toolbox Session."""

    def __init__(self, directory: str, logging_level: str = 'info'):
        self._control_dict = None
        self._config_dict = None
        self.directory = directory
        self._config_manager = ConfigurationManager(directory)
        self.logging_level = logging_level

    def parameter(self, name: str):
        """Return key from the session config or control dictionary.

        The config dictionary is prioritised over the control dictionary.

        Parameters
        ----------
        name
            The name of the property value being requested.

        Raises
        ------
        AttributeError: If the name is not in either of the dictionaries.
        """
        return self.config_dict.get(name, self.control_dict.get(name))

    @property
    def control_dict(self) -> dict:
        """Session control dictionary defining program parameters."""
        return self._control_dict

    @property
    def config_dict(self) -> dict:
        """Session config dictionary defining system parameters."""
        return self._config_dict
