"""
Congruent Engineering Toolbox (CET) Session
===========================================

This file defines a CET program session, which defines the configuration,
program parameters, and working directory.
"""

import logging
from importlib import reload
from os.path import join, split, dirname, isdir, isfile
from os import mkdir
from shutil import copyfile
import sys

import cetpy.Configuration
from cetpy.Configuration import ConfigurationManager


class Session:
    """Congruent Engineering Toolbox Session.

    The sessions aim to improve the repeatability and traceability of
    conducted work in cetpy. The session can read TOML config files within
    the specified folder and also moves the cetpy.log file in the specified
    folder. The log file can act as the proof of work, that parameters were
    recognised and processed correctly.

    Remember to call cetpy.active_session.refresh() if you've made a change
    to the config files.
    """

    def __init__(self, directory: str | None, logging_level: str = 'info'):
        self.logger = None
        self.config_manager = ConfigurationManager(directory)
        self.logging_level = logging_level
        self.directory = directory
        cetpy.sessions += [self]
        cetpy.active_session = self

    def parameter(self, name: str):
        """Return key from the session config dictionary.

        Parameters
        ----------
        name
            The name of the property value being requested.

        Raises
        ------
        KeyError: If the name is not in the dictionary.
        """
        return self.config_manager.parameter(name)

    @property
    def config_dict(self) -> dict:
        """Session config dictionary defining system and program parameters."""
        return self.config_manager.config

    @property
    def name(self) -> str:
        """Return session name."""
        return split(self.directory)[-1]

    @property
    def directory(self) -> str:
        """Session working directory, for session configs and saving."""
        return self._directory

    @directory.setter
    def directory(self, val: str) -> None:
        self._directory = val
        if val is not None:
            if not isdir(val):
                # Create directory if it doesn't exist yet
                mkdir(val)
            config_path = join(val, self.config_manager.session_config_name)
            if not isfile(config_path):
                # Copy default config if no config is present within the target directory.
                # Applies in both cases, if the folder already exists and if it doesn't.
                copyfile(join(dirname(__file__), 'default_session_config.toml'), config_path)
        self.config_manager.directory = val
        self.start_logging()

    def refresh(self) -> None:
        """Refresh this session. Does not restart logging."""
        self.config_manager.reset()
        self.config_manager.load()
        cetpy.Configuration.refresh_module_dict()

    def restart(self) -> None:
        """Restart this session. This both refreshes the config files and
        restarts the logging."""
        self.shutdown()
        self.refresh()

    @staticmethod
    def shutdown() -> None:
        """Shutdown session and logging."""
        logging.shutdown()

    def start_logging(self) -> None:
        """Start a logging process for this session."""
        reload(logging)

        match self.logging_level:
            case 'debug':
                log_level = logging.DEBUG
            case 'info':
                log_level = 15
            case 'warning':
                log_level = logging.WARNING
            case 'critical':
                log_level = logging.CRITICAL
            case _:
                log_level = self.logging_level

        # Log File
        formatter = logging.Formatter(
            '[%(asctime)s:%(filename)15s:%(funcName)25s:%(lineno)3d:%(levelname)8s]: %(message)s')

        directory = self.directory
        if directory is None:
            directory = dirname(__file__)
            name = 'Base'
        else:
            name = self.name

        handler = logging.FileHandler(join(directory, 'cetpy.log'), mode='w')
        handler.setFormatter(formatter)
        handler.setLevel(log_level)

        # Console
        handler_console = logging.StreamHandler()
        handler_console.setStream(sys.stdout)
        handler_console.setLevel(logging.INFO)

        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.addHandler(handler_console)
        logger.propagate = False
        logging.captureWarnings(False)
        self.logger = logger
