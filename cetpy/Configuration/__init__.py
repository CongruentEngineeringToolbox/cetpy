from os.path import dirname, join

from cetpy.Configuration._ConfigurationManager import ConfigurationManager
from cetpy.Configuration._Session import Session

config_locations = [join(dirname(__file__), 'Configurations')]
module_locations = [join(dirname(__file__), '..', 'Modules')]
