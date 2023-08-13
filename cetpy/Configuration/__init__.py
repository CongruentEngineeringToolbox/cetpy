from os.path import dirname, join

from cetpy.Configuration._ConfigurationManager import ConfigurationManager
from cetpy.Configuration._Session import Session
from cetpy.Configuration._ModuleManager import \
    generate_block_class_list, get_module

config_locations = [join(dirname(__file__), 'Configurations')]
module_locations = [join(dirname(__file__), '..', 'Modules')]
module_dict = {}


def refresh_module_dict() -> None:
    """Refresh the list of modules of cetpy and extension modules."""
    module_dict.clear()
    module_dict.update(generate_block_class_list())
