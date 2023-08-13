"""
Module Manager
==============

This file defines function to aggregate a list of modules used in cetpy and
extension modules. It pulls from the cetpy.Configuration module_location
list and aggregates all modules deriving from the base SysML Block class.
"""
import sys
import pkgutil
import os.path
from types import ModuleType
from typing import Dict


import cetpy
import cetpy.Configuration
from cetpy.Modules.SysML import Block


def get_module(name: str) -> cetpy.Modules.SysML.Block:
    """Find and return a module based on its base name from the blocks in
    the module locations of the cetpy configuration."""
    return cetpy.Configuration.module_dict[name]


def generate_block_class_list() -> Dict[str, Block]:
    """Return dictionary of all Block derived classes in across cetpy and
    extension modules."""
    blocks = {}
    for location in cetpy.Configuration.module_locations:
        location = os.path.abspath(location)
        # The issue here is that the location, while on the OS path, can be
        # n-levels deep in this path. So we must strip out the overlap
        # between the path entry and the module locations. Finally, we must
        # swap out the directory separators for '.' package separators.
        overlap = [p for p in sys.path if p in location][-1]
        location_new = location.replace(overlap, '').replace(
            os.path.sep, '.')[1:]

        module_instance = get_submodule_instance(location_new)

        # Verify that the correct module was loaded, also important for
        # security.
        if module_instance.__path__[0] != location:
            raise ValueError(f"The path match failed on the module locations. "
                             f"The desired import path was {location}, "
                             f"the attempted match resulted in "
                             f"{module_instance.__path__[0]}.")

        blocks.update(get_module_recursive_classes(module_instance))
    return blocks


def get_module_recursive_classes(ref_module: ModuleType) -> Dict[str, Block]:
    """Return dictionary of name and type of all classes in a module.
    Function is recursive also pulling from submodules."""
    blocks = {}
    blocks.update(get_module_classes(ref_module))
    for submodule in get_submodules(ref_module).values():
        blocks.update(get_module_recursive_classes(submodule))
    return blocks


def get_module_classes(ref_module: ModuleType) -> Dict[str, Block]:
    """Return dictionary of name and type of all classes in a module."""
    blocks = {}
    module_dict = ref_module.__dict__
    for c in module_dict:
        if isinstance(module_dict[c], type) and Block in module_dict[c].mro():
            blocks.update({c: module_dict[c]})
    return blocks


def get_submodules(ref_module: ModuleType) -> Dict[str, ModuleType]:
    """Return dictionary of name and Module Instance of all submodules of a
    module."""
    mod_list = list([
        i for i in pkgutil.iter_modules(ref_module.__path__,
                                        prefix=ref_module.__name__ + '.')
        if i.ispkg])
    modules = {}
    for submodule in mod_list:
        submodule_instance = get_submodule_instance(submodule.name)
        modules.update({submodule.name: submodule_instance})

    return modules


def get_submodule_instance(name: str) -> ModuleType:
    """Return instance of a submodule."""
    submodule = __import__(name)
    if '.' not in name:
        return submodule
    name_split = name.split('.')[1:]
    for n in name_split:
        if n in submodule.__dict__.keys():
            submodule = submodule.__dict__[n]
    return submodule
