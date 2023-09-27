"""
Report
======

This file specifies an output formatter for a SysML Block providing
user-friendly and accessible views on the created blocks.
"""
from typing import List, Iterable
import pandas as pd

from cetpy.Modules.SysML import ValueProperty


class Report:
    """SysML Block Output Formatter of the Congruent Engineering Toolbox."""

    __slots__ = ['_parent', '_value_properties']

    def __init__(self, parent):
        self._parent = parent
        self.__generate_value_property_list__()

    # region Value Properties
    def __generate_value_property_list__(self) -> None:
        """Generate a list of value properties of the parent block."""
        block = self._parent
        if block is None:
            self._value_properties = []
        else:
            vp = {}
            cls_list = type(block).mro()
            cls_list.reverse()
            for cls in cls_list:
                cls_vp = [p for p in cls.__dict__.values()
                          if isinstance(p, ValueProperty)]
                cls_dict = dict(zip([p.name for p in cls_vp], cls_vp))
                vp.update(cls_dict)
            self._value_properties = list(vp.values())

    @property
    def value_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block."""
        return self._value_properties

    @property
    def input_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block that are
        fixed and used as inputs."""
        return [p for p in self._value_properties if p.fixed(self._parent)]

    @property
    def output_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block that are not
        fixed and calculated as outputs."""
        return [p for p in self._value_properties
                if p not in self.input_properties]
    # endregion

    # region Report Output
    def __call__(self, *args, **kwargs) -> None:
        return self.report()

    def report(self) -> None:
        """Print report of block and parts to console."""
        [print(line[:-1]) for line in self.get_report_text()]

    def report_self(self) -> None:
        """Print report of block to console."""
        [print(line[:-1]) for line in self.get_report_self_text()]
    # endregion

    # region Report Text
    def get_report_text(self) -> List[str]:
        """Return list of lines for the report of the block and its parts."""
        block = self._parent
        parts = block.parts

        lines = self.get_report_self_text()

        if len(parts) > 0:
            lines += ['\n']
            parts_header = ' ' + block.name_display + "'s Part Reports "
            lines += [parts_header.center(80, '=') + '\n']

            for part in block.parts:
                part_lines = part.report.get_report_text()
                # Add indent for better system depth visualisation
                lines += ['  ' + line for line in part_lines]
                lines += ['\n']

            parts_header = (
                ' ' + block.name_display + "'s Part Reports Complete ")
            lines += [parts_header.center(80, '-') + '\n\n']

        return lines

    def get_report_self_text(self) -> List[str]:
        """Return list of lines for the report of just the block."""
        block = self._parent
        solvers = block.solvers
        ports = block.ports

        lines = self.__get_report_header_text__()
        lines += ['\n']
        lines += self.__get_report_input_text__()
        lines += ['\n']
        lines += self.__get_report_output_text__()

        if len(solvers) > 0:
            lines += ['\n']
            solvers_header = ' ' + block.name_display + "'s Solver Reports "
            lines += ['  ' + solvers_header.center(80, '=') + '\n']

            for sol in solvers:
                solver_lines = sol.report.get_report_text()
                # Add indent for better system depth visualisation
                lines += ['  ' + line for line in solver_lines]

            solvers_header = (
                ' ' + block.name_display + "'s Solver Reports Complete ")
            lines += ['  ' + solvers_header.center(80, '-') + '\n\n']

        if len(ports) > 0:
            lines += ['\n']
            ports_header = ' ' + block.name_display + "'s Port Reports "
            lines += ['  ' + ports_header.center(80, '=') + '\n']

            for port in ports:
                port_lines = port.report.get_report_text()
                # Add indent for better system depth visualisation
                lines += ['  ' + line for line in port_lines]

            ports_header = (
                ' ' + block.name_display + "'s Port Reports Complete ")
            lines += ['  ' + ports_header.center(80, '-') + '\n\n']

        lines += ['\n']
        header = ' ' + self._parent.name_display + ' Complete '
        lines += [header.center(80, '-') + '\n\n\n']
        return lines

    def __get_report_header_text__(self) -> List[str]:
        """Return header list of lines for the report."""
        block = self._parent
        solvers = block.solvers
        ports = block.ports
        parts = block.parts

        lines = []

        lines += ['=' * 80 + '\n']
        header = ' ' + block.name_display + ' '
        lines += [header.center(80, '=') + '\n']
        lines += ['=' * 80 + '\n']

        lines += ['Name: {:>33s}\n'.format(block.name_display)]
        lines += ['Abbreviation: {:>25s}\n'.format(block.abbreviation)]
        lines += ['Class: {:>32s}\n'.format(type(block).__name__)]
        if block.parent is not None:
            lines += ['Parent: {:>31s}\n'.format(block.parent.name_display)]
        lines += ['# Solvers: {:>28d}\n'.format(len(solvers))]
        lines += ['# Ports: {:>30d}\n'.format(len(ports))]
        lines += ['# Parts: {:>30d}\n'.format(len(parts))]

        if len(solvers) > 0:
            solver_lines = ['Solvers: {:30s}\n'.format(
                solvers[0].__class__.__name__)]
            for sol in solvers[1:]:
                solver_lines += ['         {:30s}\n'.format(
                    sol.__class__.__name__)]
            lines += solver_lines
        if len(ports) > 0:
            ports_lines = ['Ports: {:32s}\n'.format(
                ports[0].__class__.__name__)]
            for port in ports[1:]:
                ports_lines += ['       {:32s}\n'.format(
                    port.__class__.__name__)]
            lines += ports_lines
        if len(parts) > 0:
            parts_lines = ['Parts: {:32s}\n'.format(
                parts[0].name_display)]
            for part in parts[1:]:
                parts_lines += ['       {:32s}\n'.format(
                    part.name_display)]
            lines += parts_lines
        lines += ['Tolerance: {:>28.2e}\n'.format(block.tolerance)]

        return lines

    def __get_report_input_text__(self) -> List[str]:
        """Return input list of lines for the report."""
        block = self._parent
        value_properties = self.input_properties

        lines = ["Input\n"]
        lines += ['-----\n']
        for p in value_properties:
            try:
                value = p.__get__(block)
                if not isinstance(value, str | float | int | Iterable):
                    try:
                        value = value.name_display
                    except AttributeError:
                        try:
                            value = value.name
                        except AttributeError:
                            value = p.str(block)
                else:
                    value = p.str(block)

            except (ValueError, AttributeError, TypeError, ZeroDivisionError,
                    NotImplementedError, IndexError):
                value = 'NaN'
            if len(value) < 100:
                lines += ["{:25s}: {:15s}\n".format(
                    p.name_display, value)]

        return lines

    def __get_report_output_text__(self) -> List[str]:
        """Return output list of lines for the report."""
        block = self._parent
        value_properties = self.output_properties

        lines = ["Output\n"]
        lines += ['------\n']
        for p in value_properties:
            try:
                value = p.__get__(block)
                if not isinstance(value, str | float | int | Iterable):
                    try:
                        value = value.name_display
                    except AttributeError:
                        try:
                            value = value.name
                        except AttributeError:
                            value = p.str(block)
                else:
                    value = p.str(block)

            except (ValueError, AttributeError, TypeError, ZeroDivisionError,
                    NotImplementedError, IndexError):
                value = 'NaN'
            if len(value) < 100:
                lines += ["{:25s}: {:15s}\n".format(
                    p.name_display, value)]

        return lines
    # endregion

    # region Data Output
    def get_data_df_self(self, include_long_arrays: bool = True):
        """Return pandas DataFrame of all settings, inputs, and outputs."""
        block = self._parent
        value_properties = self.value_properties
        input_properties = self.input_properties

        df = pd.DataFrame(dtype=object)
        for p in value_properties:
            try:
                value = p.__get__(block)
            except (ValueError, AttributeError, TypeError, ZeroDivisionError,
                    NotImplementedError, IndexError):
                continue
            if not include_long_arrays and isinstance(value, Iterable) \
                    and not isinstance(value, str) and len(value) > 5:
                continue  # Skip long arrays
            n = p.name_display
            if p in input_properties:
                df.loc[n, 'type'] = 'input'
            else:
                df.loc[n, 'type'] = 'output'
            df.loc[n, 'value'] = value
            df.loc[n, 'unit'] = p.unit
            df.loc[n, 'axis_label'] = p.axis_label
            df.loc[n, 'precision'] = 0
            if (p not in input_properties and isinstance(
                    df.loc[n, 'value'], float | int | Iterable)):
                df.loc[n, 'precision'] = block.tolerance

        return df

    def get_data_df(self, include_long_arrays: bool = True):
        """Return pandas DataFrame of all properties including parts."""
        block = self._parent
        df = self.get_data_df_self(include_long_arrays=include_long_arrays)
        if df.shape[0] == 0:
            return df
        df.loc[:, 'element'] = block.name
        element = df.pop('element')
        df.insert(0, 'element', element)
        df.insert(1, 'property', df.index)
        df.reset_index(inplace=True, drop=True)

        try:
            dfs = [(e, e.report.get_data_df(
                include_long_arrays=include_long_arrays))
                   for e in block.solvers + block.ports + block.parts]

            dfs = [(e, d) for e, d in dfs if d.shape[0] > 0]

            # Ensure unique port, part, and solver names
            names = []
            for e, d in dfs:
                if e.name not in names:
                    names += [e.name]
                else:
                    new_name = e.name + '2'
                    d.loc[:, 'element'] = d.element.str.replace(
                        e.name, new_name)
                    names += new_name

            for e, d in dfs:
                d.loc[:, 'element'] = [block.name + '.' + s
                                       for s in d.loc[:, 'element']]
            if len(dfs) > 0:
                df = pd.concat((df, *[d for e, d, in dfs]))
            df.reset_index(inplace=True, drop=True)
        except AttributeError:
            pass
        return df
    # endregion
