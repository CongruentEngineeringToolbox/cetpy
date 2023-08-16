"""
Report
======

This file specifies an output formatter for a SysML Block providing
user-friendly and accessible views on the created blocks.
"""
from typing import List, Iterable

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
            self._value_properties = [p for p in type(block).__dict__.values()
                                      if isinstance(p, ValueProperty)]

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
        """Print report to console."""
        [print(line[:-1]) for line in self.get_report_text()]

    def report_self(self) -> None:
        """Print report to console."""
        [print(line[:-1]) for line in self.get_report_text()]
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
            lines += [solvers_header.center(80, '=') + '\n']

            for sol in solvers:
                solver_lines = sol.report.get_report_text()
                # Add indent for better system depth visualisation
                lines += ['  ' + line for line in solver_lines]

            solvers_header = (
                ' ' + block.name_display + "'s Solver Reports Complete ")
            lines += [solvers_header.center(80, '-') + '\n\n']

        if len(ports) > 0:
            lines += ['\n']
            ports_header = ' ' + block.name_display + "'s Port Reports "
            lines += [ports_header.center(80, '=') + '\n']

            for port in ports:
                port_lines = port.report.get_report_text()
                # Add indent for better system depth visualisation
                lines += ['  ' + line for line in port_lines]

            ports_header = (
                ' ' + block.name_display + "'s Port Reports Complete ")
            lines += [ports_header.center(80, '-') + '\n\n']

        header = ' ' + self._parent.name_display + ' Complete '
        lines += ['\n' + header.center(80, '-') + '\n']
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

        lines += ['Name: {:>23s}\n'.format(block.name_display)]
        lines += ['Abbreviation: {:>15s}\n'.format(block.abbreviation)]
        if block.parent is not None:
            lines += ['Parent: {:>21s}\n'.format(block.parent)]
        lines += ['# Solvers: {:>18d}\n'.format(len(solvers))]
        lines += ['# Ports: {:>20d}\n'.format(len(ports))]
        lines += ['# Parts: {:>20d}\n'.format(len(parts))]

        if len(solvers) > 0:
            solver_lines = ['Solvers: {:20s}\n'.format(
                solvers[0].__class__.__name__)]
            for sol in solvers[1:]:
                solver_lines += ['         {:20s}\n'.format(
                    sol.__class__.__name__)]
            lines += solver_lines
        if len(ports) > 0:
            ports_lines = ['Ports: {:22s}\n'.format(
                ports[0].__class__.__name__)]
            for port in ports[1:]:
                ports_lines += ['       {:22s}\n'.format(
                    port.__class__.__name__)]
            lines += ports_lines
        if len(parts) > 0:
            parts_lines = ['Parts: {:22s}\n'.format(
                parts[0].name_display)]
            for part in ports[1:]:
                parts_lines += ['       {:22s}\n'.format(
                    part.name_display)]
            lines += parts_lines
        lines += ['Tolerance: {:>18.2e}\n'.format(block.tolerance)]

        return lines

    def __get_report_input_text__(self) -> List[str]:
        """Return input list of lines for the report."""
        block = self._parent
        value_properties = self.input_properties

        lines = ["Input\n"]
        lines += ['-----\n']
        for p in value_properties:
            if not isinstance(p.__get__(block), Iterable):
                lines += ["{:20s}: {:15s}\n".format(
                    p.name_display, p.str(block))]

        return lines

    def __get_report_output_text__(self) -> List[str]:
        """Return output list of lines for the report."""
        block = self._parent
        value_properties = self.output_properties

        lines = ["Output\n"]
        lines += ['------\n']
        for p in value_properties:
            if not isinstance(p.__get__(block), Iterable):
                lines += ["{:20s}: {:15s}\n".format(
                    p.name_display, p.str(block))]

        return lines
    # endregion
