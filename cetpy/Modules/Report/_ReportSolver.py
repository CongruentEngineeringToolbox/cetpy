"""
ReportSolver
============

This file adapts the SysML Block Report to the Solver class.
"""
from typing import List

from cetpy.Modules.Report import Report


class ReportSolver(Report):
    """SysML Solver Output Formatter of the Congruent Engineering Toolbox."""

    # region Report Text
    def get_report_text(self) -> List[str]:
        """Return list of lines for the report of the solver.

        Unlike the block Report this function is functionally the same as
        the get_report_self_text function.
        """
        return self.get_report_self_text()

    def get_report_self_text(self) -> List[str]:
        lines = self.__get_report_header_text__()
        lines += ['\n']
        lines += self.__get_report_input_text__()
        lines += ['\n']
        lines += self.__get_report_output_text__()

        header = ' ' + self._parent.__class__.__name__ + ' Complete '
        lines += ['\n' + header.center(80, '-') + '\n']
        return lines

    def __get_report_header_text__(self) -> List[str]:
        solver = self._parent

        lines = []

        lines += ['=' * 80 + '\n']
        header = ' ' + solver.__class__.__name__ + ' '
        lines += [header.center(80, '=') + '\n']
        lines += ['=' * 80 + '\n']

        lines += ['Name: {:>23s}\n'.format(solver.__class__.__name__)]
        lines += ['Tolerance: {:>18.2e}\n'.format(solver.tolerance)]

        return lines
    # endregion
