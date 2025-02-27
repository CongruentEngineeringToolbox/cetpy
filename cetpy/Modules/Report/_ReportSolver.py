"""
ReportSolver
============

This file adapts the Report class to the Solver class.
"""
from typing import List

from cetpy.Modules.Report import Report


class ReportSolver(Report):
    """SysML Solver Output Formatter of the Congruent Engineering Toolbox."""

    # region Report Text
    def __get_report_header_text__(self) -> List[str]:
        solver = self._parent

        lines = []

        lines += ['-' * 80 + '\n']
        header = ' ' + solver.__class__.__name__ + ' '
        lines += [header.center(80, '=') + '\n']
        lines += ['-' * 80 + '\n']

        lines += ['Name: {:>23s}\n'.format(solver.__class__.__name__)]
        if solver.parent is not None:
            lines += ['Parent: {:>21s}\n'.format(solver.parent.name_display)]
        lines += ['Tolerance: {:>18.2e}\n'.format(solver.tolerance)]
        lines += ['Solved: {:>21s}\n'.format(str(solver.solved))]
        lines += ['Calculating: {:>16s}\n'.format(str(solver.calculating))]

        return lines
    # endregion

    # region Graph Output
    def get_header_attributes(self) -> dict:
        parent = self._parent
        super_dict = super().get_header_attributes()
        super_dict.update({
            'solved': parent.solved,
            'calculating': parent.calculating,
        })
        if parent.parent is None:
            super_dict.update({'parent': 'None'})
        else:
            super_dict.update({'parent': parent.parent.name_display})
        return super_dict
    # endregion

