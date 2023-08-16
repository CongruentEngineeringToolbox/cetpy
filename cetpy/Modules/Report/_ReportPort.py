"""
ReportPort
==========

This file adapts the SysML Block Report to the SysML Port class.
"""
from typing import List

from cetpy.Modules.Report import Report


class ReportPort(Report):
    """SysML Port Output Formatter of the Congruent Engineering Toolbox."""

    # region Report Text
    def get_report_text(self) -> List[str]:
        """Return list of lines for the report of the port.

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
        port = self._parent

        lines = []

        lines += ['=' * 80 + '\n']
        header = ' ' + port.__class__.__name__ + ' '
        lines += [header.center(80, '=') + '\n']
        lines += ['=' * 80 + '\n']

        lines += ['Name: {:>23s}\n'.format(port.__class__.__name__)]
        if port.upstream is not None:
            lines += ['Upstream: {:19s}\n'.format(port.upstream.name_display)]
        if port.downstream is not None:
            lines += ['Downstream: {:17s}\n'.format(
                port.downstream.name_display)]
        lines += ['Endpoint: {:>19s}\n'.format(str(port.is_endpoint))]
        lines += ['Endpoint Upstream: {:>10s}\n'.format(
            str(port.is_upstream_endpoint))]
        lines += ['Endpoint Downstream: {:>8s}\n'.format(
            str(port.is_downstream_endpoint))]
        lines += ['Tolerance: {:>18.2e}\n'.format(port.tolerance)]

        return lines
    # endregion
