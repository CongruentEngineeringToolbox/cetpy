"""
ReportPort
==========

This file adapts the Report class to the SysML Port class.
"""
from typing import List
from os.path import join, isdir
from os import mkdir

from cetpy.Modules.Report import Report


class ReportPort(Report):
    """SysML Port Output Formatter of the Congruent Engineering Toolbox."""

    # region Report Text
    def __get_report_header_text__(self) -> List[str]:
        port = self._parent

        lines = []

        lines += ['-' * 80 + '\n']
        header = ' ' + port.__class__.__name__ + ' '
        lines += [header.center(80, '=') + '\n']
        lines += ['-' * 80 + '\n']

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

    # region Data Output
    @staticmethod
    def _add_element_specific_data_df_self(df, block):
        super()._add_element_specific_data_df_self(df, block)
        if block.upstream is not None:
            df.loc['upstream', 'value'] = block.upstream.name
        if block.downstream is not None:
            df.loc['downstream', 'value'] = block.downstream.name

    def save_data_df_self(self, include_long_arrays: bool = True) -> None:
        upstream = self._parent.upstream
        downstream = self._parent.downstream
        df = self.get_data_df_self(include_long_arrays=include_long_arrays)
        if upstream is not None:
            directory = join(upstream.directory, self._parent._upstream_dict_name.strip('_'))
            if not isdir(directory):
                mkdir(directory)
            df.to_csv(join(directory, 'data_df_self.csv'))
        if downstream is not None:
            directory = join(downstream.directory, self._parent._downstream_dict_name.strip('_'))
            if not isdir(directory):
                mkdir(directory)
            df.to_csv(join(directory, 'data_df_self.csv'))

    def save_data_df(self, include_long_arrays: bool = True) -> None:
        upstream = self._parent.upstream
        downstream = self._parent.downstream
        df = self.get_data_df(include_long_arrays=include_long_arrays)
        if upstream is not None:
            directory = join(upstream.directory, self._parent._upstream_dict_name.strip('_'))
            if not isdir(directory):
                mkdir(directory)
            df.to_csv(join(directory, 'data_df.csv'))
        if downstream is not None:
            directory = join(downstream.directory, self._parent._downstream_dict_name.strip('_'))
            if not isdir(directory):
                mkdir(directory)
            df.to_csv(join(directory, 'data_df.csv'))
    # endregion
