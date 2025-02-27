"""
ReportBlock
============

This file specifies an output formatter for a SysML Block providing
user-friendly and accessible views on the created blocks.
"""
from typing import List, Dict, Sized
import matplotlib.pyplot as plt

from cetpy.Modules.Report import Report


class ReportBlock(Report):
    """SysML Block Output Formatter of the Congruent Engineering Toolbox."""

    # region Saving
    def save_all(self, include_data_df: bool = True):
        self.save_all_self()
        block = self._parent
        if include_data_df:
            self.save_data_df()
        for s in block.solvers:
            s.report.save_all(include_data_df=False)
        for p in block.parts:
            p.report.save_all(include_data_df=False)
        for p in block.ports:
            p.report.save_all(include_data_df=False)
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

            parts_header = ' ' + block.name_display + "'s Part Reports Complete "
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

            solvers_header = ' ' + block.name_display + "'s Solver Reports Complete "
            lines += ['  ' + solvers_header.center(80, '-') + '\n\n']

        if len(ports) > 0:
            lines += ['\n']
            ports_header = ' ' + block.name_display + "'s Port Reports "
            lines += ['  ' + ports_header.center(80, '=') + '\n']

            for port in ports:
                port_lines = port.report.get_report_text()
                # Add indent for better system depth visualisation
                lines += ['  ' + line for line in port_lines]

            ports_header = ' ' + block.name_display + "'s Port Reports Complete "
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
            solver_lines = ['Solvers: {:30s}\n'.format(solvers[0].__class__.__name__)]
            for sol in solvers[1:]:
                solver_lines += ['         {:30s}\n'.format(sol.__class__.__name__)]
            lines += solver_lines
        if len(ports) > 0:
            ports_lines = ['Ports: {:32s}\n'.format(ports[0].__class__.__name__)]
            for port in ports[1:]:
                ports_lines += ['       {:32s}\n'.format(port.__class__.__name__)]
            lines += ports_lines
        if len(parts) > 0:
            parts_lines = ['Parts: {:32s}\n'.format(parts[0].name_display)]
            for part in parts[1:]:
                parts_lines += ['       {:32s}\n'.format(part.name_display)]
            lines += parts_lines
        lines += ['Tolerance: {:>28.2e}\n'.format(block.tolerance)]

        return lines
    # endregion

    # region Data Output
    @staticmethod
    def _add_element_specific_data_df_self(df, block):
        super()._add_element_specific_data_df_self(df, block)
        block.solve()
        df.loc['solved', 'value'] = block.solved
        df.loc['solved_self', 'value'] = block.solved_self
        df.loc['fixed', 'value'] = block.fixed
        df.loc['fixed_self', 'value'] = block.fixed_self
    # endregion

    # region Plot Output
    def get_all_plots(self) -> Dict[str, plt.Figure]:
        """Return all plt.Figure objects of the associated block. Generated from the Block __plot_functions__
        attribute and list of vector ValueProperties.

        See Also
        --------
        plot_all: visualize all plots
        save_all_plots: save all plots to files.
        """
        plots = {}
        block = self._parent
        for k, p in block.__plot_functions__.items():
            fig = p(block)
            if isinstance(fig, plt.Figure):
                plots[k] = fig
            else:
                plots[k] = fig[0]
        if block.__default_plot_x_axis__ is not None:
            def_x = block.__default_plot_x_axis__
            if isinstance(def_x, str):
                def_x = [def_x]
            def_x = [block.__getattribute__(dx) for dx in def_x]
            for vp in [vp for vp in self.value_properties if vp.name not in block.__plot_functions__.keys()]:
                try:
                    value = vp.__get__(block)
                except (ValueError, AttributeError, TypeError, ZeroDivisionError, NotImplementedError, IndexError):
                    continue
                if isinstance(value, Sized):
                    for xp in def_x:
                        if len(value) == len(xp):
                            fig = type(block).plot.line_function(block, xp, vp)
                            if isinstance(fig, plt.Figure):
                                plots[vp.name] = fig
                            else:
                                plots[vp.name] = fig[0]
                            break
        return plots
    # endregion

    # region Graph Output
    def get_header_attributes(self) -> dict:
        parent = self._parent
        super_dict = super().get_header_attributes()
        super_dict.update({
            'abbreviation': parent.abbreviation,
            'fixed': parent.fixed,
            'solved': parent.solved,
            'solvers': [type(p).__name__ for p in parent.solvers],
            'ports': [type(p).__name__ for p in parent.ports],
            'parts': [p.name_display for p in parent.parts],
        })
        if parent.parent is None:
            super_dict.update({'parent': 'None'})
        else:
            super_dict.update({'parent': parent.parent.name_display})
        return super_dict
    # endregion
