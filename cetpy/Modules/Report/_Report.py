"""
Report
======

This file specifies an output formatter baseline for the ReportBlock, ReportSolver, and ReportPort specialized
formatters providing user-friendly and accessible views on the created blocks.
"""
from typing import List, Dict, Iterable
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join

from cetpy.Modules.SysML import ValueProperty, ReferenceProperty


class Report:
    """SysML Block Output Formatter of the Congruent Engineering Toolbox."""

    __slots__ = ['_parent', '_value_properties', '_reference_properties']

    def __init__(self, parent):
        self._parent = parent
        self.__generate_property_lists__()

    def save_all_self(self) -> None:
        """Save all block output to the block directory excluding parts, ports, and solvers."""
        self.save_data_df_self()
        self.save_all_plots()

    def save_all(self, include_data_df: bool = True):
        """Save all block output to the block directory including parts, ports, and solvers."""
        self.save_all_self()
        if include_data_df:
            self.save_data_df()

    # region Value Properties
    def __generate_property_lists__(self) -> None:
        """Generate a list of value properties of the parent block."""
        block = self._parent
        if block is None:
            self._value_properties = []
            self._reference_properties = []
        else:
            vp = {}
            rp = {}
            cls_list = type(block).mro()
            cls_list.reverse()  # Process highest priority class last, so its value overwrites any previous
            for cls in cls_list:
                cls_vp = [p for p in cls.__dict__.values() if isinstance(p, ValueProperty)]
                cls_rp = [p for p in cls.__dict__.values() if isinstance(p, ReferenceProperty)]
                cls_dict_vp = dict(zip([p.name for p in cls_vp], cls_vp))
                cls_dict_rp = dict(zip([p.name for p in cls_rp], cls_rp))
                vp.update(cls_dict_vp)
                rp.update(cls_dict_rp)
            self._value_properties = list(vp.values())
            self._reference_properties = list(rp.values())

    @property
    def value_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block."""
        return self._value_properties + []  # Protect list

    @property
    def input_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block that are
        fixed and used as inputs."""
        return [p for p in self._value_properties if p.fixed(self._parent)]

    @property
    def output_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block that are not
        fixed and calculated as outputs."""
        return [p for p in self._value_properties if p not in self.input_properties]

    @property
    def reference_properties(self) -> List[ValueProperty]:
        """Return list of ValueProperties of the parent block."""
        return self._reference_properties + []  # Protect list
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
        """Return list of lines for the report of the system element and its parts."""
        return self.get_report_self_text()

    def get_report_self_text(self) -> List[str]:
        """Return list of lines for the report of just the block."""
        lines = self.__get_report_header_text__()
        lines += ['\n']
        lines += self.__get_report_input_text__()
        lines += ['\n']
        lines += self.__get_report_output_text__()

        lines += ['\n']
        header = ' ' + type(self._parent).__name__ + ' Complete '
        lines += [header.center(80, '-') + '\n\n']
        return lines

    def __get_report_header_text__(self) -> List[str]:
        """Return header list of lines for the report."""
        block = self._parent
        lines = []

        lines += ['=' * 80 + '\n']
        header = ' ' + block.name_display + ' '
        lines += [header.center(80, '=') + '\n']
        lines += ['=' * 80 + '\n']

        lines += ['Name: {:>33s}\n'.format(block.name_display)]
        lines += ['Class: {:>32s}\n'.format(type(block).__name__)]
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

            except (ValueError, AttributeError, TypeError, ZeroDivisionError, NotImplementedError, IndexError):
                value = 'NaN'
            if len(value) < 100:
                lines += ["{:25s}: {:15s}\n".format(p.name_display, value)]

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

            except (ValueError, AttributeError, TypeError, ZeroDivisionError, NotImplementedError, IndexError):
                value = 'NaN'
            if len(value) < 100:
                lines += ["{:25s}: {:15s}\n".format(p.name_display, value)]

        return lines
    # endregion

    # region Data Output
    @staticmethod
    def _add_element_specific_data_df_self(df, block):
        """Add additional entries to data_df_self, specific to the element type that is reported on."""
        df.loc['model', 'value'] = str(block.__class__)

    def get_data_df_self(self, include_long_arrays: bool = True) -> pd.DataFrame:
        """Return pandas DataFrame of all settings, inputs, and outputs."""
        block = self._parent
        value_properties = self.value_properties
        input_properties = self.input_properties

        # Initialise array with one object column since the value can be
        # anything, an object instance, float, str, list, array, etc.
        df = pd.DataFrame(columns=['value'], dtype=object)
        self._add_element_specific_data_df(df, block)

        for p in value_properties:
            try:
                value = p.__get__(block)
            except (ValueError, AttributeError, TypeError, ZeroDivisionError, NotImplementedError, IndexError):
                continue
            if not include_long_arrays and isinstance(value, Iterable) \
                    and not isinstance(value, str) and len(value) > 5:
                continue  # Skip long arrays
            n = p.name_display
            if p in input_properties:
                df.loc[n, 'type'] = 'input'
            else:
                df.loc[n, 'type'] = 'output'
            df.at[n, 'value'] = value  # use at for inserting iterable
            df.loc[n, 'unit'] = p.unit
            df.loc[n, 'axis_label'] = p.axis_label
            # df.loc[n, 'precision'] = 0
            # if p not in input_properties and isinstance(df.loc[n, 'value'], float | int | Iterable):
            #     df.loc[n, 'precision'] = block.tolerance

        return df

    def get_data_df(self, include_long_arrays: bool = True) -> pd.DataFrame:
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
                include_long_arrays=include_long_arrays)) for e in block.solvers + block.ports + block.parts]

            dfs = [(e, d) for e, d in dfs if d.shape[0] > 0]

            # Ensure unique port, part, and solver names
            names = []
            for e, d in dfs:
                if e.name not in names:
                    names += [e.name]
                else:
                    new_name = e.name + '2'
                    d.loc[:, 'element'] = d.element.str.replace(e.name, new_name)
                    names += new_name

            for e, d in dfs:
                d.loc[:, 'element'] = [block.name + '.' + s for s in d.loc[:, 'element']]
            if len(dfs) > 0:
                df = pd.concat((df, *[d for e, d, in dfs]))
            df.reset_index(inplace=True, drop=True)
        except AttributeError:
            pass
        return df

    def save_data_df_self(self, include_long_arrays: bool = True) -> None:
        """Save a csv of the data_df_self to file into the associated block directory."""
        self.get_data_df_self(include_long_arrays=include_long_arrays).to_csv(join(
            self._parent.directory, 'data_df_self.csv'))

    def save_data_df(self, include_long_arrays: bool = True) -> None:
        """Save a csv of the data_df to file into the associated block directory."""
        self.get_data_df(include_long_arrays=include_long_arrays).to_csv(join(self._parent.directory, 'data_df.csv'))
    # endregion

    # region Plot Output
    def get_all_plots(self) -> Dict[str, plt.Figure]:
        """Return all plt.Figure objects of the associated model element.

        See Also
        --------
        plot_all: visualize all plots
        save_all_plots: save all plots to files.
        """
        return {}  # Modified in Report classes for specific model elements

    def plot_all(self) -> None:
        """Plot and display all plots of the associated block.

        See Also
        --------
        get_all_plots: plot Figure objects
        save_all_plots: save all plots to files.
        """
        [fig.show() for fig in self.get_all_plots().values()]

    def save_all_plots(self) -> None:
        """Save all plots of the associated block into its directory.

        See Also
        --------
        get_all_plots: plot Figure objects
        plot_all: visualize all plots
        """
        [fig.savefig(join(self._parent.directory, key + '.png')) for key, fig in self.get_all_plots().items()]
    # endregion

    # region Graph Output
    def get_header_attributes(self) -> dict:
        """Return dictionary of attributes for report headers.

        This includes name, class, and configuration parameters.
        """
        parent = self._parent
        return {
            'name': parent.name_display,
            'class': type(parent).__name__,
            'tolerance': parent.tolerance
        }
    # endregion
