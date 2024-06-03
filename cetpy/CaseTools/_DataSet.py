"""
DataSet
=======

This file defines a wrapper around a pandas DataFrame to add further convenient analysis using the knowledge of the
cetpy Iterator of its system.
"""
from __future__ import annotations

from os.path import isdir, dirname
from typing import List, Dict, Iterable, Tuple, Sequence
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm

from cetpy.Modules.SysML import Block, ValueProperty, value_property

from cetpy.CaseTools.FilterFunctions import apply_filter, drop_no_variance_columns, apply_one_hot_encoding
from cetpy.Modules.Utilities.Labelling import round_sig_figs, floor_sig_figs, ceil_sig_figs, unit_2_latex, \
    name_2_unit, name_2_axis_label


binary_color_maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                     'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'] * 3


def get_numerical_correlation_threshold(threshold: str | float) -> float:
    """Return float threshold for correlation label. Return float if a float is passed."""
    if isinstance(threshold, float):
        return threshold
    else:
        match threshold:
            case 'perfect':
                return 0.98
            case 'high':
                return 0.5
            case 'moderate':
                return 0.3
            case 'low':
                return 0.02
            case _:
                return 0.


class DataSet(Block):
    """Congruent Engineering Toolbox DataSet, a wrapper around a pandas DataFrame with additional analysis
    functionality."""

    drop_errors = ValueProperty(permissible_types_list=bool)
    drop_no_variance = ValueProperty(permissible_types_list=bool)
    one_hot_encoding = ValueProperty(permissible_types_list=bool)
    filter_list = ValueProperty(permissible_types_list=[str, list, type(None)])
    error_filter = ValueProperty(permissible_types_list=[str, list, type(None)])
    input_df = ValueProperty(permissible_types_list=[pd.DataFrame, type(None)])
    units = ValueProperty(permissible_types_list=[dict, type(None)])
    axis_labels = ValueProperty(permissible_types_list=[dict, type(None)])
    last_input_column = ValueProperty(permissible_types_list=[str, type(None)])

    __init_parameters__ = Block.__init_parameters__.copy() + [
        'drop_errors', 'drop_no_variance', 'one_hot_encoding', 'filter_list', 'error_filter', 'input_df', 'units',
        'axis_labels', 'last_input_column'
    ]

    _reset_dict = Block._reset_dict.copy()
    _reset_dict.update({'_output_df': None})

    def __init__(self,
                 raw_df: pd.DataFrame,
                 drop_errors: bool = True,
                 drop_no_variance: bool = True,
                 one_hot_encoding: bool = True,
                 filter_list: str | List[str] | None = None,
                 error_filter: str | List[str] | None = None,
                 units: Dict[str, str] | None = None,
                 axis_labels: Dict[str, str] | None = None,
                 last_input_column: str | None = None,
                 **kwargs):
        self._output_df = None
        self._output_last_input_column = None
        super().__init__(
            kwargs.pop('name', 'data_set'),
            kwargs.pop('abbreviation', 'DS'),
            drop_errors=drop_errors,
            drop_no_variance=drop_no_variance,
            one_hot_encoding=one_hot_encoding,
            filter_list=filter_list,
            error_filter=error_filter,
            units=units,
            axis_labels=axis_labels,
            last_input_column=last_input_column,
            **kwargs)
        self.raw_df = raw_df
        if units is None or axis_labels is None:
            names = [s.split('.')[-1] for s in raw_df.columns]
            if units is None:
                self.units = {col: name_2_unit(name) for col, name in zip(raw_df.columns, names)}
            if axis_labels is None:
                self.axis_labels = {col: name_2_axis_label(name) for col, name in zip(raw_df.columns, names)}

    @value_property(input_permissible=True,
                    permissible_types_list=[str, type(None)])
    def last_input_column(self) -> str:
        """Return key of last input column.

        Can be supplied manually, if None is provided, the column is evaluated automatically based on the structure
        of the cetpy CaseRunner.
        """
        df = self.raw_df
        try:
            idx_solved = df.columns.get_loc('solved') - 1
        except AttributeError:
            raise ValueError('The last input column of the dataframe could not be evaluated automatically. Please set '
                             'the column manually with DataSet.last_input_column')
        return df.columns[idx_solved]

    @property
    def __idx_last_input_column_raw__(self) -> int:
        """Return index of last input column of raw dataframe."""
        return self.raw_df.columns.get_loc(self.last_input_column)

    @property
    def __idx_last_input_column__(self) -> int:
        """Return index of last input column of the processed dataframe."""
        return self.df.columns.get_loc(self.last_input_column)

    # region DataFrames
    @value_property(permissible_types_list=pd.DataFrame)
    def raw_df(self) -> pd.DataFrame:
        """Unprocessed dataset as a pandas Dataframe.

        The dataframe should first have the input columns, then the output columns. If the dataframe still contains
        the original solved or error related columns, the separation can be detected automatically. Otherwise,
        a last_input_column value will need to be specified.

        See Also
        --------
        DataSet.df
        """
        return self._df_raw.copy()  # Protect original

    @raw_df.setter
    def raw_df(self, val: pd.DataFrame) -> None:
        self._df_raw = val

    @property
    def df_errors(self) -> pd.DataFrame:
        """Dataframe of only cases with errors.

        Includes both detected errors and those selected by the custom error filter.

        See Also
        --------
        DataSet.df_no_errors
        DataSet.df
        """
        df = self.raw_df
        base_filter = ['errored == True']
        if 'errored' not in df.columns:
            base_filter = []
        error_filter = self.error_filter
        if error_filter is None or len(error_filter) == 0:
            error_filter = []
        if len(base_filter + error_filter) > 0:
            return apply_filter(df, base_filter + error_filter, join_and=False)
        else:
            return pd.DataFrame(columns=df.columns)

    @property
    def df_no_errors(self) -> pd.DataFrame:
        """Raw Dataframe without errored cases.

        See Also
        --------
        DataSet.df_errors
        DataSet.df
        """
        return self.raw_df.drop(index=self.df_errors.index,
                                columns=['errored', 'error_class', 'error_message', 'error_location'],
                                errors='ignore')

    @property
    def df(self) -> pd.DataFrame:
        """DataFrame with all enabled processing functions applied.

        Processed dataframe is stored for performance. This can be reset with the DataSet.reset() function.

        Filters are applied in the following order:
            1. Drop Errors (including custom error filter)
            2. Filter List
            3. Drop No Variance

        See Also
        --------
        DataSet.reset
        DataSet.drop_errors
        DataSet.drop_no_variance
        DataSet.filter_list
        DataSet.custom_error_filter
        """
        if self._output_df is None:
            if self.drop_errors:
                df = self.df_no_errors.copy()
            else:
                df = self.raw_df
            filter_list = self.filter_list
            if filter_list is not None and len(filter_list) > 0:
                df = apply_filter(df, filter_list, join_and=True)
            if self.drop_no_variance:
                df = drop_no_variance_columns(df)

            # Dropping columns with no variance can drop the detected
            # transition input column. Verify and re-detect if necessary.
            if self.last_input_column not in df.columns:
                self._output_last_input_column = [
                    col for col in df.columns if col in list(self.raw_df.keys())][-1]

            if self.one_hot_encoding:
                # One-Hot Encoding appends the newly created columns at the
                # end. The application function splits the dataframe into
                # inputs and outputs and applies the encoding separately in
                # order to maintain separation. Given new columns are
                # appended, the last column of the inputs needs to be
                # reevaluated.
                df_input = pd.get_dummies(df.loc[:, :self.last_input_column])
                self._output_last_input_column = df_input.columns[-1]
                df = apply_one_hot_encoding(df, df_input.columns)

            self._output_df = df
        return self._output_df

    # region Keys
    @property
    def input_keys_raw(self) -> List[str]:
        """Return list of input keys in raw data frame.

        See Also
        --------
        DataSet.last_input_column
        DataSet.output_keys_raw
        DataSet.input_keys
        """
        return self.raw_df.columns[:self.__idx_last_input_column_raw__ + 1]

    @property
    def output_keys_raw(self) -> List[str]:
        """Return list of output keys in raw data frame.

        See Also
        --------
        DataSet.last_input_column
        DataSet.input_keys_raw
        DataSet.output_keys
        """
        return self.raw_df.columns[self.__idx_last_input_column_raw__ + 1:]

    @property
    def input_keys(self) -> List[str]:
        """Return list of input keys.

        See Also
        --------
        DataSet.last_input_column
        DataSet.output_keys
        DataSet.input_keys_raw
        """
        return self.df.columns[:self.__idx_last_input_column__ + 1]

    @property
    def output_keys(self) -> List[str]:
        """Return list of output keys.

        See Also
        --------
        DataSet.last_input_column
        DataSet.input_keys
        DataSet.output_keys_raw
        """
        return self.df.columns[self.__idx_last_input_column__ + 1:]

    @property
    def input_columns(self) -> pd.DataFrame:
        """Return columns representing inputs to the model.

        See Also
        --------
        DataSet.last_input_column
        """
        return self.df.loc[:, self.input_keys]

    @property
    def output_columns(self) -> pd.DataFrame:
        """Return columns representing outputs from the model.

        See Also
        --------
        DataSet.last_input_column
        """
        return self.df.loc[:, self.output_keys]

    @property
    def keys_categorical(self) -> List[str]:
        """Return list of keys of the raw dataframe which are categorical.

        See Also
        --------
        DataSet.last_input_column
        DataSet.output_keys
        DataSet.input_keys_raw
        """
        return list(self.raw_df.select_dtypes(exclude=['bool_', 'number']).columns)
    # endregion

    def get_df_custom_level(self, idx_start_filter: int = None,
                            idx_end_filter: int = None,
                            drop_errors: bool = None,
                            drop_no_variance: bool = None) -> pd.DataFrame:
        """Return dataframe with custom processing level.

        Parameters
        ----------
        idx_start_filter: optional, default = None
            Start index of the filter list. If this index is higher than the length of the filter list, a copy of the
            original dataframe is returned.

        idx_end_filter: optional, default = None
            End index of the filter list. If this index is higher than the length of the filter list, the list is
            returned until the end. if this index is smaller than the start index, just the filter at the start index is
            applied.

        drop_errors: optional, default = None
            Bool flag whether to drop errors from the dataframe. Also drops error related columns. If set to None,
            the DataSet instance default value is used.

        drop_no_variance: optional, default = None
            Bool flag whether to drop columns with no variance from the dataframe. If set to None, the DataSet
            instance default value is used.

        Returns
        -------
        pd.DataFrame
            Processed dataframe from the raw dataframe with the chosen filters applied.
        """
        if drop_errors or (drop_errors is None and self.drop_errors):
            df = self.df_no_errors
        else:
            df = self.raw_df

        # Slice Filter
        filter_list = self.filter_list
        if idx_start_filter is None and idx_end_filter is None or isinstance(filter_list, str):
            pass
        else:
            filter_list = filter_list[idx_start_filter: idx_end_filter]

        df = apply_filter(df, filter_list, join_and=True)

        if drop_no_variance or (drop_no_variance is None and self.drop_no_variance):
            df = drop_no_variance_columns(df)

        if self.one_hot_encoding:
            df = apply_one_hot_encoding(df, self.input_keys_raw)
        return df

    def get_dfs_filter_levels(self, stacked: bool = True) -> Dict[str, pd.DataFrame]:
        """Return list of dataframes corresponding to the raw, no errors (if DataSet.drop_errors is True),
        and subsequently each cumulative filter level.

        Useful to analyse scope, validity, and detail changes with filtering.

        Parameters
        ----------
        stacked: optional, default = True
            Bool flag whether to stack filters. E.g. show the incremental improvement of the full filter set (True)
            or the impact of the individual filters on the original dataset (False).
        """
        # Go through custom level function, to match no variance and
        # one-hot-encoding post-processing.
        dfs = {'raw': self.get_df_custom_level(None, 0, False, None)}
        if self.drop_errors:
            dfs['no_errors'] = self.get_df_custom_level(None, 0, True, None)
        idx_start = None
        for i in range(len(self.filter_list)):
            dfs[self.filter_list[i]] = self.get_df_custom_level(idx_start, i + 1, None, None)
            if not stacked:
                idx_start = i + 1
        return dfs

    def get_split_dataset(self, split_key: str,
                          split_set: None | int | np.ndarray = None,
                          maintain_raw: bool = None
                          ) -> List[DataSet]:
        """Return list of datasets split along a specified key.

        Parameters
        ----------
        split_key
            Dataframe column key to split the dataset. If this is a categorical key and one-hot-encoding is enabled,
            both pre- and post-encoding keys are supported.

        split_set: optional, default = None
            A definition of the set to be created. If the split_key is a categorical value. This property is ignored,
            and a subset for each categorical value is created. If the split_key is a continuous property,
            the split_set can either be an integer, in which case the integer describes the amount of sets to split
            the continuous variable into. Alternatively the set can be a 1d numpy array where a set is created from
            each value to the next of the split_key data.

        maintain_raw: optional, default = None
            Bool flag whether the new dataset should maintain the full depth to the raw dataframe. This can sometimes
            be easier for categorical filtering with one-hot encoding. Otherwise, it is recommendable to drop the
            already filtered rows to improve performance. When set to None, attempts False, if the split_key is
            categorical then True.

        Notes
        -----
        if the split_set of a continuous variable is defined as an integer.
        The created sets are automatically rounded to sensible significant figure.
        """
        if maintain_raw is None:
            maintain_raw = split_key not in self.df.columns
        if maintain_raw:
            if split_key not in self.raw_df.columns:
                raise KeyError("Split key not in raw dataframe, consider setting maintain_raw to False or applying "
                               "one-hot-encoding on the raw dataset.")
            split_data = self.raw_df.loc[:, split_key]
        else:
            if split_key not in self.df.columns:
                raise KeyError("Split key not in processed dataframe, consider setting maintain_raw to True or "
                               "adjusting the categorical split key to a specific post one-hot-encoding category.")
            split_data = self.df.loc[:, split_key]

        datasets = []
        if split_data.nunique() in [1, 2] and not isinstance(split_data.iloc[0], str):
            # Binary Data
            ds_0 = self.subset(split_key + ' == ' + str(np.unique(split_data)[0]), maintain_raw=maintain_raw)
            ds_1 = self.subset(split_key + ' != ' + str(np.unique(split_data)[0]), maintain_raw=maintain_raw)
            datasets = [ds_0, ds_1]
        elif split_key in self.keys_categorical:
            # Categorical Data
            for val in np.unique(split_data):
                datasets += [self.subset(split_key + ' == "' + val + '"', maintain_raw=maintain_raw)]
        else:
            # Numerical Data
            limit_data = self.df.loc[:, split_key]
            if isinstance(split_set, int):
                sig_figs = int(np.ceil(np.log10(split_set)) + 1)
                split_set_new = np.linspace(
                    floor_sig_figs(limit_data.min(), sig_figs),
                    ceil_sig_figs(limit_data.max(), sig_figs), split_set + 1)
                # Round to sig fig in the center as well.
                for i in range(1, len(split_set_new) - 1):
                    split_set_new[i] = round_sig_figs(split_set_new[i], sig_figs)
                split_set = split_set_new
            for i in range(len(split_set) - 1):
                datasets += [self.subset(
                    str(split_set[i]) + " <= " + split_key + " <= " + str(split_set[i + 1]), maintain_raw=maintain_raw)]

        return datasets

    def subset(self, new_filter: None | str | List[str] = None, maintain_raw: bool = False) -> DataSet:
        """Return a further dataset utilising the current processed dataframe as its base dataframe. Disables one-hot
        encoding and drop errors on the new subset if raw is not maintained. These actions only need to be performed
        once.

        Parameters
        ----------
        new_filter: optional, default = None
            Filter to define the sub-set. Can be none, just to match the current dataset, a string for a single
            filter, or a list of strings for multiple filters.

        maintain_raw: optional, default = False
            Bool flag whether the new dataset should maintain the full depth to the raw dataframe. This can sometimes
            be easier for categorical filtering with one-hot encoding. Otherwise, it is recommendable to drop the
            already filtered rows to improve performance.
        """
        subset = self.copy()

        # Base subset on own processed dataset.
        if maintain_raw:
            if new_filter is not None:
                if isinstance(new_filter, list):
                    subset.filter_list += new_filter
                else:
                    subset.filter_list += [new_filter]
                subset.reset()
        else:
            subset.raw_df = self.df
            subset.last_input_column = self.last_input_column

            # Disable unnecessary post-processing
            subset.drop_errors = False
            subset.one_hot_encoding = False

            # Clear already applied filters.
            subset.filter_list = new_filter

        return subset
    # endregion

    # region Extensions
    def get_model_xy(self, output_key: str, input_keys: str | List[str] | None = None,
                     add_constant: bool = True) -> (np.ndarray, np.ndarray):
        """Return prepared X, y values of the dataset for input into a regression model."""
        if input_keys is None:
            input_keys = self.input_keys
        x = self.df.loc[:, input_keys].astype(float)
        if add_constant:
            x = sm.add_constant(x)
        y = self.df.loc[:, output_key]
        return x, y
    # endregion

    # region Saving
    def copy(self) -> DataSet:
        """Return a copy of the DataSet using the copy package."""
        return deepcopy(self)

    def to_pickle(self, file_path: str) -> None:
        """Save dataset instance to pickle."""
        if not isdir(dirname(file_path)):
            pass
        if file_path[-4:] != '.pkl' and file_path[-7:] != '.pickle':
            file_path += '.pkl'

        with open(file_path, "wb+") as file_write:
            # pickle.dump(self, file_write)
            file_write.write(pickle.dumps(self))
    # endregion

    # region Analysis
    # region Meta Analysis
    @property
    def n_cases(self) -> int:
        """Return number of cases post filtering.

        See Also
        --------
        DataSet.df
        """
        return self.df.shape[0]

    @property
    def n_cases_fraction(self) -> float:
        """Return fraction of cases selected post filtering compared to original dataframe.

        See Also
        --------
        DataSet.df
        DataSet.raw_df
        """
        return self.n_cases / self.raw_df.shape[0]

    @property
    def n_errors(self) -> int:
        """Return number of cases with recognised errors. Includes the custom error filter.

        See Also
        --------
        DataSet.df_errors
        """
        return self.df_errors.shape[0]

    @property
    def n_errors_fraction(self) -> float:
        """Return fraction of cases with recognised errors compared to original dataframe.

        See Also
        --------
        DataSet.df_errors
        DataSet.raw_df
        """
        return self.n_errors / self.raw_df.shape[0]
    # endregion

    # region Correlations
    def corr(self, key: str = None, sort: bool = True,
             split_input_output: bool = False, alternate_df: pd.DataFrame = None) -> pd.DataFrame:
        """Return Pearson correlation coefficient matrix for the filtered dataframe or for a selected key therein.
        The matrix is optionally sorted in descending order.

        Degree of correlation:
            - Perfect: |C| ~ 1
            - High degree: 0.5 <= |C| < 1
            - Moderate degree: 0.3 <= |C| < 0.5
            - Low degree: 0 < |C| < 0.3
            - No correlation: |C| ~ 0

        Parameters
        ----------
        key: optional, default = None
            Input or output key to down-select correlation matrix. If the default none is passed, the full matrix is
            returned.

        sort: optional, default = True
            Bool flag whether to sort the down-selected matrix by absolute descending order. If no key is passed,
            this option is ignored.

        split_input_output: optional, default = False
            Bool flag whether to format the resulting symmetrical matrix to split input and output columns. If set to
            True, rows are the model inputs, and columns are the model outputs. If examining a model as a black box,
            this may be preferable to reduce clutter.

        alternate_df: optional, default = None
            An alternate dataframe to use for the analysis. If None is passed, the main dataframe is used. This
            functionality is used for example by the filter analysis functions. Please note, the dataframe is still
            split with the main input and output keys.

        Returns
        -------
        pd.DataFrame
            Correlation matrix.
        """
        if alternate_df is None:
            df = self.df
        else:
            df = alternate_df
        corr = df.corr()
        if split_input_output:
            corr.drop(columns=self.input_keys, index=self.output_keys, inplace=True)
        if key is not None:
            if key in self.input_keys:
                corr = corr.loc[key, :]
            else:
                corr = corr.loc[:, key]
            if sort:
                # Don't use in-place due to some pandas weirdness
                corr = corr.sort_values(ascending=False, key=abs)
        return corr

    def get_significant_keys(self, key: str, threshold: str | float = 'high', alternate_df: pd.DataFrame = None
                             ) -> List[str]:
        """Return list of columns whose Pearson correlation coefficient is above the given threshold for a given
        input key.

        Degree of correlation:
            - Perfect: |C| ~ 1, here defined as 0.98 <= |C| < 1
            - High degree: 0.5 <= |C| < 1
            - Moderate degree: 0.3 <= |C| < 0.5
            - Low degree: 0 < |C| < 0.3, here defined as 0.02 <= |C| < 0.3
            - No correlation: |C| ~ 0, here defined as 0 <= |C| < 0.02

        Parameters
        ----------
        key
            Input or output key for which the significant keys should be identified.

        threshold: optional, default = 'high'
            Threshold above which a correlation is deemed significant.

        alternate_df: optional, default = None
            An alternate dataframe to use for the analysis. If None is passed, the main dataframe is used. This
            functionality is used for example by the filter analysis functions. Please note, the dataframe is still
            split with the main input and output keys.

        See Also
        --------
        DataSet.corr

        Notes
        -----
        Pearson Correlation coefficient is suitable for linear relationships.
        """
        threshold = get_numerical_correlation_threshold(threshold)
        corr = self.corr(key=key, sort=True, split_input_output=False, alternate_df=alternate_df)
        sig_keys = corr.index[corr.abs() >= threshold]
        return sig_keys[sig_keys != key]

    def correlation_count_analysis(self, key: str, alternate_df: pd.DataFrame = None) -> pd.Series:
        """Return dataframe of correlation counts at various thresholds for requested key. Notably each threshold
        does not contain the entries of higher levels.

        Parameters
        ----------
        key
            Input or output key for which the analysis should be conducted.

        alternate_df: optional, default = None
            An alternate dataframe to use for the analysis. If None is passed, the main dataframe is used. This
            functionality is used for example by the filter analysis functions. Please note, the dataframe is still
            split with the main input and output keys.

        Returns
        -------
        pd.Series
            Pandas Dataframe with the input columns as rows and the thresholds perfect, high, moderate, low,
            and total as columns.
        """
        df = pd.Series(dtype=int)
        for threshold in ['perfect', 'high', 'moderate', 'low']:
            df.loc[threshold] = len(self.get_significant_keys(key, threshold, alternate_df=alternate_df))
        df.name = key
        df.loc['total'] = df.low.copy()
        # Subtract higher levels
        df.low -= df.moderate
        df.moderate -= df.high
        df.high -= df.perfect
        return df.astype(int)

    def input_correlation_count_analysis(self, alternate_df: pd.DataFrame = None) -> pd.DataFrame:
        """Return dataframe of correlation counts at various thresholds for each input key. Notably each threshold
        does not contain the entries of higher levels.

        Parameters
        ----------
        alternate_df: optional, default = None
            An alternate dataframe to use for the analysis. If None is passed, the main dataframe is used. This
            functionality is used for example by the filter analysis functions. Please note, the dataframe is still
            split with the main input and output keys.

        Returns
        -------
        pd.DataFrame
            Pandas Dataframe with the input columns as rows and the thresholds perfect, high, moderate, low,
            and total as columns.

        See Also
        --------
        DataSet.correlation_count_analysis
        """
        df = pd.DataFrame()
        for key in self.input_keys:
            df = pd.concat((df, self.correlation_count_analysis(key, alternate_df=alternate_df)), axis=1)
        return df.T

    # endregion

    # region Filter Analysis
    def correlation_count_filter_progression(self, key: str, stacked: bool = True) -> pd.DataFrame:
        """Evaluate how each filter level affects the progression of correlations on a specific input or output key.

        Parameters
        ----------
        key
            Input or output key for which the analysis should be conducted.

        stacked: optional, default = True
            Bool flag whether to stack filters. E.g. show the incremental improvement of the full filter set (True)
            or the impact of the individual filters on the original dataset (False).

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with the rows representing each filter level and the columns representing the thresholds.
        """
        df = pd.DataFrame(dtype=int)
        dfs = self.get_dfs_filter_levels(stacked=stacked)
        for level, value in dfs.items():
            df_new = self.correlation_count_analysis(key, alternate_df=value)
            df_new.name = level
            df = pd.concat((df, df_new), axis=1)
        return df

    def get_case_count_with_filter_level(self, stacked: bool = True, normed: bool = False) -> dict:
        """Return dictionary with filter levels and number of cases in the dataset.

        Parameters
        ----------
        stacked: optional, default = True
            Bool flag whether to stack filters. E.g. show the incremental improvement of the full filter set (True)
            or the impact of the individual filters on the original dataset (False).

        normed: optional, default = False
            Bool flag whether the values should be normalised against the size of the initial dataset.
        """
        case_counts = {}
        dfs = self.get_dfs_filter_levels(stacked=stacked)
        for level, value in dfs.items():
            case_counts[level] = value.shape[0]
            if normed:
                case_counts[level] /= list(case_counts.values())[0]
        return case_counts

    def get_filter_reduction_degrees(self, individual: bool = False, stacked: bool = True) -> dict:
        """Get reduction degree of each filter level.

        Parameters
        ----------
        individual: optional, default = False
            Bool flag whether the reduction level of each filter should be evaluated individually relative to the raw
            data frame or to a previous reduction (select using the stacked parameter).

        stacked: optional, default = True
            Bool flag whether to stack filters. E.g. show the incremental improvement of the full filter set (True)
            or the impact of the individual filters on the original dataset (False). Note, the original dataset
            already includes the no_errors correction. If this is not desired, set individual to True.
        """
        reductions = {}
        raw_df = self.raw_df
        n_no_errors = n_raw = raw_df.shape[0]
        if self.drop_errors:
            n_no_errors = self.df_no_errors.shape[0]
            reductions['no_errors'] = n_no_errors / n_raw
        if individual:
            for s_filter in self.filter_list:
                reductions[s_filter] = raw_df.query(s_filter).shape[0] / n_raw
        else:
            dfs = self.get_dfs_filter_levels(stacked=stacked)
            if len(dfs) > 2 or (len(dfs) > 1 and list(dfs.keys())[1] != 'no_errors'):
                n_last = n_no_errors
                for level, df in dfs.items():
                    if level in ['raw', 'no_errors']:
                        continue
                    reductions[level] = df.shape[0] / n_last
                    if stacked:
                        n_last = df.shape[0]

        return reductions

    def get_filter_describes(self, key: str, stacked: bool = True) -> pd.DataFrame:
        """Get dataframe of pandas describe for each filter level in respect to a specific column key.

        Parameters
        ----------
        key
            Evaluation key on which pd.Describe is called.

        stacked: optional, default = True
            Bool flag whether to stack filters. E.g. show the incremental impact of the full filter set (True) or the
            impact of the individual filters on the original dataset (False). Note, the original dataset already
            includes the no_errors correction.
        """
        dfs = self.get_dfs_filter_levels(stacked=stacked)
        describes = pd.DataFrame()
        for level, df in dfs.items():
            describe = df.loc[:, key].describe()
            describe.name = level
            describes = pd.concat((describes, describe), axis=1)
        return describes
    # endregion
    # endregion

    # region Visualisation
    def plot_input_correlation_count_analysis(self, stacked: bool = False, ax: plt.Axes = None,
                                              title: str = None) -> plt.Figure:
        """Plot a bar-graph of the correlation threshold counts for input parameters."""
        df = self.input_correlation_count_analysis()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        x_axis = np.arange(len(df.index))

        if stacked:
            ax.bar(x_axis, df.low, label='low', color='r')
            ax.bar(x_axis, df.moderate, bottom=df.low, label='moderate', color='orange')
            ax.bar(x_axis, df.high, bottom=df.low + df.moderate, label='high', color='g')
            ax.bar(x_axis, df.perfect, bottom=df.low + df.moderate + df.high, label='perfect', color='b')
        else:
            ax.bar(x_axis + 0.3, df.low, 0.2, label='low', color='r')
            ax.bar(x_axis + 0.1, df.moderate, 0.2, label='moderate', color='orange')
            ax.bar(x_axis - 0.1, df.high, 0.2, label='high', color='g')
            ax.bar(x_axis - 0.3, df.perfect, 0.2, label='perfect', color='b')

        ax.set_xticks(x_axis, df.index, rotation=45, ha="right")
        if title is None:
            ax.set_title("Input Correlation Thresholds")
        else:
            ax.set_title(title)
        ax.legend(title='Threshold')
        ax.grid()
        if fig is not None:
            fig.tight_layout()
        return fig

    def plot_significant_keys(self, key: str,
                              split_input_output: bool = True,
                              threshold: str | float = 'low',
                              ax: plt.Axes = None, absolute: bool = True,
                              title: str = None, y_label: str = None
                              ) -> plt.Figure:
        """Plot a bar-graph of the correlation coefficients of relevant keys for a given input key."""
        df = self.corr(key=key, sort=True, split_input_output=split_input_output)
        threshold = get_numerical_correlation_threshold(threshold)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        df = df[df.abs() > threshold]

        x_axis = np.arange(len(df.index))
        if absolute:
            df = df.abs()

        ax.bar(x_axis, df)

        ax.set_xticks(x_axis, df.index, rotation=45, ha="right")
        ax.set_ylim(0, 1)
        if title is None:
            ax.set_title(f"{key.title().replace('_', ' ')} - Significant Correlations")
        elif title == 'empty':
            pass
        else:
            ax.set_title(title)

        if y_label == 'empty':
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [])
        elif y_label is not None:
            ax.set_ylabel(y_label)
        elif absolute:
            ax.set_ylabel('Absolute Correlation Coefficient, -')
        else:
            ax.set_ylabel('Correlation Coefficient, -')
        ax.grid()
        if fig is not None:
            fig.tight_layout()
        return fig

    def plot_correlation_count_filter_progression(self, key: str, ax: plt.Axes = None) -> plt.Figure:
        """Plot a bar-graph of the correlation threshold counts for input
        parameters."""
        df = self.correlation_count_filter_progression(key)
        df.drop(index='total', inplace=True)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        x_axis = np.arange(len(df.columns))

        ax.stackplot(x_axis, df[::-1], baseline='zero', labels=df.index[::-1], colors=['r', 'orange', 'g', 'b'])

        ax.set_xticks(x_axis, df.columns, rotation=45, ha="right")
        ax.set_title(f"{key}Input Correlation Thresholds")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Threshold', loc='upper left')
        plt.grid()
        plt.tight_layout()
        return fig

    def plot_case_counts_with_filter_levels(self, stacked: bool = True,
                                            normed: bool = False,
                                            y_scale: str = 'linear',
                                            ax: plt.Axes = None
                                            ) -> plt.Figure:
        """Plot line plot as number of cases decreases with filter levels."""
        case_counts = self.get_case_count_with_filter_level(stacked=stacked, normed=normed)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        x_axis = np.arange(len(case_counts))

        ax.plot(x_axis, case_counts.values())

        ax.set_xticks(x_axis, case_counts.keys(), rotation=45, ha="right")
        ax.set_yscale(y_scale)
        ax.set_title("Case Counts with Filter Levels")
        plt.grid()
        plt.tight_layout()
        return fig

    def plot_filter_reduction_degree(self, individual: bool = False, stacked: bool = True,
                                     ax: plt.Axes = None) -> plt.Figure:
        """Plot line plot as number of cases decreases with filter levels."""
        reduction_degrees = self.get_filter_reduction_degrees(individual=individual, stacked=stacked)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        x_axis = np.arange(len(reduction_degrees))

        ax.bar(x_axis, reduction_degrees.values())

        ax.set_xticks(x_axis, reduction_degrees.keys(), rotation=45, ha="right")
        ax.set_title("Filter Reduction Degrees")
        plt.grid()
        plt.tight_layout()
        return fig

    def plot_filter_progression_box_plots(self, key: str, stacked: bool = True,
                                          ax: plt.Axes = None) -> plt.Figure:
        """Plot line plot as number of cases decreases with filter levels.

        Nan values are removed from the raw dataframe if needed.
        """
        dfs = self.get_dfs_filter_levels(stacked)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        x_axis = np.arange(1, len(dfs) + 1)
        y_values = [df.loc[:, key] for df in dfs.values()]
        if list(dfs.keys())[0] == 'raw':
            y_values[0].dropna(inplace=True)
        ax.boxplot(y_values, labels=dfs.keys())

        ax.set_xticks(x_axis, dfs.keys(), rotation=45, ha="right")
        ax.set_title(key.title().replace('_', ' ') + "Filter Distribution Impact")
        plt.grid()
        plt.tight_layout()
        return fig

    def plot_2d(self, key_x: str, keys_y: str | List[str],
                key_categorical: str = None, key_size: str = None,
                key_color: str = None, size_adj_factor: float = 1.0,
                ax: plt.Axes = None, title: str = None,
                x_label: str = None,
                y_label: str = None, legend_kwargs: dict = None,
                alternate_df: pd.DataFrame = None, plot_raw: bool = False,
                plot_errors: bool = False, plot_only_errors: bool = False,
                plot_kwargs: dict = None, **kwargs
                ) -> (plt.Figure, plt.Axes):
        """Plot a 2d line plot for an x and y value property set.

        Parameters
        ----------
        key_x
            X-axis value key

        keys_y
            y-axis value key or keys

        key_categorical: optional, default = None
            Categorical key to split the plotting over. E.g. fuel.

        key_size: optional, default = None
            Marker size value key

        key_color: optional, default = None
            Marker color value key. If a categorical key is given, each color gets a monotone color gradient.

        size_adj_factor: optional, default = 1.0
            Linear scaling factor on the marker size if a key_size is given.

        ax: optional, default = None
            Optionally pass an existing Axes to plot to same figure.

        title: optional, default = None
            Manually set plot title.

        x_label: optional, default = None
            Manually set x-axis label.

        y_label: optional, default = None
            Manually set y-axis label

        legend_kwargs: optional, default = None
            Dictionary of keyword argument to pass to the legend command.

        alternate_df: optional, default = None
            An alternate dataframe to use as source. If None is passed, the main dataframe is used. This
            functionality is used for example by the filter analysis functions. Please note, the dataframe is still
            interpreted with the stored unit and axis_label dataframes.

        plot_raw: optional, default = False
            Bool flag whether the raw data should be plotted beneath the target date. Data is plotted in grey.

        plot_errors: optional, default = False
            Bool flag whether the error data should be plotted beneath the target date. Data is plotted in red.

        plot_only_errors: optional, default = False
            Bool flag whether only the error data should be plotted.

        plot_kwargs: Optional
            Plot keyword arguments passed to the plotting function.

        kwargs: Optional
            Keyword arguments used for additional plot adjustment, e.g. disabling the color bar with colorbar=False,
            xlim, clim, ylim

        See Also
        --------
        plt.Figure
        plt.scatter
        """
        if legend_kwargs is None:
            legend_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        if alternate_df is not None:
            df = alternate_df
        else:
            df = self.df
        if isinstance(keys_y, str):
            key_y_single = keys_y
            title_y_label = keys_y.title().replace('_', ' ').replace('C ', '')
            keys_y_plotting = [keys_y]
        elif isinstance(keys_y, Iterable | Sequence):
            key_y_single = keys_y[0]
            axis_label = self.axis_labels[key_y_single]
            if axis_label[-1] == 's':
                title_y_label = axis_label + 'es'
            else:
                title_y_label = axis_label + 's'
            keys_y_plotting = keys_y
        else:
            raise ValueError("keys_y must be a str or Iterable[str].")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None
        ax.grid()

        if plot_raw and not plot_only_errors:
            ax.scatter(self.raw_df[key_x], self.raw_df[key_y_single], c='grey', **plot_kwargs, label='raw')
        if plot_errors:
            ax.scatter(self.df_errors[key_x], self.df_errors[key_y_single], c='red', **plot_kwargs, label='errors')
        if plot_only_errors:
            try:
                df_errors = alternate_df['df_errors']
                df_raw = alternate_df['df_raw']
            except (KeyError, TypeError):
                df_errors = self.df_errors
                df_raw = self.raw_df
            error_filter = self.error_filter
            if error_filter is not None and error_filter != []:
                df_errors_filter = apply_filter(df_raw, error_filter, join_and=False)
                df_errors_detected = df_errors.drop(index=df_errors_filter.index, errors='ignore')
            else:
                df_errors_filter = pd.DataFrame(columns=df_errors.columns)
                df_errors_detected = df_errors.copy()
            ax.scatter(df_errors_detected[key_x], df_errors_detected[key_y_single],
                       c='red', **plot_kwargs, label='Detected')
            if error_filter is not None and error_filter != []:
                ax.scatter(df_errors_filter[key_x], df_errors_filter[key_y_single],
                           c='tab:orange', **plot_kwargs, label='Filter')

        for key in keys_y_plotting:
            if plot_only_errors:
                break  # Don't plot anything else

            if key_categorical is None:
                if key_size is None and key_color is None:
                    ax.scatter(df[key_x], df[key], label=key, **plot_kwargs)
                elif key_size is None:
                    ax.scatter(df[key_x], df[key], c=df[key_color], cmap='jet', label=key, **plot_kwargs)
                elif key_color is None:
                    ax.scatter(df[key_x], df[key], s=df[key_size] * size_adj_factor, label=key, **plot_kwargs)
                else:
                    ax.scatter(df[key_x], df[key], s=df[key_size] * size_adj_factor, c=df[key_color],
                               cmap='jet', label=key, **plot_kwargs)
            else:
                keys_categorical = df.columns[
                    df.columns.str.contains(key_categorical)]
                for i_k, key_cat in enumerate(keys_categorical):
                    sub_df = df[df.loc[:, key_cat] == 1]
                    label_str = key_cat.replace(key_categorical + '_', '')
                    c_map = binary_color_maps[i_k + 2]
                    if key_size is None and key_color is None:
                        ax.scatter(sub_df[key_x], sub_df[key], label=label_str, **plot_kwargs)
                    elif key_size is None:
                        ax.scatter(sub_df[key_x], sub_df[key], c=sub_df[key_color], cmap=c_map,
                                   label=label_str, **plot_kwargs)
                    elif key_color is None:
                        ax.scatter(sub_df[key_x], sub_df[key], s=sub_df[key_size] * size_adj_factor,
                                   label=label_str, **plot_kwargs)
                    else:
                        ax.scatter(sub_df[key_x], sub_df[key], s=sub_df[key_size] * size_adj_factor,
                                   c=sub_df[key_color], label=label_str, cmap=c_map, **plot_kwargs)

        if (not isinstance(keys_y, str) or key_categorical is not None) and not plot_only_errors:
            if key_categorical is not None and 'title' not in legend_kwargs.keys():
                legend_kwargs['title'] = key_categorical.title().replace('_', ' ')
            legend_1 = ax.legend(**legend_kwargs)
        elif plot_only_errors:
            if 'title' not in legend_kwargs.keys():
                legend_kwargs['title'] = "Error Source"
            legend_1 = ax.legend(**legend_kwargs)
        else:
            legend_1 = None

        if x_label is None:
            ax.set_xlabel(self.axis_labels[key_x] + ", " + unit_2_latex(self.units[key_x]))
        else:
            ax.set_xlabel(x_label)
        if y_label is None:
            ax.set_ylabel(self.axis_labels[key_y_single] + ", " + unit_2_latex(self.units[key_y_single]))
        else:
            ax.set_ylabel(y_label)

        if key_color is not None and kwargs.get('colorbar', True) and not plot_only_errors:
            # Color bar location
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.15)

            # Colormap
            if key_categorical is None:
                cmap = mpl.cm.jet
            else:
                cmap = mpl.cm.binary
            if kwargs.get('clim', False):
                clim = kwargs.get('clim')
            else:
                clim = [df[key_color].min(), df[key_color].max()]
            norm = mpl.colors.Normalize(clim[0], clim[1])

            cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            cbar.set_label(self.axis_labels[key_color] + ", " + unit_2_latex(self.units[key_color]))

        if key_size is not None and not plot_only_errors:
            sizes = np.round(np.linspace(df[key_size].min(),
                                         df[key_size].max(), 3))
            labels = ["{:.3g}".format(s) for s in sizes]

            points = [ax.scatter([], [], s=s * size_adj_factor, c='gray') for s in sizes]
            ax.legend(points, labels, scatterpoints=1, title=key_size.title().replace('_', ' '), loc=7)
            if legend_1 is not None:
                ax.add_artist(legend_1)  # Re-add original legend.

        if kwargs.get('xlim', False):
            ax.set_xlim(*kwargs.get('xlim'))
        if kwargs.get('ylim', False):
            ax.set_ylim(*kwargs.get('ylim'))

        if title is None:
            ax.set_title(title_y_label)
        else:
            ax.set_title(title)
        plt.tight_layout()

        return fig, ax

    def multiplot_2d(
            self, key_x: str, keys_y: str | List[str],
            key_categorical: str = None, key_size: str = None,
            key_color: str = None, size_adj_factor: float = 1.0,
            key_multiplot: str = None,
            multiplot_set: None | int | np.ndarray = None,
            ax: plt.Axes = None, title: str = None, x_label: str = None,
            y_label: str = None, legend_kwargs: dict = None,
            plot_raw: bool = False, plot_errors: bool = False,
            plot_only_errors: bool = False,
            plot_kwargs: dict = None, **kwargs
    ) -> (plt.Figure, Tuple[Tuple[plt.Axes]], List[DataSet]):
        """Plot a collection of 2d line plot for an x and y value property set.
        Key value proposition of the function is the alignment of plots and unification of axis limits.

        Parameters
        ----------
        key_x
            X-axis value key

        keys_y
            y-axis value key or keys

        key_categorical: optional, default = None
            Categorical key to split the plotting over. E.g. fuel.

        key_size: optional, default = None
            Marker size value key

        key_color: optional, default = None
            Marker color value key. If a categorical key is given, each color gets a monotone color gradient.

        size_adj_factor: optional, default = 1.0
            Linear scaling factor on the marker size if a key_size is given.

        key_multiplot: optional, default = None
            Key to differentiate dataset for each sub-plot

        multiplot_set: optional, default = None
            Differentiation of the multiplot key. This is optional if key_multiplot is None or is a categorical
            value. In this case, as many plots are created as there are values. If the key_multiplot data is
            continuous, the multiplot_set can be an integer, which defines how many plots are created or a vector
            array which more closely defines

        ax: optional, default = None
            Optionally pass an existing Axes to plot to same figure.

        title: optional, default = None
            Manually set plot title.

        x_label: optional, default = None
            Manually set x-axis label.

        y_label: optional, default = None
            Manually set y-axis label

        legend_kwargs: optional, default = None
            Dictionary of keyword argument to pass to the legend command.

        plot_raw: optional, default = False
            Bool flag whether the raw data should be plotted beneath the target date. Data is plotted in grey.

        plot_errors: optional, default = False
            Bool flag whether the error data should be plotted beneath the target date. Data is plotted in red.

        plot_only_errors: optional, default = False
            Bool flag whether only the error data should be plotted.

        plot_kwargs: Optional
            Plot key word arguments passed to the plotting function.

        kwargs: Optional
            Keyword arguments used for additional plot adjustment, e.g. disabling the color bar with colorbar=False,
            xlim, clim, ylim

        See Also
        --------
        DataSet.plot_2d
        """
        if key_multiplot is None:
            fig, axes = self.plot_2d(
                key_x, keys_y, key_categorical, key_size, key_color, size_adj_factor, ax, title, x_label, y_label,
                legend_kwargs, None, plot_raw, plot_errors, plot_only_errors, plot_kwargs, **kwargs)
            datasets = [self]
        else:
            if plot_kwargs is None:
                plot_kwargs = {}

            datasets = self.get_split_dataset(key_multiplot, multiplot_set)
            n_plots = len(datasets)

            n_plots_minor = int(np.floor(np.sqrt(n_plots)))
            n_plots_major = int(np.ceil(n_plots / n_plots_minor))
            fig, axes = plt.subplots(n_plots_minor, n_plots_major)

            axes_1d = axes.reshape(-1)
            df = self.df
            xlim = kwargs.get('xlim', [df.loc[:, key_x].min(), df.loc[:, key_x].max()])
            ylim = kwargs.get('ylim', [df.loc[:, keys_y].min(), df.loc[:, keys_y].max()])
            if kwargs.get('clim', None) is None:
                if key_color is not None:
                    clim = [df.loc[:, key_color].min(), df.loc[:, key_color].max()]
                    norm = mpl.colors.Normalize(clim[0], clim[1])
                else:
                    clim = False
                    norm = None
            else:
                clim = kwargs.get('clim')
                norm = mpl.colors.Normalize(clim[0], clim[1])
            kwargs.update({'xlim': xlim, 'ylim': ylim, 'clim': clim, 'colorbar': False})
            plot_kwargs.update({'norm': norm})

            for i in range(n_plots):
                if plot_only_errors:
                    # Assumes the subset filter was added last in line. Typically true.
                    alternate_df = {'df_errors': datasets[i].df_errors.query(datasets[i].filter_list[-1]),
                                    'df_raw': datasets[i].raw_df.query(datasets[i].filter_list[-1])}
                else:
                    alternate_df = datasets[i].df

                # Simplify title for categorical splits
                if isinstance(datasets[i].filter_list, list):
                    sub_title = datasets[i].filter_list[-1]
                else:
                    sub_title = datasets[i].filter_list
                sub_cat = [k for k in self.keys_categorical if k + " == " in sub_title]
                if len(sub_cat) > 0:
                    sub_title = sub_title.replace(sub_cat[0] + " == ", "").replace('"', '')

                self.plot_2d(key_x, keys_y, key_categorical, key_size, key_color, size_adj_factor, axes_1d[i],
                             sub_title, x_label, y_label, legend_kwargs, alternate_df, plot_raw, plot_errors,
                             plot_only_errors, plot_kwargs, **kwargs)

            if title is not None:
                plt.suptitle(title)

            fig.tight_layout()  # Before color bar

            if key_color is not None:
                if key_categorical is None:
                    cmap = mpl.cm.jet
                else:
                    cmap = mpl.cm.binary
                cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist(), shrink=0.95)
                cbar.set_label(self.axis_labels[key_color] + ", " + unit_2_latex(self.units[key_color]))
        return fig, axes, datasets
    # endregion
