"""
Case Generator
==============

This file defines a CET case generator which wraps around itertools and
SurrogateModellingToolbox SMT Latin-Hypercube-Functions to generate cases
based on user input ranges.
"""
from __future__ import annotations

from typing import List, Callable
import numpy as np
import pandas as pd
import itertools
import smt.sampling_methods

import cetpy.Configuration
import cetpy.CaseTools
import cetpy.Modules.SysML
import cetpy.Modules.Solver


class CaseGeneratorSolver(cetpy.Modules.Solver.Solver):

    def _solve_function(self) -> None:
        block: cetpy.CaseTools.CaseGenerator = self.parent
        match block.method:
            case 'direct':
                case_df = block.__get_direct__()
            case 'full_factorial':
                case_df = block.__get_full_factorial__()
            case 'monte_carlo':
                case_df = block.__get_monte_carlo__()
            case 'lhs':
                case_df = block.__get_lhs__()
            case _:
                raise ValueError("Invalid method selector.")

        # Apply User Post-Processing Function if available
        if block.case_df_postprocess_function is not None:
            case_df = block.case_df_postprocess_function(case_df)

        block._case_df = case_df


class CaseGenerator(cetpy.Modules.SysML.Block):
    """Congruent Engineering Toolbox Case Generator.

    The CaseGenerator combines a set of input parameters to create large
    sets of evaluation cases for cetpy Blocks. It can do so in four key ways:
    - direct: the boring way, what every you put in, is what you get out.
              Used to read in pre-generated lists into the CaseRunner.
    - monte_carlo: Randomize each parameter either with an even distribution or
                   a normal distribution.
    - lhs: Latin Hypercube Sampling, a.k.a Monte Carlo on Steroids. Evenly
           resolve each design parameter and combine them in a way that doesn't
           leave gaps in the design space.
    - full_factorial: Every value of every parameter with every value of
                      every other parameter. Easy on the analysis and
                      punishing on the CPU. Beware, the case count applies
                      to each parameter individually and the true case count
                      quickly explodes.
    """

    input_df = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=pd.DataFrame)
    method = cetpy.Modules.SysML.ValueProperty(
        permissible_list=['direct', 'monte_carlo', 'lhs', 'full_factorial'],
        permissible_types_list=str)
    sub_method = cetpy.Modules.SysML.ValueProperty(
        permissible_list=['ese', 'center', 'maximin', 'centermaximin',
                          'correlation', 'corr', None],
        permissible_types_list=[type(None), str])
    n_cases = cetpy.Modules.SysML.ValueProperty(
        permissible_list=(0, None),
        permissible_types_list=[type(None), int])
    case_df_postprocess_function = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=[type(None), Callable])

    __init_parameters__ = \
        cetpy.Modules.SysML.Block.__init_parameters__.copy() + [
            'input_df', 'method', 'sub_method', 'n_cases',
            'case_df_postprocess_function'
        ]

    _reset_dict = cetpy.Modules.SysML.Block._reset_dict.copy()
    _reset_dict.update({'_case_df': None, '_module_instances': None,
                        '_instance': None})

    def __init__(self, input_df: pd.DataFrame = None,
                 method: str = 'direct',
                 sub_method: str = None,
                 n_cases: int = 2,
                 case_df_postprocess_function:
                 Callable[[pd.DataFrame], pd.DataFrame] = None,
                 **kwargs):
        """Initialise a Case Generator for combination of parameters to 
        create a large evaluation case list.
        
        Parameters
        ----------
        input_df
            Pandas dataframe of input parameters and their ranges. In case of 
            the 'direct' method, the input_df is directly the case list, 
            where each row represents one set of inputs for the system. 
            Otherwise, each parameter is a column and the rows can be one of 
            'min', 'max', 'list', 'mean', 'std'. 'mean' and 'std' are only 
            permitted for the 'monte_carlo' method. Min/Max specify the 
            lower and upper bounds of the range, while list specifies a list of 
            potential parameters, these can also include strings. An easy 
            way to generate a valid DataFrame is as follows:
            
            input_df = pd.DataFrame({
                'prop1': [1 , 2, None],
                'prop2': [None, None, [3, 4, 5]]
            }, index = ['min', 'max', 'list'])
            
        method: optional, default = 'direct'
            Method selector for case generation, one of:
            - direct:           Direct pass-through of input_df, each row 
                                represents a system input.
            - monte_carlo:      Generate a random distribution of each 
                                parameter individually. Can be an even 
                                distribution (min + max, list) or a normal 
                                distribution (mean + std).
            - lhs:              Latin-Hypercube Sampling, sample each 
                                parameter evenly in its design space, then
                                combine the parameter samplings such that no
                                gaps are left in the design space and the
                                design space is evenly filled.
            - full_factorial:   Discretise each parameter in accordance to 
                                the n_cases parameter. Then combine each
                                combination of every value of every 
                                parameter. Beware the case counts grow very 
                                quickly!
                                
        sub_method: optional, default = None
            Sub methods of the available methods, currently relevant for 'lhs'.
            Supports: 'ese' (recommended), 'center', 'maximin', 
                      'centermaximin', 'correlation', and 'corr'
            See documentation of smt.sampling_methods for more information.
            
        n_cases: optional, default = 2
            Number of cases to generate. Has no effect in the 'direct' 
            method. For full-factorial applies on a per parameter basis. So 
            3 parameters with an n_cases of 3 results in 27 cases, for 10 
            parameters that is 59049 cases. For 10 parameters and an n_cases of
            5 it is 10^7 cases. Behaviour with a single case is untested, 
            it is recommended to simply generate an instance of the system 
            directly.
        
        case_df_postprocess_function: optional, default = None
            A function to modify the case dataframe after generation. Allows 
            the operator to procedurally modify the case list after generation 
            but before execution in a CaseRunner. The function has to follow 
            the pattern:
            
            def post_process(df: pd.DataFrame) -> pd.DataFrame:
                # Insert your modifications here
                return df
        """
        super().__init__(
            kwargs.pop('name', 'case_generator'),
            input_df=input_df,
            method=method,
            sub_method=sub_method,
            n_cases=n_cases,
            case_df_postprocess_function=case_df_postprocess_function,
            **kwargs)
        self._case_df = None
        self.case_solver = CaseGeneratorSolver(parent=self)

    @cetpy.Modules.SysML.value_property()
    def input_keys(self) -> List[str]:
        """List of keys in input_df."""
        return self.input_df.keys()

    @cetpy.Modules.SysML.value_property()
    def n_input_keys(self) -> int:
        """Return number of input keys."""
        return len(self.input_keys)

    @property
    def case_df(self) -> pd.DataFrame:
        """Return generated case list."""
        self.case_solver.solve()
        return self._case_df

    # region Input Interpretation
    @property
    def idx_min_max(self) -> np.ndarray:
        """Return filter of input keys defined with a min and max limit."""
        input_df = self.input_df
        if 'min' not in input_df.index or 'max' not in input_df.index:
            return np.zeros(self.n_input_keys, dtype=bool)
        else:
            return np.bitwise_not(np.isnan(input_df.loc['min', :])
                                  + np.isnan(input_df.loc['max', :]))

    @property
    def idx_list(self) -> np.ndarray:
        """Return filter of input keys defined with a list of values."""
        input_df = self.input_df
        if 'list' not in input_df.index:
            return np.zeros(self.n_input_keys, dtype=bool)
        else:
            return np.bitwise_not(pd.isna(input_df.loc['list', :]))

    @property
    def idx_normal_distribution(self) -> np.ndarray:
        """Return filter of input keys defined with a normal distribution."""
        input_df = self.input_df
        if 'mean' not in input_df.index or 'std' not in input_df.index:
            return np.zeros(self.n_input_keys, dtype=bool)
        else:
            return np.bitwise_not(np.isnan(input_df.loc['mean', :])
                                  + np.isnan(input_df.loc['std', :]))

    def __get_preprocessed_input_df__(self) -> pd.DataFrame:
        """Return version of the input_df with lists converted to a minmax
        range.

        See Also
        --------
        CaseGenerator.get_post_processed_case_df
        """
        input_df = self.input_df
        idx_list = self.idx_list
        if not any(idx_list):
            return input_df.drop(index='list', errors='ignore')
        if 'min' not in input_df.index:
            input_df.loc['min', :] = np.nan
        if 'max' not in input_df.index:
            input_df.loc['max', :] = np.nan

        for key in self.input_keys[self.idx_list]:
            # Replace list by an even distribution between 0 and the number
            # of items in the list. The methods for monte carlo and
            # Latin-Hypercube sampling can only sample on numeric intervals.
            # So the dataset is converted back after case generation. Here
            # the first item in the list corresponds to the interval 0 - 1,
            # the second to 1-2, etc.
            input_df.loc['min', key] = 0
            input_df.loc['max', key] = len(input_df.loc['list', key])
        return input_df.drop(index='list', errors='ignore')

    def __get_post_processed_case_df__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the case list with any list input keys returned to their
        list values.

        See Also
        --------
        CaseGenerator.get_preprocessed_input_df
        """
        input_df = self.input_df
        for key in self.input_keys[self.idx_list]:
            df.loc[:, key] = input_df.loc['list', key].take(
                np.floor(df.loc[:, key]).astype(int))
        return df
    # endregion

    # region Solvers
    def __get_direct__(self) -> pd.DataFrame:
        """Return direct pass through of input dataframe."""
        return self.input_df.copy()

    def __get_full_factorial__(self) -> pd.DataFrame:
        """Return full-factorial sampled dataframe of the input dataframe."""
        input_df = self.input_df
        n_cases = self.n_cases
        # 1. Expand each input key to its full set
        key_array = np.empty(self.n_input_keys, dtype=object)
        idx_minmax = self.idx_min_max
        idx_list = self.idx_list
        idx_normal_distribution = self.idx_normal_distribution
        for i, key in enumerate(self.input_keys):
            if idx_minmax[i]:
                key_array[i] = np.linspace(input_df.loc['min', key],
                                           input_df.loc['max', key],
                                           n_cases)
            elif idx_list[i]:
                key_array[i] = np.asarray(input_df.loc['list', key])
            elif idx_normal_distribution[i]:
                key_array[i] = np.random.normal(input_df.loc['mean', key],
                                                input_df.loc['std', key],
                                                n_cases)
            else:
                raise ValueError(f"Key {key} could not be parsed. Please "
                                 f"make sure the key is defined with a valid "
                                 f"input set. That is either a 'min' and "
                                 f"'max' value or a 'list' of valid values "
                                 f"or a 'mean' and 'std'.")
        # 2. Multiply each key list with each other using itertools
        return pd.DataFrame(itertools.product(*key_array),
                            columns=input_df.columns)

    def __get_monte_carlo__(self) -> pd.DataFrame:
        """Return monte carlo sampled dataframe of the input dataframe."""
        input_df = self.__get_preprocessed_input_df__()
        n_cases = self.n_cases
        idx_minmax = np.bitwise_or(self.idx_min_max, self.idx_list)
        case_df = pd.DataFrame(np.empty((n_cases, self.n_input_keys)),
                               columns=input_df.columns)

        array_minmax = np.random.random((n_cases, sum(idx_minmax)))
        minimum = input_df.loc['min', idx_minmax].to_numpy()
        maximum = input_df.loc['max', idx_minmax].to_numpy()
        array_minmax = minimum + array_minmax * (maximum - minimum)
        case_df.iloc[:, idx_minmax] = array_minmax

        for key in self.input_keys[self.idx_normal_distribution]:
            case_df.loc[:, key] = np.random.normal(
                input_df.loc['mean', key], input_df.loc['std', key], n_cases)

        return self.__get_post_processed_case_df__(case_df)

    def __get_lhs__(self) -> pd.DataFrame:
        """Return monte carlo sampled dataframe of the input dataframe."""
        input_df = self.__get_preprocessed_input_df__().drop(
            index=['mean', 'std'], errors='ignore')
        sub_method = self.sub_method
        if sub_method is None:
            sub_method = 'ese'

        case_df = pd.DataFrame(
            smt.sampling_methods.LHS(xlimits=np.fliplr(input_df.to_numpy().T),
                                     criterion=sub_method)(self.n_cases),
            columns=self.input_keys
        )

        return self.__get_post_processed_case_df__(case_df)
    # endregion
