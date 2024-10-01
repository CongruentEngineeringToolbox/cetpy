"""
Case Runner
===========

This file defines a CET case runner designed to generate and execute thousands of design or operating variations.
"""
from __future__ import annotations

import logging
from typing import List, Callable
from time import perf_counter
import numpy as np
import pandas as pd
from traceback import format_tb

import cetpy.Configuration
import cetpy.Modules.SysML
import cetpy.Modules.Solver
import cetpy.CaseTools


class CaseSolver(cetpy.Modules.Solver.Solver):

    def _solve_function(self) -> None:
        runner: CaseRunner = self.parent
        output_df = runner.output_df
        instances = runner.module_instances

        for case in output_df.loc[~output_df.solved, :].itertuples():
            i = int(getattr(case, 'Index'))
            logging.info(f'Executing case {i}.')
            print(f'Executing case {i}.')
            t1 = perf_counter()
            if runner.catch_errors:
                # noinspection PyBroadException
                try:
                    runner.__initialise_case__(case)
                    runner.__evaluate_case__(case)
                    output_df.loc[i, 'solved'] = True
                except Exception as err:
                    output_df.loc[i, 'solved'] = True
                    output_df.loc[i, 'errored'] = True
                    output_df.loc[i, 'error_class'] = err.__class__.__name__
                    output_df.loc[i, 'error_message'] = err.args[0]
                    output_df.loc[i, 'error_location'] = format_tb(err.__traceback__)[-1]
                    output_df.loc[i, 'code_time'] = perf_counter() - t1
                    continue
            else:
                runner.__initialise_case__(case)
                runner.__evaluate_case__(case)
            output_df.loc[i, 'code_time'] = perf_counter() - t1

            if runner.save_instances:
                # noinspection PyProtectedMember
                instances[i] = runner._instance


class CaseRunner(cetpy.Modules.SysML.Block):
    """Congruent Engineering Toolbox Case Runner."""

    module = cetpy.Modules.SysML.ValueProperty(permissible_types_list=[type(cetpy.Modules.SysML.Block), str])
    save_instances = cetpy.Modules.SysML.ValueProperty(permissible_types_list=bool)
    catch_errors = cetpy.Modules.SysML.ValueProperty(permissible_types_list=bool)
    additional_module_kwargs = cetpy.Modules.SysML.ValueProperty(permissible_types_list=[dict, type(None)])
    output_properties = cetpy.Modules.SysML.ValueProperty(permissible_types_list=[list, type(None)])
    output_df_postprocess_function = cetpy.Modules.SysML.ValueProperty(permissible_types_list=[type(None), Callable])
    custom_evaluation_function = cetpy.Modules.SysML.ValueProperty(permissible_types_list=[type(None), Callable])
    enable_default_evaluation = cetpy.Modules.SysML.ValueProperty(permissible_types_list=bool)

    __init_parameters__ = \
        cetpy.Modules.SysML.Block.__init_parameters__.copy() + [
            'module', 'save_instances', 'catch_errors', 'additional_module_kwargs', 'output_properties',
            'output_df_postprocess_function', 'custom_evaluation_function', 'enable_default_evaluation'
        ]

    _reset_dict = cetpy.Modules.SysML.Block._reset_dict.copy()
    _reset_dict.update({'_output_df': None, '_module_instances': None, '_instance': None})

    def __init__(self, module: cetpy.Modules.SysML.Block | str = None,
                 input_df: pd.DataFrame = None,
                 save_instances: bool = False,
                 catch_errors: bool = True,
                 additional_module_kwargs: dict = None,
                 output_properties: List[str] = None,
                 method: str = 'direct',
                 sub_method: str | None = None,
                 n_cases: int = 1,
                 case_df_postprocess_function: Callable[[pd.DataFrame], pd.DataFrame] = None,
                 output_df_postprocess_function: Callable[[pd.DataFrame], pd.DataFrame] = None,
                 custom_evaluation_function:
                 Callable[[pd.DataFrame, pd.DataFrame, cetpy.Modules.SysML.Block], None] = None,
                 enable_default_evaluation: bool = True,
                 **kwargs):
        """Initialise a Case Runner for large-scale execution of many system evaluations.

        Parameters
        ----------
        module
            A cetpy Block derived system or name thereof as a string. This is the system that is instantiated and 
            tested in each case.

        input_df
            Pandas dataframe of input parameters and their ranges. In case of the 'direct' method, the input_df is 
            directly the case list, where each row represents one set of inputs for the system. Otherwise, 
            each parameter is a column and the rows can be one of 'min', 'max', 'list', 'mean', 'std'. 'mean' and 
            'std' are only permitted for the 'monte_carlo' method. Min/Max specify the lower and upper bounds of the 
            range, while list specifies a list of potential parameters, these can also include strings. An easy 
            way to generate a valid DataFrame is as follows:

            input_df = pd.DataFrame({
                'prop1': [1 , 2, None],
                'prop2': [None, None, [3, 4, 5]]
            }, index = ['min', 'max', 'list'])
            
        save_instances: optional, default = False
            Bool flag whether each generated instance should be saved to a list. Beware, this is very memory 
            intensive for large evaluations.
            
        catch_errors: optional, default = True
            Bool flag whether errors should be logged and the evaluation continued (True) or if the run should be 
            aborted (False) if an error occurs. For the occurring errors, the type, message, and location is logged.
            
        additional_module_kwargs: optional, default = None
            Any additional keyword arguments that should be given to the module on initialisation. Any properties in 
            both the input_df and the additional_module_kwargs list, are prioritised in the input_df.
            
        output_properties: optional, default = None
            A string list of properties that should automatically be written to the output dataframe. These do not 
            have to be on the top level, but can also be from parts, ports, solvers, and their sub parts, ports, 
            solver. Separate each level with a '.'.

        method: optional, default = 'direct'
            Method selector for case generation, one of:
            - direct:           Direct pass-through of input_df, each row represents a system input.
            - monte_carlo:      Generate a random distribution of each parameter individually. Can be an even 
                                distribution (min + max, list) or a normal distribution (mean + std).
            - lhs:              Latin-Hypercube Sampling, sample each parameter evenly in its design space, then
                                combine the parameter samplings such that no gaps are left in the design space and the
                                design space is evenly filled.
            - full_factorial:   Discretise each parameter in accordance to the n_cases parameter. Then combine each
                                combination of every value of every parameter. Beware the case counts grow very 
                                quickly!

        sub_method: optional, default = None
            Sub methods of the available methods, currently relevant for 'lhs'.
            Supports: 'ese' (recommended), 'center', 'maximin', 'centermaximin', 'correlation', and 'corr'
            See documentation of smt.sampling_methods for more information.

        n_cases: optional, default = 2
            Number of cases to generate. Has no effect in the 'direct' method. For full-factorial applies on a per 
            parameter basis. So 3 parameters with an n_cases of 3 results in 27 cases, for 10 parameters that is 
            59049 cases. For 10 parameters and an n_cases of 5 it is 10^7 cases. Behaviour with a single case is 
            untested, it is recommended to simply generate an instance of the system directly.

        case_df_postprocess_function: optional, default = None
            A function to modify the case dataframe after generation. Allows the operator to procedurally modify the 
            case list after generation but before execution in a CaseRunner. The function has to follow the pattern:

            def post_process(df: pd.DataFrame) -> pd.DataFrame:
                # Insert your modifications here
                return df
                
        output_df_postprocess_function: optional, default = None
            A function to modify the output dataframe after evaluation. Allows the operator to procedurally run 
            evaluations before saving. Useful for calculating ratios or derived values using vector math.

            def post_process(df: pd.DataFrame) -> pd.DataFrame:
                # Insert your modifications here
                return df
                
        custom_evaluation_function: optional, default = None
            A function to specify custom evaluation behaviour. Follows the pattern:

            def custom_evaluation_function(df_in, case, instance) -> None:
                # Insert your evaluation here.
                
        enable_default_evaluation: optional, default = True
            Bool flag whether the default evaluation using the output_properties should be conducted. Convenient way 
            to disable this aspect if a custom evaluation function is specified.
        """
        super().__init__(
            kwargs.pop('name', 'case_runner'),
            module=module,
            save_instances=save_instances,
            catch_errors=catch_errors,
            additional_module_kwargs=additional_module_kwargs,
            output_properties=output_properties,
            output_df_postprocess_function=output_df_postprocess_function,
            custom_evaluation_function=custom_evaluation_function,
            enable_default_evaluation=enable_default_evaluation,
            **kwargs)
        self._output_df = None
        self._module_instances = None
        self._instance = None
        self.case_solver = CaseSolver(parent=self)
        self.case_generator = cetpy.CaseTools.CaseGenerator(
            input_df=input_df, method=method, sub_method=sub_method, n_cases=n_cases,
            case_df_postprocess_function=case_df_postprocess_function)

    @module.setter
    def module(self, val: cetpy.Modules.SysML.Block | str) -> None:
        if isinstance(val, str):
            self._module = cetpy.Configuration.get_module(val)
        else:
            self._module = val

    @property
    def input_df(self) -> pd.DataFrame:
        """Case Runner input dataframe. Direct pass through of CaseGenerator.input_df setter and
        CaseGenerator.case_df getter."""
        return self.case_generator.case_df

    @input_df.setter
    def input_df(self, val: pd.DataFrame) -> None:
        self.case_generator.input_df = val

    @property
    def output_df(self) -> pd.DataFrame:
        """Return completed run of cases with input followed by output values."""
        if self._output_df is None:
            # Initialise output dataframe from input dataframe and add
            # columns for solver progress, any occurring error diagnostic,
            # and performance timing.
            self._output_df = self.input_df.copy()
            cols = ['solved', 'errored', 'error_class', 'error_message', 'error_location']
            self._output_df.loc[:, cols] = False
            self._output_df.loc[:, 'code_time'] = np.nan
        self.case_solver.solve()
        return self._output_df

    @property
    def output_df_post_processed(self) -> pd.DataFrame:
        """Return the completed run of cases with input followed by output values and the user post-processing
        function applied."""
        df = self.output_df
        if self.output_df_postprocess_function is not None:
            df = self.output_df_postprocess_function(df.copy())
        return df

    @property
    def module_instances(self) -> List[cetpy.Modules.SysML.Block]:
        """Return list of solved module instances if save_instances is enabled."""
        if self._module_instances is None:
            self._module_instances = [None] * self.input_df.shape[0]
        self.case_solver.solve()
        return self._module_instances

    def __initialise_case__(self, case) -> None:
        """Initialise the instance for a single specific case."""
        kwargs = self.additional_module_kwargs.copy()
        if kwargs is None:
            kwargs = {}

        for col in [c for c in case._fields if c in self.input_df.columns]:
            kwargs.update({col: getattr(case, col)})

        # noinspection PyPropertyAccess
        self._instance = self.module(**kwargs)

    def __evaluate_case__(self, case) -> None:
        """Evaluate and write the output for a single_specific_case."""
        output_properties = self.output_properties
        df = self.output_df
        i = getattr(case, 'Index')
        instance = self._instance
        if self.enable_default_evaluation:
            if output_properties is None:
                instance.solve()
                for vp in [type(instance).__getattr__(p) for p in instance.__dir__() if isinstance(
                        type(instance).__getattr__(p), cetpy.Modules.SysML.ValueProperty)]:
                    df.loc[i, vp.name] = vp.__get__(instance)
            else:
                for col in output_properties:
                    df.loc[i, col] = instance.__deep_getattr__(col)
        if self.custom_evaluation_function is not None:
            self.custom_evaluation_function(df, case, instance)
