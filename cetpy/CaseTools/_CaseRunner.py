"""
Case Runner
===========

This file defines a CET case runner designed to generate and execute
thousands of design or operating variations.
"""
from __future__ import annotations

import logging
from typing import List
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
                except Exception as err:
                    output_df.loc[i, 'solved'] = True
                    output_df.loc[i, 'errored'] = True
                    output_df.loc[i, 'error_class'] = err.__class__.__name__
                    output_df.loc[i, 'error_message'] = err.args[0]
                    output_df.loc[i, 'error_location'] = format_tb(
                        err.__traceback__)[-1]
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

    module = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=[type(cetpy.Modules.SysML.Block), str])
    save_instances = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=bool)
    catch_errors = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=bool)
    additional_module_kwargs = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=[dict, type(None)])
    output_properties = cetpy.Modules.SysML.ValueProperty(
        permissible_types_list=[list, type(None)])

    __init_parameters__ = cetpy.Modules.SysML.Block.__init_parameters__ + [
        'module', 'save_instances', 'catch_errors',
        'additional_module_kwargs', 'output_properties'
    ]

    _reset_dict = cetpy.Modules.SysML.Block._reset_dict
    _reset_dict.update({'_output_df': None, '_module_instances': None,
                        '_instance': None})

    def __init__(self, module: cetpy.Modules.SysML.Block | str = None,
                 input_df: pd.DataFrame = None,
                 save_instances: bool = False,
                 catch_errors: bool = True,
                 additional_module_kwargs: dict = None,
                 output_properties: List[str] = None,
                 method: str = 'direct',
                 sub_method: str | None = None,
                 n_cases: int = 1, **kwargs):
        super().__init__(kwargs.pop('name', 'case_runner'),
                         module=module,
                         save_instances=save_instances,
                         catch_errors=catch_errors,
                         additional_module_kwargs=additional_module_kwargs,
                         output_properties=output_properties,
                         **kwargs)
        self._output_df = None
        self._module_instances = None
        self._instance = None
        self.case_solver = CaseSolver(parent=self)
        self.case_generator = cetpy.CaseTools.CaseGenerator(
            input_df=input_df, method=method, sub_method=sub_method,
            n_cases=n_cases)

    @module.setter
    def module(self, val: cetpy.Modules.SysML.Block | str) -> None:
        if isinstance(val, str):
            self._module = cetpy.Configuration.get_module(val)
        else:
            self._module = val

    @property
    def input_df(self) -> pd.DataFrame:
        """Case Runner input dataframe. Direct pass through of
        CaseGenerator.input_df setter and CaseGenerator.case_df getter."""
        return self.case_generator.case_df

    @input_df.setter
    def input_df(self, val: pd.DataFrame) -> None:
        self.case_generator.input_df = val

    @property
    def output_df(self) -> pd.DataFrame:
        """Return completed run of cases with input followed by output
        values."""
        if self._output_df is None:
            # Initialise output dataframe from input dataframe and add
            # columns for solver progress, any occurring error diagnostic,
            # and performance timing.
            self._output_df = self.input_df.copy()
            cols = ['solved', 'errored', 'error_class', 'error_message',
                    'error_location']
            self._output_df.loc[:, cols] = False
            self._output_df.loc[:, 'code_time'] = np.nan
        self.case_solver.solve()
        return self._output_df

    @property
    def module_instances(self) -> List[cetpy.Modules.SysML.Block]:
        """Return list of solved module instances if save_instances is
        enabled."""
        if self._module_instances is None:
            self._module_instances = [None] * self.input_df.shape[0]
        self.case_solver.solve()
        return self._module_instances

    def __initialise_case__(self, case) -> None:
        """Initialise the instance for a single specific case."""
        kwargs = self.additional_module_kwargs.copy()
        if kwargs is None:
            kwargs = {}

        for col in [c for c in case._fields
                    if c in self.case_generator.input_keys]:
            kwargs.update({col: getattr(case, col)})

        # noinspection PyPropertyAccess
        self._instance = self.module(**kwargs)

    def __evaluate_case__(self, case) -> None:
        """Evaluate and write the output for a single_specific_case."""
        output_properties = self.output_properties
        df = self.output_df
        i = getattr(case, 'Index')
        instance = self._instance
        if output_properties is None:
            instance.solve()
            for vp in [type(instance).__getattr__(p)
                       for p in instance.__dir__() if isinstance(
                    type(instance).__getattr__(p),
                    cetpy.Modules.SysML.ValueProperty)]:
                df.loc[i, vp.name] = vp.__get__(instance)
        else:
            for col in output_properties:
                df.loc[i, col] = instance.__deep_getattr__(col)
