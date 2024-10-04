"""
test_Combustion.py
==================

This file declares tests concerning the Combustion Class, which is a wrapper
around Cantera to provide common and protected interfaces.
"""

import pytest
import numpy as np
import pandas as pd

from cetpy.tests.TestModules.TestSysML.test_Block import TestBlock

from cetpy.CaseTools import CaseRunner
from cetpy.Modules.SysML import Block, value_property

input_df_1 = pd.DataFrame(
    {
        "property_1": [0.1, 0.3, None, None, None],
        "property_2": [4, 9, None, None, None],
        "property_3": [None, None, [True, False], None, None],
        "property_4": [None, None, [2, 3, 4, 5], None, None],
    },
    index=["min", "max", 'list', 'mean', 'std'],
)


class TestRunnerExampleClass(Block):
    pass

    @value_property()
    def errored_calculation(self) -> float:
        raise ValueError()


class TestCaseRunner(TestBlock):

    resolution = 5
    save_instances_range = [False, True]
    catch_errors_range = [False, True]
    method_choice_range = ['direct', 'monte_carlo', 'lhs', 'full_factorial']
    sub_method_range = ['ese', 'center', 'maximin', 'centermaximin', 'correlation', 'corr']
    n_cases_range = [2, 20]

    @pytest.fixture
    def test_class(self) -> type(CaseRunner):
        return CaseRunner

    @pytest.fixture
    def case_runner_example_class(self) -> type(TestRunnerExampleClass):
        return TestRunnerExampleClass

    @pytest.fixture
    def init_kwargs(self, case_runner_example_class) -> dict:
        return {case_runner_example_class, input_df_1}

    def pytest_generate_tests(self, metafunc):
        # Direct range use
        parameters = ['save_instances', 'catch_errors', 'method_choice',
                      'sub_method', 'n_cases']
        for par in parameters:
            if par in metafunc.fixturenames:
                metafunc.parametrize(par, getattr(self, par + '_range'))
        # Linear space distribution
        parameters = []
        for par in parameters:
            if par in metafunc.fixturenames:
                metafunc.parametrize(
                    par, np.round(np.linspace(*getattr(self, par + '_range'), self.resolution), 3))

    def test_solver_function(self, test_class, init_kwargs, save_instances, catch_errors, method_choice, sub_method,
                             n_cases):
        if method_choice != 'lhs' and sub_method != 'ese':
            pytest.skip()
        c = test_class(save_instances=save_instances, catch_errors=catch_errors, method=method_choice,
                       sub_method=sub_method, n_cases=n_cases, **init_kwargs)
        output_df = c.output_df
        assert isinstance(output_df, pd.DataFrame)
        assert np.all(output_df.solved == True)

    def test_catch_errors_true(self, test_class, init_kwargs):
        c = test_class(save_instances=False, catch_errors=True, method='lhs',
                       sub_method='ese', n_cases=2, output_properties=['errored_calculation'], **init_kwargs)
        output_df = c.output_df
        assert isinstance(output_df, pd.DataFrame)
        assert np.all(output_df.errored)
        assert np.all(output_df.error_class == ValueError)
        assert np.all([isinstance(s, str) for s in output_df.error_message])
        assert np.all([isinstance(s, str) for s in output_df.error_location])

    def test_catch_errors_false(self, test_class, init_kwargs):
        c = test_class(save_instances=False, catch_errors=False, method='lhs',
                       sub_method='ese', n_cases=2, output_properties=['errored_calculation'], **init_kwargs)
        try:
            _ = c.output_df
            assert False
        except ValueError:
            assert True
