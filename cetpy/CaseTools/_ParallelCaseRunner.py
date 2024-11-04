"""
Parallel Case Runner
===========

This file defines a CET case runner, that extends the original one and allows
parallel processing on n cores.
"""

from multiprocessing import Pool

import cetpy.CaseTools
from cetpy.CaseTools._CaseRunner import CaseSolver, CaseRunner
from cetpy.Modules.SysML import ValueProperty
import numpy as np
import pandas as pd


class ParallelCaseSolver(CaseSolver):

    def _solve_function(self) -> None:
        """Split the generated cases into n parts and feeds them to n processes.

        Every process has its own (regular) `CaseRunner` and processes the
        cases in direct manner. Then the result is merged back into `output_df`
        of the `ParallelCaseRunner`, that this object belongs to.
        """
        runner: ParallelCaseRunner = self.parent
        output_df = runner.output_df
        n_cores = runner.n_cores

        # Identify unsolved cases and only to prevent recomputing solved ones
        idx_unsolved = runner.output_df.solved == False

        dfs = np.array_split(runner.case_generator.case_df[idx_unsolved], n_cores)
        # `CaseRunner` objects are not picklable, so we pass arguments to
        # other processes to create own `CaseRunner`s. It is necessary,
        # because the objects have internal state, which cannot be shared
        # safely over multiple threads
        # noinspection PyProtectedMember
        args = [(runner._serialize(), d) for d in dfs]
        with Pool(n_cores) as pool:
            # noinspection PyProtectedMember
            evals = pool.starmap(ParallelCaseRunner._compute_cases, args)
            result = pd.concat(evals)

        # This comment is so long, because I spent like 2 days fighting this
        # bug, so bear with me.
        # Create columns, that don't exist in current version of the
        # `output_df`. This is necessary, because python refuses to create them
        # automatically and so the next line of code doesn't work without it.
        # This will only create columns, that don't exist in `output_df`, and
        # have no effect on columns, that exist in `output_df`, but not in
        # `result`.
        rescols = set(result.columns)
        outcols = set(output_df.columns)
        output_df[list(rescols.difference(outcols))] = 0

        # Rematch results to the unsolved cases, the `output_df` is mutable so
        # changes are made within the `ParallelCaseRunner`s `output_df`
        output_df.loc[idx_unsolved] = result


class ParallelCaseRunner(CaseRunner):
    """Same as `CaseRunner`, but can perform calculations in n parallel processes."""

    n_cores = ValueProperty(permissible_types_list=int, permissible_list=(1, None))
    save_instances = ValueProperty(permissible_types_list=bool, permissible_list=[False])

    __init_parameters__ = CaseRunner.__init_parameters__.copy() + ['n_cores']

    # This is necessary to use `ParallelCaseSolver` instead of normal one
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.solvers.remove(self.case_solver)
        self.case_solver = ParallelCaseSolver(parent=self)

    def _serialize(self) -> dict:
        """Returns all arguments, needed to create a copy of original
        `CaseRunner` in a different process.

        Rationale
        ---------
        `CaseRunner` objects are not picklable due to decorators (idk what
        pythons problem with them is), so we cannot just pass the original
        object to other thread and make a copy there.
        Also we could just use the `self.__dict__` but
        `self.case_generator.__dict__` is not picklable either.
        """
        return {vp.name: vp.__get__(self) for vp in self.report.input_properties}

    @staticmethod
    def _compute_cases(cr_args: dict, cases: pd.DataFrame) -> pd.DataFrame:
        """The function, passed to pool workers. It has to be static, because
        python.

        Parameters
        ----------
        cr_args
            Dictionary, that contains all information, needed to create (almost)
            copies of original `CaseRunner`. Every worker has own `CaseRunner`,
            because these objects have internal state and thus can't be shared
            between processes.
        cases
            Pandas `DataFrame` with cases to compute in "direct" mode. The idea
            is to generate cases in original process, split them in equal parts
            and pass to worker processes. This is not exactly efficient, but
            splitting ranges and generating cases in each process would result
            in gaps in design space (as far as I understand).
        """
        input_dict = cr_args.copy()
        input_dict["input_df"] = cases
        input_dict["method"] = "direct"
        input_dict["save_instances"] = False  # Not supported currently
        cr = cetpy.CaseTools.CaseRunner(**input_dict)
        return cr.output_df
