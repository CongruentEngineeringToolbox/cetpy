"""
Parallel Case Runner
===========

This file defines a CET case runner, that extends the original one and allows
parallel processing on n cores.

NOTE: I am not familiar enough with cetpy and it's inner workings, so I 
implemented the class in a way, that doesn't require to dig too deeply into 
cetpys internal structure.
"""

from multiprocessing import Pool

import cetpy.CaseTools
from cetpy.CaseTools._CaseRunner import CaseSolver
from cetpy.Modules.SysML import ValueProperty
import numpy as np
import pandas as pd


class ParallelCaseSolver(CaseSolver):

    def _solve_function(self) -> None:
        runner: ParallelCaseRunner = self.parent
        output_df = runner.output_df
        n_cores = runner.n_cores

        # Identify unsolved cases and only send these cases to the parallel case runners
        idx_unsolved = output_df.solved == False

        dfs = np.array_split(runner.case_generator.case_df[idx_unsolved], n_cores)
        # `CaseRunner` objects are not picklable, so we pass arguments to
        # other processes to create own CaseRunners. It is necessary,
        # because the objects have internal state, which cannot be shared
        # safely over multiple threads
        args = [(runner._serialize(), d) for d in dfs]
        with Pool(n_cores) as pool:
            evals = pool.starmap(ParallelCaseRunner._compute_cases, args)
            result = pd.concat(evals)

        # Rematch results to the unsolved cases, the output_df is mutable so changes are made within the
        # ParallelCaseRunner output_df
        output_df[idx_unsolved] = result


class ParallelCaseRunner(cetpy.CaseTools.CaseRunner):
    """Same as `CaseRunner`, but can perform calculations in n parallel
    processes.
    """

    # TODO: this couldn't be read from the config file, idk why
    n_cores = ValueProperty(permissible_types_list=int, permissible_list=(1, None))
    save_instances = ValueProperty(
        permissible_types_list=bool, permissible_list=[False]
    )

    # This is necessary to use ParallelCaseSolver instead of normal one
    def __init__(self, *args, n_cores=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_solver = ParallelCaseSolver(parent=self)
        self.n_cores = n_cores

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
