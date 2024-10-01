"""
ParallelCaseRunner Test
=======================

This file implements tests for ParallelCaseRunner.
WARNING: this test is incomplete, bc I am not familiar with pytest yet.
"""

import os

import RockIt.Modules.CombustionChamber
import cetpy.CaseTools
from cetpy.CaseTools import _ParallelCaseRunner as ParallelCaseRunner
import cetpy.Modules.SysML
import pandas as pd
from time import perf_counter


OUTPUT_PROPERTIES = ["thrust", "isp_sl", "contour.cooling_channel.outlet.p"]


def custom_evaluation_function(df_in: pd.DataFrame, case, instance) -> None:
    """Evaluate the engine."""

    i = getattr(case, "Index")
    print("#######")
    # print(type(instance))
    print(instance.thrust)
    print(max(instance.contour.cooling_channel.t_s))
    print("#######")

    # Default evaluation
    for col in OUTPUT_PROPERTIES:
        df_in.loc[i, col] = instance.__deep_getattr__(col)
    df_in.loc[i, "t_coolant_max"] = max(instance.contour.cooling_channel.t_s)
    df_in.loc[i, "t_hot_wall_max"] = max(instance.contour.liner.t_hot_wall)
    df_in.loc[i, "chamber_pressure"] = instance.chamber_pressure
    df_in.loc[i, "dhdx_min"] = L(instance)[
        "dhdx_min"
    ]  # the smaller this value, the better the contour smoothness
    df_in.loc[i, "dhdx_max"] = L(instance)[
        "dhdx_max"
    ]  # the smaller this value, the better the contour smoothness


# contour channel height roughness evaluation:
def L_idx(instance, pt):
    if pt == 0:
        pt = 1
        print("USER WARNING: error for some contours at position 0, executing for pt=1")

    h0 = instance.contour.cooling_channel.height[pt]
    x0 = instance.contour.x[pt]

    L_mx = (
        instance.contour.cooling_channel.height[-1]
        - instance.contour.cooling_channel.height[0]
    ) / instance.contour.x[-1]
    L_mn = (
        instance.contour.cooling_channel.height[-1]
        - instance.contour.cooling_channel.height[0]
    ) / instance.contour.x[-1]

    for i in range(1, len(instance.contour.x)):
        f_mx = L_mx * (instance.contour.x[i] - x0) + h0
        f_mn = L_mn * (instance.contour.x[i] - x0) + h0

        if i != pt:
            if instance.contour.cooling_channel.height[i] > f_mx:
                temp = instance.contour.cooling_channel.height[i]
                L_mx = (temp - h0) / (instance.contour.x[i] - x0)
            if instance.contour.cooling_channel.height[i] < f_mn:
                temp = instance.contour.cooling_channel.height[i]
                L_mn = (temp - h0) / (instance.contour.x[i] - x0)
        if i == pt:
            temp = L_mn
            L_mn = L_mx
            L_mx = temp

    ref = (L_mx - L_mn) * 10**3  # distance between the 2 curves at 1m from x0 in mm
    return {"L1": L_mx, "L2": L_mn, "ref": ref}


def L(instance):
    sol = []
    for i in range(1, len(instance.contour.x)):
        sol.append(L_idx(instance, pt=i)["ref"])
    return {"dhdx_min": min(sol), "dhdx_max": max(sol)}


if __name__ == "__main__":
    cetpy.new_session(os.path.realpath("./"))

    input_df = pd.DataFrame(
        {
            "chamber_diameter": [0.1, 0.3],
            "contraction_ratio": [4, 9],
            "expansion_ratio": [4, 9],
            "characteristic_length": [0.7, 1.1],
        },
        index=["min", "max"],
    )
    cores = 11

    start = perf_counter()
    pcr = ParallelCaseRunner.ParallelCaseRunner(
        RockIt.Modules.CombustionChamber.RegenerativeCombustionChamber,
        input_df,
        catch_errors=True,
        additional_module_kwargs={},
        method="lhs",
        sub_method="ese",
        n_cases=int(cores * 4),
        custom_evaluation_function=custom_evaluation_function,
        enable_default_evaluation=False,
    )
    res = pcr.compute_output_df(n_cores=cores)
    stop = perf_counter()

    print(f"\nTime elapsed: {stop - start}")
