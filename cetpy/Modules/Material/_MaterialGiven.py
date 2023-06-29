"""
Implementation of a Material with given parameters
==================================================

This file specifies an implementation of a Material with given parameters.
"""

from typing import List
import numpy as np

from cetpy.Modules.Material._MaterialSkeleton import MaterialSkeleton


class MaterialGiven(MaterialSkeleton):
    """CET Material with given properties."""

    __slots__ = ['_rho', '_ultimate_tensile_strength', '_yield_strength',
                 '_yield_strength', '_thermal_conductivity',
                 '_thermal_expansion', '_compatibility']

    def __init__(self, name: str, rho: float = None,
                 ultimate_tensile_strength: float = None,
                 yield_strength: float = None,
                 thermal_conductivity: float = None,
                 thermal_expansion: float = None,
                 compatibility: List[str] = None):
        super().__init__(name)
        self._rho = rho
        self._ultimate_tensile_strength = ultimate_tensile_strength
        self._yield_strength = yield_strength
        self._thermal_conductivity = thermal_conductivity
        self._thermal_expansion = thermal_expansion
        self._compatibility = compatibility

    def rho(self, t: float | np.ndarray | None = None) -> float | np.ndarray:
        return self._rho

    def ultimate_tensile_strength(self, t: float | np.ndarray | None = None
                                  ) -> float | np.ndarray:
        return self._ultimate_tensile_strength

    def yield_strength(self, t: float | np.ndarray | None = None
                       ) -> float | np.ndarray:
        return self._yield_strength

    def thermal_conductivity(self, t: float | np.ndarray | None = None
                             ) -> float | np.ndarray:
        return self._thermal_conductivity

    def thermal_expansion(self, t: float | np.ndarray | None = None
                          ) -> float | np.ndarray:
        return self._thermal_expansion

    def is_compatible(self, substance: str | List[str]) -> bool | List[bool]:
        if isinstance(substance, str):
            return substance in self._compatibility
        else:
            return [s in self._compatibility for s in substance]
