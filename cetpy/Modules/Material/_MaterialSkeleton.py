"""
Class Skeleton for a generic Material
=====================================

This file specifies the skeleton of a generic Material, that is the base
parameters that every Material implementation should implement.
"""

from typing import List
import numpy as np


class MaterialSkeleton:
    """Elemental functions of all Material class implementations."""

    __slots__ = ['_name']

    def __init__(self, name: str):
        self.name = name

    @property
    def name(self) -> str:
        """Fluid name (human-readable)."""
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        self._name = val

    def rho(self, t: float | np.ndarray | None = None) -> float | np.ndarray:
        """Return material density from temperature [kg/m^3].

        If no temperature is specified. Room Temperature 298.15 K (25 deg C)
        is assumed.
        """
        raise NotImplementedError

    def ultimate_tensile_strength(self, t: float | np.ndarray | None = None
                                  ) -> float | np.ndarray:
        """Return material ultimate tensile strength from temperature [kg/m^3].

        If no temperature is specified. Room Temperature 298.15 K (25 deg C)
        is assumed.
        """
        raise NotImplementedError

    def yield_strength(self, t: float | np.ndarray | None = None
                       ) -> float | np.ndarray:
        """Return material yield strength from temperature [kg/m^3].

        If no temperature is specified. Room Temperature 298.15 K (25 deg C)
        is assumed.
        """
        raise NotImplementedError

    def thermal_conductivity(self, t: float | np.ndarray | None = None) -> \
            float | np.ndarray:
        """Return material thermal conductivity from temperature [kg/m^3].

        If no temperature is specified. Room Temperature 298.15 K (25 deg C)
        is assumed.
        """
        raise NotImplementedError

    def thermal_expansion(self, t: float | np.ndarray | None = None
                          ) -> float | np.ndarray:
        """Return material thermal expansion coefficient from temperature [-].

        If no temperature is specified. Room Temperature 298.15 K (25 deg C)
        is assumed.
        """
        raise NotImplementedError

    def is_compatible(self, substance: str | List[str]) -> bool | List[bool]:
        """Return bool if fluid is compatible with a given substance."""
        raise NotImplementedError
