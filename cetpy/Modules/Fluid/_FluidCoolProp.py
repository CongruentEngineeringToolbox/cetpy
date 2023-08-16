"""
CoolProp Implementation of a CET Fluid
======================================

This file specifies an implementation of a CET Fluid using the open-source
CoolProp Package.
"""

import numpy as np
from CoolProp.CoolProp import PropsSI, PhaseSI

from cetpy.Modules.Fluid import FluidSkeleton


class FluidCoolProp(FluidSkeleton):
    """Implements a CET Fluid using CoolProp."""

    __slots__ = ['_cpid']

    def __init__(self, name: str, function: str = None, cpid: str = None):
        self._cpid = cpid
        self._name = None  # for comparison with cpid.
        super().__init__(name, function)

    @property
    def name(self) -> str:
        """Fluid name (human-readable).

        Inherited function is extended to write the coolprop ID (cpid) if
        the cpid is currently not set or the same as the existing name."""
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        if self._cpid is None or self._cpid == self._name:
            self._cpid = val
        self._name = val

    @property
    def cpid(self) -> str:
        """Fluid CoolProp ID."""
        return self._cpid

    @cpid.setter
    def cpid(self, val: str) -> None:
        self._cpid = val

    def rho(self, t: float | np.ndarray, p: float | np.ndarray
            ) -> float | np.ndarray:
        return PropsSI('D', 'T', t, 'P', p, self.cpid)

    def p(self, t: float | np.ndarray, rho: float | np.ndarray
          ) -> float | np.ndarray:
        return PropsSI('P', 'T', t, 'D', rho, self.cpid)

    def h(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        return PropsSI('H', 'T', t, 'P', p, self.cpid)

    def t_kappa(self, t1: float | np.ndarray, p1: float | np.ndarray,
                p2: float | np.ndarray) -> float | np.ndarray:
        h = self.h(t1, p1)
        return PropsSI('T', 'P', p2, 'H', h, self.cpid)

    def t_boil(self, p: float | np.ndarray) -> float | np.ndarray:
        return PropsSI('T', 'P', p, 'Q', 0, self.cpid)

    def p_boil(self, t: float | np.ndarray) -> float | np.ndarray:
        return PropsSI('P', 'T', t, 'Q', 0, self.cpid)

    def t_crit(self) -> float:
        return PropsSI('Tcrit', self.cpid)

    def p_crit(self) -> float:
        return PropsSI('Pcrit', self.cpid)

    def phase(self, t: float | np.ndarray, p: float | np.ndarray
              ) -> str | np.ndarray:
        return PhaseSI('T', t, 'P', p, self.cpid)

    def t_rhoh(self, rho: float | np.ndarray, h: float | np.ndarray
               ) -> float | np.ndarray:
        return PropsSI('T', 'D', rho, 'H', h, self.cpid)

    def t_ph(self, p: float | np.ndarray, h: float | np.ndarray
             ) -> float | np.ndarray:
        return PropsSI('T', 'P', p, 'H', h, self.cpid)

    def mu(self, t: float | np.ndarray, p: float | np.ndarray
           ) -> float | np.ndarray:
        return PropsSI('V', 'T', t, 'P', p, self.cpid)

    def c(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        return PropsSI('C', 'T', t, 'P', p, self.cpid)

    def a(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        return PropsSI('A', 'T', t, 'P', p, self.cpid)

    def pr(self, t: float | np.ndarray, p: float | np.ndarray
           ) -> float | np.ndarray:
        return PropsSI('PRANDTL', 'T', t, 'P', p, self.cpid)

    def k(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        return PropsSI('CONDUCTIVITY', 'T', t, 'P', p, self.cpid)

    def kappa(self, t: float | np.ndarray, p: float | np.ndarray
              ) -> float | np.ndarray:
        return PropsSI('ISENTROPIC_EXPANSION_COEFFICIENT', 'T', t, 'P', p,
                       self.cpid)

    def m(self) -> float | np.ndarray:
        return PropsSI('M', self.cpid)
