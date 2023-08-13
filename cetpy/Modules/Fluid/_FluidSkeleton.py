"""
Class Skeleton for a generic Fluid
==================================

This file specifies the skeleton of a Fluid, that is the base parameters that
every Fluid implementation should implement.
"""

import numpy as np


class FluidSkeleton:
    """Elemental functions of all Fluid class implementations."""

    __slots__ = ['_name', '_function']

    def __init__(self, name: str, function: str = None):
        self.name = name
        self.function = function

    @property
    def name(self) -> str:
        """Fluid name (human-readable)."""
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        self._name = val

    @property
    def function(self) -> str:
        """Fluid function in a system."""
        return self._function

    @function.setter
    def function(self, val: str | None) -> None:
        self._function = val

    def rho(self, t: float | np.ndarray, p: float | np.ndarray
            ) -> float | np.ndarray:
        """Return fluid density from temperature and pressure [kg/m^3]."""
        raise NotImplementedError

    def p(self, t: float | np.ndarray, rho: float | np.ndarray
          ) -> float | np.ndarray:
        """Return fluid pressure from temperature and density [Pa]."""
        raise NotImplementedError

    def h(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        """Return fluid mass specific enthalpy from temperature and
        pressure [J/kg]."""
        raise NotImplementedError

    def t_kappa(self, t1: float | np.ndarray, p1: float | np.ndarray,
                p2: float | np.ndarray) -> float | np.ndarray:
        """Return fluid isentropic expansion temperature from initial
         temperature and pressure and final pressure [K]."""
        raise NotImplementedError

    def t_boil(self, p: float | np.ndarray) -> float | np.ndarray:
        """Return fluid boiling temperature from pressure [K]."""
        raise NotImplementedError

    def p_boil(self, t: float | np.ndarray) -> float | np.ndarray:
        """Return fluid boiling pressure from temperature [Pa]."""
        raise NotImplementedError

    def t_crit(self) -> float:
        """Return critical temperature [K]."""
        raise NotImplementedError

    def p_crit(self) -> float:
        """Return critical pressure [Pa]."""
        raise NotImplementedError

    def phase(self, t: float | np.ndarray, p: float | np.ndarray
              ) -> str | np.ndarray:
        """Return fluid phase from temperature and pressure."""
        raise NotImplementedError

    def is_critical(self, t: float | np.ndarray, p: float | np.ndarray
                    ) -> float | np.ndarray:
        """Return bool true if fluid is supercritical."""
        return self.phase(t, p) in ['supercritical',
                                    'supercritical_liquid',
                                    'supercritical_gas']

    def is_gas(self, t: float | np.ndarray, p: float | np.ndarray
               ) -> float | np.ndarray:
        """Return bool true if fluid is gas or supercritical gas."""
        return self.phase(t, p) in ['gas', 'supercritical',
                                    'supercritical_gas']

    def t_rhoh(self, rho: float | np.ndarray, h: float | np.ndarray
               ) -> float | np.ndarray:
        """Return fluid temperature from density and enthalpy [K]."""
        raise NotImplementedError

    def t_ph(self, p: float | np.ndarray, h: float | np.ndarray
             ) -> float | np.ndarray:
        """Return fluid temperature from pressure and enthalpy [K]."""
        raise NotImplementedError

    def mu(self, t: float | np.ndarray, p: float | np.ndarray
           ) -> float | np.ndarray:
        """Return fluid dynamic viscosity from temperature and
        pressure [Pas]."""
        raise NotImplementedError

    def nu(self, t: float | np.ndarray, p: float | np.ndarray
           ) -> float | np.ndarray:
        """Return fluid kinematic viscosity from temperature and
        pressure [m^2/s]."""
        return self.mu(t, p) / self.rho(t, p)

    def c(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        """Return fluid mass specific heat capacity from temperature and
         pressure [J/(kgK)]."""
        raise NotImplementedError

    def a(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        """Return fluid speed of sound from temperature and pressure [m/s]."""
        raise NotImplementedError

    def pr(self, t: float | np.ndarray, p: float | np.ndarray
           ) -> float | np.ndarray:
        """Return fluid prandtl number from temperature and pressure [-]."""
        raise NotImplementedError

    def k(self, t: float | np.ndarray, p: float | np.ndarray
          ) -> float | np.ndarray:
        """Return fluid thermal conductivity from temperature and
        pressure [W/(mK)]."""
        raise NotImplementedError

    def kappa(self, t: float | np.ndarray, p: float | np.ndarray
              ) -> float | np.ndarray:
        """Return fluid isentropic expansion coefficient from temperature and
         pressure [-]."""
        raise NotImplementedError

    def m(self) -> float | np.ndarray:
        """Return fluid molecular mass [kg/mol]."""
        raise NotImplementedError
