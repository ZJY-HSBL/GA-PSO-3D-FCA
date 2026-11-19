"""GA-PSO-3D-FCA: auditable reproduction of a 3D fuzzy CA diffusion study."""
from .core import Structure, Params, reference_field, simulate, metrics, fitness
from .optimize import SearchConfig, optimize
__all__ = ['Structure','Params','reference_field','simulate','metrics','fitness','SearchConfig','optimize']
