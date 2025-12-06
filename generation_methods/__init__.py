"""
Generation methods package for 2048 tile generation.
"""

from generation_methods.base import GenerationMethod
from generation_methods.random import Random2
from generation_methods.default import Default
from generation_methods.scaling import Scaling

__all__ = ['GenerationMethod', 'Random2', 'Default', 'Scaling']

