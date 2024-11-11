from dataclasses import dataclass

import numpy as np


@dataclass
class MotifRepresentative:
    """A data class to represent a motif."""

    representative: tuple[int, int]
    motif_set: list[tuple[int, int]]
    induced_paths: list[np.ndarray]
    fitness: float
