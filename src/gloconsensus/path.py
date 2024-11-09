import numpy as np
from numba import float32, int32
from numba.experimental import jitclass

class_setup = [
    ('path', int32[:, :]),
    ('path_similarities', float32[:]),
    ('cumulative_path_similarity', float32[:]),
    ('row_indices', int32[:]),
    ('column_indices', int32[:]),
    ('row_start', int32),
    ('row_end', int32),
    ('column_start', int32),
    ('column_end', int32),
]


@jitclass(class_setup)
class Path:
    """Represents a warping path between two timeseries segments, including similarity
    information.

    Attributes:
        - path: the path as an array of (row, column) indices.
        - path_similarities: the similarity values along the path.
        - cumulative_path_similarity: the cumulative sum of similarities along the path.
        - row_indices: mapping from timeseries row indices to path indices.
        - column_indices: mapping from timeseries column indices to path indices.
    """

    def __init__(
        self,
        path: np.ndarray,
        path_similarities: np.ndarray,
    ):
        self.path = path
        self.path_similarities = path_similarities
        self.cumulative_path_similarity = np.concatenate(
            (np.array([0.0], dtype=np.float32), np.cumsum(self.path_similarities))
        )
        self.row_start = self.path[0][0]
        self.row_end = self.path[-1][0] + 1
        self.column_start = self.path[0][1]
        self.column_end = self.path[-1][1] + 1

        self.construct_indices()

    def construct_indices(self) -> None:
        """Constructs mappings from timeseries indices to path indices."""
        current_row = self.row_start
        current_column = self.column_start

        row_indices = np.zeros(self.row_end - self.row_start, dtype=np.int32)
        column_indices = np.zeros(self.column_end - self.column_start, dtype=np.int32)

        for path_index in range(1, len(self.path)):
            if self.path[path_index][0] != current_row:
                row_indices[
                    current_row - self.row_start + 1 : self.path[path_index][0]
                    - self.row_start
                    + 1
                ] = path_index
                current_row = self.path[path_index][0]

            if self.path[path_index][1] != current_column:
                column_indices[
                    current_column - self.column_start + 1 : self.path[path_index][1]
                    - self.column_start
                    + 1
                ] = path_index
                current_column = self.path[path_index][1]

        self.row_indices = row_indices
        self.column_indices = column_indices

    def find_row(self, row: int) -> int:
        """Find the index in the path corresponding to the given row timeseries index."""
        return self.row_indices[row - self.row_start]

    def find_column(self, column: int) -> int:
        """Find the index in the path corresponding to the given column timeseries index."""
        return self.column_indices[column - self.column_start]

    def __getitem__(self, index: int) -> np.ndarray:
        """Get the (row, column) pair at a specific index in the path."""
        return self.path[index, :]
