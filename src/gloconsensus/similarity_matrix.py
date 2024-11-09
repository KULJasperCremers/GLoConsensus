import numpy as np
from numba import boolean, float32, float64, int32, njit


@njit(float32[:, :](float32[:, :], float32[:, :], int32, boolean))
def calculate_similarity_matrixV1(
    timeseries1: np.ndarray,
    timeseries2: np.ndarray,
    GAMMA: int,
    only_triu: bool = False,
) -> np.ndarray:
    """Calculate the similarity matrix between two timeseries using specified GAMMA value."""
    n, m = len(timeseries1), len(timeseries2)
    similarity_matrix = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        j_start = i if only_triu else 0
        j_end = m
        similarities = np.exp(
            -GAMMA
            * np.sum(
                np.power(timeseries1[i, :] - timeseries2[j_start:j_end, :], 2), axis=1
            )
        )
        similarity_matrix[i, j_start:j_end] = similarities
    return similarity_matrix


@njit(float32[:, :](float32[:, :], int32[:, :], float64, float64, float64))
def calculate_cumulative_similarity_matrixV1(
    similarity_matrix: np.ndarray,
    STEP_SIZES: np.ndarray,
    tau: float,
    delta_a: float,
    delta_m: float,
) -> np.ndarray:
    """Calculate the cumulative similarity matrix from the similarity matrix."""
    max_vertical_step = np.max(STEP_SIZES[:, 0])
    max_horizontal_step = np.max(STEP_SIZES[:, 1])
    n, m = similarity_matrix.shape
    cumulative_similarity_matrix = np.zeros(
        (
            n + max_vertical_step,
            m + max_horizontal_step,
        ),
        dtype=np.float32,
    )

    for row in range(n):
        for column in range(m):
            similarity = similarity_matrix[row, column]
            indices = (
                np.array([row + max_vertical_step, column + max_horizontal_step])
                - STEP_SIZES
            )
            max_cumulative_similarity = np.amax(
                np.array([cumulative_similarity_matrix[i_, j_] for (i_, j_) in indices])
            )
            if similarity < tau:
                cumulative_similarity_matrix[
                    row + max_vertical_step, column + max_horizontal_step
                ] = max(0, delta_m * max_cumulative_similarity - delta_a)
            else:
                cumulative_similarity_matrix[
                    row + max_vertical_step, column + max_horizontal_step
                ] = max(0, max_cumulative_similarity + similarity)

    return cumulative_similarity_matrix
