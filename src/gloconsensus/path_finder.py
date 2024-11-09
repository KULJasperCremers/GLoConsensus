import numpy as np
import path as path_class
from numba import boolean, float32, int32, njit, types


@njit(
    int32[:, :](float32[:, :], boolean[:, :], int32, int32, int32[:, :], int32, int32)
)
def max_warping_pathV1(
    cumulative_similarity_matrix: np.ndarray,
    mask: np.ndarray,
    best_row_index: int,
    best_column_index: int,
    STEP_SIZES: np.ndarray,
    max_horizontal_step: int,
    max_vertical_step: int,
) -> np.ndarray:
    """Trace back the maximum warping path from a given position in the csm."""
    path = []
    while (
        best_row_index >= max_vertical_step and best_column_index >= max_horizontal_step
    ):
        path.insert(
            0,
            (
                best_row_index - max_vertical_step,
                best_column_index - max_horizontal_step,
            ),
        )
        indices = (
            np.array([best_row_index, best_column_index], dtype=np.int32) - STEP_SIZES
        )
        values = np.array(
            [cumulative_similarity_matrix[_row, _column] for (_row, _column) in indices]
        )
        masked = np.array([mask[_row, _column] for (_row, _column) in indices])
        argmax = np.argmax(values)
        if masked[argmax]:
            break

        best_row_index = best_row_index - STEP_SIZES[argmax, 0]
        best_column_index = best_column_index - STEP_SIZES[argmax, 1]

    return np.array(path, dtype=np.int32)


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32))
def mask_pathV1(
    path: np.ndarray, mask: np.ndarray, max_horizontal_step: int, max_vertical_step: int
) -> np.ndarray:
    """Update the mask to include the positions covered by the given path."""
    for rows, cols in path:
        mask[rows + max_horizontal_step, cols + max_vertical_step] = True
    return mask


@njit(boolean[:, :](int32[:, :], boolean[:, :], int32, int32, int32))
def mask_vicinityV0(
    path: np.ndarray,
    mask: np.ndarray,
    max_horizontal_step: int,
    max_vertical_step: int,
    V_WIDTH: int,
) -> np.ndarray:
    """Update the mask to include the vicinity around the given path."""
    (row_start, column_start) = path[0] + np.array(
        (max_vertical_step, max_horizontal_step)
    )
    for row_next, column_next in path[1:] + np.array(
        [max_vertical_step, max_horizontal_step]
    ):
        row_difference = row_next - row_start
        column_difference = column_start - column_next
        error = row_difference + column_difference
        while row_start != row_next or column_start != column_next:
            mask[row_start - V_WIDTH : row_start + V_WIDTH + 1, column_start] = True
            mask[row_start, column_start - V_WIDTH : column_start + V_WIDTH + 1] = True

            step = 2 * error
            if step > column_difference:
                error += column_difference
                row_start += 1
            if step < row_difference:
                error += row_difference
                column_start += 1

    mask[row_next - V_WIDTH : row_next + V_WIDTH + 1, column_next] = True
    mask[row_next, column_next - V_WIDTH : column_next + V_WIDTH + 1] = True
    return mask


@njit(types.List(int32[:, :])(float32[:, :], int32[:, :], int32, int32, boolean[:, :]))
def find_warping_pathsV1(
    cumulative_similarity_matrix: np.ndarray,
    STEP_SIZES: np.ndarray,
    L_MIN: int,
    V_WIDTH: int,
    mask: np.ndarray,
) -> list[np.ndarray]:
    """Identify the best warping paths in the cumulative similarity matrix."""
    max_vertical_step = np.max(STEP_SIZES[:, 0])
    max_horizontal_step = np.max(STEP_SIZES[:, 1])

    rows_, cols_ = np.nonzero(cumulative_similarity_matrix <= 0)
    for best_index in range(len(rows_)):
        mask[rows_[best_index], cols_[best_index]] = True

    row_indices, column_indices = np.nonzero(cumulative_similarity_matrix)
    non_zero_values = np.array(
        [
            cumulative_similarity_matrix[row_indices[i], column_indices[i]]
            for i in range(len(row_indices))
        ]
    )
    sorting_indices = np.argsort(non_zero_values)
    sorted_row_indices: np.ndarray = row_indices[sorting_indices]
    sorted_column_indices: np.ndarray = column_indices[sorting_indices]

    best_index: int = len(sorted_row_indices) - 1
    paths: list[np.ndarray] = []

    while best_index >= 0:
        path = np.empty((0, 0), dtype=np.int32)
        path_found = False
        while not path_found:
            while mask[
                sorted_row_indices[best_index], sorted_column_indices[best_index]
            ]:
                best_index -= 1
                if best_index < 0:
                    return paths

            best_row_index: int = sorted_row_indices[best_index]
            best_column_index: int = sorted_column_indices[best_index]
            if (
                best_row_index < max_vertical_step
                or best_column_index < max_horizontal_step
            ):
                return paths

            path: np.ndarray = max_warping_pathV1(
                cumulative_similarity_matrix,
                mask,
                best_row_index,
                best_column_index,
                STEP_SIZES,
                max_horizontal_step,
                max_vertical_step,
            )

            mask = mask_pathV1(path, mask, max_horizontal_step, max_vertical_step)

            if (path[-1][0] - path[0][0] + 1) >= L_MIN or (
                path[-1][1] - path[0][1] + 1
            ) >= L_MIN:
                path_found = True

        mask = mask_vicinityV0(
            path, mask, max_horizontal_step, max_vertical_step, V_WIDTH
        )
        paths.append(path)

    return paths


def find_induced_pathsV0(
    start_index: int,
    end_index: int,
    paths: list[path_class.Path],
    mask: np.ndarray,
) -> list[np.ndarray]:
    """Find paths induced by given start and end indices that do not overlap with masked
    positions.
    """
    induced_paths = []
    for path in paths:
        if path.column_start <= start_index and end_index <= path.column_end:
            start_column = path.find_column(start_index)
            end_column = path.find_column(end_index - 1)
            motif_start = path[start_column][0]
            motif_end = path[end_column][0] + 1

            if not np.any(mask[motif_start:motif_end]):
                induced_path = np.copy(path.path[start_column : end_column + 1])
                induced_paths.append(induced_path)
    return induced_paths
