import numpy as np
import path as path_class
from numba import boolean, float32, int32, njit, prange, types


@njit(
    types.Tuple((types.UniTuple(int32, 2), float32))(
        boolean[:],
        boolean[:],
        boolean[:],
        types.ListType(path_class.Path.class_type.instance_type),  # type: ignore
        int32,
        int32,
        int32,
        float32,
    ),
    parallel=True,
)
def find_candidatesV3(
    start_mask: np.ndarray,
    end_mask: np.ndarray,
    mask: np.ndarray,
    paths: types.ListType(path_class.Path.class_type.instance_type),  # type: ignore
    start_offset: np.int32,
    L_MIN: np.int32,
    L_MAX: np.int32,
    OVERLAP: np.float32,
):
    """Identify the best candidate motif within specified masks and paths using fitness
    evaluation.
    """
    # n for local indices
    n = len(start_mask)
    # global n for global indices
    global_n = len(mask)
    num_paths = len(paths)

    # global path indices
    path_start_indices = np.array([path.column_start for path in paths])
    path_end_indices = np.array([path.column_end for path in paths])

    num_start_indices = n - L_MIN + 1
    if num_start_indices <= 0:
        return ((0, 0), 0.0)

    # store per-thread results
    best_fitness_arr = np.zeros(n - L_MIN + 1, dtype=np.float32)
    best_candidate_arr = np.zeros((n - L_MIN + 1, 2), dtype=np.int32)

    # parallel loop over start_index
    for start_index in prange(n - L_MIN + 1):
        if not start_mask[start_index]:
            continue

        thread_best_fitness = 0.0
        thread_best_candidate = (0, n)

        # map local indices to global indices
        global_start_index = start_index + start_offset
        start_indices_mask = path_start_indices <= global_start_index

        # local column indices
        for end_index in range(
            start_index + L_MIN, min(n + 1, start_index + L_MAX + 1)
        ):
            if not end_mask[end_index - 1]:
                continue

            # map local indices to global indices
            global_end_index = end_index + start_offset
            if np.any(mask[global_start_index:global_end_index]):
                break
            end_indices_mask = path_end_indices >= global_end_index

            path_mask = start_indices_mask & end_indices_mask
            if sum(path_mask) < 2:
                break

            motif_start_indices = np.zeros(num_paths, dtype=np.int32)
            motif_end_indices = np.zeros(num_paths, dtype=np.int32)
            crossing_start_indices = np.zeros(num_paths, dtype=np.int32)
            crossing_end_indices = np.zeros(num_paths, dtype=np.int32)

            for path_index in np.flatnonzero(path_mask):
                path = paths[path_index]
                # global path indices
                crossing_start_indices[path_index] = path_row = path.find_column(
                    global_start_index
                )
                crossing_end_indices[path_index] = path_column = path.find_column(
                    global_end_index - 1
                )
                motif_start_indices[path_index] = path[path_row][0]
                motif_end_indices[path_index] = path[path_column][0] + 1

                # global mask
                if np.any(
                    mask[
                        motif_start_indices[path_index] : motif_end_indices[path_index]
                    ]
                ):
                    path_mask[path_index] = False

            if sum(path_mask) < 2:
                break

            sorted_motif_start_indices = motif_start_indices[path_mask]
            sorted_motif_end_indices = motif_end_indices[path_mask]
            sorted_best_indices = np.argsort(sorted_motif_start_indices)
            sorted_motif_start_indices = sorted_motif_start_indices[sorted_best_indices]
            sorted_motif_end_indices = sorted_motif_end_indices[sorted_best_indices]

            sorted_length = sorted_motif_end_indices - sorted_motif_start_indices
            sorted_length[:-1] = np.minimum(sorted_length[:-1], sorted_length[1:])
            overlaps = np.maximum(
                sorted_motif_end_indices[:-1] - sorted_motif_start_indices[1:] - 1, 0
            )

            if np.any(overlaps > OVERLAP * sorted_length[:-1]):
                continue

            coverage = np.sum(
                sorted_motif_end_indices - sorted_motif_start_indices
            ) - np.sum(overlaps)
            # mask and coverage are on global scale, so global n
            coverage_amount = coverage / float(global_n)

            score = 0
            for path_index in np.flatnonzero(path_mask):
                score += (
                    paths[path_index].cumulative_path_similarity[
                        crossing_end_indices[path_index] + 1
                    ]
                    - paths[path_index].cumulative_path_similarity[
                        crossing_start_indices[path_index]
                    ]
                )
            score_amount = score / float(
                np.sum(
                    crossing_end_indices[path_mask]
                    - crossing_start_indices[path_mask]
                    + 1
                )
            )

            denominator = coverage_amount + score_amount
            if denominator != 0:
                fit = 2 * (coverage_amount * score_amount) / denominator
            else:
                fit = 0.0

            if fit == 0.0:
                continue

            if fit > thread_best_fitness:
                # map the candidate to the global matrix
                thread_best_candidate = (global_start_index, global_end_index)
                thread_best_fitness = fit

        best_fitness_arr[start_index] = thread_best_fitness
        best_candidate_arr[start_index] = thread_best_candidate

    best_index = np.argmax(best_fitness_arr)
    best_fitness = best_fitness_arr[best_index]
    best_candidate = (
        best_candidate_arr[best_index][0],
        best_candidate_arr[best_index][1],
    )

    return best_candidate, best_fitness
