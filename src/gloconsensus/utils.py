import pickle
from multiprocessing import shared_memory
from typing import Generator, Optional, Union

import motif_finder as mf
import numpy as np
import pandas as pd
import path_finder as pf
import similarity_matrix as sm
from motif_representative import MotifRepresentative

RHO = 0.8


def calculate_similarity_matrices(
    ts1: np.ndarray, ts2: np.ndarray, diagonal: bool, GAMMA: int
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if diagonal:
        di_similarity_matrix = sm.calculate_similarity_matrixV1(
            ts1, ts2, GAMMA, only_triu=True
        )
        return di_similarity_matrix, None, None
    else:
        ut_similarity_matrix = sm.calculate_similarity_matrixV1(
            ts1, ts2, GAMMA, only_triu=False
        )
        lt_similarity_matrix = sm.calculate_similarity_matrixV1(
            ts2, ts1, GAMMA, only_triu=False
        )
        return None, ut_similarity_matrix, lt_similarity_matrix


def calculate_cumulative_similarity_matrices(
    similarity_matrix: Optional[np.ndarray], is_diagonal: bool, STEP_SIZES: np.ndarray
) -> np.ndarray:
    if is_diagonal:
        tau = estimate_tau_symmetric(similarity_matrix)
    else:
        tau = estimate_tau_assymmetric(similarity_matrix)
    if tau:
        delta_a = 2 * tau
        delta_m = 0.5
        csm = sm.calculate_cumulative_similarity_matrixV1(
            similarity_matrix, STEP_SIZES, tau, delta_a, delta_m
        )
    return csm


def find_local_warping_paths(
    sm_tuple: tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
    csm: np.ndarray,
    diagonal: bool,
    STEP_SIZES: np.ndarray,
    L_MIN: int,
    V_WIDH: int,
    mask: np.ndarray,
    global_start_index_tuple: tuple[int, int],
) -> tuple[
    Optional[list[tuple[np.ndarray, np.ndarray]]],
    Optional[list[tuple[np.ndarray, np.ndarray]]],
    Optional[list[tuple[np.ndarray, np.ndarray]]],
]:
    di_sm, ut_sm, lt_sm = sm_tuple
    row_start, col_start = global_start_index_tuple

    found_paths = pf.find_warping_pathsV1(csm, STEP_SIZES, L_MIN, V_WIDH, mask)
    di_paths = []
    ut_paths = []
    lt_paths = []
    if diagonal and di_sm is not None:
        # hardcode the diagonal path because we already masked it
        diagonal_indices = np.arange(len(di_sm))
        diagonal_path = np.stack((diagonal_indices, diagonal_indices), axis=1)
        diagonal_path_similarities = di_sm[diagonal_indices, diagonal_indices]
        # map the local path to the global comparison matrix using the global indices
        global_diagonal_path = diagonal_path + (row_start, col_start)
        di_paths.append(
            # ensure paths are serializable for multiprocessing!
            (
                global_diagonal_path.astype(np.int32),
                diagonal_path_similarities.astype(np.float32),
            )
        )

    for found_path in found_paths:
        # path logic for the local comparison
        rows = found_path[:, 0]
        # map the local row to the global comparison matrix using the global indices
        global_rows = rows + row_start
        cols = found_path[:, 1]
        # map the local cols to the global comparison matrix using hte global indices
        global_cols = cols + col_start
        if diagonal and di_sm is not None:
            # similarities use the local sm and indices
            found_path_similarities = di_sm[rows, cols]
            di_paths.append(
                # ensure paths are serializable for multiprocessing!
                (
                    np.stack((global_rows, global_cols), axis=1).astype(np.int32),
                    found_path_similarities.astype(np.float32),
                )
            )
        # if non-diagonal comparison paths are upper triangular
        elif ut_sm is not None:
            found_path_similarities = ut_sm[rows, cols]
            ut_paths.append(
                # ensure paths are serializable for multiprocessing!
                (
                    np.stack((global_rows, global_cols), axis=1).astype(np.int32),
                    found_path_similarities.astype(np.float32),
                )
            )

        # if non-diagonal comparison paths can be mirrored to be lower triangular
        # comparison (0,1) is mirrored to (1,0) in the global comparison matrix
        if not diagonal and lt_sm is not None:
            mirrored_rows = found_path[:, 1]
            global_mirorred_rows = mirrored_rows + col_start  # also mirror start
            mirrored_cols = found_path[:, 0]
            global_mirorred_cols = mirrored_cols + row_start  # also mirror start
            mirrored_found_path_similarities = lt_sm[mirrored_rows, mirrored_cols]
            lt_paths.append(
                # ensure paths are serializable for multiprocessing
                (
                    np.stack(
                        (global_mirorred_rows, global_mirorred_cols), axis=1
                    ).astype(np.int32),
                    mirrored_found_path_similarities.astype(np.float32),
                )
            )

    if diagonal:
        return (di_paths, None, None)
    else:
        return (None, ut_paths, lt_paths)


def transfer_global_columns_to_shared_memory(global_column_dict_lists_paths):
    shared_memory_block_dict = {}
    for column_index in global_column_dict_lists_paths.keys():
        global_column = list(global_column_dict_lists_paths[column_index])
        serialized_global_column = pickle.dumps(global_column)
        shared_memory_block = shared_memory.SharedMemory(
            create=True, size=len(serialized_global_column)
        )
        shared_memory_block.buf[: len(serialized_global_column)] = (
            serialized_global_column
        )
        shared_memory_block_dict[column_index] = shared_memory_block

    return shared_memory_block_dict


def find_motif_representatives(
    x: int,
    global_offsets: np.ndarray,
    # global_column_dict_path: dict[
    # int, types.ListType(path_class.Path.class_type.instance_type)  # type: ignore
    # ],
    shared_memory_block_dict,
    L_MIN: int,
    L_MAX: int,
    OVERLAP: float,
) -> Generator[MotifRepresentative, None, None]:
    return mf.find_motifs_representativesV3(
        x, global_offsets, shared_memory_block_dict, L_MIN, L_MAX, OVERLAP
    )


def estimate_tau_symmetric(
    similarity_matrix: Optional[np.ndarray],
) -> Optional[np.float32]:
    if similarity_matrix is not None:
        return np.quantile(
            similarity_matrix[np.triu_indices(len(similarity_matrix))], RHO, axis=None
        )


def estimate_tau_assymmetric(
    similarity_matrix: Optional[np.ndarray],
) -> Optional[np.float32]:
    if similarity_matrix is not None:
        return np.quantile(similarity_matrix, RHO, axis=None)


def z_normalize(timeseries: np.ndarray) -> np.ndarray:
    mean = np.mean(timeseries)
    std = np.std(timeseries)
    return (timeseries - mean) / std


def offsets_indexer(n: int) -> list[list[tuple[int, int]]]:
    offsets_indices = []
    for i in range(n):
        for j in range(n):
            if j >= i:
                offset_index = [(i, i + 1), (j, j + 1)]
                offsets_indices.append(offset_index)
    return offsets_indices


def load_patient_data(file_path: str) -> dict[str, list[pd.DataFrame]]:
    with open(file_path, 'rb') as f:
        patient_data = pickle.load(f)
    return patient_data


def filter_time_series(
    patient_data: dict[str, list[pd.DataFrame]],
    patient_ids: list[str],
    scenario_ids: Union[str, list[str]],
    time_ids: Union[str, list[str]],
) -> list[np.ndarray]:
    filtered_data = []
    for patient_id in patient_ids:
        patient_dfs = patient_data.get(patient_id, [])

        matching_dfs = [
            df
            for df in patient_dfs
            if df['scenario'].iloc[0] in scenario_ids and df['time'].iloc[0] in time_ids
        ]

        for df in matching_dfs:
            data_array = df[['x', 'y']].to_numpy(dtype=np.float32)
            filtered_data.append(data_array)

    return filtered_data
