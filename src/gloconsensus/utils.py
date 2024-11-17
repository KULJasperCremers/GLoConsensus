import pickle
from typing import Generator, Union

import motif_finder as mf
import numpy as np
import pandas as pd
import path_finder as pf
import similarity_matrix as sm
from motif_representative import MotifRepresentative

from numba import boolean, float32, int32, njit, typed, types

RHO = 0.8

@njit(types.Tuple((
    float32[:,:], float32[:,:], float32[:,:]
    ))(
        float32[:,:],
        float32[:,:],
        boolean,
        int32
    )
)
def calculate_similarity_matrices(
    ts1: np.ndarray, ts2: np.ndarray, diagonal: bool, GAMMA: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    zero_array = np.zeros((0,0), dtype=np.float32)
    if diagonal:
        di_similarity_matrix = sm.calculate_similarity_matrixV1(
            ts1, ts2, GAMMA, only_triu=True
        )
        return di_similarity_matrix, zero_array, zero_array
    else:
        ut_similarity_matrix = sm.calculate_similarity_matrixV1(
            ts1, ts2, GAMMA, only_triu=False
        )
        lt_similarity_matrix = sm.calculate_similarity_matrixV1(
            ts2, ts1, GAMMA, only_triu=False
        )
        return zero_array, ut_similarity_matrix, lt_similarity_matrix

def estimate_tau_symmetric(
    similarity_matrix: np.ndarray
) -> np.float32:
    return np.quantile(
        similarity_matrix[np.triu_indices(len(similarity_matrix))], RHO, axis=None
    )

def estimate_tau_assymmetric(
    similarity_matrix: np.ndarray
) -> np.float32:
    return np.quantile(similarity_matrix, RHO, axis=None)

@njit(float32[:,:](float32[:,:], float32, int32[:,:]))
def calculate_cumulative_similarity_matrices(
    similarity_matrix: np.ndarray, tau: np.float32, STEP_SIZES: np.ndarray
) -> np.ndarray:
    delta_a = 2 * tau
    delta_m = 0.5
    csm = sm.calculate_cumulative_similarity_matrixV1(
        similarity_matrix, STEP_SIZES, tau, delta_a, delta_m
    )
    return csm

tuple_type = types.Tuple((int32[:], int32[:], float32[:]))
@njit(
    types.Tuple((
        types.ListType(tuple_type),
        types.ListType(tuple_type),
        types.ListType(tuple_type),
    ))(
        float32[:, :],
        float32[:, :],
        float32[:, :],
        float32[:, :],
        boolean,       
        int32[:, :],
        int32,         
        int32,         
        boolean[:, :],  
        int32,
        int32        
    )
)
def find_local_warping_paths(
    di_sm: np.ndarray,
    ut_sm: np.ndarray,
    lt_sm: np.ndarray,
    csm: np.ndarray,
    diagonal: bool,
    STEP_SIZES: np.ndarray,
    L_MIN: int,
    V_WIDTH: int,
    mask: np.ndarray,
    row_start: int,
    col_start: int,
) -> tuple[
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    found_paths = pf.find_warping_pathsV1(csm, STEP_SIZES, L_MIN, V_WIDTH, mask)

    di_paths = typed.List.empty_list(tuple_type)
    ut_paths = typed.List.empty_list(tuple_type)
    lt_paths = typed.List.empty_list(tuple_type)
    
    if diagonal and di_sm.shape[0] > 0:
        # hardcode the diagonal path because we already masked it
        diagonal_indices = np.arange(len(di_sm))
        rows = diagonal_indices
        cols = diagonal_indices
        global_rows = rows + row_start
        global_cols = cols + col_start
        sims = np.empty(len(rows))
        for i in range(len(rows)):
            sims[i] = di_sm[rows[i], cols[i]]
        di_paths.append(
            (
                global_rows.astype(np.int32),
                global_cols.astype(np.int32),
                sims.astype(np.float32)
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
        if diagonal and di_sm.shape[0] > 0:
            # similarities use the local sm and indices
            sims = np.empty(len(rows))
            for i in range(len(rows)):
                sims[i] = di_sm[rows[i], cols[i]]
            di_paths.append(
                # ensure paths are serializable for multiprocessing!
                (
                    global_rows.astype(np.int32),
                    global_cols.astype(np.int32),
                    sims.astype(np.float32),
                )
            )
        # if non-diagonal comparison paths are upper triangular
        elif ut_sm.shape[0] > 0:
            sims = np.empty(len(rows))
            for i in range(len(rows)):
                sims[i] = ut_sm[rows[i], cols[i]]
            ut_paths.append(
                # ensure paths are serializable for multiprocessing!
                (
                    global_rows.astype(np.int32),
                    global_cols.astype(np.int32),
                    sims.astype(np.float32),
                )
            )

        # if non-diagonal comparison paths can be mirrored to be lower triangular
        # comparison (0,1) is mirrored to (1,0) in the global comparison matrix
        if not diagonal and lt_sm.shape[0] > 0:
            mirrored_rows = found_path[:, 1]
            global_mirrored_rows = mirrored_rows + col_start  # also mirror start
            mirrored_cols = found_path[:, 0]
            global_mirrored_cols = mirrored_cols + row_start  # also mirror start
            sims = np.empty(len(mirrored_rows))
            for i in range(len(mirrored_rows)):
                sims[i] = lt_sm[mirrored_rows[i], mirrored_cols[i]]
            lt_paths.append(
                # ensure paths are serializable for multiprocessing
                (
                    global_mirrored_rows.astype(np.int32),
                    global_mirrored_cols.astype(np.int32),
                    sims.astype(np.float32),
                )
            )

    return di_paths, ut_paths, lt_paths


def find_motif_representatives(
    x: int,
    global_offsets: np.ndarray,
    global_column_dict_lists_paths,
    L_MIN: int,
    L_MAX: int,
    OVERLAP: float,
) -> Generator[MotifRepresentative, None, None]:
    return mf.find_motifs_representativesV3(
        x, global_offsets, global_column_dict_lists_paths, L_MIN, L_MAX, OVERLAP
    )



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