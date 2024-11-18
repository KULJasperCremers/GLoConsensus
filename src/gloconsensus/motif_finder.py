import multiprocessing
import threading
from typing import Generator

import numpy as np
import path as path_class
import path_finder as pf
from joblib import Parallel, delayed
from motif_representative import MotifRepresentative
from numba import typed
from process import process_candidate


def find_motifs_representativesV3(
    max_amount: int,
    global_offsets: np.ndarray,
    global_column_dict_lists_paths,
    L_MIN: int,
    L_MAX: int,
    OVERLAP: float,
) -> Generator[
    MotifRepresentative,
    None,
    None,
]:
    """Generate motifs by finding and masking the best candidates based on fitness scores."""
    n = global_offsets[-1]
    start_mask = np.full(n, True, dtype=np.bool)
    end_mask = np.full(n, True, dtype=np.bool)
    mask = np.full(n, False, dtype=np.bool)

    amount = 0
    num_threads = multiprocessing.cpu_count()

    # thread-safe acces to mask
    mask_lock = threading.Lock()

    while max_amount is None or amount < max_amount:
        best_fitness = 0.0
        best_candidate = None
        best_column_index = None

        start_mask &= ~mask
        end_mask &= ~mask

        if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
            break

        args_list = []
        for column_index in global_column_dict_lists_paths.keys():
            global_column_lists_paths = global_column_dict_lists_paths[column_index]

            start_offset = global_offsets[column_index]
            end_offset = global_offsets[column_index + 1]

            # map global mask to local column mask for candidate finding
            column_smask = start_mask[start_offset:end_offset]
            column_emask = end_mask[start_offset:end_offset]

            # all the arguments that need to be passed to the candidate worker
            args = (
                column_index,
                column_smask,
                column_emask,
                mask,
                global_column_lists_paths,
                np.int32(start_offset),
                np.int32(L_MIN),
                np.int32(L_MAX),
                np.float32(OVERLAP),
            )
            args_list.append(args)

        results = Parallel(n_jobs=num_threads, backend='threading')(
            delayed(process_candidate)(args) for args in args_list
        )

        for column_index, candidate, fitness in results:
            if candidate is not None and fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate
                best_column_index = column_index

        if best_fitness == 0.0 or best_candidate is None:
            break

        (start_index, end_index) = best_candidate
        global_column_paths = typed.List.empty_list(
            path_class.Path.class_type.instance_type  # type: ignore
        )
        for path_tuple in global_column_dict_lists_paths[best_column_index]:
            path = path_class.Path(
                np.stack((path_tuple[0], path_tuple[1]), axis=1), path_tuple[2]
            )
            global_column_paths.append(path)

        with mask_lock:
            induced_paths = pf.find_induced_pathsV0(
                start_index,
                end_index,
                global_column_paths,
                mask,
            )
            motif_set = [(path[0][0], path[-1][0] + 1) for path in induced_paths]

            for motif_start, motif_end in motif_set:
                motif_length = motif_end - motif_start
                overlap = int(OVERLAP * motif_length)
                mask_start = motif_start + overlap - 1
                mask_end = motif_end - overlap
                mask[mask_start:mask_end] = True

        amount += 1
        yield MotifRepresentative(
            representative=(start_index, end_index),
            motif_set=motif_set,
            induced_paths=induced_paths,
            fitness=best_fitness,
        )
