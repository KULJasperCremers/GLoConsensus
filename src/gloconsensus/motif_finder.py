from typing import Generator

import candidate_finder as cf
import numpy as np
import path as path_class
import path_finder as pf
from motif_representative import MotifRepresentative
from numba import types


def find_motifs_representativesV3(
    max_amount: int,
    global_offsets: np.ndarray,
    global_column_dict_path: dict[
        int, types.ListType(path_class.Path.class_type.instance_type)  # type: ignore
    ],
    L_MIN: int,
    L_MAX: int,
    OVERLAP: float,
) -> Generator[
    MotifRepresentative,
    None,
    None,
]:
    """Generate motifs by finding and masking the best candidates based on fitness scores."""
    # initialize global masks to ensure proper masking for consensus motifs
    n = global_offsets[-1]
    start_mask = np.full(n, True, dtype=np.bool)
    end_mask = np.full(n, True, dtype=np.bool)
    mask = np.full(n, False, dtype=np.bool)

    amount = 0
    while max_amount is None or amount < max_amount:
        best_fitness = 0.0
        best_candidate = None
        best_column_index = None
        start_mask &= ~mask
        end_mask &= ~mask

        if np.all(mask) or not np.any(start_mask) or not np.any(end_mask):
            break

        # loop over each global column:
        #   because the ut part of the global matrix is a mirror of the lt part
        for column_index, global_column in global_column_dict_path.items():
            global_column = global_column_dict_path[column_index]

            start_offset = global_offsets[column_index]
            end_offset = global_offsets[column_index + 1]

            # map global mask to local column mask for candidate finding
            column_smask = start_mask[start_offset:end_offset]
            column_emask = end_mask[start_offset:end_offset]

            candidate, fitness = cf.find_candidatesV3(
                column_smask,
                column_emask,
                mask,
                global_column,
                start_offset,
                np.int32(L_MIN),
                np.int32(L_MAX),
                np.float32(OVERLAP),
            )

            if candidate is not None:
                candidate = (candidate[0], candidate[1])

            if fitness > best_fitness:
                best_fitness = fitness
                best_candidate = candidate
                best_column_index = column_index

        if best_fitness == 0.0:
            break

        (start_index, end_index) = best_candidate
        induced_paths = pf.find_induced_pathsV0(
            start_index,
            end_index,
            global_column_dict_path[best_column_index],
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
