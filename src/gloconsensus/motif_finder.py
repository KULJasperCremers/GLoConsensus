import multiprocessing
from itertools import chain
from multiprocessing import shared_memory
from typing import Generator

import logger
import numpy as np
import path as path_class
import path_finder as pf
from motif_representative import MotifRepresentative
from numba import typed
from process import process_candidate


def find_motifs_representativesV3(
    max_amount: int,
    global_offsets: np.ndarray,
    # global_column_dict_path: dict[
    # int, types.ListType(path_class.Path.class_type.instance_type)  # type: ignore
    # ],
    global_column_dict_lists_path,
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
    logger.start_listener()
    log_queue = logger.get_log_queue()

    n = global_offsets[-1]
    start_mask = np.full(n, True, dtype=np.bool)
    end_mask = np.full(n, True, dtype=np.bool)
    mask = np.full(n, False, dtype=np.bool)
    mask_shared_memory_block = shared_memory.SharedMemory(create=True, size=mask.nbytes)
    mask_name = mask_shared_memory_block.name
    mask_shape = mask.shape
    mask_dtype = mask.dtype
    shared_mask = np.ndarray(
        mask_shape, dtype=mask_dtype, buffer=mask_shared_memory_block.buf
    )
    shared_mask[:] = mask[:]

    amount = 0

    args_list = []
    for (
        column_index,
        global_column_dict_list,
    ) in global_column_dict_lists_path.items():
        start_offset = global_offsets[column_index]
        end_offset = global_offsets[column_index + 1]

        # map global mask to local column mask for candidate finding
        column_smask = start_mask[start_offset:end_offset]
        column_emask = end_mask[start_offset:end_offset]

        args = (
            column_index,
            column_smask,
            column_emask,
            mask_name,
            mask_shape,
            mask_dtype,
            global_column_dict_list,
            np.int32(start_offset),
            np.int32(L_MIN),
            np.int32(L_MAX),
            np.float32(OVERLAP),
        )
        args_list.append(args)

    num_processes = multiprocessing.cpu_count()
    try:
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=logger.worker_configurer,
            initargs=(log_queue,),
        ) as pool:
            while max_amount is None or amount < max_amount:
                best_fitness = 0.0
                best_candidate = None
                best_column_index = None

                start_mask &= ~shared_mask
                end_mask &= ~shared_mask

                if (
                    np.all(shared_mask)
                    or not np.any(start_mask)
                    or not np.any(end_mask)
                ):
                    break

                results = pool.map(process_candidate, args_list)
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
                for path_tuple in list(
                    chain.from_iterable(
                        global_column_dict_lists_path[best_column_index]
                    )
                ):
                    path = path_class.Path(path_tuple[0], path_tuple[1])
                    global_column_paths.append(path)
                induced_paths = pf.find_induced_pathsV0(
                    start_index,
                    end_index,
                    global_column_paths,
                    shared_mask,
                )
                motif_set = [(path[0][0], path[-1][0] + 1) for path in induced_paths]

                for motif_start, motif_end in motif_set:
                    motif_length = motif_end - motif_start
                    overlap = int(OVERLAP * motif_length)
                    mask_start = motif_start + overlap - 1
                    mask_end = motif_end - overlap
                    shared_mask[mask_start:mask_end] = True

                amount += 1
                yield MotifRepresentative(
                    representative=(start_index, end_index),
                    motif_set=motif_set,
                    induced_paths=induced_paths,
                    fitness=best_fitness,
                )

    finally:
        logger.stop_listener()
        mask_shared_memory_block.close()
        mask_shared_memory_block.unlink()
