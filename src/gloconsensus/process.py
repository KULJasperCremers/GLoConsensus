import logging
from itertools import chain

import candidate_finder as cf
import numpy as np
import path as path_class
import utils
from numba import typed

process_logger = logging.getLogger()


def process_candidate(args) -> tuple[np.int32, tuple[int, int], float]:
    (
        column_index,
        column_smask,
        column_emask,
        mask,
        global_column_dict_list,
        start_offset,
        L_MIN,
        L_MAX,
        OVERLAP,
    ) = args
    process_logger.info(f'Processing column {column_index + 1}.')

    global_column_paths = typed.List.empty_list(
        path_class.Path.class_type.instance_type  # type: ignore
    )
    for path_tuple in list(chain.from_iterable(global_column_dict_list)):
        path = path_class.Path(path_tuple[0], path_tuple[1])
        global_column_paths.append(path)

    # TODO: make utils call here
    candidate, fitness = cf.find_candidatesV3(
        column_smask,
        column_emask,
        mask,
        global_column_paths,
        start_offset,
        L_MIN,
        L_MAX,
        OVERLAP,
    )

    return column_index, candidate, fitness


def process_comparison(args) -> list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]:
    (
        comparison_index,
        ts1,
        ts2,
        diagonal,
        offsets_indices,
        global_offsets,
        GAMMA,
        STEP_SIZES,
        L_MIN,
        V_WIDTH,
    ) = args

    process_logger.info(f'Processing comparison {comparison_index + 1}.')

    # similarity matrix calculations:
    #   diagonal sm: ts1 and ts2 are equal
    #   upper triangular sm: ts1 is the row perspective
    #   lower triangular sm: ts2 is the row perspective
    #
    # if diagonal:
    #   diagional sm is calculated and upper/lower triangular == None
    # else:
    #   upper/lower triangular is calculated and diagonal == None
    di_similarity_matrix, ut_similarity_matrix, lt_similarity_matrix = (
        utils.calculate_similarity_matrices(ts1, ts2, diagonal, GAMMA)
    )

    # line 51 for offsets_indices explanation
    ts1_offsets, ts2_offsets = offsets_indices[comparison_index]

    # offsets_indices[i] returns a tuple, e.g. [(0, 1), (1, 2)]
    # mapped to the global_offets:
    #   - the first tuple element returns the start index of the global rows for ts1
    #   - the second tuple element returns the end index of the global rows for ts1
    #   - the first tuple element returns the start index of the global columns for ts2
    #   - the second tuple element returns the end index of the global columns for ts2
    row_start, _ = (
        global_offsets[ts1_offsets[0]],
        global_offsets[ts1_offsets[1]],
    )
    col_start, _ = (
        global_offsets[ts2_offsets[0]],
        global_offsets[ts2_offsets[1]],
    )

    if diagonal:
        cumulative_similarity_matrix = utils.calculate_cumulative_similarity_matrices(
            di_similarity_matrix, diagonal, STEP_SIZES
        )
    else:
        cumulative_similarity_matrix = utils.calculate_cumulative_similarity_matrices(
            ut_similarity_matrix, diagonal, STEP_SIZES
        )

    # create a mask for fnding the local warping paths
    mask = np.full(cumulative_similarity_matrix.shape, False)
    # add the diagonal + v_width to the mask for self comparisons
    if diagonal:
        for i in range(len(mask)):
            mask[i, max(0, i - V_WIDTH) : i + V_WIDTH + 1] = True

    sm_tuple = (di_similarity_matrix, ut_similarity_matrix, lt_similarity_matrix)
    global_start_index_tuple = (row_start, col_start)
    # depending on the type of comparison either return:
    #   diagonal comparison: diagonal paths
    #   non-diagonal comparison:
    #       upper triangular paths of the comparison itself, e.g. global index (0,1)
    #       lower triangular paths are mirrored paths of the mirrored comparison e.g. global index (1,0)
    di_paths, ut_paths, lt_paths = utils.find_local_warping_paths(
        sm_tuple,
        cumulative_similarity_matrix,
        diagonal,
        STEP_SIZES,
        L_MIN,
        V_WIDTH,
        mask,
        global_start_index_tuple,
    )

    result: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]] = []
    if diagonal and di_paths is not None:
        # (r_start, r_end), (c_start, c_end)
        # (i, i+1)        , (j, j+1)
        # 0: (0,1) (0,1) //                //
        #                // 3: (1,2) (1,2) //
        #                                  // 5: (2,3) (2,3)
        # append to the global column of ts1_offets[0] (=current) for diagonal paths
        result.append((ts1_offsets[0], di_paths))
    elif ut_paths is not None and lt_paths is not None:
        # (r_start, r_end), (c_start, c_end)
        # (i, i+1)        , (j, j+1)
        #                // 1: (0,1) (1,2) // 2: (0,1) (2,3)
        #                //                // 4: (1,2) (2,3)
        #                                  //
        # append to the global column of ts2_offsets[0] (=current) for upper triangular paths
        result.append((ts2_offsets[0], ut_paths))
        # append to the global column of ts1_offsets[0] (=previous) for lower triangular paths
        result.append((ts1_offsets[0], lt_paths))

    return result