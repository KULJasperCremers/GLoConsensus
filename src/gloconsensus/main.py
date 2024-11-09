import glob
import logging
import os
from itertools import combinations, combinations_with_replacement

import numpy as np
import utils
from logger import BASE_LOGGER

logger = BASE_LOGGER
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# clean the plot map before running GLoConsensus
for plot in glob.glob('./plots/*'):
    os.remove(plot)

# constants setup (RHO in utils.py)
GAMMA = 1
STEP_SIZES = np.array([[1, 1], [2, 1], [1, 2]], np.int32)
INCLUDE_DIAGONAL = True
# INCLUDE_DIAGONAL = False
L_MIN = 100
L_MAX = 1000
V_WIDTH = L_MIN // 2
OVERLAP = 0.0

if __name__ == '__main__':
    # DATA SETUP:
    #   can chose any number of patients ids:
    #                                           ALS01 - ALS05
    #                                           id1   - id170
    #   can chose any number of scenario ids:
    #                                           scenario1 - scenario9
    #   can chose any number of time ids:
    #                                           for ALS: time1 - time 10
    #                                           for id:  time1

    patient_data = utils.load_patient_data('./data/patient_data_scaled.pkl')
    patient_ids = ['ALS01']
    scenario_ids = ['scenario1']
    time_ids = ['time1', 'time2', 'time3']
    timeseries_list = utils.filter_time_series(
        patient_data, patient_ids, scenario_ids, time_ids
    )

    # GLoConsensus LOGIC
    timeseries_length = [len(ts) for ts in timeseries_list]
    n = len(timeseries_list)

    # offsets_indices example /w INCLUDE_DIAGONAL=False and n=3:
    #   [[(0, 1), (1, 2)], [(0, 1), (2, 3)], [(1, 2), (2, 3)]]
    #       - the row indices for the first comparison are (0,1)
    #       - the column indices for the first comparison are (1,2)
    #       - etc...
    #
    # offsets_indices example /w INCLUDE_DIAGONAL=True and n=3:
    # [[(0, 1), (0, 1)], [(0, 1), (1, 2)], [(0, 1), (2, 3)],
    #  [(1, 2), (1, 2)], [(1, 2), (2, 3)], [(2, 3), (2, 3)]]
    #       - the row indices for the first comparison are (0,1)
    #       - the column indices for the first comparison are (0,1)
    #       - etc...
    offsets_indices = (
        utils.offsets_indexer(n)
        if INCLUDE_DIAGONAL
        else [
            offset_index
            for offset_index in utils.offsets_indexer(n)
            if offset_index[0] < offset_index[1]
        ]
    )

    # creates a np.array /w for each global index the cutoff point
    # e.g. [0, 1735, 2722, 3955] for n=3
    global_offsets = np.cumsum([0] + timeseries_length)
    global_n = max(global_offsets)

    # create a global similarity matrix to hold the similarity values for each individual comparison
    global_similarity_matrix = np.zeros((global_n, global_n), dtype=np.float32)

    # map each timeseries to an unique index
    timeseries_index_map = dict(enumerate(timeseries_list))
    # reverse map for each timeseries to an index for easy lookup
    reverse_timeseries_index_map = {
        id(ts): index for index, ts in enumerate(timeseries_list)
    }

    # set up a list to hold all the paths for each global column
    global_column_lists_paths = {i: [] for i in range(n)}

    # the amount of comparisons needed to set up the loop to fill the global column lists
    total_comparisons = n * (n + 1) // 2 if INCLUDE_DIAGONAL else n * (n - 1) // 2
    logger.info(
        msg=f'Performing {total_comparisons} comparisons in total to set up the global column lists.\n'
    )
    # timeseries comparisons set up for the loop to fill the global column lists
    #   combinations_with_replacements returns self comparisons, e.g. (ts1, ts1)
    #   combinations includes only distinct comparisons, e.g. (ts1, ts2)
    comparisons = (
        combinations_with_replacement(timeseries_list, 2)
        if INCLUDE_DIAGONAL
        else combinations(timeseries_list, 2)
    )

    # loop to set up the global column lists
    for comparison_index, (ts1, ts2) in enumerate(comparisons):
        logger.info(msg=f'Performing comparison {comparison_index + 1}.')
        # reverse lookup to get the index for each timeseries
        ts1_index = reverse_timeseries_index_map[id(ts1)]
        ts2_index = reverse_timeseries_index_map[id(ts2)]

        # boolean to determine if the current comparison is a self comparison or not
        diagonal = ts1_index == ts2_index and INCLUDE_DIAGONAL

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
        row_start, row_end = (
            global_offsets[ts1_offsets[0]],
            global_offsets[ts1_offsets[1]],
        )
        col_start, col_end = (
            global_offsets[ts2_offsets[0]],
            global_offsets[ts2_offsets[1]],
        )

        if diagonal:
            global_similarity_matrix[row_start:row_end, col_start:col_end] = (
                di_similarity_matrix
            )
            cumulative_similarity_matrix = (
                utils.calculate_cumulative_similarity_matrices(
                    di_similarity_matrix, diagonal, STEP_SIZES
                )
            )
        else:
            global_similarity_matrix[row_start:row_end, col_start:col_end] = (
                ut_similarity_matrix
            )
            cumulative_similarity_matrix = (
                utils.calculate_cumulative_similarity_matrices(
                    ut_similarity_matrix, diagonal, STEP_SIZES
                )
            )

        # create a mask for fnding the local warping paths
        mask = np.full(cumulative_similarity_matrix.shape, False)
        # add the diagonal + v_width to the mask for self comparisons
        if diagonal:
            for i in range(len(mask)):
                mask[i, max(0, i - V_WIDTH) : i + V_WIDTH + 1] = True

        sm_tuple = (di_similarity_matrix, ut_similarity_matrix, lt_similarity_matrix)
        global_start_index_tuple = (row_start, col_start)
        paths = utils.find_local_warping_paths(
            sm_tuple,
            cumulative_similarity_matrix,
            diagonal,
            STEP_SIZES,
            L_MIN,
            V_WIDTH,
            mask,
            global_start_index_tuple,
        )

        print(f'sm tuple: {sum(1 for sm in sm_tuple if sm is not None)}')
        logger.info(
            msg=f'Found {len(paths)} paths in total for comparison {comparison_index + 1}.'
        )
