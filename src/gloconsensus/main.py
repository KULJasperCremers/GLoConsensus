import glob
import logging
import multiprocessing
import os
import time
from itertools import chain, combinations, combinations_with_replacement

import logger
import numpy as np
import path as path_class
import utils
import visualize as vis
from numba import typed
from process import process_comparison

main_logger = logging.getLogger()

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
    # PARALLELIZATION logging setup
    logger.start_listener()
    log_queue = logger.get_log_queue()

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
    patient_ids = [
        'ALS01',
        'ALS02',
        'ALS03',
        'ALS04',
        'ALS05',
    ]
    scenario_ids = ['scenario1']
    time_ids = [
        'time1',
        'time2',
        'time3',
        'time4',
        'time5',
        'time6',
        'time7',
        'time8',
        'time9',
    ]
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

    # map each timeseries to an unique index
    timeseries_index_map = dict(enumerate(timeseries_list))
    # reverse map for each timeseries to an index for easy lookup
    reverse_timeseries_index_map = {
        id(ts): index for index, ts in enumerate(timeseries_list)
    }

    # the amount of comparisons needed to set up the loop to fill the global column lists
    total_comparisons = n * (n + 1) // 2 if INCLUDE_DIAGONAL else n * (n - 1) // 2
    print(
        f'Performing {total_comparisons} comparisons in total to set up the global column lists.\n'
    )
    # timeseries comparisons set up for the loop to fill the global column lists
    #   combinations_with_replacements returns self comparisons, e.g. (ts1, ts1)
    #   combinations includes only distinct comparisons, e.g. (ts1, ts2)
    comparisons = (
        combinations_with_replacement(timeseries_list, 2)
        if INCLUDE_DIAGONAL
        else combinations(timeseries_list, 2)
    )

    comparison_timer_start = time.perf_counter()
    # PARALLELIZATION setup:
    args_list = []
    for comparison_index, (ts1, ts2) in enumerate(comparisons):
        # reverse lookup to get the index for each timeseries
        ts1_index = reverse_timeseries_index_map[id(ts1)]
        ts2_index = reverse_timeseries_index_map[id(ts2)]

        # boolean to determine if the current comparison is a self comparison or not
        diagonal = ts1_index == ts2_index and INCLUDE_DIAGONAL

        args = (
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
        )
        args_list.append(args)

    # set up a dict to hold all the lists of path objects for each global column
    global_column_dict_lists_path = {i: [] for i in range(n)}
    num_processes = multiprocessing.cpu_count()
    try:
        with multiprocessing.Pool(
            processes=num_processes,
            initializer=logger.worker_configurer,
            initargs=(log_queue,),
        ) as pool:
            results = pool.map(process_comparison, args_list)
            for result in results:
                for column_index, paths in result:
                    global_column_dict_lists_path[column_index].append(paths)
    finally:
        logger.stop_listener()

    comparison_timer_end = time.perf_counter()
    comparison_timer = comparison_timer_end - comparison_timer_start

    motif_timer_start = time.perf_counter()
    print(f'Processing {len(global_column_dict_lists_path)} global columns.')
    # find x motif representatives in the global column paths
    x = 100
    motif_representatives = utils.find_motif_representatives(
        x, global_offsets, global_column_dict_lists_path, L_MIN, L_MAX, OVERLAP
    )
    print(f'Found {len(motif_representatives)} motif representatives in total.\n')
    motif_timer_end = time.perf_counter()
    motif_timer = motif_timer_end - motif_timer_start

    print(
        f'Performed {total_comparisons} comparisons in {comparison_timer:.2f} seconds.\nFound {len(motif_representatives)} motifs in {motif_timer:.2f} seconds.'
    )

    for mr in motif_representatives:
        vis.plot_motif_set(
            timeseries_list,
            mr.representative,
            mr.motif_set,
            mr.induced_paths,
            mr.fitness,
        )
