import glob
import logging
import multiprocessing
import os
import time
from itertools import combinations, combinations_with_replacement

from logger import configure_logging
import numpy as np
import utils
import visualize as vis
from process import process_comparison

from joblib import Parallel, delayed


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
    configure_logging()
    logger = logging.getLogger()

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
        #'ALS03',
        #'ALS04',
        #'ALS05',
    ]
    scenario_ids = [
        'scenario1',
        'scenario2',
        #'scenario3',
        #'scenario4',
        #'scenario5',
        #'scenario6',
        #'scenario7',
        #'scenario8',
        #'scenario9',
    ]
    time_ids = [
        'time1',
        'time2',
        #'time3',
        #'time4',
        #'time5',
        #'time6',
        #'time7',
        #'time8',
        #'time9',
    ]
    # all ALS patient data
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
    logger.info(
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

        # all the arguments that need to be passed to each comparison worker
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

    global_column_dict_lists_paths = {column_index: [] for column_index in range(n)}

    num_threads = multiprocessing.cpu_count()
    worker_results = Parallel(n_jobs=num_threads, backend='threading')(
        delayed(process_comparison)(args) for args in args_list
    )

    for result in worker_results:
        for column_index, paths in result:
            global_column_dict_lists_paths[column_index].extend(paths)

    comparison_timer_end = time.perf_counter()
    comparison_timer = comparison_timer_end - comparison_timer_start


    logger.info(f'Processing comparisons took {comparison_timer:.2f} seconds.')

    motif_timer_start = time.perf_counter()
    logger.info(f'Processing {len(global_column_dict_lists_paths)} global columns.')
    # find x motif representatives in the global column paths
    x = 10
    motif_representatives = []
    motif_representatives_gen = utils.find_motif_representatives(
        x, global_offsets, global_column_dict_lists_paths, L_MIN, L_MAX, OVERLAP
    )
    for motif_representative in motif_representatives_gen:
        motif_representatives.append(motif_representative)

    motif_timer_end = time.perf_counter()
    motif_timer = motif_timer_end - motif_timer_start

    logger.info(
        f'Performed {total_comparisons} comparisons in {comparison_timer:.2f} seconds.\nFound {len(motif_representatives)} motifs in {motif_timer:.2f} seconds.'
    )

    motif_representatives_gen.close()

    for mr in motif_representatives:
        vis.plot_motif_set(
            timeseries_list,
            mr.representative,
            mr.motif_set,
            mr.induced_paths,
            mr.fitness,
        )
