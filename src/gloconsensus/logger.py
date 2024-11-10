import logging
import multiprocessing
import sys
from logging.handlers import QueueHandler

LOG_LEVEL = logging.INFO
log_queue = None
listener = None


def listener_configurer():
    root = logging.getLogger()
    log_format = '%(asctime)s | %(processName)s | %(levelname)8s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(LOG_LEVEL)


def listener_process(queue):
    listener_configurer()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger()
            logger.handle(record)
        except Exception:
            import traceback

            print('Error in logging listener:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker_configurer(queue):
    root = logging.getLogger()
    queue_handler = QueueHandler(queue)
    root.addHandler(queue_handler)
    root.setLevel(LOG_LEVEL)


def start_listener():
    global log_queue, listener
    log_queue = multiprocessing.Queue()
    listener = multiprocessing.Process(target=listener_process, args=(log_queue,))
    listener.start()


def stop_listener():
    global log_queue, listener
    if log_queue is not None:
        log_queue.put_nowait(None)
        listener.join()
        log_queue = None
        listener = None


def get_log_queue():
    return log_queue
