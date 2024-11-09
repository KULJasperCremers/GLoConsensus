import logging
import sys

LOG_LEVEL = logging.INFO
BASE_LOGGER = logging.getLogger('rfp-logger')
BASE_LOGGER.setLevel(LOG_LEVEL)
log_format = '%(asctime)s | %(levelname)8s | %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(log_format, date_format)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(formatter)
BASE_LOGGER.addHandler(stdout_handler)
