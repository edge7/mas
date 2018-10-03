import os
import logging
import pandas as pd
from constants.constants import FLAG_GO, ORDERS

logger = logging.getLogger(__name__)


def can_i_run(path):
    return os.access(os.path.join(path, 'flag_go'), os.F_OK)


def write_response(path, write):
    with open(os.path.join(path, 'action'), 'w') as f:
        f.write(write)

    with open(os.path.join(path, 'strategy_done'), 'w') as f:
        f.write(' ')


def end_loop(PATH, response):
    os.remove(PATH + FLAG_GO)
    try:
        os.remove((PATH + ORDERS))
    except Exception:
        pass
    write_response(PATH, response)

def get_orders(PATH):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(os.path.join(PATH, ORDERS))
    except FileNotFoundError:
        logger.warning("File is not found, ignore this error if this is the first run ")
    finally: return df