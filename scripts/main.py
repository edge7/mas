import pandas as pd
import time
import os
import logging
from os import path
from logging.config import fileConfig

from dateutil.parser import parse

from utility.send_nots import notify

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging_config.ini')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger(__name__)

from utility.mql4_socket import can_i_run, end_loop, get_orders
from constants.constants import OUT, PATH, ACTION
from utility.discovery import get_info, search_for_bullish_and_bearish_candlestick, check_for_orders


def insert_bearish(b, bearish_candles, limit=13):
    if len(bearish_candles) >= limit:
        bearish_candles = bearish_candles[1:]
    bearish_candles.append(b)
    return bearish_candles


def insert_bullish(b, bullish_candles, limit=3):
    if len(bullish_candles) >= limit:
        bullish_candles = bullish_candles[1:]

    bullish_candles.append(b)
    return bullish_candles


def insert_market_info(mi, mis, limit=15):
    if len(mis) >= limit:
        mis = mis[1:]

    mis.append(mi)
    return mis


def run_strategy(send_email = False):
    def get_time(x):
        try:
            x = parse(x)
        except Exception as e:
            x = None
        return x

    #df = pd.read_csv('/home/edge7/Desktop/Documenti/dataEc/macroData.csv', sep=";")
    #df['Time'] = df['Time'].apply(lambda x: get_time(x))
    news = None
    bearish_candles = []
    bullish_candles = []
    market_infos = []
    old = None
    while True:
        response = "OUT"
        # FLAG_GO
        while not can_i_run(PATH):
            time.sleep(0.1)

        try:
            os.remove(PATH + ACTION)
        except Exception:
            pass

        logger.info("Started . . .")
        orders = get_orders(PATH)
        df = pd.read_csv(PATH + 'o.csv', sep=",").tail(2000).reset_index(drop = True)
        if df.shape[0] < 51:
            end_loop(PATH, "OUT")
            continue

        market_info = get_info(df, news)
        market_infos = insert_market_info(market_info, market_infos)
        bear_candle, bull_candle = search_for_bullish_and_bearish_candlestick(market_info)
        bearish_candles = insert_bearish(bear_candle, bearish_candles)
        bullish_candles = insert_bullish(bull_candle, bullish_candles)

        buy, sell, close, scalp, tp, sl = check_for_orders(orders, bearish_candles, bullish_candles, market_infos, old)
        old = "OUT"
        size = "0.2,noscalp,"+str(tp)+","+str(sl)
        if scalp: size = "0.11,scalp,"+str(tp)+","+str(sl)
        if buy and not sell:
            response = "BUY," + size
            old = 'BUY'
        if sell and not buy:
            response = "SELL," + size
            old = 'SELL'
        if close is not None:
            response = close

        if send_email:
            if buy or sell or (close is not None):
                notify(response)
                response = "OUT"

        # Write response and delete FLAG_GO
        end_loop(PATH, response)


if __name__ == '__main__':
    run_strategy(send_email=False)

