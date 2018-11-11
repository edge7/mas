import logging
import os
import time
from datetime import timedelta
from logging.config import fileConfig
from os import path


log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging_config.ini')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger(__name__)
import pandas as pd
from dateutil.parser import parse
from ml.create_model import create_random_forest

from market_info.market_info import CROSS
#from risk_management.low_risk import risk_management
from utility.adjust import adjust_data
from utility.send_nots import notify



from utility.mql4_socket import can_i_run, end_loop, get_orders, get_balance
from constants.constants import PATH, ACTION
from utility.discovery import get_info, check_london_break, check_avg_behav, check_daily_shit, check_random_order, \
    ask_me_what_to_do, mini_trend, doCane, in_pips, find_trends, search_buy_trend, check_if_buy, figa


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


def insert_balance(b, ba):
    if len(ba) >= 15500:
        ba = ba[1:]
    ba.append(b)
    return ba


def run_strategy(send_email=False):
    def pp(t):
        x = None
        try:
            x = parse(t)
        except Exception:
            return "REMOVE"
        return x
    def modify_actual(ac):
        ac = ac.replace(',', '')
        if '%' in ac:
            ac = ac.replace("%", '')
            ac = float(ac)
            ac = ac/ 100.0
        elif 'B' in ac:
            ac = ac.replace('B', '')
            ac = float(ac)
            ac = ac * 1000000
        elif 'K' in ac:
            ac = ac.replace('K', '')
            ac = float(ac) * 1000.0
        elif 'M' in ac:
            ac = ac.replace('M', '')
            ac = float(ac) * 1000000
        elif ac == '':
            ac = 'REMOVE'
        else:
            try:
                ac = float(ac)
            except:
                ac = "REMOVE"
        return ac

    # df = pd.read_csv('/home/edge7/Desktop/Documenti/dataEc/macroData.csv', sep=";")
    # df['Time'] = df['Time'].apply(lambda x: pp(x))
    # df = df[df['Time'] != "REMOVE"]
    # news = df
    # currency_one = CROSS[:3]
    # currency_second = CROSS[3:6]
    # news = news[(news['Volatility'] == 'High Volatility Expected') | (
    #         news['Volatility'] == 'g2 Volatility Expected')]
    # news = news[(news['Currency'] == currency_one) | (
    #         news['Currency'] == currency_second)]
    # news['Event'] = 'FUND_' + news['Currency'] + news['Event']
    # news = news[['Actual','Event', 'Time']]
    # news['Actual'] = news['Actual'].apply(lambda x: modify_actual(x))
    # news = news[news['Actual'] != 'REMOVE']
    # news['Actual'] = news['Actual'].apply(lambda x:float(x))
    # def mm(t):
    #     t = t + timedelta(days = 1)
    #     t = t.replace(hour = 0, minute = 0)
    #     return t
    # news['Time'] = news['Time'].apply(lambda x: mm(x))
    # def rm(x):
    #     x = x.replace('Jan', '').replace('Feb', '').replace('Mar', '').replace('Apr', '').replace('May', '').replace('Jun', '').replace('Jul', '').replace('Aug', '').replace('Sep','').replace('Oct','').replace('Nov','').replace('Dec','')
    #     x = x.replace('Q1','').replace('Q2','').replace('Q3','').replace('Q4','')
    #     return x
    # news['Event'] = news['Event'].apply(lambda x: rm(x))

    import warnings
    warnings.filterwarnings("ignore")
    # all_events = list(set(news['Event']))
    # dfs = []
    # res = news.copy()
    # minn = res['Time'].min()
    # maxx = res['Time'].max()
    #
    # daterange = [minn + timedelta(days=x) for x in range(0, (maxx - minn).days)]
    # res = res['Time']
    # res = pd.DataFrame(daterange)
    # res['Time']=res[0]
    # del res[0]
    # for event in all_events:
    #     x = news[news['Event'] == event]
    #     x = x.copy()
    #     x[event] = x['Actual']
    #     del x['Event']
    #     del x['Actual']
    #     res = res.merge(x, how='left', left_on='Time', right_on='Time')
    #     res.drop_duplicates(inplace=True)
    #     #dfs.append(res)
    #
    #
    #
    #
    #
    # result = []
    # news = res.fillna(method='ffill')
    bearish_candles = []
    bullish_candles = []
    market_infos = []
    balances = []
    old = None
    ccc = 0
    res = None
    while True:
        response = "OUT"
        # FLAG_GO
        while not can_i_run(PATH):
            time.sleep(0.1)

        try:
            os.remove(PATH + ACTION)
        except Exception:
            pass
        si = True
        while si:
            try:
                balance = get_balance(PATH)
                balance.split(',')[1]
                si = False
            except Exception:
                pass
        balances = insert_balance(balance, balances)
        orders = get_orders(PATH)
        df = pd.read_csv(PATH + 'o.csv', sep=",").tail(500).reset_index(drop=True)
        df = adjust_data(df, CROSS, candle=4)
        if df.shape[0] < 60:
            end_loop(PATH, "OUT")
            continue

        market_info = get_info(df, None)
        market_infos = insert_market_info(market_info, market_infos)
        # bear_candle, bull_candle = search_for_bullish_and_bearish_candlestick(market_info)
        # bearish_candles = insert_bearish(bear_candle, bearish_candles)
        # bullish_candles = insert_bullish(bull_candle, bullish_candles)

        trigger, tp, sl, action, close = figa(market_infos[-1].df, orders)

        response = str(trigger) + "," + str(tp) + "," + str(sl) + "," + str(action)
        if close is not None:
            response = close
        if trigger is not None or close is not None:
            response = response + "," + str(in_pips(50))
            logger.info(response)


        if send_email:
            if buy or sell or (close is not None):
                notify(response)
                # response = "OUT"

        # Write response and delete FLAG_GO
        end_loop(PATH, response)


if __name__ == '__main__':
    run_strategy(send_email=False)
