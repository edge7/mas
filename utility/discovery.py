import logging
import datetime
from dateutil.parser import parse
import pandas as pd
from utility.mql4_socket import write_sup_res

TP = 30
SL = 20

logger = logging.getLogger(__name__)

from market_info.market_info import MarketInfo, LONG_TREND, CROSS, get_pips


def get_info(df, news):
    mi = MarketInfo(df, news)
    mi.search_for_info()
    return mi


def search_for_bullish_and_bearish_candlestick(market_info):
    bearish = {'datetime': str(datetime.datetime.now())}
    bullish = {'datetime': str(datetime.datetime.now())}
    market_info = market_info.candle_info
    # Hammer is bullish
    if market_info.hammer:
        bullish['hammer'] = True
    # Hanging man is bearish
    if market_info.hanging:
        bearish['hanging'] = True

    if market_info.inverted_hammer:
        bullish['inv_hammer'] = True

    if market_info.shooting_start:
        bearish['shooting_star'] = True

    if market_info.bullish_engulfing:
        bullish['bullish_eng'] = True

    if market_info.bearish_engulfing:
        bearish['bearish_eng'] = True

    if market_info.tweezer_tops:
        bullish['tweezer_tops'] = True

    if market_info.tweezer_bottoms:
        bearish['tweezer_bottoms'] = True

    if market_info.morning_star:
        bullish['morning_star'] = True

    if market_info.evening_star:
        bearish['evening_star'] = True

    if market_info.three_white:
        bullish['3_white'] = True

    if market_info.three_black:
        bearish['3_black'] = True

    if market_info.three_ins_up:
        bullish['3_ins_up'] = True

    if market_info.three_ins_down:
        bearish['3_ins_down'] = True

    if market_info.marubozu is not None:
        if market_info.marubozu == 'white':
            bullish['marubozu'] = True
        else:
            bearish['marubozu'] = True

    if market_info.dragonfly:
        bullish['dragonfly'] = True

    if market_info.gravestone:
        bearish['gravestone'] = True

    return bearish, bullish


def search_for_buy(bearish_candles, bullish_candles, market_infos):
    long_trends = [market_info.long_trend_flag for market_info in market_infos]
    medium_trends = [market_info.medium_trend_flag for market_info in market_infos]
    short_trends = [market_info.short_trend_flag for market_info in market_infos]

    long_trend_sum = sum(long_trends)
    medium_trends_sum = sum(medium_trends)
    short_trends_sum = sum(short_trends)

    if long_trend_sum > 0 and medium_trends_sum > 0:
        logger.info("Potential buy given long and medium trend")
        logger.info("Waiting for other signals")
        number_bullish = sum([len(list(bullish_candle.keys())) for bullish_candle in bullish_candles])
        number_bearish = sum([len(list(bearish_candle.keys())) for bearish_candle in bearish_candles])
        if number_bearish < number_bullish:
            return True

    return False


def search_for_sell(bearish_candles, bullish_candles, market_infos):
    long_trends = [market_info.long_trend_flag for market_info in market_infos]
    medium_trends = [market_info.medium_trend_flag for market_info in market_infos]
    short_trends = [market_info.short_trend_flag for market_info in market_infos]

    long_trend_sum = sum(long_trends)
    medium_trends_sum = sum(medium_trends)
    short_trends_sum = sum(short_trends)

    if long_trend_sum < 0 and medium_trends_sum < 0:
        logger.info("Potential sell given long and medium trend")
        logger.info("Waiting for other signals")
        number_bullish = sum([len(list(bullish_candle.keys())) for bullish_candle in bullish_candles])
        number_bearish = sum([len(list(bearish_candle.keys())) for bearish_candle in bearish_candles])
        if number_bearish > number_bullish:
            return True

    return False


def scalping_buy(bearish_candles, bullish_candles, market_infos):
    s = market_infos[-1].low_band_50.tail(1)
    try:
        yes_scalp_up = s.groupby(s).count()[True]
    except KeyError:
        yes_scalp_up = 0
    try:
        no_scalp_up = s.groupby(s).count()[False]
    except KeyError:
        no_scalp_up = 0

    number_bullish = sum([len(list(bullish_candle.keys())) for bullish_candle in bullish_candles])
    number_bearish = sum([len(list(bearish_candle.keys())) for bearish_candle in bearish_candles])

    if yes_scalp_up > no_scalp_up and number_bullish > number_bearish:
        logger.info("Scalpellino suuuu")
        return True
    return False


def scalping_sell(bearish_candles, bullish_candles, market_infos):
    s = market_infos[-1].high_band_50.tail(1)
    try:
        yes_scalp_down = s.groupby(s).count()[True]
    except KeyError:
        yes_scalp_down = 0
    try:
        no_scalp_down = s.groupby(s).count()[False]
    except KeyError:
        no_scalp_down = 0

    number_bullish = sum([len(list(bullish_candle.keys())) for bullish_candle in bullish_candles])
    number_bearish = sum([len(list(bearish_candle.keys())) for bearish_candle in bearish_candles])

    if yes_scalp_down > no_scalp_down and number_bullish < number_bearish:
        logger.info("Scalpellino giùùùù")
        return True
    return False


def scalping_avg_buy(bearish_candles, bullish_candles, market_infos):
    def iterate(l):
        res = False
        for i in range(3, 10):
            res = res or l.iloc[-i]
        return res

    short_trend = market_infos[-1].short_trend

    if short_trend.iloc[-1] and short_trend.iloc[-2] and not iterate(short_trend):
        return True
    return False


def scalping_avg_sell(bearish_candles, bullish_candles, market_infos):
    def iterate(l):
        res = True
        for i in range(3, 10):
            res = res and l.iloc[-i]
        return res

    short_trend = market_infos[-1].short_trend
    if not short_trend.iloc[-1] and not short_trend.iloc[-2] and iterate(short_trend):
        return True
    return False


def regime_switching_up(market_infos):
    mi = market_infos[-1]
    if mi.long_trend.iloc[-1] and mi.long_trend.iloc[-2] and not mi.long_trend.iloc[-3]:
        logger.info("Regime is switching ... going UP")
        return True
    return False


def regime_switching_down(market_infos):
    mi = market_infos[-1]
    if not mi.long_trend.iloc[-1] and mi.long_trend.iloc[-2] and mi.long_trend.iloc[-3]:
        logger.info("Regime is switching ... going Down")
        return True
    return False


def in_pips(param):
    if 'JPY' in CROSS or 'XAU' in CROSS:
        multiply = 100.0
    else:
        multiply = 10000.0
    return param / multiply


def at_least_one(self, attr):
    for i in range(2):
        i = i + 1
        if getattr(self[-i], attr):
            return True
    return False


history_buy = []
history_sell = []


def search_for_buy_sell_comb(market_infos):
    global history_buy
    global history_sell
    if len(market_infos) < 6:
        return "OUT"
    buy = 0
    sell = 0
    if at_least_one(market_infos, 'superata_da_poco_ora_scende_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_scende_long'):
        sell += 1
    if at_least_one(market_infos, 'superata_da_poco_ora_scende_short') and \
            at_least_one(market_infos, 'sotto_ma_avvicinando_long'):
        sell += 1
    if at_least_one(market_infos, 'superata_da_poco_ora_scende_short') and \
            at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_long'):
        sell += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_scende_short') and \
            at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_long'):
        buy += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_scende_short') and \
            at_least_one(market_infos, 'sopra_ma_avvicinando_long'):
        sell += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_sale_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_sale_long'):
        buy += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_sale_short') and \
            at_least_one(market_infos, 'sotto_ma_avvicinando_long'):
        sell += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_sale_short') and \
            at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_long'):
        sell += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_sale_short') and \
            at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_long'):
        buy += 1

    if at_least_one(market_infos, 'superata_da_poco_ora_sale_short') and \
            at_least_one(market_infos, 'sopra_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_ma_avvicinando_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_scende_long'):
        sell += 1

    if at_least_one(market_infos, 'sotto_ma_avvicinando_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_sale_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sotto_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_long'):
        sell += 1

    if at_least_one(market_infos, 'sotto_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sopra_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_short') and \
            at_least_one(market_infos, 'sotto_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_short') and \
            at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_long'):
        sell += 2

    if at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_short') and \
            at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_long'):
        buy += 1

    if at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_short') and \
            at_least_one(market_infos, 'sopra_ma_avvicinando_long'):
        sell += 1

    if at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_scende_long'):
        sell += 1

    if at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_sale_long'):
        buy += 1

    if at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_short') and \
            at_least_one(market_infos, 'sotto_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_short') and \
            at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_long'):
        buy += 2

    if at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_short') and \
            at_least_one(market_infos, 'sopra_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sopra_ma_avvicinando_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_scende_long'):
        sell += 1

    if at_least_one(market_infos, 'sopra_ma_avvicinando_short') and \
            at_least_one(market_infos, 'superata_da_poco_ora_sale_long'):
        buy += 1

    if at_least_one(market_infos, 'sopra_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sotto_ma_avvicinando_long'):
        buy += 1

    if at_least_one(market_infos, 'sopra_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sotto_e_ancora_piu_sotto_long'):
        sell += 1

    if at_least_one(market_infos, 'sopra_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sopra_e_ancora_piu_sopra_long'):
        buy += 1

    if at_least_one(market_infos, 'sopra_ma_avvicinando_short') and \
            at_least_one(market_infos, 'sopra_ma_avvicinando_long'):
        sell += 1

    res = "OUT"
    if len(history_buy) > 3:
        history_buy = history_buy[1:]
    if len(history_sell) > 3:
        history_sell = history_sell[1:]

    history_buy.append(buy)
    history_sell.append(sell)

    if sum(history_sell) > sum(history_buy) and sum(history_sell) > 1:
        res = "SELL"
    if sum(history_buy) > sum(history_sell) and sum(history_buy) > 1:
        res = "BUY"
    return res


def describe_what_is_going_on(market, start, end):
    import copy

    RANGING_THRES = 30  # Improve
    TREND_THRES = 50  # Improve
    if end is not None:
        market = market.iloc[start:end]
    else:
        market = market.iloc[start:]
    max = market.max()
    min = market.min()
    mean = market.mean()
    last = market.iloc[-1]
    first = market.iloc[0]
    range_ = abs(max - min)
    ranging = False
    down = False
    up = False
    if get_pips(range_) < RANGING_THRES:
        ranging = True
    if last - first < 0 and get_pips(abs(last - first)) > TREND_THRES:
        down = True
    if last - first > 0 and get_pips(abs(last - first)) > TREND_THRES:
        up = True

    res = []
    supp = []
    THRE = 85

    def confirm_res(l, base):
        for i in range(1, 6):
            res = get_pips(l[base] - l[base + i])
            if res > THRE:
                return True
            if res < 0:
                return False
        return False

    def confirm_supp(l, base):
        for i in range(1, 5):
            res = get_pips(l[base + i] - l[base])
            if res > THRE:
                return True
            if res < 0:
                return False
        return False

    l = list(market)
    for index, value in enumerate(l):
        if index == 0: continue
        THRE = 4460
        try:
            if l[index - 1] < l[index] and (get_pips(l[index - 1] - l[index + 1]) > THRE or get_pips(
                    l[index - 1] - l[index + 2]) > THRE) or get_pips(
                l[index - 1] - l[index + 4]) > THRE:
                res.append(l[index - 1])
        except Exception:
            pass
        try:
            if l[index - 1] > l[index] and (get_pips(l[index + 1] - l[index - 1]) > THRE or get_pips(
                    l[index + 2] - l[index - 1]) > THRE) or get_pips(l[index + 4] - l[index - 1]) > THRE:
                supp.append(
                    l[index - 1])

        except Exception:
            pass

    orig = copy.deepcopy(supp)
    new_supp = copy.deepcopy(supp)

    notmod = False
    while not notmod:
        notmod = True
        for item in supp:
            for jitem in supp:
                if item != jitem and get_pips(abs(item - jitem)) < 15:
                    if item in new_supp:
                        new_supp.remove(item)
                    if jitem in new_supp:
                        new_supp.remove(jitem)
                    m = round((item + jitem) / 2, 5)
                    if m not in new_supp:
                        notmod = False
                        new_supp.append(m)
        supp = new_supp

    orig = copy.deepcopy(res)
    new_res = copy.deepcopy(res)

    notmod = False
    while not notmod:
        notmod = True
        for item in res:
            for jitem in res:
                if item != jitem and get_pips(abs(item - jitem)) < 15:
                    if item in new_res:
                        new_res.remove(item)
                    if jitem in new_res:
                        new_res.remove(jitem)
                    m = round((item + jitem) / 2, 5)
                    if m not in new_res:
                        notmod = False
                        new_res.append(m)
        res = new_res

    return ranging, up, down, res, supp, list(set(res + supp))


def mas(item, p, mi):
    last_closes = mi.df[CROSS + "CLOSE"].tail(20)
    last_closes_10 = mi.df[CROSS + "CLOSE"].tail(10)
    # Check if last 10 erano vicine
    counter = -1
    sopra = 0
    sotto = 0
    for i in list(last_closes_10):
        if get_pips(abs(i - item)) < 5:
            counter += 1
            if item <= i:
                sopra += 1
            else:
                sotto += 1
    if counter:
        logger.info("NN UNICA")
    else:
        logger.info("UNICA")

    if item > p:
        logger.info("SOTTO")

    if item <= p:
        logger.info("SOPRA")


def analyse(last_45, last_25, before, price, candles, df):
    debug = True
    gu = False
    gd = False
    if df.get_close(1) < df.get_close(6) and df.get_close(1) < df.get_close(3):
        gd = True
    if df.get_close(1) > df.get_close(6) and df.get_close(1) > df.get_close(3):
        gu = True
    xxx = set(last_45[5] + last_25[5] + before[5])
    xxx = list(xxx)
    if debug:
        logger.info("PRINTING RES/SUPP")
        logger.info(";".join(map(str, xxx)))
        write_sup_res(xxx, df.get_close(1))
    bearish_candles = candles[0]
    bullish_candles = candles[1]
    is_ranging_now = last_25[0]
    is_ranging_since_a_while = last_45[0]
    has_ranged = before[0]

    is_going_up_now = last_25[1]
    is_going_up_since_a_while = last_45[1]
    was_going_up = before[1]

    is_going_down_now = last_25[2]
    is_going_down_since_a_while = last_45[2]
    was_going_down = last_45[2]

    def is_close_to_xxx(p, xxx):
        for item in xxx:
            if abs(get_pips(item - p)) < 5:
                logger.info("Close To Key Point")
                mas(item, p, df)
                return item
        return False

    close = is_close_to_xxx(price, xxx)
    if not close:
        return None
    if len(bearish_candles.keys()) > 1 and gu:
        logger.info("SCENDE")
        return "SELL"
    if len(bullish_candles.keys()) > 1 and gd:
        logger.info("SALE")
        return "BUY"


def search_buy_sell_h_l(market_infos):
    last_45 = describe_what_is_going_on(market_infos[-1].df[CROSS + "CLOSE"], -1, None)
    last_25 = describe_what_is_going_on(market_infos[-1].df[CROSS + "CLOSE"], -2, None)
    before = describe_what_is_going_on(market_infos[-1].df[CROSS + "CLOSE"], -1000, None)

    return analyse(last_45, last_25, before, market_infos[-1].df[CROSS + "CLOSE"].iloc[-1],
                   search_for_bullish_and_bearish_candlestick(market_infos[-1]), market_infos[-1])


def get_take_profit(mi):
    avg = mi.df[CROSS + "BODY"].tail(10).abs().mean()
    return get_pips(avg) * 4.5


def get_stop_loss(mi):
    avg = mi.df[CROSS + "BODY"].tail(10).abs().mean()
    return get_pips(avg) * 3.5


def build_ml_for_sell(medium_avg, fast_avg, close):
    df = pd.DataFrame([])
    df['medium_avg'] = medium_avg
    df['fast_avg'] = fast_avg
    df['close'] = close
    print("A")


def search_buy_sell_mas(market_infos):
    mi = market_infos[-1]
    last_price = mi.get_close(1)
    avg = mi.df[CROSS + "CLOSE"].ewm(span=100).mean()
    diff = mi.df[CROSS + "CLOSE"] - avg
    medium_avg = diff.ewm(span=50).mean()
    fast_avg = diff.rolling(window=25).mean()
    # Negative values mean che sta andando verso il basso
    ultimo_fast = fast_avg.iloc[-1]
    penultimo_fast = fast_avg.iloc[-2]
    last_medium = medium_avg.iloc[-1]
    logger.info("PRINTIN AVG")
    logger.info(str(ultimo_fast))
    logger.info(str(last_medium))
    if ultimo_fast * penultimo_fast < 0:  # cambiamento di segno
        if ultimo_fast < 0 and last_medium < 0:
            return "SELL"
        if ultimo_fast > 0 and last_medium > 0:
            return "BUY"


def search_buy_sell_mas2(market_infos):
    mi = market_infos[-1]
    last_price = mi.get_close(1)
    avg_slow = mi.df[CROSS + "CLOSE"].ewm(span=300).mean()
    avg_fast = mi.df[CROSS + "CLOSE"].ewm(span=50).mean()
    dist = abs(last_price - avg_fast.iloc[-1])
    dist = get_pips(dist)
    if avg_fast.iloc[-1] > avg_slow.iloc[-1] and \
            avg_fast.iloc[-2] < avg_slow.iloc[-2]:
        return "BUY"

    if avg_fast.iloc[-1] < avg_slow.iloc[-1] and \
            avg_fast.iloc[-2] > avg_slow.iloc[-2]:
        return "SELL"


def get_sequence_state(avg_slow, avg_medium, avg_fast, closes, bodies):
    avg_slow = list(avg_slow)[-50:]
    avg_medium = list(avg_medium)[-50:]
    avg_fast = list(avg_fast)[-50:]
    closes = list(closes)[-50:]
    states = []
    from itertools import groupby

    for index, close in enumerate(closes):
        current_slow = avg_slow[index]
        current_medium = avg_medium[index]
        current_fast = avg_fast[index]
        if close > current_slow and close > current_medium and close > current_fast:
            states.append("SOPRATUTTE")

        elif close < current_slow and close < current_medium and close < current_fast:
            states.append("SOTTOTUTTE")

        else:
            states.append("MIDDLE")

    g = [(x[0], len(list(x[1]))) for x in groupby(states)]
    if len(g) > 1:
        last_state = g[-1][0]
        last_state_count = g[-1][1]
        num_su_last = list(bodies)[-last_state_count:]
        num_su_last = sum(1 if x > 0 else 0 for x in num_su_last)

        num_giu_last = list(bodies)[-last_state_count:]
        num_giu_last = sum(1 if x < 0 else 0 for x in num_giu_last)

        pen_state = g[-2][0]
        pen_state_count = g[-2][1]
        if (pen_state == "MIDDLE" or pen_state == "SOTTOTUTTE") and last_state == "SOPRATUTTE" and num_su_last > 20:
            return "BUY"

        if (pen_state == "MIDDLE" or pen_state == "SOPRATUTTE") and last_state == "SOTTOTUTTE" and num_giu_last > 20:
            return "SELL"


def get_sequence_state2(lookbacks, closes, bodies):
    sell = 0
    buy = 0
    current_price = closes.iloc[-1]
    for l in lookbacks:
        min = closes.rolling(window=l).min().iloc[-1]
        max = closes.rolling(window=l).max().iloc[-1]
        if current_price <= min:
            sell += 1
        if current_price >= max:
            buy += 1

    if buy > sell:
        return "BUY", buy - sell
    if sell > buy:
        return "SELL", sell - buy
    return None, None


avg_slow_base = 100
avg_medium_base = 75
avg_medium_fast = 50


def optimize_avg(df):
    from numpy import isnan

    closes = df[CROSS + "CLOSE"]
    bodies = df[CROSS + "BODY"]
    all_res = []
    for i in range(5, 200, 20):
        slow = list(closes.ewm(span=avg_slow_base + i).mean())
        medium = list(closes.ewm(span=avg_medium_base + i).mean())
        fast = list(closes.ewm(span=avg_medium_fast + i).mean())
        diff = closes - closes.shift(-20)
        diff = list(diff)
        diff = [get_pips(x) for x in diff]
        res = 0
        for index, value in enumerate(diff):
            if index < 5: continue
            s = slow[0: index]
            m = medium[0:index]
            f = fast[0:index]
            d = diff[index - 1]
            if index % 20 != 0: continue
            cl = closes.iloc[0:index]
            bo = list(bodies)[0:index]
            if isnan(d): continue
            pos, _ = get_sequence_state2([i], cl, bodies)
            if pos == "BUY" and diff[index] > 0:
                res += diff[index]
            if pos == "BUY" and diff[index] < 0:
                res -= abs(diff[index])

            if pos == "SELL" and diff[index] < 0:
                res += abs(diff[index])
            if pos == "SELL" and diff[index] > 0:
                res -= diff[index]

        all_res.append((i, res))
    all_res_sorted = sorted(all_res, key=lambda tup: tup[1], reverse=True)
    logger.info(str(all_res_sorted))
    logger.info("BEST IS: " + str(all_res_sorted[0][0]))
    return all_res_sorted[0][0], all_res_sorted[0][1]


def optimize_lb(df):
    from numpy import isnan

    closes = df[CROSS + "CLOSE"]
    bodies = df[CROSS + "BODY"]
    all_res = []
    for i in range(200, 1000, 50):

        diff = closes - closes.shift(-2)
        diff = list(diff)
        diff = [get_pips(x) for x in diff]
        res = 0
        last = None
        last_pos = None
        for index, value in enumerate(diff):
            if index < 5: continue
            d = diff[index - 1]
            cl = closes.iloc[0:index]
            if isnan(d): continue
            pos, _ = get_sequence_state2([i], cl, bodies)
            if pos == "BUY":
                if last is None:
                    last = cl.iloc[-1]
                    last_pos = "BUY"
                else:
                    if last_pos == "SELL":
                        res += (last - cl.iloc[-1])
                    else:
                        res -= (last - cl.iloc[-1])
                    last_pos = "BUY"
                    last = cl.iloc[-1]

            if pos == "SELL":
                if last is None:
                    last = cl.iloc[-1]
                    last_pos = "SELL"
                else:
                    if last_pos == "SELL":
                        res += (last - cl.iloc[-1])
                    else:
                        res -= (last - cl.iloc[-1])
                    last_pos = "SELL"
                    last = cl.iloc[-1]

            if last_pos == "SELL":
                cp = cl.iloc[-1]
                if get_pips(last - cp) >= TP:
                    res += in_pips(TP)
                    last_pos = None
                    last = None
                if get_pips(cp - last) >= SL:
                    res -= in_pips(SL)
                    last_pos = None
                    last = None

            if last_pos == "BUY":
                cp = cl.iloc[-1]
                if get_pips(cp - last) >= TP:
                    res += in_pips(TP)
                    last_pos = None
                    last = None
                if get_pips(last - cp) >= SL:
                    res -= in_pips(SL)
                    last_pos = None
                    last = None

        logger.info(str((i, res)))
        all_res.append((i, res))
    all_res_sorted = sorted(all_res, key=lambda tup: tup[1], reverse=True)
    logger.info(str(all_res_sorted))
    # logger.info("BEST IS: " + str(all_res_sorted[0][0]))
    pos = [x[0] for x in all_res_sorted if x[0] >= 0]
    if len(pos) > 0:
        return pos[0:4], True
    return [x[0] for x in all_res_sorted][0], False


cc = 0
xx = []
jr = False



def search_buy_sell_mas3(market_infos):
    global cc
    global xx
    global jr
    mi = market_infos[-1]
    ori = mi.df.copy(deep=True).tail(1290).reset_index(drop=True)
    ans = []
    b = 0
    s = 0
    cc += 1
    if cc == 200 or cc == 1:
        res, jr = optimize_lb(ori)
        if len(res) > 0:
            xx = res
        else:
            xx = []
        cc = 1

    res = xx
    if res is None:
        logger.info("Non apro")
        return

    return get_sequence_state2(res, mi.df[CROSS + "CLOSE"], mi.df[CROSS + "BODY"])


to_open = None


def cazzo_fo(balances):
    return False
    from itertools import groupby
    balances = [x[0] for x in groupby(balances)]
    if len(balances) < 3:
        return False
    if balances[-1] < balances[-2] < balances[-3]:
        return True
    return False


no_trade = 0
last = None

def is_not_asian(time):
    from dateutil.parser import parse
    time = parse(time).time()
    current_hour = time.hour
    if current_hour > 21 or current_hour < 6:
        return False
    return True

def check_asian_orders(market, reverse = False):
    current_price = market.df[CROSS + "CLOSE"].iloc[-1]
    current_big_avg =  market.df[CROSS + "CLOSE"].ewm(span=250).mean().iloc[-1]
    last_body = market.get_close(1)
    if last_body >= 0:
        high = market.df[CROSS + "CLOSE"] + market.df[CROSS + "HIGHINPIPS"]
        low = market.df[CROSS + "OPEN"] - market.df[CROSS + "LOWINPIPS"]
    else:
        high = market.df[CROSS + "OPEN"] + market.df[CROSS + "HIGHINPIPS"]
        low = market.df[CROSS + "CLOSE"] - market.df[CROSS + "LOWINPIPS"]

    high = high.iloc[-1]
    low = low.iloc[-1]
    current_up_avg = market.df[CROSS + "CLOSE"].rolling(window=50).mean() +  market.df[CROSS + "CLOSE"].rolling(window=50).std() * 1.8
    current_down_avg = market.df[CROSS + "CLOSE"].rolling(window=50).mean() -  market.df[CROSS + "CLOSE"].rolling(window=50).std() * 1.8
    current_up_avg = current_up_avg.iloc[-1]
    current_down_avg = current_down_avg.iloc[-1]
    if (low <= current_down_avg) and (current_price > current_big_avg) and (current_price<current_down_avg):
        return "BUY"
    if (high >= current_up_avg) and (current_price < current_big_avg) and (current_price>current_up_avg):
        return "SELL"


def optimize_asian(market):
    import copy
    m = copy.copy(market)
    ori = market.df.copy(deep = True)
    ori["filt"] = ori["TIME"].apply(lambda x: not is_not_asian(x))
    asian = ori[ori["filt"] == True]
    for index in range(1, asian.shape[0]):
        m.df = asian.iloc[0:index]
        res = check_asian_orders(m)
        if res == "SELL":
            last_pos = res
            last = 1


def check_for_ord(orders, bearish_candles, bullish_candles, market_infos, old, balances):
    if not orders.empty:
        try:
            seconds_bars = (
                    parse(market_infos[-1].df["TIME"].iloc[-1]) - parse(
                market_infos[-1].df["TIME"].iloc[-2])).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (
                parse(market_infos[-1].df["TIME"].iloc[-1]) - parse(orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]

        open_at = orders["OPEN_AT"].iloc[-1]
        if bar_ago > 15 and profit > 5 or (bar_ago>11 and profit<0):
            to_close = round(float(lots) / 1, 0)
            if to_close == 0.0:
                to_close = lots

            close = 'CLOSE,' + id + "," + str(to_close)
            return None, None, close, None, None, None, None, None

    if orders.empty:
        current_time = market_infos[-1].df["TIME"].iloc[-1]
        if is_not_asian(current_time):
            return None, None, None, None, None, None, None, None
        #optimize_asian(market_infos[-1])
        res = check_asian_orders(market_infos[-1])

        if res == "BUY":
            buy = True
            last = "BUY"
            sell = False
            lots = 1.0
            tp = market_infos[-1].get_close(1) + in_pips(TP)
            sl = market_infos[-1].get_close(1) - in_pips(SL)
            return buy, sell, None, None, tp, sl, lots, None

        if res == "SELL":
            sell = True
            lots = 1.0
            buy = False
            last = "SELL"
            tp = market_infos[-1].get_close(1) - in_pips(TP)
            sl = market_infos[-1].get_close(1) + in_pips(SL)
            return buy, sell, None, None, tp, sl, lots, None

    return None, None, None, None, None, None, None, None

def check_for_orders(orders, bearish_candles, bullish_candles, market_infos, old, balances):
    global to_open
    global no_trade
    global last
    buy = False
    sell = False
    close = None
    scalp = False
    tp = None
    sl = None

    if cazzo_fo(balances) and no_trade == 0:
        no_trade = 30
        to_open = None
        return None, None, None, None, None, None, None, jr
    if no_trade > 0:
        no_trade -= 1
        return None, None, None, None, None, None, None, jr
    if not orders.empty:
        try:
            seconds_bars = (
                    parse(market_infos[-1].df["TIME"].iloc[-1]) - parse(
                market_infos[-1].df["TIME"].iloc[-2])).total_seconds()

        except Exception:
            seconds_bars = 1
        if seconds_bars == 0: seconds_bars = 1
        seconds_from_orders = (
                parse(market_infos[-1].df["TIME"].iloc[-1]) - parse(orders['TIME'].iloc[-1])).total_seconds()
        bar_ago = int(seconds_from_orders / seconds_bars) + 1
        id = str(orders["ID"].iloc[-1])
        current_price = market_infos[-1].get_close(1)
        lots = str(orders["LOTS"].iloc[-1])
        profit = orders["PROFIT"].iloc[-1]

        open_at = orders["OPEN_AT"].iloc[-1]

        if bar_ago % 5 == 0 and profit == 0:
            to_close = round(float(lots) / 1, 0)
            if to_close == 0.0:
                to_close = lots

            close = 'CLOSE,' + id + "," + str(to_close)
            return None, None, close, None, None, None, None, jr

        if bar_ago % 150 == 0 and profit > 0 and bar_ago >= 10:
            to_close = 1.0
            if to_close == 0.0:
                to_close = lots
            close = 'CLOSE,' + id + "," + str(to_close)
            return None, None, close, None, None, None, None, jr

        if bar_ago % 150 == 0 and profit < 0 and bar_ago >= 30:
            to_close =  1.0
            if to_close == 0.0:
                to_close = lots

            close = 'CLOSE,' + id + "," + str(to_close)
            return None, None, close, None, None, None, None, jr

        if bar_ago < 0 and profit < 0 and abs(get_pips(open_at - current_price)) > 35:
            if open_at > current_price:
                sell = True
            else:
                buy = True

            if buy:
                tp = market_infos[-1].get_close(-1) + in_pips(100)
                sl = market_infos[-1].get_close(-1) - in_pips(80)
            if sell:
                tp = market_infos[-1].get_close(-1) - in_pips(100)
                sl = market_infos[-1].get_close(-1) + in_pips(80)
            elif bar_ago > 200 and orders["PROFIT"].iloc[-1] < 0:
                logger.info("Closing as it is too much time")
                close = 'CLOSE,' + id + "," + lots  # Close
            elif bar_ago > 2200 and orders["PROFIT"].iloc[-1] > 0:
                logger.info("Close all and lock in profit")
                close = 'CLOSE,' + id + "," + lots

    if orders.empty or to_open is not None or True:
        # res = search_buy_sell_h_l(market_infos)
        res, lots = search_buy_sell_mas3(market_infos)

        if lots is not None:
            lots = lots * 3
        # if res == "SELL": res = "BUY"
        # elif res == "BUY": res = "SELL"
        if not orders.empty and res == last:
            return None, None, None, None, None, None, None, jr
        if not orders.empty and res is not None:
            to_open = res
            id = str(orders["ID"].iloc[-1])
            lots = str(orders["LOTS"].iloc[-1])
            close = 'CLOSE,' + id + "," + str(lots)
            return buy, sell, close, scalp, tp, sl, lots, jr

        if to_open is not None:
            res = to_open
            to_open = None

        if res == "SELL":
            sell = True
            buy = False
            last = "SELL"
            tp = market_infos[-1].get_close(1) - in_pips(TP)
            sl = market_infos[-1].get_close(1) + in_pips(SL)

        if res == "BUY":
            buy = True
            last = "BUY"
            sell = False
            tp = market_infos[-1].get_close(1) + in_pips(TP)
            sl = market_infos[-1].get_close(1) - in_pips(SL)

    return buy, sell, close, scalp, tp, sl, lots, jr
