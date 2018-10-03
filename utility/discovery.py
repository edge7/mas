import logging
import datetime
from dateutil.parser import parse

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
    if 'JPY' in CROSS:
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
    range = abs(max - min)
    ranging = False
    down = False
    up = False
    if get_pips(range) < RANGING_THRES:
        ranging = True
    if last - first < 0 and get_pips(abs(last - first)) > TREND_THRES:
        down = True
    if last - first > 0 and get_pips(abs(last - first)) > TREND_THRES:
        up = True

    res = []
    supp = []

    l = list(market)
    for index, value in enumerate(l):
        if index == 0: continue
        THRE = 15
        try:
            if l[index - 1] <= l[index] and (get_pips(l[index - 1] - l[index + 1]) > THRE or get_pips(
                    l[index - 1] - l[index + 2]) > THRE or get_pips(l[index - 1] - l[index + 4]) > THRE):
                res.append(l[index - 1])
        except Exception:
            pass
        try:
            if l[index - 1] >= l[index] and (get_pips(l[index + 1] - l[index - 1]) > THRE or get_pips(
                    l[index + 2] - l[index - 1]) > THRE or get_pips(l[index + 4] - l[index - 1]) > THRE):
                supp.append(l[index - 1])
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


def analyse(last_45, last_25, before, price, candles, df):
    debug = True
    gu = False
    gd = False
    if df.get_close(1) < df.get_close(10) and df.get_close(1) < df.get_close(3):
        gd = True
    if df.get_close(1) > df.get_close(10) and df.get_close(1) > df.get_close(3):
        gu = True
    xxx = set(last_45[5] + last_25[5] + before[5])
    xxx = list(xxx)
    if debug:
        logger.info("PRINTING RES/SUPP")
        logger.info(str(xxx))
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
            if abs(get_pips(item - p)) < 15:
                logger.info("Close To Key Point")
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
    last_45 = describe_what_is_going_on(market_infos[-1].df[CROSS + "CLOSE"], -100, None)
    last_25 = describe_what_is_going_on(market_infos[-1].df[CROSS + "CLOSE"], -10, None)
    before = describe_what_is_going_on(market_infos[-1].df[CROSS + "CLOSE"], -2000, -45)

    return analyse(last_45, last_25, before, market_infos[-1].df[CROSS + "CLOSE"].iloc[-1],
                   search_for_bullish_and_bearish_candlestick(market_infos[-1]), market_infos[-1])


last = None


def check_for_orders(orders, bearish_candles, bullish_candles, market_infos, old):
    buy = False
    sell = False
    close = None
    scalp = False
    tp = None
    sl = None
    # buy = search_for_buy(bearish_candles, bullish_candles, market_infos)
    # sell = search_for_sell(bearish_candles, bullish_candles, market_infos)
    if buy:
        tp = market_infos[-1].get_close(-1) + in_pips(60)
        sl = market_infos[-1].get_close(-1) - in_pips(30)
    if sell:
        tp = market_infos[-1].get_close(-1) - in_pips(60)
        sl = market_infos[-1].get_close(-1) + in_pips(30)

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
        if bar_ago <= 6 and profit < 0 and abs(get_pips(open_at - current_price)) > 25:
            if open_at > current_price:
                sell = True
            else:
                buy = True
            if buy:
                tp = market_infos[-1].get_close(1) + in_pips(60)
                sl = market_infos[-1].get_close(1) - in_pips(30)
            if sell:
                tp = market_infos[-1].get_close(1) - in_pips(60)
                sl = market_infos[-1].get_close(1) + in_pips(30)
        elif bar_ago > 15 and orders["PROFIT"].iloc[-1] < 0:
            logger.info("Closing as it is too much time")
            close = 'CLOSE,' + id + "," + lots
            # Close
        elif bar_ago > 30 and orders["PROFIT"].iloc[-1] > 0:
            logger.info("Close all and lock in profit")
            close = 'CLOSE,' + id + "," + lots

    # if buy is False and sell is False and close is None:
    #     buy = scalping_buy(bearish_candles, bullish_candles, market_infos)
    #     sell = scalping_sell(bearish_candles, bullish_candles, market_infos)
    #     scalp = buy or sell
    #     if buy:
    #         tp = market_infos[-1].get_close(-1) + in_pips(45)
    #         sl = market_infos[-1].get_close(-1) - in_pips(45)
    #     if sell:
    #         tp = market_infos[-1].get_close(-1) - in_pips(45)
    #         sl = market_infos[-1].get_close(-1) + in_pips(45)

    # if buy is False and sell is False and close is None:
    #    buy = regime_switching_up(market_infos)
    #    sell = regime_switching_down(market_infos)

    # if buy is False and sell is False and close is None:
    #   buy = scalping_avg_buy(bearish_candles, bullish_candles, market_infos)
    #   sell = scalping_avg_sell(bearish_candles, bullish_candles, market_infos)
    #  scalp = buy or sell
    # if market_infos[-1].going_up_quickly:
    #     logger.info("going UP QUICKLY")
    #     if not buy:
    #         buy = True
    #         tp = market_infos[-1].get_close(1) + in_pips(100)
    #         sl = market_infos[-1].get_close(1) - in_pips(75)
    #         #scalp = True
    #
    #
    # if market_infos[-1].going_down_quickly:
    #     logger.info("Diving down")
    #     if not sell:
    #         sell = True
    #         tp = market_infos[-1].get_close(1) - in_pips(100)
    #         sl = market_infos[-1].get_close(1) + in_pips(75)
    # scalp = True

    # res = search_for_buy_sell_comb(market_infos)
    # if orders.empty:
    #     if res == "BUY":
    #         buy = True
    #         sell = False
    #         tp = market_infos[-1].get_close(1) + in_pips(200)
    #         sl = market_infos[-1].get_close(1) - in_pips(75)
    #     if res == "SELL":
    #         buy = False
    #         sell = True
    #         tp = market_infos[-1].get_close(1) - in_pips(200)
    #         sl = market_infos[-1].get_close(1) + in_pips(75)

    # buy = buy and market_infos[-1].buy_fund
    # sell = sell and market_infos[-1].sell_fund
    if orders.empty:
        res = search_buy_sell_h_l(market_infos)

        if res == "SELL":
            sell = True
            buy = False
            last = "SELL"
            tp = market_infos[-1].get_close(1) - in_pips(60)
            sl = market_infos[-1].get_close(1) + in_pips(30)

        if res == "BUY":
            buy = True
            last = "BUY"
            sell = False
            tp = market_infos[-1].get_close(1) + in_pips(60)
            sl = market_infos[-1].get_close(1) - in_pips(30)

    return buy, sell, close, scalp, tp, sl
