from constants.constants import PATH
from market_info.market_info import CROSS, in_pips, get_pips
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
from utility.adjust import adjust_data
from sklearn.ensemble import RandomForestClassifier

from utility.discovery import TP, SL


def create_random_forest(df):
    # Create target
    df_copy = df.copy()
    df_copy['TARGET'] = - df_copy[CROSS + "CLOSE"] + df_copy[CROSS + "CLOSE"].shift(-5)
    df_copy['BODY_IN_PIPS'] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "OPEN"]
    #for i in range(1, 5):
    #    df_copy['BODY_IN_PIPS' + str(i)] = df_copy['BODY_IN_PIPS'].shift(i)
    high = []
    low = []
    # for index, row in df_copy.iterrows():
    #     body = row['BODY_IN_PIPS']
    #     open = row[CROSS + "OPEN"]
    #     close = row[CROSS + "CLOSE"]
    #     low_ = row[CROSS + "LOW"]
    #     high_ = row[CROSS + "HIGH"]
    #     if body >= 0:
    #         high.append(high_ - close)
    #         low.append(open - low_)
    #     else:
    #         high.append(high_ - open)
    #         low.append(close - low_)

    # low_pips = pd.Series(low)
    # high_pips = pd.Series(high)

    # df_copy['LOW_P'] = low_pips
    # df_copy['HIGH_P'] = high_pips

    df_copy['AVG'] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].ewm(span=25).mean()
    # df_copy['AVG_50'] = df_copy[CROSS + "CLOSE"] - df_copy[CROSS + "CLOSE"].ewm(span=10).mean()
    df_copy['SUMAVG'] = df_copy['AVG'].rolling(window=20).sum()
    #df_copy['SUMAVG2'] = df_copy['AVG'].rolling(window=30).sum()
    # for i in range(1,2):
    #    df_copy['AVG' + str(i)] = df_copy['AVG'].shift(i)
    #    df_copy['SUMAVG' + str(i)] = df_copy['SUMAVG'].shift(i)
    # df_copy['AVGS'] = df_copy['AVG'].shift(1)

    # df_copy['AVG_D'] = df_copy['AVG'] - df_copy['AVGS']
    df_copy = df_copy[df_copy['AVG'] * df_copy['AVG'].shift(1) < 0]
    # del df_copy['AVGS']
    df_copy['H'] = df_copy['TIME'].apply(lambda x: x.hour)
    df_copy = df_copy[(df_copy['H'] < 8) & (df_copy['H'] <= 8)]
    del df_copy[CROSS + 'OPEN']
    del df_copy[CROSS + 'CLOSE']
    del df_copy[CROSS + 'HIGH']
    del df_copy[CROSS + 'LOW']
    # del df_copy['']
    del df_copy['H']
    del df_copy['TIME']

    def ap_target(x):
        x = get_pips(x)
        if x >= 10:
            return "BUY"
        if x <= -10:
            return "SELL"
        return "out"

    df_copy['TARGET'] = df_copy['TARGET'].apply(lambda x: ap_target(x))

    df_copy = df_copy.dropna()
    train = df_copy.head(df_copy.shape[0] - 50)
    test = df_copy.tail(50)

    target_train = train['TARGET']
    target_test = test['TARGET']

    del train['TARGET']
    del test['TARGET']

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    clf.fit(train, target_train)
    t = clf.predict(test)
    pd.crosstab(target_test, t, rownames=['Actual Species'], colnames=['Predicted Species'])
    f = f1_score(target_test, t, average='weighted')
    fi = list(zip(train.columns, clf.feature_importances_))
    c = confusion_matrix(target_test, t)
    print(c)
    print("A")


df = pd.read_csv(PATH + 'o.csv', sep=",").reset_index(drop=True)
df = adjust_data(df, CROSS, candle=4)
create_random_forest(df)
