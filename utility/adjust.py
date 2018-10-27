from datetime import datetime
from datetime import timedelta
import pandas as pd

def adjust_data(df, CROSS, candle = 4, adjust_time=0):
    # original = df.copy()
    #df = df.tail(400).reset_index(drop = True)
    to_keep = ["TIME", CROSS + "CLOSE", CROSS + "OPEN", CROSS + "HIGH", CROSS + "LOW" ]
    df =  df[to_keep]
    df["TIME"] = df["TIME"].apply(lambda x: datetime.strptime(x, '%Y.%m.%d %H:%M') -timedelta(hours=adjust_time) )
    return df
    res = df[df["TIME"].apply(lambda x: x.hour) == 1].index[0]
    df_1 = df.iloc[res:].reset_index(drop=True)

    accumulate_time = []
    accumulate_close = []
    accumulate_open = []
    accumulate_high = []
    accumulate_low = []

    emit_time = []
    emit_close = []
    emit_open = []
    emit_high = []
    emit_low = []

    def is_last(h, g):
        for x in g:
            if h == x[-1]:
                return True
        return False

    groups = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20], [21,22,23,0]]
    for index, row in df_1.iterrows():
        hour = row["TIME"].hour

        if is_last(hour, groups):
            if accumulate_close:
                accumulate_time.append(row["TIME"])
                accumulate_low.append(row[CROSS + "LOW"])
                accumulate_high.append(row[CROSS + "HIGH"])
                accumulate_close.append(row[CROSS + "CLOSE"])
                accumulate_open.append(row[CROSS + "OPEN"])

                emit_close.append(accumulate_close[-1])
                emit_high.append(max(accumulate_high))
                emit_low.append(min(accumulate_low))
                emit_time.append(max(accumulate_time))
                emit_open.append(accumulate_open[-1])
                if len(accumulate_low) !=4:
                    pass

                accumulate_time = []
                accumulate_close = []
                accumulate_open = []
                accumulate_high = []
                accumulate_low = []

            else:
                pass
                #accumulate_time.append(row["TIME"])
                #accumulate_low.append(row[CROSS + "LOW"])
                #accumulate_high.append(row[CROSS + "HIGH"])
                #accumulate_close.append(row[CROSS + "CLOSE"])
                #accumulate_open.append(row[CROSS + "OPEN"])


        else:
            accumulate_time.append(row["TIME"])
            accumulate_low.append(row[CROSS + "LOW"])
            accumulate_high.append(row[CROSS + "HIGH"])
            accumulate_close.append(row[CROSS + "CLOSE"])
            accumulate_open.append(row[CROSS + "OPEN"])

    to_return = pd.DataFrame({"TIME": emit_time, CROSS + "OPEN": emit_open, CROSS + "CLOSE": emit_close, CROSS + "HIGH": emit_high, CROSS + "LOW": emit_low})
    return to_return


