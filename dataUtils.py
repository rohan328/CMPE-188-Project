import finnhub
import pandas as pd
import numpy as np

fc = finnhub.Client(api_key="c23hkuqad3ieeb1ld9jg")

def getCryptoData(ticker, resolution, fr, to):
    json = fc.crypto_candles(ticker, resolution, fr, to)
    df = pd.DataFrame.from_dict(json)
    df["Date"] = pd.to_datetime(df['t'], unit='s')
    df.set_index(["Date"], inplace=True)
    df.drop(columns=['h', 'l', 'o', 's', 'v', 't'], inplace=True)
    df.rename(columns={'c': 'Price'}, inplace=True)
    df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
    return df

def createFeatures(df):
    df['Returns'] = np.log(df['close']/ df['close'].shift())
    window=20
    df['Direction'] = np.where(df['Returns'] > 0, 1, 0)
    df['SMA'] = df['close'].rolling(window).mean()
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Min'] = df['close'].rolling(window).min() / df['close'] - 1
    df['Max'] = df['close'].rolling(window).max() / df['close'] - 1
    df['RSI'] = get_RSI(df['close'])
    SD = df['close'].rolling(window).std()
    df['UpperBB'] = df['SMA'] + (2 * SD)
    df['LowerBB'] = df['SMA'] - (2 * SD)
    df.dropna(inplace=True)
    return df

def createLags(df, lags):
    cols = ['open', 'high', 'low', 'close', 'VolETH']
    features = ['Direction', 'SMA', 'EMA', 'Min', 'Max', 'RSI', 'UpperBB', 'LowerBB']
    for f in features:
        for lag in range(1, lags+1):
            col = "{}_lag_{}".format(f, lag)
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return cols

def get_RSI(prices, n=14):
    
    deltas = (prices-prices.shift(1)).fillna(0)

    avg_of_gains = deltas[1:n+1][deltas > 0].sum() / n
    avg_of_losses = -deltas[1:n+1][deltas < 0].sum() / n

    rsi_series = pd.Series(0.0, deltas.index)

    up = lambda x: x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = n+1
    for d in deltas[n+1:]:
        avg_of_gains = ((avg_of_gains * (n-1)) + up(d)) / n
        avg_of_losses = ((avg_of_losses * (n-1)) + down(d)) / n
        if avg_of_losses != 0:
            rs = avg_of_gains / avg_of_losses
            rsi_series[i] = 100 - (100 / (1 + rs))
        else:
            rsi_series[i] = 100
        i += 1

    return rsi_series