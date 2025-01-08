
import pandas as pd

def calculate_bollinger_bands(df, window=20):
    df['MA'] = df['close'].rolling(window).mean()
    df['STD'] = df['close'].rolling(window).std()
    df['Upper_Band'] = df['MA'] + (df['STD'] * 2)
    df['Lower_Band'] = df['MA'] - (df['STD'] * 2)
    return df

def generate_signal(df):
    df = calculate_bollinger_bands(df)
    df['Signal'] = 0
    df.loc[df['close'] < df['Lower_Band'], 'Signal'] = 1  # Buy signal
    df.loc[df['close'] > df['Upper_Band'], 'Signal'] = -1  # Sell signal
    return df
