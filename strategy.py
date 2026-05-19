import numpy as np

import pandas as pd

def gen_position(df: pd.DataFrame) -> pd.DataFrame:

    """

    Simple MA Cross strategy.

    Logic:

    - Calculate short moving average and long moving average from Close price.

    - Buy signal = 1 when short MA is above long MA.

    - Sell signal = -1 when short MA is below long MA.

    - Do nothing = 0 when there is not enough data.

    """

    df = df.copy()

    # Use the correct column name from QuantVN data: "Close"

    df["ma_short"] = df["Close"].rolling(window=10).mean()

    df["ma_long"] = df["Close"].rolling(window=30).mean()

    df["signal"] = 0

    df.loc[df["ma_short"] > df["ma_long"], "signal"] = 1

    df.loc[df["ma_short"] < df["ma_long"], "signal"] = -1

    df["position"] = df["signal"]

    return df