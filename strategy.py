import numpy as np
import pandas as pd


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    return (series - mean) / std.replace(0, np.nan)


def gen_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    z_window = 80
    smooth_window = 10
    trend_span = 120
    trend_slope_window = 20

    entry_threshold = 0.65
    exit_threshold = 0.20
    min_hold_bars = 18

    max_vol_z = 1.8

    df["ret"] = df["Close"].pct_change()
    df["ret_lag"] = df["ret"].shift(1)

    df["ret_z"] = rolling_zscore(df["ret_lag"], window=z_window)

    # Flipped reversion = momentum
    df["signal_raw"] = df["ret_z"]
    df["signal"] = df["signal_raw"].rolling(window=smooth_window).mean()

    # Trend filter
    df["ema_trend"] = df["Close"].ewm(span=trend_span, adjust=False).mean()
    df["trend_slope"] = df["ema_trend"].pct_change(periods=trend_slope_window)

    # Volatility filter
    df["vol"] = df["ret_lag"].rolling(window=60).std()
    df["vol_z"] = rolling_zscore(df["vol"], window=120)

    df["desired_position"] = 0

    long_cond = (
        (df["signal"] > entry_threshold)
        & (df["Close"] > df["ema_trend"])
        & (df["trend_slope"] > 0)
        & (df["vol_z"] < max_vol_z)
    )

    short_cond = (
        (df["signal"] < -entry_threshold)
        & (df["Close"] < df["ema_trend"])
        & (df["trend_slope"] < 0)
        & (df["vol_z"] < max_vol_z)
    )

    df.loc[long_cond, "desired_position"] = 1
    df.loc[short_cond, "desired_position"] = -1

    positions = []
    current_position = 0
    bars_held = 0

    for _, row in df.iterrows():
        signal = row["signal"]
        desired_position = int(row["desired_position"])

        if pd.isna(signal):
            positions.append(current_position)
            continue

        if current_position == 0:
            if desired_position != 0:
                current_position = desired_position
                bars_held = 1
            else:
                bars_held = 0

        else:
            bars_held += 1

            if bars_held >= min_hold_bars:
                if abs(signal) < exit_threshold:
                    current_position = 0
                    bars_held = 0

                elif desired_position != 0 and desired_position != current_position:
                    current_position = desired_position
                    bars_held = 1

        positions.append(current_position)

    df["position"] = positions

    return df
