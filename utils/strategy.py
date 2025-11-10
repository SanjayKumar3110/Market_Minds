import numpy as np

# === Relative Strength Index (RSI) ===
def calculate_rsi(prices, window=14):
    if len(prices) < window + 1:
        return np.zeros_like(prices)

    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / (down + 1e-6)
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)

    for i in range(window, len(prices)):
        delta = deltas[i - 1]
        upval = max(delta, 0)
        downval = -min(delta, 0)
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        rs = up / (down + 1e-6)
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

# === Moving Average Crossover ===
def calculate_ma_crossover(prices, short_window=5, long_window=20):
    if len(prices) < long_window + 1:
        return np.zeros_like(prices), np.zeros_like(prices)

    short_ma = np.convolve(prices, np.ones(short_window) / short_window, mode='same')
    long_ma = np.convolve(prices, np.ones(long_window) / long_window, mode='same')

    # Make sure output lengths match input
    short_ma = short_ma[:len(prices)]
    long_ma = long_ma[:len(prices)]

    return short_ma, long_ma

# === Price Momentum ===
def calculate_momentum(prices, window=5):
    momentum = np.zeros_like(prices)
    if len(prices) < window + 1:
        return momentum

    for i in range(window, len(prices)):
        momentum[i] = prices[i] - prices[i - window]
    return momentum

# === Combined Strategy Decision ===
# 0 = Hold, 1 = Buy, 2 = Sell
def strategy_decision(index, prices, rsi, short_ma, long_ma, momentum):
    if (
        index >= len(prices)
        or index >= len(rsi)
        or index >= len(short_ma)
        or index >= len(long_ma)
        or index >= len(momentum)
    ):
        return 0  # Hold by default

    rsi_val = rsi[index]
    ma_diff = short_ma[index] - long_ma[index]
    mom = momentum[index]

    score = 0
    if rsi_val < 30: score += 1
    if ma_diff > 0: score += 1
    if mom > 0: score += 1

    if score >= 2:
        return 1  # Buy
    elif rsi_val > 70 or ma_diff < 0:
        return 2  # Sell
    else:
        return 0  # Hold
