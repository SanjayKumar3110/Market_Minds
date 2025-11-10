import yfinance as yf
import pandas as pd

from core.envi.test_env import TestEnvironment
from core.RL_agent import StrategyRLAgent
from utils.strategy import calculate_rsi, calculate_ma_crossover, calculate_momentum
from utils.logger import log_ticker, log_plot

# === Configuration ===
SEQUENCE_LENGTH = 10
INPUT_DIM = 7 # Prices, RSI, Short MA, Long MA, Momentum, Cash Ratio
HIDDEN_DIM = 64
ACTION_SIZE = 3
MODEL_PATH_PT = "agents/pt_model/pro_agent_0829_1936.pt" # Update with your saved model's timestamp

START_DATE_TEST = "2025-04-01"
END_DATE_TEST = "2025-07-30"
MAX_STEPS = 1000
TIMEFRAME_PERIOD_LIMITS = {
    "1m": "5d", "2m": "5d", "5m": "60d", "15m": "60d",
    "30m": "60d", "1h": "60d", "90m": "60d",
    "1d": None, "1wk": None, "1mo": None 
}

# === Load PyTorch Model ===
agent = StrategyRLAgent(
    sequence_length=SEQUENCE_LENGTH,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    action_size=ACTION_SIZE
)
agent.load(MODEL_PATH_PT)
agent.epsilon = 0.0

print(" PyTorch model loaded and set to test mode.")

# === Test Loop ===
try:
    while True:
        print("\n--- Testing Console ---")
        ticker = input("Enter the ticker (or type 'xx' to quit): ").strip().upper()
        if ticker.lower() == "xx":
            break

        timeframe = input("Enter the timeframe (e.g. 1m, 5m, 1d): ").strip().lower()
        if timeframe.lower() == "xx":
            break

        period = TIMEFRAME_PERIOD_LIMITS.get(timeframe)

        try:
            print(f"\nFetching {ticker} [{timeframe}] data...")
            if period:
                data = yf.download(ticker, interval=timeframe, period=period, progress=False)
            else:
                data = yf.download(ticker, interval=timeframe, start=START_DATE_TEST, end=END_DATE_TEST, progress=False)

            if 'Close' not in data.columns or data.empty:
                print("No valid 'Close' data.")
                continue

            prices = data['Close'].dropna().values.flatten()
            print(f"Loaded {len(prices)} raw price points.")

            # --- Calculate all technical indicators ---
            rsi = calculate_rsi(prices)
            short_ma, long_ma = calculate_ma_crossover(prices)
            momentum = calculate_momentum(prices)
            
            # --- Align all data into a single DataFrame ---
            indicators_df = pd.DataFrame({
                'price': prices,
                'rsi': rsi,
                'short_ma': short_ma,
                'long_ma': long_ma,
                'momentum': momentum
            })
            indicators_df.dropna(inplace=True)

            aligned_data = indicators_df.values
            
            # Check for sufficient data
            required_length = SEQUENCE_LENGTH + 1
            if len(aligned_data) < required_length:
                print(f"Not enough aligned data. Required: {required_length}, Found: {len(aligned_data)}.")
                continue
            
            # Init Environment
            env = TestEnvironment(
                price_data=aligned_data[:, 0], # Price data is the first column
                rsi_values=aligned_data[:, 1],
                short_ma_values=aligned_data[:, 2],
                long_ma_values=aligned_data[:, 3],
                momentum_values=aligned_data[:, 4],
                window_size=SEQUENCE_LENGTH,
                initial_cash=100.01
            )

            state = env.reset()
            step_count = 0
            log_data = []
            action_map = {0: "Hold", 1: "Buy", 2: "Sell"}

            print("Running test simulation...")

            while not env.done and env.current_step < len(env.price_data) - 1:
                if step_count >= MAX_STEPS:
                    print(" Max steps reached.")
                    break

                # The agent's `act` method handles the trading logic and epsilon
                action = agent.act(state)

                next_state, done = env.step(action)
                state = next_state
                step_count += 1

                current_price = env.price_data[env.current_step - 1] if env.current_step > 0 else env.price_data[0]
                asset_value = env.stock_held * current_price
                portfolio_value = env.cash + asset_value

                log_data.append({
                    "ticker": ticker,
                    "timeframe": timeframe,
                    "step": step_count,
                    "price": round(current_price, 2),
                    "action": action_map[action],
                    "cash": round(env.cash, 2),
                    "stock_held": round(env.stock_held, 4),
                    "asset_value": round(asset_value, 2),
                    "portfolio_value": round(portfolio_value, 2)
                })

                if step_count < 20 or action in [1, 2]:
                    print(f"Step {step_count:>3}: {action_map[action]} @ {current_price:.2f} | Held: {env.stock_held:.4f} | Cash: ${env.cash:.2f}")

                if done:
                    break

            final_value = env.cash + env.stock_held * env.price_data[-1]
            print(f"\nFinal Portfolio Value: ${final_value:.2f}")
            print(f"Net P/L: ${final_value - env.initial_cash:.2f}")

            # === Save log and plot ===
            csv_path = log_ticker(log_data, ticker)
            log_plot(csv_path, timeframe)

        except Exception as e:
            print(f" Error during simulation: {e}")

except KeyboardInterrupt:
    print("\nExiting.")

print("All done.")