import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch

from datetime import datetime
from core.envi.trade_env import TradingEnvironment
from core.RL_agent import StrategyRLAgent
from utils.strategy import calculate_rsi, calculate_ma_crossover, calculate_momentum

# === Configuration ===
TICKER = ["BTC-USD"]
TIMEFRAMES = ["1d"]
TIMEFRAME_PERIOD_LIMITS = {
    "15m": "60d", "30m": "60d", "1h": "60d",
    "1d": None, "1wk": None, "1mo": None
}
START_DATE = "2020-01-01"
END_DATE = "2025-08-01"

# Updated hyperparameters for CNN-LSTM
SEQUENCE_LENGTH = 10
INPUT_DIM = 7  # For prices, rsi, short_ma, long_ma, momentum
HIDDEN_DIM = 64
WINDOW_SIZE = 10
ACTION_SIZE = 3 # Hold, Buy, Sell

EPISODES = 25
MIN_REPLAY_SIZE = 64
STRATEGY_USE_PROB = 0.1

TIMESTAMP = datetime.now().strftime("%m%d_%H%M")

# === Initialize Agent ===
agent = StrategyRLAgent(
    sequence_length=SEQUENCE_LENGTH,
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    action_size=ACTION_SIZE,
    learning_rate=0.0001,
    gamma=0.95,
    epsilon=0.5,
    epsilon_decay=0.995,
    epsilon_min=0.05
)

# === Train on multiple timeframes ===
for ticker in TICKER:
    for timeframe in TIMEFRAMES:
        print(f"\nTraining on {ticker} - Timeframe: {timeframe}")

        period = TIMEFRAME_PERIOD_LIMITS.get(timeframe)

        try:
            if period:
                data = yf.download(ticker, interval=timeframe, period=period, progress=False)
            else:
                data = yf.download(ticker, interval=timeframe, start=START_DATE, end=END_DATE, progress=False)

            # Ensure 'Close' exists and is a Series
            close_data = data.get("Close")

            if close_data is None:
                print(f"Skipping {ticker} [{timeframe}]: 'Close' column missing.")
                continue

            # If Close is a DataFrame (can happen with some download edge cases), convert it
            if isinstance(close_data, pd.DataFrame):
                if ticker in close_data.columns:
                    close_data = close_data[ticker]
                else:
                    print(f"Skipping {ticker} [{timeframe}]: 'Close' data not found for ticker.")
                    continue

            if close_data.isna().all():
                print(f"Skipping {ticker} [{timeframe}]: 'Close' data is all NaN.")
                continue

            data.dropna(inplace=True)
            prices = close_data.dropna().values.flatten()

            # --- Calculate all technical indicators ---
            rsi_values = calculate_rsi(prices)
            short_ma_values, long_ma_values = calculate_ma_crossover(prices)
            momentum_values = calculate_momentum(prices)
            
            # --- Align all data for a single input tensor ---
            indicators_df = pd.DataFrame({
                'price': prices,
                'rsi': rsi_values,
                'short_ma': short_ma_values,
                'long_ma': long_ma_values,
                'momentum': momentum_values
            })
            indicators_df.dropna(inplace=True)
            
            # The input dimension for the agent is the number of features
            INPUT_DIM = indicators_df.shape[1]

            # The sequence length for the agent is defined by the WINDOW_SIZE
            SEQUENCE_LENGTH = WINDOW_SIZE

            # Convert the aligned data into a numpy array
            aligned_data = indicators_df.values

            if len(aligned_data) < SEQUENCE_LENGTH + 1:
                print(f"Skipping {timeframe}: Not enough aligned data for environment ({len(aligned_data)} available).")
                continue

            # --- Initialize Environment with ALIGNED data ---
            env = TradingEnvironment(
                price_data=aligned_data[:, 0],
                rsi_values=aligned_data[:, 1],
                short_ma_values=aligned_data[:, 2],
                long_ma_values=aligned_data[:, 3],
                momentum_values=aligned_data[:, 4],
                window_size=WINDOW_SIZE,
                initial_cash=100.01,
                episode=25,
                ticker=ticker,
                timeframe=timeframe
            )

            # --- Training Loop ---
            for episode in range(EPISODES):
                state = env.reset() 
                total_reward = 0
                steps = 0
                episode_trades = []

                while True:
                    action = agent.act(state)

                    next_state, reward, done, trade_executed = env.step(action)
                    
                    episode_trades.append({
                        "step": env.current_step,
                        "action": action,
                        "price": round(env.price_data[env.current_step], 3),
                        "cash": round(env.cash,3),
                        "stock": round(env.stock_held,3),
                        "reward": round(reward,5),
                        "asset_value":round(env.asset_value, 3),
                    })

                    if trade_executed:
                        agent.store_experience(state, action, reward, next_state, done)
                        
                    state = next_state
                    total_reward += reward
                    steps += 1

                    if len(agent.memory) >= MIN_REPLAY_SIZE:
                        loss = agent.train()
                    else:
                        loss = None

                    if done:
                        # === Final forced sell if stock remains
                        if env.stock_held >= env.min_trade_size:
                            final_price = env.price_data[-1]
                            proceeds = env.stock_held * final_price
                            fee = 0.0001 * proceeds
                            env.cash += proceeds - fee
                            print(f"[FORCED FINAL SELL] Step {env.current_step} | Price: {final_price:.2f} | "
                                  f"Proceeds: {proceeds:.4f} | Fee: {fee:.4f} | Final Cash: {env.cash:.2f}")
                            env.stock_held = 0.0
                            env.asset_value = 0.0

                        print(
                            f"[{timeframe}] Ep {episode+1}/{EPISODES} - "
                            f"Reward: {total_reward:.2f} Steps: {steps} Eps: {agent.epsilon:.3f} "
                            + (f", Loss: {loss:.3f}" if loss is not None else "")
                        )
                        print(f"Last(Cash: ${env.cash:.2f}, Stock: {env.stock_held:.4f}, Asset: ${env.asset_value:.2f})")
                        print("--------------")

                        # Log trades
                        if episode_trades:
                            df = pd.DataFrame(episode_trades)
                            df["episode"] = episode + 1
                            df["ticker"] = ticker
                            df["timeframe"] = timeframe
                            df.to_csv(f"logs/train_logs/trade_log_{TIMESTAMP}.csv", mode="a", index=False,
                                     header=not os.path.exists(f"logs/train_logs/train_{TIMESTAMP}.csv"))

                        break
            print(f"Finished training on {ticker}_{timeframe}. Final epsilon: {agent.epsilon:.4f}")

        except Exception as e:
            if 'prices' in locals() and prices.shape == (0,):
                print(f"Error training on {TICKER} [{timeframe}]: No valid price data after dropna. {e}")
            else:
                print(f"Error training on {TICKER} [{timeframe}]: {e}")

# === Save model for reuse ===
os.makedirs("agents/pt_model", exist_ok=True)
model_name = f"agents/pt_model/pro_agent_{TIMESTAMP}.pt"


# Save raw PyTorch weights
torch.save(agent.model.state_dict(), model_name)
print(f" PyTorch model saved: {model_name}")
