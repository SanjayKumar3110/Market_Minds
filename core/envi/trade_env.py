import numpy as np
from utils.strategy import strategy_decision

class TradingEnvironment:
    def __init__(self, price_data, rsi_values, short_ma_values, long_ma_values, momentum_values,
                 window_size=10, initial_cash=100.0, min_trade_size=0.001, cooldown_steps=3,
                 episode=1, ticker="BTC", timeframe="1d"):

        self.price_data = price_data
        self.rsi_values = rsi_values
        self.short_ma_values = short_ma_values
        self.long_ma_values = long_ma_values
        self.momentum_values = momentum_values
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.min_trade_size = min_trade_size
        self.cooldown_steps = cooldown_steps
        self.episode = episode
        self.ticker = ticker
        self.timeframe = timeframe
        self.last_buy_price = 0.0 # Added to track buy price for profit calculation
        self.reset()

    def reset(self):
        self.current_step = self.window_size - 1
        self.cash = self.initial_cash
        self.stock_held = 0.0
        self.asset_value = 0.0
        self.total_value = self.cash
        self.trade_cooldown = 0
        self.last_trade_step = -self.cooldown_steps
        self.last_buy_price = 0.0
        self.trades = [] # Keep a log of trades
        return self._get_state()

    def _get_state(self):
        end = self.current_step + 1
        start = end - self.window_size
        if start < 0:
            start = 0
            end = self.window_size
        window_prices = self.price_data[start:end]
        window_rsi = self.rsi_values[start:end]
        window_short_ma = self.short_ma_values[start:end]
        window_long_ma = self.long_ma_values[start:end]
        window_momentum = self.momentum_values[start:end]
        norm_prices = (window_prices - np.mean(window_prices)) / (np.std(window_prices) + 1e-6)
        cash_ratio = np.full(self.window_size, self.cash / (self.total_value + 1e-6))
        stock_held_ratio = np.full(self.window_size, self.stock_held * self.price_data[end-1] / (self.total_value + 1e-6))
        state_matrix = np.vstack([
            norm_prices,
            window_rsi,
            window_short_ma,
            window_long_ma,
            window_momentum,
            cash_ratio,
            stock_held_ratio
        ]).T
        return state_matrix.astype(np.float32)

    def get_strategy_action(self):
        return strategy_decision(
            self.current_step,
            self.price_data,
            self.rsi_values,
            self.short_ma_values,
            self.long_ma_values,
            self.momentum_values
        )

    def step(self, action):
        done = False
        trade_executed = False
        reward = 0.0
        price = self.price_data[self.current_step]

        last_total_value = self.total_value

        # Enforce cooldown
        if self.current_step - self.last_trade_step < self.cooldown_steps:
            action = 0 # Hold

        if action == 1 and self.cash >= self.min_trade_size * price:
            # Buy
            max_affordable = self.cash / price
            buy_amount = max(self.min_trade_size, max_affordable * 0.25)
            quantity = min(max_affordable, buy_amount)
            if quantity > 0:
                cost = quantity * price
                fee = 0.0001 * cost
                self.cash -= (cost + fee)
                self.stock_held += quantity
                trade_executed = True
                self.last_trade_step = self.current_step
                self.last_buy_price = price # Update last buy price
                reward -= 0.001

        elif action == 2 and self.stock_held >= self.min_trade_size:
            # Sell
            sell_amount = self.stock_held * 0.25
            proceeds = sell_amount * price
            fee = 0.0001 * proceeds
            self.cash += proceeds - fee
            self.stock_held -= sell_amount
            trade_executed = True
            self.last_trade_step = self.current_step

            # Reward based on profit/loss from the trade cycle
            profit_loss = (price - self.last_buy_price) * sell_amount
            reward += profit_loss / (self.initial_cash + 1e-6)

            # Reward/Penalty for trading
            if price > self.last_buy_price:
                reward += 0.05 # Bonus for a profitable trade
            else:
                reward -= 0.05 # Penalty for a losing trade

        self.asset_value = self.stock_held * price
        new_total_value = self.cash + self.asset_value

        # Calculate base reward
        base_reward = (new_total_value - last_total_value) / (last_total_value + 1e-6)
        reward += base_reward * 0.1

        # Penalize holding
        if action == 0 and not trade_executed:
            reward -= 0.001

        # Combine rewards
        reward += base_reward
        self.total_value = new_total_value

        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            done = True
            
        next_state = self._get_state()
        return next_state, reward, done, trade_executed