import numpy as np

from utils.strategy import strategy_decision
from core.reward import RewardCalculator

class TradingEnvironment:
    def __init__(self, price_data, rsi_values, short_ma_values, long_ma_values, momentum_values,
                 window_size=10, initial_cash=10.01, ticker="", timeframe="", episode=0,
                 slippage_pct=0.001, min_trade_size=0.001):
        self.price_data = price_data
        self.rsi = rsi_values
        self.short_ma = short_ma_values
        self.long_ma = long_ma_values
        self.momentum = momentum_values
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.ticker = ticker
        self.timeframe = timeframe
        self.episode = episode

        self.slippage_pct = slippage_pct
        self.min_trade_size = min_trade_size
        self.reward_calc = RewardCalculator(trading_fee=0.01, min_trade_size=min_trade_size)

        self.reset()

    def reset(self):
        self.current_step = self.window_size - 1
        self.cash = self.initial_cash
        self.stock_held = 0.0
        self.asset_value = 0.0
        self.total_value = self.initial_cash
        self.last_buy_price = 0.0
        self.done = False
        self.trade_rewards = []
        self.forced_first_buy_done = False
        self.steps_since_last_action = 0
        self.last_action_step = -10
        return self._get_state()

    def _get_state(self):
        if self.current_step >= len(self.price_data):
            return np.zeros(self.window_size + 5)

        price_window = self.price_data[self.current_step - self.window_size + 1:self.current_step + 1]
        norm_price = (price_window - np.mean(price_window)) / (np.std(price_window) + 1e-7)

        rsi_value = self.rsi[self.current_step] if self.current_step < len(self.rsi) else 50.0
        rsi_value = 50.0 if np.isnan(rsi_value) else rsi_value

        current_price = self.price_data[self.current_step]
        portfolio_value = self.cash + self.stock_held * current_price
        cash_ratio = self.cash / (portfolio_value + 1e-8)
        asset_ratio = (self.stock_held * current_price) / (portfolio_value + 1e-8)

        return np.concatenate([
            norm_price,
            [self.stock_held, rsi_value / 100, cash_ratio, asset_ratio, portfolio_value / (self.initial_cash + 1e-8)]
        ])

    def get_strategy_action(self):
        return strategy_decision(
            self.current_step,
            self.price_data,
            self.rsi,
            self.short_ma,
            self.long_ma,
            self.momentum
        )

    def step(self, action):
        if self.current_step >= len(self.price_data):
            self.done = True
            return self._get_state(), 0.0, self.done, False

        price = self.price_data[self.current_step] * (
            1 + self.slippage_pct if action == 1 else 1 - self.slippage_pct)
        prev_total = self.cash + self.stock_held * price
        reward = 0.0
        trade_executed = False

        # Enforce cooldown to avoid overtrading
        if self.current_step - self.last_action_step < 3:
            action = 0

        # Forced first BUY
        if not self.forced_first_buy_done:
            action = 1
            self.forced_first_buy_done = True

        # Handle invalid actions
        if action == 1 and self.cash <= 0:
            reward = self.reward_calc.apply_cash_penalty(reward)
            action = 0
        elif action == 2 and self.stock_held <= 0:
            reward = self.reward_calc.apply_cash_penalty(reward)
            action = 0

        # Auto-sell half if held too long
        if self.steps_since_last_action >= 5 and self.stock_held > self.min_trade_size:
            shares_to_sell = self.stock_held / 2
            proceeds = shares_to_sell * price
            fee = proceeds * self.reward_calc.trading_fee
            self.stock_held -= shares_to_sell
            self.cash += proceeds - fee
            trade_executed = True
            reward = self.reward_calc.apply_liquidity_penalty(reward)
            self.steps_since_last_action = 0
            self.last_action_step = self.current_step

        # Execute BUY
        elif action == 1 and self.cash >= 1.0:
            shares_to_buy = self.cash / price
            if shares_to_buy >= self.min_trade_size:
                cost = shares_to_buy * price
                fee = self.reward_calc.trading_fee * cost
                self.stock_held += shares_to_buy
                self.asset_value += cost
                self.cash = 0.0
                self.last_buy_price = price
                trade_executed = True
                reward += 0.005
                self.steps_since_last_action = 0
                self.last_action_step = self.current_step

        # Execute SELL
        elif action == 2 and self.stock_held >= self.min_trade_size:
            proceeds = self.stock_held * price
            fee = proceeds * self.reward_calc.trading_fee
            trade_profit = price > self.last_buy_price
            reward += self.reward_calc.compute_trade_reward(price, self.last_buy_price, trade_profit)
            self.cash += proceeds - fee
            self.trade_rewards.append(reward)
            self.stock_held = 0.0
            self.asset_value = 0.0
            trade_executed = True
            self.steps_since_last_action = 0
            self.last_action_step = self.current_step

        # HOLD
        elif action == 0:
            reward = self.reward_calc.apply_hold_penalty(reward)
            self.steps_since_last_action += 1

        self.current_step += 1
        self.done = self.current_step >= len(self.price_data) - 1

        # Final forced SELL
        if self.done and self.stock_held > self.min_trade_size:
            final_price = self.price_data[self.current_step]
            proceeds = self.stock_held * final_price
            fee = proceeds * self.reward_calc.trading_fee
            trade_profit = final_price > self.last_buy_price
            reward += self.reward_calc.compute_trade_reward(final_price, self.last_buy_price, trade_profit)
            self.cash += proceeds - fee
            self.stock_held = 0.0
            self.asset_value = 0.0
            trade_executed = True

        new_total = self.cash + self.stock_held * price
        raw_reward = (new_total - prev_total) / (prev_total + 1e-8)
        reward = self.reward_calc.add_reward_shaping(reward, raw_reward, trade_executed)

        # Avoid tiny floating stock balances
        if self.stock_held < self.min_trade_size:
            self.stock_held = 0.0

        # Logging
        if trade_executed and (self.current_step == self.window_size or self.current_step % 50 == 0 or self.done):
            print(f"[LOG] {'First' if self.current_step == self.window_size else 'Final' if self.done else 'Step'} {self.current_step} | "
                  f"Action: {action} | Price: {price:.2f} | Cash: {self.cash:.2f} | Crypto: {self.stock_held:.6f}")

        return self._get_state(), reward, self.done, trade_executed

    def get_trade_rewards(self):
        return self.trade_rewards
