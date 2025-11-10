import numpy as np

class RewardCalculator:
    def __init__(self, trading_fee=0.0001, min_trade_size=0.0005):
        self.trading_fee = trading_fee
        self.min_trade_size = min_trade_size
        self.last_action_step = None
        self.last_action = None

    def compute_trade_reward(self, price, last_buy_price, is_profit, steps_since_last_trade):
        trade_reward = (price - last_buy_price) / (last_buy_price + 1e-8)

        if not is_profit:
            trade_reward -= 0.05  # Increased penalty for loss

        if steps_since_last_trade < 3:
            trade_reward -= 0.01

        return trade_reward

    def apply_liquidity_penalty(self, reward):
        return reward - 0.01

    def apply_cash_penalty(self, reward):
        return reward - 0.03

    def apply_hold_penalty(self, reward):
        # Apply a consistent penalty for every hold action
        return reward - 0.005 # Increased and consistent penalty

    def discourage_churn(self, reward, current_action):
        if current_action == self.last_action and current_action in [1, 2]:
            reward -= 0.003
        self.last_action = current_action
        return reward

    def add_reward_shaping(self, reward, raw_reward, trade_executed):
        if trade_executed:
            reward += 0.01  # Increased bonus for executing any trade
        reward += 0.05 * raw_reward # Increased weight on raw reward
        return np.clip(reward, -0.05, 2.0)