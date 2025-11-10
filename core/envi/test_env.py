import numpy as np

class TestEnvironment:
    def __init__(self, price_data, rsi_values, short_ma_values, long_ma_values, momentum_values, window_size=10, initial_cash=10.01):
        self.price_data = price_data
        self.rsi = rsi_values
        self.short_ma = short_ma_values
        self.long_ma = long_ma_values
        self.momentum = momentum_values
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.min_trade_size = 0.0001
        self.total_sell_value = 0
        self.total_stocks_sold = 0

        self.reset()

    def reset(self):
        self.current_step = self.window_size - 1
        self.cash = self.initial_cash
        self.stock_held = 0.0
        self.last_buy_price = 0.0
        self.done = False
        self.trades = [] 
        return self._get_state()

    def _get_state(self):
        end = self.current_step + 1
        start = end - self.window_size

        if start < 0:
            start = 0

        # Extract data windows for all features
        price_window = self.price_data[start:end]
        rsi_window = self.rsi[start:end]
        short_ma_window = self.short_ma[start:end]
        long_ma_window = self.long_ma[start:end]
        momentum_window = self.momentum[start:end]

        # Normalize the price data
        norm_prices = (price_window - np.mean(price_window)) / (np.std(price_window) + 1e-7)

        # Broadcast scalar features to the window size
        portfolio_value = self.cash + self.stock_held * self.price_data[self.current_step]
        cash_ratio_window = np.full(self.window_size, self.cash / (portfolio_value + 1e-8))
        asset_ratio_window = np.full(self.window_size, (self.stock_held * self.price_data[self.current_step]) / (portfolio_value + 1e-8))

        # Stack features to create a 2D state matrix
        state_matrix = np.vstack([
            norm_prices,
            rsi_window,
            short_ma_window,
            long_ma_window,
            momentum_window,
            cash_ratio_window,
            asset_ratio_window
        ]).T  # Transpose to get the shape (window_size, num_features)

        return state_matrix.astype(np.float32)

    def step(self, action):
        price = self.price_data[self.current_step]
        trade_executed = False
        
        # Enforce minimum trade size
        if self.cash < self.min_trade_size * price and action == 1:
            action = 0 # Hold if not enough cash to buy
            
        if self.stock_held < self.min_trade_size and action == 2:
            action = 0 # Hold if not enough stock to sell
        
        if action == 1:
            # Buy a percentage of available cash
            buy_amount = (self.cash * 0.25) / price
            cost = buy_amount * price
            self.stock_held += buy_amount
            self.cash -= cost
            self.trades.append(('buy', price, buy_amount))
            trade_executed = True

        elif action == 2:
            # Sell a percentage of held stock
            sell_amount = self.stock_held * 0.25
            proceeds = sell_amount * price
            self.cash += proceeds
            self.stock_held -= sell_amount
            self.trades.append(('sell', price, sell_amount))
            trade_executed = True
        
        # The environment needs to know the next state
        self.current_step += 1
        
        # Check if the end of data is reached
        if self.current_step >= len(self.price_data):
            self.done = True
            
        next_state = self._get_state()
        
        return next_state, self.done
