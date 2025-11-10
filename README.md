##  Crypto Trading Agent

An intelligent **cryptocurrency trading agent** built using **deep reinforcement learning (DRL)** and **neural networks** (LSTM + CNN).  
The model learns to make real-time trading decisions — **Buy**, **Sell**, or **Hold** — based on historical price patterns and technical indicators.  

This project combines **pattern recognition**, **reinforcement learning**, and **rule-based strategies** to continuously evolve a self-learning trading model capable of identifying profitable opportunities in crypto markets.  

**Anyone interested in this idea is warmly welcome to contribute, share improvements, or experiment with new strategies in this repo!**

---

##  Core Features

- **Neural Network Hybrid Architecture**
  - Combines **CNN** (for price pattern recognition from candlestick data) and **LSTM** (for sequential temporal modeling).
- **Reinforcement Learning Framework**
  - Learns through rewards and penalties for actions: Buy / Sell / Hold.
  - Dynamic reward system based on profit/loss, position holding time, and volatility.
- **Trading Strategy Integration**
  - Rule-based conditions (e.g., RSI, MA crossover, Momentum) assist in decision-making.
- **Training & Testing Modes**
  - Training phase saves logs to CSV for performance tracking.
  - Testing phase evaluates the trained `.pt` model on unseen market data.
- **Torch Model Export**
  - Model is saved as a portable `.pt` file for deployment or further fine-tuning.
- **Extensive Logging**
  - All training and test iterations are logged with timestamp, actions, rewards, and balances.

---

##  Technical Stack

- **Language:** Python 3.10+
- **Deep Learning:** PyTorch
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Plotly
- **Market Data:** yfinance
- **Environment:** Custom RL environment (`TradingEnvironment`)

---

##  Model Architecture

- **CNN Layer:** Extracts spatial patterns from candlestick or indicator images.
- **LSTM Layer:** Captures sequential dependencies in time-series price movements.
- **Fully Connected Layers:** Combine CNN and LSTM outputs to predict action probabilities.
- **Action Space:** `{ Buy, Sell, Hold }`
- **Reward System:** 
  - Positive reward for profitable trades.
  - Negative reward for losses or unnecessary holds.

---

##  How It Works

1. **Data Collection:** Historical crypto data fetched using API.
2. **Feature Engineering:** Technical indicators and image-like inputs created.
3. **Model Training:** 
   - The agent interacts with the environment, learning optimal actions.
   - Rewards reinforce profitable strategies.
   - Logs recorded to `training_logs.csv`.
4. **Model Saving:** Best-performing model saved as `.pt` file.
5. **Testing:** 
   - Loaded model tested on unseen data (`test_agent.py`).
   - Results logged to `test_logs.csv`.
6. **Evaluation:** 
   - Accuracy and performance metrics visualized via `plot_results.py`.

---

##  Future Improvements

* Integrate **ONNX export** for model portability.
* Implement **real-time inference engine** for live trading.
* Add **Binance Testnet** and **paper trading** integration.
* Introduce **adaptive reward functions** for dynamic market regimes.
* Deploy on **cloud instance** for 24/7 operation.

---

##  Version

**Current Version:** `2.01`
**Status:** Stable (Training & Testing Functional)
**Next Milestone:** Real-time Trading Integration

**Task to complete**

| Area                | Guidelines                                                                                                                  |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Risk Management** | Never risk more than 1-2% of capital per trade. Implement stop-loss and max drawdown controls.                              |
| **Regulatory**      | Consult local laws. Some countries require registering as an investment advisor or firm if using AI to manage client funds. |
| **Logging**         | Log all predictions, actions, slippage, and latency.                                                                        |
| **Backtesting**     | Should be statistically rigorous and avoid lookahead bias or overfitting.                                                   |
| **Paper Trading**   | Run for **2–3 months** before live launch.                                                                                  |
| **Model Updating**  | Models should be retrained periodically or adapt online (carefully)                                                         |
| **Kill Switch**     | Emergency stop logic for market anomalies (e.g., flash crashes)                                                             |

---

##  License

This project is licensed under the **MIT License** — feel free to use and modify with attribution.

---

##  Author

**Sanjay Kumar**

Machine Learning & Trading Systems Developer

 [Gmail](mailto:sanjaykumar78523sk@gmail.com)
 [GitHub Profile](https://github.com/SanjayKumar3110)
 [LinkedIn URL](https://www.linkedin.com/in/sanjaykumar785)

---

