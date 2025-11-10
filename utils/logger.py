# log_manager.py

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === 1. Save per-ticker log ===
def log_ticker(log_data, ticker, folder="logs"):
    """Save per-ticker portfolio log with timestamp."""
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    filename = f"{folder}/portfolio_log_{ticker}_{timestamp}.csv"

    df = pd.DataFrame(log_data)
    df["ticker"] = ticker
    df.to_csv(filename, index=False)

    print(f"[INFO] Log saved to {filename}")
    return filename

# === 2. Plot per-ticker log ===
def log_plot(csv_path, timeframe="Unknown"):
    """Plot a per-ticker portfolio log with Buy/Sell markers."""
    df = pd.read_csv(csv_path)

    ticker = df['ticker'].iloc[0] if 'ticker' in df.columns else "UNKNOWN"
    os.makedirs("logs/report_img", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    report_img = f"logs/report_img/report_{ticker}_{timestamp}.png"

    plt.figure(figsize=(14, 6))
    plt.plot(df['step'], df['portfolio_value'], label="Portfolio Value", linewidth=2)

    if 'action' in df.columns:
        buys = df[df['action'].str.contains("Buy", case=False, na=False)]
        sells = df[df['action'].str.contains("Sell", case=False, na=False)]
        plt.scatter(buys['step'], buys['portfolio_value'], marker='^', color='green', label='Buy', s=100)
        plt.scatter(sells['step'], sells['portfolio_value'], marker='v', color='red', label='Sell', s=100)

    plt.title(f"Portfolio Value for {ticker} | Timeframe: {timeframe}")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(report_img)

    print(f"[INFO] Report saved to {report_img}")

# === 3. Save overall (aggregated) log ===
def overall_log(log_data, folder="logs"):
    """Save overall simulation log across multiple runs/tickers."""
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    filename = f"{folder}/simulation_report_{timestamp}.csv"

    df = pd.DataFrame(log_data)
    df.to_csv(filename, index=False)

    print(f"[INFO] Log saved to {filename}")
    return filename

# === 4. Plot overall (aggregated) log ===
def overall_plot(csv_path):
    """Plot overall portfolio performance with Buy/Sell markers."""
    df = pd.read_csv(csv_path)

    x_column = 'global_step' if 'global_step' in df.columns else 'step'
    x_label = "Global Step" if x_column == 'global_step' else "Step"

    os.makedirs("logs/report_img", exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    report_img = f"logs/report_img/simulation_report_{timestamp}.png"

    plt.figure(figsize=(14, 6))
    plt.plot(df[x_column], df['portfolio_value'], label="Portfolio Value", linewidth=2)

    if 'action' in df.columns:
        buys = df[df['action'].str.contains("Buy", case=False, na=False)]
        sells = df[df['action'].str.contains("Sell", case=False, na=False)]
        plt.scatter(buys[x_column], buys['portfolio_value'], marker='^', color='green', label='Buy', s=100)
        plt.scatter(sells[x_column], sells['portfolio_value'], marker='v', color='red', label='Sell', s=100)

    plt.title("Portfolio Value Over Time")
    plt.xlabel(x_label)
    plt.ylabel("Total Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(report_img)

    print(f"[INFO] Report saved to {report_img}")

# === 4. store csv about model training ===

def log_trade_step(step, action, price, cash, stock, reward, asset_value, portfolio_value, 
                   ticker="", timeframe="", episode=0, filename="logs/trade_log.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Ticker", "Timeframe", "Episode", "Step", "Action", 
                "Price", "Cash", "Stock", "Reward", "Asset_Value", "Portfolio_Value"
            ])
        
        writer.writerow([
            ticker,
            timeframe,
            episode,
            step,
            action,
            round(price, 2),
            round(cash, 2),
            round(stock, 4),
            round(reward, 4),
            round(asset_value, 2),
            round(portfolio_value, 2)
        ])
