# === Alpaca Paper Trading Integration ===
import re
from bs4 import BeautifulSoup
from transformers import pipeline
import yfinance as yf
import argparse
import praw
import tweepy
import json

# Register and get API keys at https://alpaca.markets
ALPACA_KEY = "your_alpaca_api_key"
ALPACA_SECRET = "your_alpaca_secret"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

import pandas as pd
import numpy as np
import requests
import alpaca_trade_api as tradeapi
import time
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import threading
import schedule

MODEL_CACHE = {}
SCALER_CACHE = {}
PORTFOLIO_POSITIONS = {}

# === Reddit & X Trend Scanner ===
REDDIT_CLIENT = praw.Reddit(
    client_id="your_reddit_client_id",
    client_secret="your_reddit_secret",
    user_agent="signal_bot"
)

def scan_reddit_trending(limit=50):
    trending = {}
    for submission in REDDIT_CLIENT.subreddit("stocks+wallstreetbets").hot(limit=limit):
        words = re.findall(r'\b[A-Z]{2,5}\b', submission.title)
        for word in words:
            trending[word] = trending.get(word, 0) + 1
    return sorted(trending, key=trending.get, reverse=True)[:10]

TWITTER_BEARER_TOKEN = "your_twitter_bearer"
TWITTER_HEADERS = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}

def scan_twitter_trending():
    try:
        response = requests.get("https://api.twitter.com/2/tweets/search/recent?query=$AAPL OR $TSLA OR $NVDA&max_results=100", headers=TWITTER_HEADERS)
        data = response.json()
        if 'data' not in data:
            return []
        symbols = re.findall(r'\$[A-Z]{2,5}', json.dumps(data))
        freq = pd.Series(symbols).value_counts()
        return freq.head(5).index.str.replace("$", "").tolist()
    except Exception as e:
        print("Twitter scan failed:", e)
        return []

# === News Prompt Analysis ===
from openai import OpenAI
openai.api_key = "your_openai_api_key"

def parse_news_ai(headlines):
    prompt = f"Analyze the following headlines and return a short 1-2 sentence insight on overall market tone:\n\n{chr(10).join(headlines)}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return "Prompt failed: " + str(e)

# === Volatility Prediction (Basic) ===
def forecast_volatility(df: pd.DataFrame) -> float:
    returns = df['close'].pct_change().dropna()
    return np.std(returns.tail(10)) * np.sqrt(252)

# === Dynamic Watchlist Expansion & Smart Summary ===
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "TSLA", "META", "AMZN"]

def expand_watchlist():
    reddit = scan_reddit_trending()
    twitter = scan_twitter_trending()
    combined = list(set(WATCHLIST + reddit + twitter))
    return combined[:20]  # Cap max size

def daily_summary():
    try:
        all_signals = []
        for symbol in WATCHLIST:
            df = fetch_latest_ohlc(symbol)
            if df is not None:
                headlines = get_yahoo_headlines(symbol)
                ai_summary = parse_news_ai(headlines)
                vol = forecast_volatility(df)
                all_signals.append(f"{symbol}: Vol={vol:.2f} | News Insight: {ai_summary}")
        message = "**Daily Market Summary**

" + "
".join(all_signals)
        send_discord(message)
    except Exception as e:
        send_discord(f"Summary failed: {e}")

# Auto-expand watchlist hourly
schedule.every(60).minutes.do(lambda: WATCHLIST.clear() or WATCHLIST.extend(expand_watchlist()))

# Send daily summary after market close
schedule.every().day.at("16:10").do(daily_summary)

# === Backtesting Engine ===
def backtest_strategy(symbol, model_type="logistic", start="2023-01-01", end="2023-12-31", initial_capital=100000):
    df = yf.download(symbol, start=start, end=end, interval="1d")
    df.dropna(inplace=True)
    df['symbol'] = symbol
    df_feat = compute_features(df)
    df_feat.dropna(inplace=True)

    train_model(df_feat, model_type=model_type)
    model = MODEL_CACHE[symbol]
    scaler = SCALER_CACHE[symbol]

    capital = initial_capital
    position = 0
    trade_log = []

    for i in range(20, len(df_feat)):
        today = df_feat.iloc[i]
        X = scaler.transform([today[['vwap_diff', 'rsi', 'zscore', 'momentum_3', 'momentum_10', 'volume_z']].values])
        prob = model.predict_proba(X)[0][1]

        price = df_feat.iloc[i]['close']

        if prob > 0.6 and capital >= price:
            qty = int(capital / price)
            capital -= qty * price
            position += qty
            trade_log.append((df_feat.index[i], "BUY", price, qty))

        elif prob < 0.4 and position > 0:
            capital += position * price
            trade_log.append((df_feat.index[i], "SELL", price, position))
            position = 0

    if position > 0:
        capital += position * df_feat.iloc[-1]['close']

    return_pct = (capital - initial_capital) / initial_capital * 100
    return round(return_pct, 2), trade_log

# Example usage:
# pct, log = backtest_strategy("AAPL", model_type="xgboost")
# print("Return:", pct, "%")

# === Performance Metrics & PnL Visualization ===
def calculate_metrics(trade_log, initial_capital=100000):
    pnl_curve = []
    capital = initial_capital
    position = 0
    equity = initial_capital
    peak = equity
    drawdowns = []
    wins, losses = 0, 0

    for i, (date, action, price, qty) in enumerate(trade_log):
        if action == "BUY":
            capital -= price * qty
            position += qty
        elif action == "SELL":
            proceeds = price * qty
            capital += proceeds
            gain = proceeds - (qty * price)
            if gain > 0:
                wins += 1
            else:
                losses += 1
            position = 0
        equity = capital + (position * price)
        pnl_curve.append(equity)
        peak = max(peak, equity)
        drawdowns.append((peak - equity) / peak)

    sharpe = np.mean(np.diff(pnl_curve)) / (np.std(np.diff(pnl_curve)) + 1e-9) * np.sqrt(252)
    max_dd = max(drawdowns) * 100 if drawdowns else 0
    win_rate = 100 * wins / (wins + losses) if wins + losses > 0 else 0
    return {
        "Final Equity": round(equity, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown %": round(max_dd, 2),
        "Win Rate %": round(win_rate, 2),
        "Trades": len(trade_log)
    }

# Example usage:
# stats = calculate_metrics(log)
# print(json.dumps(stats, indent=2))

# === Integrate to Elite Signal (Inline Comments for Expansion) ===
# You can now call scan_reddit_trending() and scan_twitter_trending() to expand WATCHLIST dynamically
# Call parse_news_ai() in alert logging for smarter LLM-generated reasoning
# Use forecast_volatility(df) before sending alerts to optionally filter based on volatility thresholds

# === Strategy DSL + Plugin Engine ===
import importlib.util
import os

STRATEGY_DIR = "./strategies"
STRATEGY_PLUGINS = {}

# Load user-defined plugins

def load_plugins():
    for filename in os.listdir(STRATEGY_DIR):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            file_path = os.path.join(STRATEGY_DIR, filename)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            STRATEGY_PLUGINS[module_name] = mod

# Call plugin strategy by name
def run_strategy(symbol, df, strategy_name="default"):
    if strategy_name in STRATEGY_PLUGINS:
        return STRATEGY_PLUGINS[strategy_name].generate_signal(symbol, df)
    else:
        raise ValueError(f"Strategy '{strategy_name}' not loaded.")

# Example plugin (saved as ./strategies/macd_cross.py):
# def generate_signal(symbol, df):
#     df['ema12'] = df['close'].ewm(span=12).mean()
#     df['ema26'] = df['close'].ewm(span=26).mean()
#     df['macd'] = df['ema12'] - df['ema26']
#     df['signal'] = df['macd'].ewm(span=9).mean()
#     if df['macd'].iloc[-1] > df['signal'].iloc[-1]:
#         return "BUY"
#     elif df['macd'].iloc[-1] < df['signal'].iloc[-1]:
#         return "SELL"
#     else:
#         return "HOLD"

load_plugins()

# === Options Flow Scanner ===
def fetch_options_flow(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/options/{symbol}"
        response = requests.get(url)
        chain = response.json()
        calls = chain['optionChain']['result'][0]['options'][0]['calls']
        puts = chain['optionChain']['result'][0]['options'][0]['puts']

        unusual_activity = []
        for opt in calls + puts:
            volume = opt.get("volume", 0)
            open_interest = opt.get("openInterest", 1)
            iv = opt.get("impliedVolatility", 0)
            if open_interest > 0 and volume / open_interest > 5 and volume > 100:
                unusual_activity.append({
                    "contract": opt['contractSymbol'],
                    "type": "CALL" if opt in calls else "PUT",
                    "strike": opt['strike'],
                    "expiry": opt['expiration'],
                    "volume": volume,
                    "oi": open_interest,
                    "iv": iv
                })
        return unusual_activity
    except Exception as e:
        return [f"Options data error: {e}"]

# Example usage:
# print(fetch_options_flow("NVDA"))

# Send Discord alert for unusual flow

def alert_options_flow(symbol):
    flow = fetch_options_flow(symbol)
    if not flow:
        return
    if isinstance(flow[0], str):
        send_discord(f"Options Flow Error: {flow[0]}")
    else:
        msg = f"📈 Unusual Options Activity for {symbol}"
        for opt in flow[:5]:
            msg += f"
→ {opt['type']} {opt['strike']} exp {opt['expiry']} | Vol: {opt['volume']} | OI: {opt['oi']} | IV: {opt['iv']:.2f}"
        send_discord(msg)

# Auto-run alert daily after market open
schedule.every().day.at("09:45").do(lambda: [alert_options_flow(sym) for sym in WATCHLIST])

# === Trade Journal + Analytics Storage ===
import sqlite3

DB_PATH = "trades.db"

# Initialize SQLite DB

def init_trade_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        symbol TEXT,
        action TEXT,
        price REAL,
        qty INTEGER,
        date TEXT
    )''')
    conn.commit()
    conn.close()

# Store each trade to DB

def log_trade(symbol, action, price, qty):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?)", (symbol, action, price, qty, datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Example usage inside trade loop:
# log_trade(symbol, "BUY", price, qty)

init_trade_db()

# === Portfolio Risk Manager ===

MAX_PORTFOLIO_ALLOCATION = 0.25  # Max 25% in any single asset
MAX_TOTAL_EXPOSURE = 0.95        # Never deploy more than 95% of capital

portfolio_cash = 100000
portfolio_holdings = {}

# Check if a trade exceeds exposure rules

def risk_check(symbol, price, qty):
    global portfolio_cash, portfolio_holdings
    current_value = sum(p * q for s, (p, q) in portfolio_holdings.items())
    new_value = price * qty
    total_value = current_value + portfolio_cash

    if new_value / total_value > MAX_PORTFOLIO_ALLOCATION:
        return False, f"❌ Risk: {symbol} exceeds per-asset cap"
    if (current_value + new_value) / total_value > MAX_TOTAL_EXPOSURE:
        return False, f"❌ Risk: Total exposure exceeds limit"
    return True, "✅ Risk checks passed"

# Add post-trade update logic:

def update_portfolio(symbol, price, qty, action):
    global portfolio_cash, portfolio_holdings
    if action == "BUY":
        portfolio_cash -= price * qty
        if symbol in portfolio_holdings:
            p, q = portfolio_holdings[symbol]
            portfolio_holdings[symbol] = (price, q + qty)
        else:
            portfolio_holdings[symbol] = (price, qty)
    elif action == "SELL" and symbol in portfolio_holdings:
        p, q = portfolio_holdings[symbol]
        proceeds = price * qty
        portfolio_cash += proceeds
        if qty >= q:
            del portfolio_holdings[symbol]
        else:
            portfolio_holdings[symbol] = (p, q - qty)

# Example use inside trade decision:
# allowed, msg = risk_check(symbol, price, qty)
# if allowed:
#     update_portfolio(symbol, price, qty, "BUY")
#     log_trade(symbol, "BUY", price, qty)
# else:
#     send_discord(msg)

# === Reinforcement Learning Stub (Phase Start) ===
# Placeholder for PPO / Q-learning based decision module
# Will eventually integrate a reward system for portfolio improvement

# === Discord Dashboard Summary ===

def discord_dashboard_summary():
    summary_lines = [
        "📊 **Portfolio Overview**",
        f"💰 Cash: ${portfolio_cash:,.2f}"
    ]
    total_value = portfolio_cash
    for symbol, (price, qty) in portfolio_holdings.items():
        position_value = price * qty
        summary_lines.append(f"• {symbol}: {qty} shares @ ${price:.2f} = ${position_value:,.2f}")
        total_value += position_value
    summary_lines.append(f"📈 Total Portfolio Value: ${total_value:,.2f}")
    send_discord("
".join(summary_lines))

# Auto-send dashboard at 4:15 PM
schedule.every().day.at("16:15").do(discord_dashboard_summary)


