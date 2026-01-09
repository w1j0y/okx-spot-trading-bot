#!/usr/bin/env python3
# Ichimoku OKX Dip-Buy Bot (Open Source Edition)
# - Plain-text config.json (no Fernet/KDF)
# - Supports any SPOT *-USDT pair via config instId
# - Limit buy ladder + last-buy-price enforcement
# - Profit sell on full stack (+ cycle reset)
# - CSV trade log + optional email notifications + weekly report

import os
import time
import json
import logging
import pandas as pd
import csv
from datetime import datetime, timedelta, timezone
import smtplib
from email.message import EmailMessage

from okx.MarketData import MarketAPI
from okx.Trade import TradeAPI
from okx.Account import AccountAPI
from okx.PublicData import PublicAPI
import talib

# =======================
# Files / Paths
# =======================
data_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(data_dir, exist_ok=True)

CONFIG_PATH      = os.path.join(data_dir, "config.json")
log_file         = os.path.join(data_dir, "trade_log.csv")
botlog_file      = os.path.join(data_dir, "bot.log")
report_file      = os.path.join(data_dir, "last_report_sent.txt")
stage_file       = os.path.join(data_dir, "buy_stage.txt")
order_file       = os.path.join(data_dir, "pending_order.txt")
sell_order_file  = os.path.join(data_dir, "pending_sell_order.txt")
last_buy_file    = os.path.join(data_dir, "last_buy_price.txt")
cycle_stats_file = os.path.join(data_dir, "cycle_stats.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(botlog_file)]
)

# =======================
# Load config.json
# =======================
def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"Missing config.json at: {path}\n"
            "Create it next to the script. See README/sample config."
        )
    with open(path, "r") as f:
        cfg = json.load(f)

    okx = cfg.get("okx") or {}
    trade = cfg.get("trade") or {}
    email = cfg.get("email") or {}

    api_key = okx.get("api_key")
    api_secret = okx.get("api_secret")
    passphrase = okx.get("passphrase")
    if not (api_key and api_secret and passphrase):
        raise SystemExit("config.json missing okx.api_key / okx.api_secret / okx.passphrase")

    instId = str(trade.get("instId") or "BTC-USDT").strip().upper().replace(" ", "")
    if "-" not in instId:
        raise SystemExit("trade.instId must look like 'BTC-USDT' (OKX spot instrument id).")

    total_usdt = trade.get("total_usdt")
    if total_usdt is None:
        raise SystemExit("config.json missing trade.total_usdt")
    try:
        total_usdt = float(total_usdt)
    except Exception:
        raise SystemExit("trade.total_usdt must be numeric")

    interval_okx = str(trade.get("interval_okx") or "30m").strip()
    use_demo = bool(trade.get("use_demo", False))

    email_enabled = bool(email.get("enabled", False))
    sender_email = email.get("sender_email")
    email_password = email.get("email_password")
    recipient_email = email.get("recipient_email")

    return {
        "okx": {"api_key": api_key, "api_secret": api_secret, "passphrase": passphrase},
        "trade": {"instId": instId, "base_ccy": instId.split("-")[0], "total_usdt": total_usdt, "interval_okx": interval_okx, "use_demo": use_demo},
        "email": {"enabled": email_enabled, "sender_email": sender_email, "email_password": email_password, "recipient_email": recipient_email},
    }

cfg = load_config(CONFIG_PATH)

# =======================
# Static Strategy Params
# =======================
portions = [0.04, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.16]
LIMIT_PRICE_OFFSET = 0.995

# Profit sell settings
PROFIT_TARGET_PCT = 0.0025    # 0.25% over avg cost
SELL_PRICE_BUFFER = 1.0005    # defined but we use quantized price directly as you fixed

instId       = cfg["trade"]["instId"]
base_ccy     = cfg["trade"]["base_ccy"]
total_usdt   = cfg["trade"]["total_usdt"]
interval_okx = cfg["trade"]["interval_okx"]
use_demo     = cfg["trade"]["use_demo"]

logging.info(f"🔧 Loaded config: instId={instId}, total_usdt={total_usdt}, interval={interval_okx}, demo={use_demo}")

# =======================
# Email
# =======================
EMAIL_ENABLED   = cfg["email"]["enabled"]
SENDER_EMAIL    = cfg["email"]["sender_email"]
EMAIL_PASSWORD  = cfg["email"]["email_password"]
RECIPIENT_EMAIL = cfg["email"]["recipient_email"]

def cancel_all_open_orders_for_inst(instId: str):
    """
    Cancel ALL live / partially filled SPOT orders
    for the given instrument (safety cleanup).
    """
    try:
        res = tradeAPI.get_order_list(instType="SPOT", instId=instId)
        orders = res.get("data", [])

        for o in orders:
            if o.get("state") in ("live", "partially_filled"):
                ord_id = o.get("ordId")
                tradeAPI.cancel_order(instId=instId, ordId=ord_id)
                logging.info(f"🛑 Cancelled leftover order {ord_id} for {instId}")

    except Exception as e:
        logging.error(f"Failed cancelling open orders for {instId}: {e}")

def send_trade_email(subject: str, body: str):
    if not EMAIL_ENABLED:
        return
    if not (SENDER_EMAIL and EMAIL_PASSWORD and RECIPIENT_EMAIL):
        logging.warning("Email enabled but sender/recipient/password missing in config.json")
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logging.info("📧 Trade email sent.")
    except Exception as e:
        logging.error(f"Trade email error: {e}")

def send_weekly_report_email(subject: str, body: str, csv_path: str):
    if not EMAIL_ENABLED:
        return
    if not (SENDER_EMAIL and EMAIL_PASSWORD and RECIPIENT_EMAIL):
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    try:
        with open(csv_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="application",
                subtype="octet-stream",
                filename=os.path.basename(csv_path),
            )
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, EMAIL_PASSWORD)
            smtp.send_message(msg)
        logging.info("📧 Weekly report sent.")
    except Exception as e:
        logging.error(f"CSV email error: {e}")

# =======================
# OKX Clients
# =======================
flag = "1" if use_demo else "0"
marketAPI  = MarketAPI(cfg["okx"]["api_key"], cfg["okx"]["api_secret"], cfg["okx"]["passphrase"], False, flag)
tradeAPI   = TradeAPI(cfg["okx"]["api_key"], cfg["okx"]["api_secret"], cfg["okx"]["passphrase"], False, flag)
accountAPI = AccountAPI(cfg["okx"]["api_key"], cfg["okx"]["api_secret"], cfg["okx"]["passphrase"], False, flag)
publicAPI  = PublicAPI(cfg["okx"]["api_key"], cfg["okx"]["api_secret"], cfg["okx"]["passphrase"], False, flag)

# =======================
# Instrument precision
# =======================
def fetch_instrument_info(instId_: str):
    try:
        res = publicAPI.get_instruments(instType="SPOT", instId=instId_)
        data = res.get("data", [])
        if not data:
            raise RuntimeError(f"No instrument data for {instId_}")
        info = data[0]
        tickSz = float(info["tickSz"])
        lotSz  = float(info["lotSz"])
        minSz  = float(info.get("minSz", lotSz))
        return tickSz, lotSz, minSz
    except Exception as e:
        logging.warning(f"Instrument info error for {instId_}: {e}. Fallback tickSz=0.0001, lotSz=0.0001, minSz=0.0001")
        return 0.0001, 0.0001, 0.0001

tickSz, lotSz, minSz = fetch_instrument_info(instId)
logging.info(f"Precision — tickSz={tickSz}, lotSz={lotSz}, minSz={minSz}")
if minSz <= 0:
    raise SystemExit(f"Instrument {instId} invalid on OKX.")

def quantize_price(px: float) -> float:
    if tickSz <= 0:
        return float(f"{px:.8f}")
    steps = int(px / tickSz)
    return float(f"{steps * tickSz:.8f}")

def quantize_size(sz: float) -> float:
    if lotSz <= 0:
        return float(f"{sz:.8f}")
    steps = int(sz / lotSz)
    q = steps * lotSz
    if q < minSz:
        return 0.0
    return float(f"{q:.8f}")

# =======================
# Time helper
# =======================
def sleep_until_next_half_hour():
    now = datetime.now(timezone.utc)
    next_minute = 30 if now.minute < 30 else 60
    next_boundary = now.replace(minute=0 if next_minute == 60 else 30, second=0, microsecond=0)
    if next_minute == 60:
        next_boundary = next_boundary + timedelta(hours=1)
    time.sleep(max(0, (next_boundary - now).total_seconds()))

# =======================
# Market data
# =======================
def get_klines_okx(instId_: str, bar: str = "30m", limit: int = 100) -> pd.DataFrame:
    def _fetch():
        res = marketAPI.get_candlesticks(instId=instId_, bar=bar, limit=str(limit))
        return res.get("data") or []

    rows = _fetch()
    if not rows:
        logging.warning(f"No kline data from OKX for {instId_} {bar}. Retrying after 60s.")
        time.sleep(60)
        rows = _fetch()
        if not rows:
            raise RuntimeError(f"No kline data for {instId_} {bar} after retry.")

    rows = list(reversed(rows))
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume","volCcy","volCcyQuote","confirm"])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df

# =======================
# Indicators
# =======================
def calculate_ichimoku(df):
    TENKAN = 18
    KIJUN  = 52
    SPANB  = 104

    high_t = df["high"].rolling(window=TENKAN).max()
    low_t  = df["low"].rolling(window=TENKAN).min()
    tenkan = (high_t + low_t) / 2

    high_k = df["high"].rolling(window=KIJUN).max()
    low_k  = df["low"].rolling(window=KIJUN).min()
    kijun  = (high_k + low_k) / 2

    span_a = ((tenkan + kijun) / 2).shift(KIJUN)
    high_b = df["high"].rolling(window=SPANB).max()
    low_b  = df["low"].rolling(window=SPANB).min()
    span_b = ((high_b + low_b) / 2).shift(KIJUN)
    chikou = df["close"].shift(-KIJUN)
    return tenkan, kijun, span_a, span_b, chikou

def calculate_macd(df):
    fast_period = 24
    slow_period = 52
    signal_period = 18
    ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def check_signals(df):
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)

    price = df["close"].iloc[-2]
    volume = df["volume"].iloc[-2]
    avg_volume = df["volume"].rolling(window=40).mean().iloc[-2]
    rsi = talib.RSI(df["close"], timeperiod=28).iloc[-2]
    macd, signal_line, hist = calculate_macd(df)

    signals = []

    if rsi < 40:
        signals.append(("rsi_oversold", 2.0))
        logging.info(f"✔ RSI Oversold ({rsi:.2f} < 40)")

    if price <= kijun.iloc[-2] * 1.005 and price > min(span_a.iloc[-2], span_b.iloc[-2]) * 0.98:
        signals.append(("kijun_dip_bullish", 1.5))
        logging.info("✔ Price at/below Kijun and above Kumo")

    if hist.iloc[-2] > 0 or (hist.iloc[-2] > hist.iloc[-3] if len(hist) > 3 else False):
        signals.append(("macd_hist_positive_or_improving", 1.5))
        logging.info(f"✔ MACD Histogram positive/improving (Hist: {hist.iloc[-2]:.4f})")

    if volume > avg_volume * 1.5:
        signals.append(("volume_surge", 1.0))
        logging.info(f"✔ Volume Surge ({volume:.2f} > 1.5x avg {avg_volume:.2f})")

    last_close = df["close"].iloc[-2]
    last_open  = df["open"].iloc[-2]
    drop_pct = (last_close - last_open) / last_open
    if drop_pct <= -0.008:
        signals.append(("rapid_dip_bar", 1.0))
        logging.info(f"✔ Rapid dip bar ({drop_pct*100:.2f}%)")

    return signals

# =======================
# Trade log / report
# =======================
def append_trade(timestamp, action, price, qty):
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp","action","price","quantity"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"timestamp": timestamp, "action": action, "price": price, "quantity": qty})

def get_trade_summary():
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        return 0, 0.0, 0.0
    df = pd.read_csv(log_file)
    n_buys = (df["action"] == "BUY").sum()
    total_qty = df["quantity"].sum()
    total_invested = sum([total_usdt * p for p in portions[:n_buys]])
    return int(n_buys), float(total_qty), float(total_invested)

def send_weekly_report():
    if not EMAIL_ENABLED:
        return
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        return

    now = datetime.now()
    if not os.path.exists(report_file):
        with open(report_file, "w") as f:
            f.write(now.isoformat())
        logging.info("First run: initialized weekly report timestamp, skipping send.")
        return

    try:
        with open(report_file, "r") as f:
            last_sent = datetime.fromisoformat(f.read().strip())
    except Exception:
        last_sent = datetime.min

    if now - last_sent >= timedelta(days=7):
        n_buys, total_asset, total_invested = get_trade_summary()
        body = f"""Dear User,

Here is your weekly report:

Instrument: {instId}
Total buys: {n_buys}
Total {base_ccy}: {total_asset:.8f}
Total USDT invested (approx): ${total_invested:.2f}

Full trade details are attached in the CSV.
"""
        send_weekly_report_email("📊 Weekly Trading Report", body, log_file)
        with open(report_file, "w") as f:
            f.write(now.isoformat())

# =======================
# Cycle state
# =======================
def read_last_buy_price():
    try:
        with open(last_buy_file, "r") as f:
            return float(f.read().strip())
    except Exception:
        return None

def write_last_buy_price(px: float):
    with open(last_buy_file, "w") as f:
        f.write(str(px))

def reset_last_buy_price():
    if os.path.exists(last_buy_file):
        os.remove(last_buy_file)

def read_cycle_stats():
    try:
        with open(cycle_stats_file, "r") as f:
            return json.load(f)
    except Exception:
        return {"total_spent": 0.0, "total_btc": 0.0}

def write_cycle_stats(total_spent: float, total_btc: float):
    with open(cycle_stats_file, "w") as f:
        json.dump({"total_spent": total_spent, "total_btc": total_btc}, f, indent=2)

def reset_cycle_stats():
    if os.path.exists(cycle_stats_file):
        os.remove(cycle_stats_file)

def read_buy_stage() -> int:
    try:
        with open(stage_file, "r") as f:
            return int(f.read().strip())
    except Exception:
        return 0

def write_buy_stage(stage: int):
    with open(stage_file, "w") as f:
        f.write(str(stage))

# =======================
# OKX balance helpers
# =======================
def _num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def _account_details():
    try:
        res = accountAPI.get_account_balance()
        return (res or {}).get("data", [])
    except Exception as e:
        logging.warning(f"Account balance fetch error: {e}")
        return []

def get_usdt_balance() -> float:
    data = _account_details()
    if not data:
        return 0.0
    for d in data[0].get("details", []):
        if d.get("ccy") == "USDT":
            eq = _num(d.get("eq"))
            avail = _num(d.get("availBal"))
            cash = _num(d.get("cashBal"))
            return eq if eq > 0 else max(avail, cash)
    return 0.0

def get_asset_balance(asset: str) -> float:
    data = _account_details()
    if not data:
        return 0.0
    for d in data[0].get("details", []):
        if d.get("ccy") == asset:
            eq = _num(d.get("eq"))
            avail = _num(d.get("availBal"))
            cash = _num(d.get("cashBal"))
            frozen = _num(d.get("frozenBal")) + _num(d.get("ordFrozen"))
            return eq if eq > 0 else max(avail + frozen, cash)
    return 0.0

# =======================
# Pending orders
# =======================
def read_pending_order():
    try:
        with open(order_file, "r") as f:
            data = json.load(f)
            return data.get("order_id"), data.get("timestamp")
    except Exception:
        return None, None

def write_pending_order(order_id: str, timestamp: str):
    with open(order_file, "w") as f:
        json.dump({"order_id": order_id, "timestamp": timestamp}, f)

def clear_pending_order():
    if os.path.exists(order_file):
        os.remove(order_file)

def read_pending_sell():
    try:
        with open(sell_order_file, "r") as f:
            data = json.load(f)
            return data.get("order_id"), data.get("price")
    except Exception:
        return None, None

def write_pending_sell(order_id: str, price: float):
    with open(sell_order_file, "w") as f:
        json.dump({"order_id": order_id, "price": price}, f)

def clear_pending_sell():
    if os.path.exists(sell_order_file):
        os.remove(sell_order_file)

# =======================
# Order actions
# =======================
def check_order_status(order_id: str):
    try:
        res = tradeAPI.get_order(instId=instId, ordId=order_id)
        data = res.get("data", [{}])[0]
        state = data.get("state")
        filled_qty = float(data.get("fillSz", 0.0))
        avg_price = float(data.get("avgPx") or 0.0)
        if state == "filled":
            return True, filled_qty, avg_price
        if state in ["live", "partially_filled"]:
            return False, 0.0, 0.0
        return False, 0.0, 0.0
    except Exception as e:
        logging.error(f"Error checking order {order_id}: {e}")
        return False, 0.0, 0.0

def place_limit_buy(price: float, amount_usdt: float):
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    usdt = get_usdt_balance()
    if usdt < amount_usdt:
        logging.warning("⚠ Not enough USDT to place BUY order.")
        return 0.0, None

    limit_price = quantize_price(price * LIMIT_PRICE_OFFSET)
    qty = quantize_size(amount_usdt / limit_price)
    if qty < minSz:
        logging.warning(f"⚠ Order size {qty} below minSz {minSz}.")
        return 0.0, None

    try:
        res = tradeAPI.place_order(
            instId=instId,
            tdMode="cash",
            side="buy",
            ordType="limit",
            px=str(limit_price),
            sz=str(qty),
            tgtCcy="base_ccy",
        )
        order_id = res.get("data", [{}])[0].get("ordId")
        if not order_id:
            logging.error(f"Failed to place limit order: {res}")
            return 0.0, None

        write_pending_order(order_id, now_str)
        logging.info(f"🟢 LIMIT BUY placed @ {limit_price:.6f} for {qty} {base_ccy} (USDT: {amount_usdt})")
        return qty, order_id

    except Exception as e:
        logging.error(f"LIMIT BUY order error: {e}")
        return 0.0, None

def place_limit_sell_all(price: float):
    qty = quantize_size(get_asset_balance(base_ccy))
    if qty < minSz:
        return None

    limit_price = quantize_price(price)
    try:
        res = tradeAPI.place_order(
            instId=instId,
            tdMode="cash",
            side="sell",
            ordType="limit",
            px=str(limit_price),
            sz=str(qty),
            tgtCcy="base_ccy",
        )
        order_id = res.get("data", [{}])[0].get("ordId")
        if not order_id:
            logging.error(f"SELL order failed: {res}")
            return None

        write_pending_sell(order_id, limit_price)
        logging.info(f"🔴 LIMIT SELL placed @ {limit_price:.6f} for {qty} {base_ccy}")
        return order_id

    except Exception as e:
        logging.error(f"SELL order error: {e}")
        return None

def check_sell_filled():
    order_id, px = read_pending_sell()
    if not order_id:
        return False

    try:
        res = tradeAPI.get_order(instId=instId, ordId=order_id)
        data = res.get("data", [{}])[0]

        if data.get("state") == "filled":
            filled_qty = float(data.get("fillSz", 0))
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            append_trade(now, "SELL", px, filled_qty)
            # Cancel any leftover BUY orders
            cancel_all_open_orders_for_inst(instId)

            # Full cycle cleanup
            clear_pending_order()
            clear_pending_sell()
            reset_last_buy_price()
            reset_cycle_stats()
            write_buy_stage(0)

            send_trade_email(
                f"[{instId}] PROFIT SELL FILLED",
                f"""Dear User,

A PROFIT SELL order has been filled:

- Instrument: {instId}
- Price:      ${px:,.6f}
- Quantity:   {filled_qty} {base_ccy}
- Time:       {now}

Cycle has been reset. Bot is ready for the next dip-buy cycle.
"""
            )

            logging.info("🔁 SELL filled — cycle fully reset")
            return True

    except Exception as e:
        logging.error(f"SELL fill check error: {e}")

    return False

def check_and_place_profit_sell():
    sell_order_id, _ = read_pending_sell()
    if sell_order_id:
        return

    stats = read_cycle_stats()
    if stats["total_btc"] <= 0:
        return

    avg_cost = stats["total_spent"] / stats["total_btc"]
    df = get_klines_okx(instId, interval_okx, limit=2)
    current_price = df["close"].iloc[-1]

    # Minimum profit threshold
    target_price = avg_cost * (1 + PROFIT_TARGET_PCT)

    if current_price >= target_price:
        # SELL ABOVE current market, not at avg-based target
        sell_price = current_price * SELL_PRICE_BUFFER

        logging.info(
            f"💰 PROFIT TARGET HIT — placing SELL | "
            f"avg={avg_cost:.6f}, "
            f"min_target={target_price:.6f}, "
            f"sell_price={sell_price:.6f}, "
            f"current={current_price:.6f}"
        )

        place_limit_sell_all(sell_price)

def check_and_reset_cycle():
    asset_balance = get_asset_balance(base_ccy)
    buy_order_id, _ = read_pending_order()
    sell_order_id, _ = read_pending_sell()

    try:
        open_orders = tradeAPI.get_order_list(instType="SPOT", instId=instId).get("data", [])
        has_live_order = any(
            o.get("state") in ("live", "partially_filled")
            for o in open_orders
        )
    except Exception as e:
        logging.warning(f"⚠ Could not fetch live orders: {e}")
        has_live_order = False

    if asset_balance < 0.00005 and not buy_order_id and not sell_order_id and not has_live_order:
        stage = read_buy_stage()
        if stage > 0:
            logging.info(f"🔁 Balance={asset_balance:.8f}, no open orders — resetting cycle.")
            write_buy_stage(0)
            clear_pending_order()
            clear_pending_sell()
            reset_last_buy_price()
            reset_cycle_stats()

            send_trade_email(
                f"[{instId}] Cycle Reset Triggered",
                f"""Dear User,

{base_ccy} balance is {asset_balance:.8f}.
No open BUY or SELL orders detected.

Cycle has been reset and the bot is ready to start again.
"""
            )

# =======================
# Main
# =======================
def main():
    logging.info(f"🚀 Starting OKX Dip-Buy Bot (instId={instId}, bar={interval_okx})...")

    if not os.path.exists(stage_file):
        write_buy_stage(0)
        reset_last_buy_price()
        reset_cycle_stats()
        logging.info("First run: buy_stage set to 0, cycle stats reset.")

    while True:
        try:
            # 1) Highest priority: did our PROFIT SELL fill?
            if check_sell_filled():
                send_weekly_report()
                sleep_until_next_half_hour()
                continue

            # 2) Place profit SELL if target hit
            check_and_place_profit_sell()

            # 3) Reset logic + stage
            check_and_reset_cycle()
            current_stage = read_buy_stage()

            if current_stage >= len(portions):
                logging.info(f"✅ All {len(portions)} buy stages completed. Bot stopping buys.")
                stats = read_cycle_stats()
                if stats["total_btc"] > 0:
                    avg_cost = stats["total_spent"] / stats["total_btc"]
                    logging.info(
                        f"📊 Cycle summary — Total {base_ccy}: {stats['total_btc']:.8f}, "
                        f"Total USDT: {stats['total_spent']:.2f}, Avg Cost: {avg_cost:.6f}"
                    )
                send_weekly_report()
                sleep_until_next_half_hour()
                continue

            # 4) Handle pending BUY
            order_id, _ = read_pending_order()
            if order_id:
                is_filled, filled_qty, avg_price = check_order_status(order_id)
                if is_filled and filled_qty > 0:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_trade(now, "BUY", avg_price, filled_qty)

                    write_last_buy_price(avg_price)

                    stats = read_cycle_stats()
                    stats["total_spent"] += avg_price * filled_qty
                    stats["total_btc"] += filled_qty
                    write_cycle_stats(stats["total_spent"], stats["total_btc"])

                    avg_cost = stats["total_spent"] / stats["total_btc"]

                    write_buy_stage(current_stage + 1)
                    clear_pending_order()

                    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    send_trade_email(
                        f"[{instId}] LIMIT BUY FILLED @ ${avg_price:,.6f}",
                        f"""Dear User,

A LIMIT BUY order has been filled:
- Instrument: {instId}
- Price:      ${avg_price:,.6f}
- Quantity:   {filled_qty} {base_ccy}
- Time:       {now_utc}

Cycle stats:
- Total {base_ccy} this cycle: {stats["total_btc"]:.8f}
- Total USDT spent this cycle: ${stats["total_spent"]:.2f}
- Average cost this cycle:     ${avg_cost:,.6f}
"""
                    )

                    logging.info("Stage advanced, skipping new buy check this cycle.")
                    send_weekly_report()
                    sleep_until_next_half_hour()
                    continue

                logging.info(f"⏳ Pending limit order {order_id} still open.")
                send_weekly_report()
                sleep_until_next_half_hour()
                continue

            # 5) No pending BUY → evaluate a new BUY
            amount_to_buy = total_usdt * portions[current_stage]
            df = get_klines_okx(instId, interval_okx, limit=100)
            price = df["close"].iloc[-1]
            signals = check_signals(df)

            bullish_score = sum(
                w for sig, w in signals
                if "rsi_oversold" in sig
                or "kijun_dip_bullish" in sig
                or "macd_hist_positive_or_improving" in sig
                or "volume_surge" in sig
            )
            logging.info(f"📈 Current Price: {price:.6f} | Dip score: {bullish_score} | Signals: {signals}")

            last_px = read_last_buy_price()
            if last_px is not None and price >= last_px:
                logging.info(f"⛔ SKIP BUY: price {price:.6f} >= last buy {last_px:.6f}")
                send_weekly_report()
                sleep_until_next_half_hour()
                continue

            if bullish_score >= 2.5:
                logging.info(f"🟢 Strong DIP-BUY stage {current_stage+1} — placing LIMIT buy (${amount_to_buy:.2f})")
                place_limit_buy(price, amount_to_buy)
            else:
                logging.info("🤝 HOLD — dip conditions not strong enough.")

            send_weekly_report()
            sleep_until_next_half_hour()

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()

