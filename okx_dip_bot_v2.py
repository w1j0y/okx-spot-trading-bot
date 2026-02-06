#!/usr/bin/env python3
# Ichimoku OKX Dip-Buy Bot v2 (Open Source Edition)
# - Plain-text config.json (no Fernet/KDF)
# - Supports any SPOT pair via config instId (USDT, USDC, etc.)
# - Limit buy ladder + progressive drop requirements
# - Crash detection + capital preservation mode
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
crash_mode_file  = os.path.join(data_dir, "crash_mode.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(botlog_file)]
)

# =======================
# Static Strategy Params
# =======================
portions = [0.04, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12, 0.13, 0.14, 0.16]
LIMIT_PRICE_OFFSET = 0.995  # Place limit buy 0.5% below market

# Profit sell settings
PROFIT_TARGET_PCT = 0.0025    # 0.25% over avg cost
SELL_PRICE_BUFFER = 1.0005
SELL_INTERVAL_OKX = "5m"
SELL_CHECK_INTERVAL = 300     # Check sell every 5 minutes

# Progressive drop requirements per stage
STAGE_DROP_REQUIREMENTS = {
    0: 0.00,   # First buy - no drop required
    1: 0.02,   # 2% drop from stage 0 fill
    2: 0.02,   # 2% drop from stage 1 fill
    3: 0.03,   # 3% drop from stage 2 fill
    4: 0.03,   # 3% drop from stage 3 fill
    5: 0.04,   # 4% drop from stage 4 fill
    6: 0.04,   # 4% drop from stage 5 fill
    7: 0.05,   # 5% drop from stage 6 fill
    8: 0.05,   # 5% drop from stage 7 fill
    9: 0.05,   # 5% drop from stage 8 fill
}

# Progressive signal strength requirements per stage
STAGE_SIGNAL_REQUIREMENTS = {
    0: 2.5,    # Stages 0-2: standard threshold
    1: 2.5,
    2: 2.5,
    3: 3.5,    # Stages 3-5: moderate threshold
    4: 3.5,
    5: 3.5,
    6: 4.5,    # Stages 6-9: strict threshold
    7: 4.5,
    8: 4.5,
    9: 4.5,
}

# Crash detection parameters
CRASH_DETECTION_WINDOW = 48   # Number of 30-min bars (24 hours)
CRASH_THRESHOLD_PCT = 0.10    # 10% drop triggers crash mode
CRASH_MODE_EXTRA_DROP = 0.02  # Extra 2% drop required in crash mode
CRASH_MODE_EXTRA_SCORE = 1.0  # Extra signal score required in crash mode
CRASH_RECOVERY_PCT = 0.05     # 5% recovery from low exits crash mode

# Pending order log interval
PENDING_LOG_INTERVAL = 600    # 10 minutes

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

    total_quote = trade.get("total_usdt") or trade.get("total_quote")
    if total_quote is None:
        raise SystemExit("config.json missing trade.total_usdt (or trade.total_quote)")
    try:
        total_quote = float(total_quote)
    except Exception:
        raise SystemExit("trade.total_usdt must be numeric")

    interval_okx = str(trade.get("interval_okx") or "30m").strip()
    use_demo = bool(trade.get("use_demo", False))

    email_enabled = bool(email.get("enabled", False))
    sender_email = email.get("sender_email")
    email_password = email.get("email_password")
    recipient_email = email.get("recipient_email")

    # Extract base and quote currencies
    base_ccy = instId.split("-")[0]
    quote_ccy = instId.split("-")[1] if "-" in instId else "USDT"

    return {
        "okx": {"api_key": api_key, "api_secret": api_secret, "passphrase": passphrase},
        "trade": {
            "instId": instId,
            "base_ccy": base_ccy,
            "quote_ccy": quote_ccy,
            "total_quote": total_quote,
            "interval_okx": interval_okx,
            "use_demo": use_demo
        },
        "email": {
            "enabled": email_enabled,
            "sender_email": sender_email,
            "email_password": email_password,
            "recipient_email": recipient_email
        },
    }

cfg = load_config(CONFIG_PATH)

instId       = cfg["trade"]["instId"]
base_ccy     = cfg["trade"]["base_ccy"]
quote_ccy    = cfg["trade"]["quote_ccy"]
total_quote  = cfg["trade"]["total_quote"]
interval_okx = cfg["trade"]["interval_okx"]
use_demo     = cfg["trade"]["use_demo"]

logging.info(f"🔧 Loaded config: instId={instId}, base={base_ccy}, quote={quote_ccy}, total={total_quote}, interval={interval_okx}, demo={use_demo}")

# =======================
# Email
# =======================
EMAIL_ENABLED   = cfg["email"]["enabled"]
SENDER_EMAIL    = cfg["email"]["sender_email"]
EMAIL_PASSWORD  = cfg["email"]["email_password"]
RECIPIENT_EMAIL = cfg["email"]["recipient_email"]


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
        logging.warning(f"Instrument info error for {instId_}: {e}. Fallback values used.")
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
# Crash Mode Functions
# =======================
def detect_crash_mode(df: pd.DataFrame) -> dict:
    """
    Detect if we're in a market crash based on recent price action.
    Returns dict with crash status and relevant metrics.
    """
    if len(df) < CRASH_DETECTION_WINDOW:
        return {"is_crash": False, "drop_pct": 0, "recent_high": 0, "recent_low": 0}
    
    recent_df = df.tail(CRASH_DETECTION_WINDOW)
    recent_high = recent_df['high'].max()
    recent_low = recent_df['low'].min()
    current_price = df['close'].iloc[-1]
    
    drop_from_high = (recent_high - current_price) / recent_high
    recovery_from_low = (current_price - recent_low) / recent_low if recent_low > 0 else 0
    
    is_crash = drop_from_high >= CRASH_THRESHOLD_PCT and recovery_from_low < CRASH_RECOVERY_PCT
    
    return {
        "is_crash": is_crash,
        "drop_pct": drop_from_high,
        "recovery_pct": recovery_from_low,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "current_price": current_price
    }


def read_crash_mode() -> dict:
    """Read crash mode state from file."""
    try:
        with open(crash_mode_file, "r") as f:
            return json.load(f)
    except:
        return {"active": False, "entered_at": None, "entry_price": None}


def write_crash_mode(active: bool, entry_price: float = None):
    """Write crash mode state to file."""
    with open(crash_mode_file, "w") as f:
        json.dump({
            "active": active,
            "entered_at": datetime.now().isoformat() if active else None,
            "entry_price": entry_price
        }, f, indent=2)


def clear_crash_mode():
    """Clear crash mode file."""
    if os.path.exists(crash_mode_file):
        os.remove(crash_mode_file)


def get_required_drop_for_stage(stage: int, crash_mode_active: bool) -> float:
    """
    Get the required percentage drop from last fill for a given stage.
    Adds extra drop requirement if in crash mode.
    """
    base_drop = STAGE_DROP_REQUIREMENTS.get(stage, 0.05)
    
    if crash_mode_active and stage > 0:
        return base_drop + CRASH_MODE_EXTRA_DROP
    
    return base_drop


def get_required_score_for_stage(stage: int, crash_mode_active: bool) -> float:
    """
    Get the required bullish score for a given stage.
    Adds extra score requirement if in crash mode.
    """
    base_score = STAGE_SIGNAL_REQUIREMENTS.get(stage, 4.5)
    
    if crash_mode_active:
        return base_score + CRASH_MODE_EXTRA_SCORE
    
    return base_score


def check_buy_conditions(current_stage: int, current_price: float, 
                         last_fill_price: float, bullish_score: float,
                         crash_info: dict) -> tuple:
    """
    Check if all buy conditions are met for the current stage.
    
    Returns:
        tuple: (can_buy: bool, reason: str)
    """
    crash_mode_active = crash_info.get("is_crash", False)
    
    required_drop = get_required_drop_for_stage(current_stage, crash_mode_active)
    required_score = get_required_score_for_stage(current_stage, crash_mode_active)
    
    # Stage 0: Only check signal strength
    if current_stage == 0:
        if bullish_score >= required_score:
            return True, f"Stage 0: Score {bullish_score:.1f} >= {required_score:.1f}"
        else:
            return False, f"Stage 0: Score {bullish_score:.1f} < {required_score:.1f} required"
    
    # Stages 1+: Check both drop AND signal strength
    if last_fill_price is None or last_fill_price <= 0:
        return False, "No last fill price recorded"
    
    actual_drop = (last_fill_price - current_price) / last_fill_price
    
    # Check drop requirement
    if actual_drop < required_drop:
        drop_needed = last_fill_price * (1 - required_drop)
        return False, (
            f"Stage {current_stage}: Price {current_price:.2f} needs to drop to "
            f"{drop_needed:.2f} ({required_drop*100:.1f}% below last fill {last_fill_price:.2f}). "
            f"Current drop: {actual_drop*100:.2f}%"
        )
    
    # Check signal strength
    if bullish_score < required_score:
        return False, (
            f"Stage {current_stage}: Drop OK ({actual_drop*100:.2f}%), but score "
            f"{bullish_score:.1f} < {required_score:.1f} required"
        )
    
    # All conditions met
    mode_str = " [CRASH MODE]" if crash_mode_active else ""
    return True, (
        f"Stage {current_stage}{mode_str}: All conditions met - "
        f"Drop {actual_drop*100:.2f}% >= {required_drop*100:.1f}%, "
        f"Score {bullish_score:.1f} >= {required_score:.1f}"
    )


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
    """
    Calculate technical signals for buy decisions.
    Returns list of (signal_name, weight) tuples.
    """
    tenkan, kijun, span_a, span_b, chikou = calculate_ichimoku(df)

    price = df["close"].iloc[-2]
    volume = df["volume"].iloc[-2]
    avg_volume = df["volume"].rolling(window=40).mean().iloc[-2]
    rsi = talib.RSI(df["close"], timeperiod=28).iloc[-2]
    macd, signal_line, hist = calculate_macd(df)

    signals = []

    # RSI Oversold (stronger weight for deeper oversold)
    if rsi < 30:
        signals.append(("rsi_deeply_oversold", 3.0))
        logging.info(f"✔ RSI Deeply Oversold ({rsi:.2f} < 30)")
    elif rsi < 40:
        signals.append(("rsi_oversold", 2.0))
        logging.info(f"✔ RSI Oversold ({rsi:.2f} < 40)")

    # Price at or below Kijun, above Kumo
    kumo_top = max(span_a.iloc[-2], span_b.iloc[-2])
    kumo_bottom = min(span_a.iloc[-2], span_b.iloc[-2])
    
    if price <= kijun.iloc[-2] * 1.005 and price > kumo_bottom * 0.98:
        signals.append(("kijun_dip_bullish", 1.5))
        logging.info("✔ Price at/below Kijun and above Kumo (Dip in Bullish Context)")
    
    # Price inside or below Kumo (deeper dip)
    if price <= kumo_bottom:
        signals.append(("below_kumo", 1.0))
        logging.info("✔ Price below Kumo (Deep dip territory)")

    # MACD Histogram Positive or Improving
    if len(hist) > 3:
        if hist.iloc[-2] > 0:
            signals.append(("macd_hist_positive", 1.5))
            logging.info(f"✔ MACD Histogram Positive (Hist: {hist.iloc[-2]:.4f})")
        elif hist.iloc[-2] > hist.iloc[-3]:
            signals.append(("macd_hist_improving", 1.0))
            logging.info(f"✔ MACD Histogram Improving (Hist: {hist.iloc[-2]:.4f} > {hist.iloc[-3]:.4f})")

    # Volume Surge
    if volume > avg_volume * 1.5:
        signals.append(("volume_surge", 1.0))
        logging.info(f"✔ Volume Surge ({volume:.2f} > 1.5x avg {avg_volume:.2f})")
    
    # Extreme volume (potential capitulation)
    if volume > avg_volume * 3.0:
        signals.append(("volume_capitulation", 1.5))
        logging.info(f"✔ Extreme Volume - Possible Capitulation ({volume:.2f} > 3x avg)")

    # Rapid dip in last closed bar
    last_close = df["close"].iloc[-2]
    last_open = df["open"].iloc[-2]
    drop_pct = (last_close - last_open) / last_open
    
    if drop_pct <= -0.015:
        signals.append(("rapid_dip_bar_severe", 1.5))
        logging.info(f"✔ Severe rapid dip bar detected ({drop_pct*100:.2f}%)")
    elif drop_pct <= -0.008:
        signals.append(("rapid_dip_bar", 1.0))
        logging.info(f"✔ Rapid dip bar detected ({drop_pct*100:.2f}%)")
    
    # Bullish engulfing pattern (recovery signal)
    if len(df) >= 3:
        prev_close = df["close"].iloc[-3]
        prev_open = df["open"].iloc[-3]
        if prev_close < prev_open:  # Previous bar was red
            if last_close > last_open and last_close > prev_open and last_open < prev_close:
                signals.append(("bullish_engulfing", 1.5))
                logging.info("✔ Bullish Engulfing Pattern detected")

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
    total_invested = sum([total_quote * p for p in portions[:n_buys]])
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
        stats = read_cycle_stats()
        avg_cost = stats["total_spent"] / stats["total_asset"] if stats["total_asset"] > 0 else 0
        
        body = f"""Dear User,

Here is your weekly report:

Instrument: {instId}
Total buys this cycle: {n_buys}
Total {base_ccy}: {total_asset:.8f}
Total {quote_ccy} invested (approx): ${total_invested:.2f}
Average cost this cycle: ${avg_cost:.2f} per {base_ccy}

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
        return {"total_spent": 0.0, "total_asset": 0.0}


def write_cycle_stats(total_spent: float, total_asset: float):
    with open(cycle_stats_file, "w") as f:
        json.dump({"total_spent": total_spent, "total_asset": total_asset}, f, indent=2)


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


def get_quote_balance() -> float:
    """Get balance of quote currency (USDT, USDC, etc.)"""
    data = _account_details()
    if not data:
        return 0.0
    for d in data[0].get("details", []):
        if d.get("ccy") == quote_ccy:
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
# Full Cycle Reset
# =======================
def full_cycle_reset():
    """Complete cycle reset - called when SELL fills or manual reset needed."""
    cancel_all_open_orders_for_inst(instId)
    clear_pending_order()
    clear_pending_sell()
    reset_last_buy_price()
    reset_cycle_stats()
    clear_crash_mode()
    write_buy_stage(0)
    logging.info("🔁 Full cycle reset completed")


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


def cancel_all_open_orders_for_inst(instId_: str):
    """Cancel ALL live / partially filled SPOT orders for the given instrument."""
    try:
        res = tradeAPI.get_order_list(instType="SPOT", instId=instId_)
        orders = res.get("data", [])

        for o in orders:
            if o.get("state") in ("live", "partially_filled"):
                ord_id = o.get("ordId")
                tradeAPI.cancel_order(instId=instId_, ordId=ord_id)
                logging.info(f"🛑 Cancelled leftover order {ord_id} for {instId_}")

    except Exception as e:
        logging.error(f"Failed cancelling open orders for {instId_}: {e}")


def cancel_pending_order():
    order_id, _ = read_pending_order()
    if not order_id:
        return
    try:
        tradeAPI.cancel_order(instId=instId, ordId=order_id)
        logging.info(f"🛑 Cancelled pending limit order {order_id}")
        clear_pending_order()
    except Exception as e:
        logging.error(f"Error cancelling order {order_id}: {e}")


def cleanup_stale_pending_sell():
    order_id, _ = read_pending_sell()
    if not order_id:
        return

    try:
        res = tradeAPI.get_order(instId=instId, ordId=order_id)
        data = res.get("data", [{}])[0]

        side = data.get("side")
        state = data.get("state")

        if side != "sell" or state not in ("live", "partially_filled"):
            logging.warning(f"🧹 Clearing stale SELL order {order_id}")
            clear_pending_sell()

    except Exception:
        clear_pending_sell()


# =======================
# Order actions
# =======================
def check_order_status(order_id: str):
    try:
        res = tradeAPI.get_order(instId=instId, ordId=order_id)
        data = res.get("data", [{}])[0]
        state = data.get("state")
        filled_qty = float(data.get("accFillSz") or data.get("fillSz") or 0.0)
        avg_price = float(data.get("avgPx") or 0.0)
        if state == "filled":
            return True, filled_qty, avg_price
        if state in ["live", "partially_filled"]:
            return False, 0.0, 0.0
        return False, 0.0, 0.0
    except Exception as e:
        logging.error(f"Error checking order {order_id}: {e}")
        return False, 0.0, 0.0


def place_limit_buy(price: float, amount_quote: float):
    """Place a limit buy order. amount_quote is in quote currency (USDT/USDC/etc.)"""
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    quote_bal = get_quote_balance()
    if quote_bal < amount_quote:
        logging.warning(f"⚠ Not enough {quote_ccy} to place BUY order. Have: {quote_bal:.2f}, Need: {amount_quote:.2f}")
        return 0.0, None

    limit_price = quantize_price(price * LIMIT_PRICE_OFFSET)
    qty = quantize_size(amount_quote / limit_price)
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
        logging.info(f"🟢 LIMIT BUY placed @ {limit_price:.6f} for {qty} {base_ccy} ({quote_ccy}: {amount_quote})")
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

            # Full cycle reset
            full_cycle_reset()

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
    cleanup_stale_pending_sell()
    sell_order_id, _ = read_pending_sell()
    if sell_order_id:
        logging.info(f"🟠 SELL blocked: pending SELL ordId exists ({sell_order_id})")
        return

    stats = read_cycle_stats()
    if stats["total_asset"] <= 0:
        return

    avg_cost = stats["total_spent"] / stats["total_asset"]
    df = get_klines_okx(instId, SELL_INTERVAL_OKX, limit=2)
    current_price = df["close"].iloc[-1]

    target_price = avg_cost * (1 + PROFIT_TARGET_PCT)

    logging.info(
        f"🔎 SELL check: avg={avg_cost:.6f}, target={target_price:.6f}, current={current_price:.6f}"
    )

    if current_price >= target_price:
        cancel_pending_order()
        sell_price = current_price * SELL_PRICE_BUFFER

        logging.info(
            f"💰 PROFIT TARGET HIT — placing SELL | "
            f"avg={avg_cost:.6f}, target={target_price:.6f}, "
            f"current={current_price:.6f}, sell_price={sell_price:.6f}"
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

    if (
        asset_balance < 0.00005
        and not buy_order_id
        and not sell_order_id
        and not has_live_order
    ):
        stage = read_buy_stage()
        if stage >= len(portions):
            return
        if stage > 0:
            logging.info(
                f"🔁 {base_ccy} balance = {asset_balance:.8f}, "
                f"no open BUY/SELL orders — resetting cycle."
            )

            full_cycle_reset()

            send_trade_email(
                f"[{instId}] Cycle Reset Triggered",
                f"""Dear User,

{base_ccy} balance is {asset_balance:.8f}.
No open BUY or SELL orders detected.

The trading cycle has been reset and the bot is ready
to start a new dip-buy cycle.
"""
            )


# =======================
# Main
# =======================
def main():
    global last_sell_check
    global last_pending_log
    
    last_sell_check = 0
    last_pending_log = 0

    logging.info(f"🚀 Starting OKX Dip-Buy Bot v2 (instId={instId}, bar={interval_okx})...")
    logging.info("📋 Features: Crash Detection, Progressive Drop Requirements, Capital Preservation")

    cleanup_stale_pending_sell()

    if not os.path.exists(stage_file):
        write_buy_stage(0)
        reset_last_buy_price()
        reset_cycle_stats()
        clear_crash_mode()
        logging.info("First run: buy_stage set to 0, cycle stats reset.")

    while True:
        try:
            # =========================================================
            # 1) Highest priority: did our PROFIT SELL fill?
            # =========================================================
            if check_sell_filled():
                send_weekly_report()
                continue

            # =========================================================
            # 2) Place profit SELL if target hit (every 5 minutes)
            # =========================================================
            now = time.time()

            if now - last_sell_check >= SELL_CHECK_INTERVAL:
                logging.info("⏱️ 5m SELL watcher tick")
                cleanup_stale_pending_sell()
                check_and_place_profit_sell()
                last_sell_check = now

            # =========================================================
            # 3) Reset logic (SELL-aware) + read stage
            # =========================================================
            check_and_reset_cycle()
            current_stage = read_buy_stage()

            if current_stage >= len(portions):
                logging.info(f"✅ All {len(portions)} buy stages completed. Bot stopping buys.")
                stats = read_cycle_stats()
                if stats["total_asset"] > 0:
                    avg_cost = stats["total_spent"] / stats["total_asset"]
                    logging.info(
                        f"📊 Cycle summary — Total {base_ccy}: {stats['total_asset']:.8f}, "
                        f"Total {quote_ccy}: {stats['total_spent']:.2f}, "
                        f"Avg Cost: {avg_cost:.2f} {quote_ccy}/{base_ccy}"
                    )
                send_weekly_report()
                time.sleep(300)
                continue

            # =========================================================
            # 4) Handle pending BUY order
            # =========================================================
            order_id, order_timestamp = read_pending_order()

            # Cancel stale FIRST buy if open too long (3 days)
            if order_id and current_stage == 0 and order_timestamp:
                try:
                    order_time = datetime.strptime(order_timestamp, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                    age = datetime.now(timezone.utc) - order_time

                    if age >= timedelta(days=3):
                        logging.warning("⏰ First BUY order open > 3 days — cancelling and resetting cycle")

                        full_cycle_reset()

                        send_trade_email(
                            f"[{instId}] First BUY expired — cycle reset",
                            f"""Dear User,

The FIRST limit BUY order remained open for more than 3 days.

To avoid being stuck below market price, the cycle was reset:

- Pending BUY cancelled
- Cycle stats cleared
- Bot will wait for a new buying opportunity

This logic applies ONLY to the first buy stage.
"""
                        )

                        time.sleep(60)
                        continue

                except Exception as e:
                    logging.error(f"Error handling stale first BUY: {e}")

            if order_id:
                is_filled, filled_qty, avg_price = check_order_status(order_id)
                if is_filled:
                    time.sleep(1)
                    is_filled, filled_qty, avg_price = check_order_status(order_id)

                if is_filled and filled_qty > 0:
                    if filled_qty < minSz:
                        logging.warning(f"⚠ Dust execution detected ({filled_qty} {base_ccy}); waiting for full fill")
                        continue

                    logging.info(f"✅ LIMIT BUY ORDER filled @ {avg_price:.2f} for {filled_qty} {base_ccy}")
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    append_trade(now_ts, "BUY", avg_price, filled_qty)

                    write_last_buy_price(avg_price)

                    stats = read_cycle_stats()
                    stats["total_spent"] += avg_price * filled_qty
                    stats["total_asset"] += filled_qty
                    write_cycle_stats(stats["total_spent"], stats["total_asset"])

                    avg_cost = stats["total_spent"] / stats["total_asset"]
                    logging.info(
                        f"🎯 Updated cycle stats — Total {base_ccy}: {stats['total_asset']:.8f}, "
                        f"Total {quote_ccy}: {stats['total_spent']:.2f}, "
                        f"Avg Cost: {avg_cost:.2f} {quote_ccy}/{base_ccy}"
                    )

                    write_buy_stage(current_stage + 1)
                    clear_pending_order()

                    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    send_trade_email(
                        f"[{instId} Bot] LIMIT BUY Order Filled at ${avg_price:,.2f}",
                        f"""Dear User,

A LIMIT BUY order has been filled:
- Instrument: {instId}
- Side:       BUY
- Price:      ${avg_price:,.2f}
- Quantity:   {filled_qty} {base_ccy}
- Time:       {now_utc}

Cycle stats so far:
- Total {base_ccy} stacked this cycle: {stats["total_asset"]:.8f}
- Total {quote_ccy} spent this cycle: ${stats["total_spent"]:.2f}
- Average cost this cycle:      ${avg_cost:.2f} per {base_ccy}
"""
                    )

                    logging.info("Stage advanced, skipping new buy check this cycle.")
                    continue
                else:
                    now = time.time()
                    if now - last_pending_log >= PENDING_LOG_INTERVAL:
                        logging.info(f"⏳ Pending limit order {order_id} still open.")
                        send_weekly_report()
                        last_pending_log = now
                    time.sleep(30)
                    continue

            # =========================================================
            # 5) No pending BUY → Evaluate new BUY opportunity
            # =========================================================
            sell_order_id, _ = read_pending_sell()
            if sell_order_id:
                logging.info("🟠 BUY skipped: SELL order is active")
                time.sleep(60)
                continue

            # Get market data
            df = get_klines_okx(instId, interval_okx, limit=100)
            price = df["close"].iloc[-1]

            # Detect crash mode
            crash_info = detect_crash_mode(df)
            crash_state = read_crash_mode()

            # Update crash mode state
            if crash_info["is_crash"] and not crash_state["active"]:
                write_crash_mode(True, price)
                logging.warning(
                    f"🚨 CRASH MODE ACTIVATED - Price dropped {crash_info['drop_pct']*100:.1f}% "
                    f"from recent high {crash_info['recent_high']:.2f}"
                )
                send_trade_email(
                    f"[{instId}] ⚠️ CRASH MODE ACTIVATED",
                    f"""Dear User,

The bot has detected a significant market crash:

- Recent High: ${crash_info['recent_high']:,.2f}
- Current Price: ${price:,.2f}
- Drop: {crash_info['drop_pct']*100:.1f}%

CRASH MODE is now active. The bot will:
- Require larger drops between buys (+2% extra)
- Require stronger signals (+1.0 score extra)
- Be more conservative to preserve capital

This helps avoid buying too aggressively during a sustained crash.
"""
                )
            elif not crash_info["is_crash"] and crash_state["active"]:
                clear_crash_mode()
                logging.info(
                    f"✅ CRASH MODE DEACTIVATED - Market recovered {crash_info['recovery_pct']*100:.1f}% "
                    f"from low {crash_info['recent_low']:.2f}"
                )
                send_trade_email(
                    f"[{instId}] ✅ Crash Mode Deactivated",
                    f"""Dear User,

Market conditions have stabilized:

- Recent Low: ${crash_info['recent_low']:,.2f}
- Current Price: ${price:,.2f}
- Recovery: {crash_info['recovery_pct']*100:.1f}%

The bot has exited CRASH MODE and will return to normal buy parameters.
"""
                )

            # Log current state
            crash_mode_active = crash_info["is_crash"]
            mode_str = "🚨 CRASH MODE" if crash_mode_active else "📊 NORMAL MODE"
            logging.info(f"{mode_str} | Price: {price:.2f} | Stage: {current_stage}")

            if crash_mode_active:
                logging.info(
                    f"   Crash stats: {crash_info['drop_pct']*100:.1f}% from high, "
                    f"{crash_info['recovery_pct']*100:.1f}% recovery from low"
                )

            # Calculate signals
            signals = check_signals(df)
            bullish_score = sum(weight for sig, weight in signals)

            logging.info(f"📊 Signals: {signals}")
            logging.info(f"📈 Bullish Score: {bullish_score:.1f}")

            # Get requirements for current stage
            required_drop = get_required_drop_for_stage(current_stage, crash_mode_active)
            required_score = get_required_score_for_stage(current_stage, crash_mode_active)
            last_fill_price = read_last_buy_price()

            logging.info(
                f"📋 Stage {current_stage} Requirements: "
                f"Score >= {required_score:.1f}, "
                f"Drop >= {required_drop*100:.1f}% from last fill"
            )
            if last_fill_price:
                target_price = last_fill_price * (1 - required_drop)
                logging.info(f"   Last fill: {last_fill_price:.2f}, Target price: {target_price:.2f}")

            # Check all buy conditions
            can_buy, reason = check_buy_conditions(
                current_stage=current_stage,
                current_price=price,
                last_fill_price=last_fill_price,
                bullish_score=bullish_score,
                crash_info=crash_info
            )

            if can_buy:
                amount_to_buy = total_quote * portions[current_stage]
                logging.info(f"🟢 {reason}")
                logging.info(f"🟢 Placing LIMIT BUY for stage {current_stage + 1} (${amount_to_buy:.2f})")

                qty, order_id = place_limit_buy(price, amount_to_buy)
                if qty > 0 and order_id:
                    logging.info(f"🟢 LIMIT BUY placed (order {order_id})")
                else:
                    logging.warning("⚠ Failed to place limit BUY.")
            else:
                logging.info(f"🤝 HOLD — {reason}")

            send_weekly_report()
            time.sleep(60)

        except Exception as e:
            logging.error(f"Main loop error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
