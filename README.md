# Open-Source OKX Spot Dip-Buying Trading Bot v2

This repository contains the exact trading execution bot I personally run on OKX Spot.

It is open-sourced for **transparency, auditability, and trust** — so anyone can review how trades are executed, how risk is managed, and how capital is allocated.

## ⚠️ Disclaimer

This project is **not financial advice**. You are fully responsible for your own risk, configuration, and capital.

---

## What This Bot Does

- Trades **SPOT markets only** on OKX
- Uses your own OKX API key, API secret, and passphrase
- Executes **limit BUY and limit SELL orders only**
- **No leverage**
- **No futures**
- **No martingale**
- **No hidden logic**

This bot is designed for **low-risk, disciplined execution**, not aggressive or high-frequency trading.

---

## Strategy Overview (How the Bot Works)

### Capital Allocation (Staged Buying)

You define:
- The total quote currency amount you want to invest (USDT, USDC, etc.)
- The trading pair (for example: `BTC-USDT`, `ETH-USDC`, `SOL-USDT`)

The bot **never invests all capital at once**.

Instead, it buys in **10 predefined portions** (laddered entries):

| Stage | Portion |
|-------|---------|
| 1 | 4% |
| 2 | 6% |
| 3 | 7% |
| 4 | 8% |
| 5 | 9% |
| 6 | 11% |
| 7 | 12% |
| 8 | 13% |
| 9 | 14% |
| 10 | 16% |

This allows the bot to:
- Survive deeper market pullbacks
- Lower the average entry price progressively
- Reduce emotional decision-making

---

### BUY Conditions (Dip-Buying Logic)

A BUY is considered only when **multiple conditions align**, including:

- RSI oversold conditions (with extra weight for deeply oversold < 30)
- Price pulling back toward Ichimoku support (Kijun / Kumo context)
- MACD momentum stabilizing or improving
- Volume confirmation (surge or capitulation)
- Strong red (dip) candles
- Bullish engulfing patterns (recovery signals)

#### Progressive Requirements (NEW in v2)

The bot uses **progressive drop requirements** — later stages require larger price drops before buying:

| Stage | Required Drop | Required Signal Score |
|-------|---------------|----------------------|
| 0 | None (first buy) | ≥ 2.5 |
| 1-2 | ≥ 2% from last fill | ≥ 2.5 |
| 3-5 | ≥ 3-4% from last fill | ≥ 3.5 |
| 6-9 | ≥ 4-5% from last fill | ≥ 4.5 |

**This guarantees:**
- No chasing price upward
- Every additional buy improves the average cost
- Capital is used only on **real dips**
- Later buys require **stronger confirmation**

---

### Crash Detection Mode (NEW in v2)

The bot automatically detects **market crashes** and becomes more conservative:

**Crash Mode Activates When:**
- Price drops **≥ 10%** from the 24-hour high
- AND price has **not recovered ≥ 5%** from the low

**When Crash Mode is Active:**
- All drop requirements increase by **+2%**
- All signal score requirements increase by **+1.0**

| Stage | Normal Mode | Crash Mode |
|-------|-------------|------------|
| 0 | Score ≥ 2.5 | Score ≥ 3.5 |
| 1-2 | 2% drop, Score ≥ 2.5 | 4% drop, Score ≥ 3.5 |
| 3-5 | 3-4% drop, Score ≥ 3.5 | 5-6% drop, Score ≥ 4.5 |
| 6-9 | 4-5% drop, Score ≥ 4.5 | 6-7% drop, Score ≥ 5.5 |

**This helps:**
- Preserve capital during sustained crashes
- Avoid buying every small bounce in a waterfall decline
- Accumulate at much lower prices during real crashes

**Crash Mode Deactivates When:**
- Price recovers **≥ 5%** from the recent low

---

### Limit BUY Orders (No Market Orders)

When a BUY signal is valid:
- The bot places a **LIMIT BUY**
- The limit price is set **0.5% below** the current market price
- If price does not dip, the order simply does not fill

This avoids slippage and emotional entries.

---

### Cycle-Based Accounting

For each cycle, the bot tracks:
- Total quote currency spent (USDT/USDC/etc.)
- Total asset accumulated
- Average cost of the position

Once all BUY stages are completed:
- The bot stops buying
- It waits for a profitable exit

---

### SELL Logic (Low-Risk Profit Taking)

When price rises above the average buy cost by **0.25%**:
- The bot places a **LIMIT SELL** for the full position
- SELL monitoring uses **5-minute candles** for faster exits

After the SELL fills:
- The cycle is fully reset
- The bot waits for the next dip cycle

**This approach favors low risk, high probability, and steady returns.**

This strategy does not aim for large profits per trade.
It is designed for **low, steady, repeatable gains**.

---

## Latest Features (v2)

| Feature | Description |
|---------|-------------|
| **Crash Detection** | Automatically detects 10%+ drops and becomes more conservative |
| **Progressive Drop Requirements** | Later stages require larger price drops (2% → 5%) |
| **Progressive Signal Requirements** | Later stages require stronger signals (2.5 → 4.5 score) |
| **Multi-Currency Support** | Works with any quote currency (USDT, USDC, etc.) |
| **Enhanced Signals** | Added deep RSI, volume capitulation, bullish engulfing detection |
| **SELL on 5m Candles** | Faster profit-taking during quick bounces |
| **First BUY Auto-Cancel** | Cancels after 3 days if unfilled to avoid being stuck |
| **Full Cycle Reset** | Clean state management including crash mode |

---

## Risk Profile

This bot is intentionally designed to be:
- **Low risk**
- **Conservative**
- **Capital-preserving**

**Trade-offs:**
- No "get rich quick" behavior
- No aggressive leverage
- No unrealistic returns

**Low risk results in low but steady income.**
This is a feature, not a flaw.

---

## Email Notifications (Optional)

The bot can send email notifications for:
- Filled BUY orders
- Filled SELL orders
- Crash Mode activation/deactivation
- Weekly reports (CSV attached)

### Email Security Note

If you enable email notifications:
- **Do not use your normal email password**
- Create an **App Password** (Gmail, Proton, etc.)
- Use that app-specific password in `config.json`

Email notifications are **disabled by default**.

---

## Installation and Usage

### Clone the Repository

```bash
git clone https://github.com/w1j0y/okx-spot-trading-bot.git
cd okx-spot-trading-bot
```

### Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure the Bot

```bash
cp config.json.example config.json
```

Edit `config.json` and set:
- Your OKX API key
- API secret
- Passphrase
- Trading pair
- Total quote amount

**Example `config.json`:**

```json
{
  "okx": {
    "api_key": "your-api-key",
    "api_secret": "your-api-secret",
    "passphrase": "your-passphrase"
  },
  "trade": {
    "instId": "BTC-USDT",
    "total_usdt": 1000,
    "interval_okx": "30m",
    "use_demo": false
  },
  "email": {
    "enabled": false,
    "sender_email": "",
    "email_password": "",
    "recipient_email": ""
  }
}
```

**Important:**
- Never share your API keys
- Use **SPOT permissions only**
- **Disable withdrawal permissions**
- IP whitelist recommended

### Run the Bot

```bash
python3 okx_dip_bot_v2.py
```

The bot will:
1. Wait for valid dip conditions
2. Place limit orders when appropriate
3. Monitor for profit targets
4. Auto-reset cycles after successful sells

---

## Configuration Parameters

### Strategy Parameters (Hardcoded)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `LIMIT_PRICE_OFFSET` | 0.995 | Buy limit placed 0.5% below market |
| `PROFIT_TARGET_PCT` | 0.0025 | 0.25% profit target for sells |
| `SELL_CHECK_INTERVAL` | 300s | Check sell conditions every 5 minutes |
| `CRASH_THRESHOLD_PCT` | 0.10 | 10% drop triggers crash mode |
| `CRASH_MODE_EXTRA_DROP` | 0.02 | Extra 2% drop required in crash mode |
| `CRASH_MODE_EXTRA_SCORE` | 1.0 | Extra signal score in crash mode |
| `CRASH_RECOVERY_PCT` | 0.05 | 5% recovery exits crash mode |

### Timeframes

| Function | Timeframe |
|----------|-----------|
| Buy signal analysis | 30-minute candles |
| Sell monitoring | 5-minute candles |
| Crash detection | 24-hour window (48 × 30m bars) |

---

## Log Files

The bot creates these files in the same directory:

| File | Description |
|------|-------------|
| `bot.log` | Full execution log |
| `trade_log.csv` | All executed trades |
| `buy_stage.txt` | Current buy stage (0-9) |
| `pending_order.txt` | Pending buy order ID |
| `pending_sell_order.txt` | Pending sell order ID |
| `last_buy_price.txt` | Last filled buy price |
| `cycle_stats.json` | Current cycle statistics |
| `crash_mode.json` | Crash mode state |

---

## Who This Project Is For

This project is ideal if you:
- Want **full transparency**
- Prefer **conservative strategies**
- Understand basic crypto risks
- Want **execution discipline** instead of emotions

If you prefer hands-off deployment, monitoring, updates, and support, a managed Telegram version is available.

---

## Support, Contact, and Referral (Optional)

If you find this project useful and want to support development:

**OKX referral link (optional):**
https://www.okx.com/join/50798543

**For onboarding or managed deployment:**
- Telegram: @w1j0y
- Email: contact@rycron.com

Nothing is forced. No hidden fees. No locked logic.

---

## Final Notes

- This code is open so you can **verify exactly** how trades are executed
- You are free to study, modify, and run it yourself
- You are also free to contact me if you prefer a managed setup

**Transparency first. Discipline over hype.**

---

## Changelog

### v2.0.0 (2026-02)
- Added **Crash Detection Mode** (10% drop triggers conservative mode)
- Added **Progressive Drop Requirements** (2% → 5% based on stage)
- Added **Progressive Signal Requirements** (2.5 → 4.5 based on stage)
- Added **Multi-Currency Support** (USDT, USDC, any quote currency)
- Enhanced signal detection (deep RSI, volume capitulation, bullish engulfing)
- Improved cycle reset with crash mode clearing
- Better logging with mode indicators

### v1.0.0 (Initial Release)
- Basic dip-buying with fixed thresholds
- 10-stage laddered buying
- 0.25% profit taking
- Email notifications

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
