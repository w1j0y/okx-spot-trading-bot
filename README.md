# Open-Source OKX Spot Dip-Buying Trading Bot

This repository contains the exact trading execution bot I personally run on OKX Spot.

It is open-sourced for transparency, auditability, and trust — so anyone can review how trades are executed, how risk is managed, and how capital is allocated.

**Disclaimer:**  
This project is not financial advice. You are fully responsible for your own risk, configuration, and capital.

## What This Bot Does

- Trades **SPOT markets only** on OKX  
- Uses your own OKX API key, API secret, and passphrase  
- Executes **limit BUY and limit SELL orders only**  
- No leverage  
- No futures  
- No martingale  
- No hidden logic  

This bot is designed for **low-risk, disciplined execution**, not aggressive or high-frequency trading.

## Strategy Overview (How the Bot Works)

### Capital Allocation (Staged Buying)

You define:

- The total USDT amount you want to invest  
- The trading pair (for example: `BTC-USDT`, `AVAX-USDT`, `SOL-USDT`)  

The bot never invests all capital at once.

Instead, it buys in **multiple predefined portions (laddered entries)**.  
This allows the bot to:

- Survive deeper market pullbacks  
- Lower the average entry price  
- Reduce emotional decision-making  

### BUY Conditions (Dip-Buying Logic)

A BUY is considered only when **multiple conditions align**, including:

- RSI oversold conditions  
- Price pulling back toward Ichimoku support (Kijun / Kumo context)  
- MACD momentum stabilizing or improving  
- Volume confirmation  
- Strong red (dip) candles  

**Important rule:**  
After the first BUY, every next BUY **must be at a lower price than the previous filled BUY**.

This guarantees:

- No chasing price upward  
- Every additional buy improves the average cost  
- Capital is used only on real dips  

### Limit BUY Orders (No Market Orders)

When a BUY signal is valid:

- The bot places a **LIMIT BUY**  
- The limit price is set **below the current market price**  
- If price does not dip, the order simply does not fill  

This avoids slippage and emotional entries.

### Cycle-Based Accounting

For each cycle, the bot tracks:

- Total USDT spent  
- Total asset accumulated  
- Average cost of the position  

Once all BUY stages are completed:

- The bot stops buying  
- It waits for a profitable exit  

### SELL Logic (Low-Risk Profit Taking)

When price rises above the average buy cost by a small, predefined margin:

- The bot places a **LIMIT SELL** for the full position  

After the SELL fills:

- The cycle is fully reset  
- The bot waits for the next dip cycle  

This approach favors low risk, high probability, and steady returns.

This strategy does **not** aim for large profits per trade.  
It is designed for **low, steady, repeatable gains**.

## Risk Profile

This bot is intentionally designed to be:

- Low risk  
- Conservative  
- Capital-preserving  

Trade-offs:

- No “get rich quick” behavior  
- No aggressive leverage  
- No unrealistic returns  

Low risk results in **low but steady income**.  
This is a feature, not a flaw.

## Email Notifications (Optional)

The bot can send email notifications for:

- Filled BUY orders  
- Filled SELL orders  
- Weekly reports (CSV attached)  

### Email Security Note

If you enable email notifications:

- Do **not** use your normal email password  
- Create an **App Password** (Gmail, Proton, etc.)  
- Use that app-specific password in `config.json`  

Email notifications are **disabled by default**.

## Installation and Usage

### Clone the Repository

```
git clone https://github.com/w1j0y/okx-spot-trading-bot.git
cd okx-spot-trading-bot
```

### Create a Virtual Environment (Recommended)

```
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Configure the Bot

```
cp config.json.example config.json
```

Edit `config.json` and set:

- Your OKX API key  
- API secret  
- Passphrase  
- Trading pair  
- Total USDT amount  

**Important:**  
Never share your API keys.  
Use **SPOT permissions only**.
Disable **withdrawal permissions**.
**IP whitelist** recommended.

### Run the Bot

```
python3 okx_dip_bot_open.py
```

The bot will wait for the next candle close, evaluate market conditions, and place limit orders when appropriate.

## Who This Project Is For

This project is ideal if you:

- Want full transparency  
- Prefer conservative strategies  
- Understand basic crypto risks  
- Want execution discipline instead of emotions  

If you prefer **hands-off deployment, monitoring, updates, and support**, a managed Telegram version is available.

## Telegram Onboarding and API Key Encryption (Managed Version)

In addition to the open-source version provided in this repository, a **managed Telegram onboarding flow** is available.

When users onboard via Telegram:

- API keys are never stored in plain text  
- The user sets a private password during registration  
- All sensitive credentials (API key, secret, passphrase) are encrypted using strong symmetric encryption (Fernet / PBKDF2)  
- Credentials are stored only in encrypted form  
- The deployed trading bot cannot decrypt credentials without the user’s password  

As a result:

- The generated `config.json` file does not reveal any secrets  
- Even if someone accesses the server or files, credentials remain protected  
- Only the user can decrypt and activate the bot  

The open-source version in this repository uses **plain-text configuration by design**, so it can be easily audited and run manually by advanced users.

## Support, Contact, and Referral (Optional)

If you find this project useful and want to support development:

- OKX referral link (optional):  
  https://www.okx.com/join/50798543  

For onboarding or managed deployment:

- Telegram: @w1j0y  
- Email: contact@rycron.com 

Nothing is forced.  
No hidden fees.  
No locked logic.

## Final Notes

- This code is open so you can verify exactly how trades are executed  
- You are free to study, modify, and run it yourself  
- You are also free to contact me if you prefer a managed setup  

Transparency first.  
Discipline over hype.

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
