# US Equity Portfolio Daily Tracker

Automated daily and cumulative performance tracker for a US equity portfolio, designed for GitHub Actions and zero-emoji Telegram HTML delivery.

## Key Features

- **Dual NAV Tracking (Buy & Hold)**: Tracks both total portfolio NAV (`Port`, including cash allocation and natural weight drift) and normalized equity-only NAV (`Eqty (100%)`) using total-return `Adj Close` ratios while displaying raw `Close` prices.
- **Benchmark & Alpha Attribution**: Computes 1D, MTD, and Cumulative returns against `SPY` and `QQQ`, along with active alpha (`Port - Benchmark`), max drawdown, 20D annualized volatility, and advancers/decliners breadth.
- **Zero-Emoji Mobile-Safe Format**: Renders a single Telegram HTML message (`< 4,096` chars) with pure-ASCII `<pre>` tables capped at `<= 36` characters per line, enforced by a runtime emoji guard.
- **NYSE Calendar & DST-Safe Scheduling**: Uses `exchange_calendars` (`XNYS`) to skip US market holidays/weekends and handle early closes automatically, paired with dual UTC cron schedules (`20:35` and `21:35` UTC) and `data/state.json` idempotency.
- **Secret-Safe Delivery**: Reads credentials strictly from environment variables / GitHub Secrets and scrubs tokens, bot IDs, and chat IDs from all error logs and exceptions.

## Repository Structure

```text
portfolio-tracker/
├── .github/workflows/daily_report.yml  # Scheduled & manual GitHub Actions workflow
├── config/portfolio.yaml               # Portfolio inception, weights, benchmarks, rebalance history
├── data/                               # Persisted nav_history.csv, holdings_history.csv, state.json
├── src/
│   ├── models.py                       # Typed dataclasses for config, prices, report, pending state
│   ├── config_loader.py                # YAML schema validation
│   ├── market_calendar.py              # NYSE XNYS holiday & early-close gate
│   ├── data_fetcher.py                 # Yahoo Finance price loader with retry & stale-ticker guard
│   ├── performance.py                  # Buy & Hold virtual share & TWRR NAV engine
│   ├── formatter.py                    # Zero-emoji Telegram HTML renderer
│   ├── telegram_client.py              # Secret-redacting Telegram Bot API client
│   └── main.py                         # CLI entrypoint
├── tests/                              # Unit & security verification suite
├── .env.example                        # Environment variable template (placeholders only)
├── requirements.txt                    # Python dependencies
└── README.md
```

## Local Setup & Dry-Run

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
   # macOS / Linux:       source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run a local dry-run (computes live Yahoo Finance data and prints the Telegram HTML message to stdout without sending to Telegram or writing to `data/`):

   ```bash
   python -m src.main --dry-run --date 2026-11-23 --skip-market-time-check
   ```

3. For local live testing (optional), copy `.env.example` to `.env` and populate your credentials:

   ```bash
   cp .env.example .env
   ```

   Never commit `.env` to version control.

## CLI Options

| Flag | Description |
| --- | --- |
| `--dry-run` | Print rendered report to stdout without sending to Telegram or updating `data/`. |
| `--date YYYY-MM-DD` | Target a specific NYSE trading date. |
| `--force` | Bypass market-close time gate and `state.json` idempotency guard. |
| `--skip-market-time-check` | Skip the post-close ET buffer check. |
| `--config PATH` | Path to portfolio YAML (default: `config/portfolio.yaml`). |
| `--data-dir PATH` | Directory for persisted CSVs and `state.json` (default: `data`). |

## GitHub Actions & Secrets Setup

1. Push this repository to GitHub.
2. Go to **Settings > Secrets and variables > Actions > New repository secret** and add:
   - `TELEGRAM_BOT_TOKEN`: Bot token issued by `@BotFather` (e.g., `<bot_id>:<bot_secret>`).
   - `TELEGRAM_CHAT_ID`: Target channel handle (e.g., `@your_channel_handle`) or numeric chat ID (e.g., `-100xxxxxxxxxx`).
3. Ensure **Settings > Actions > General > Workflow permissions** allows **Read and write permissions** (the workflow also declares `permissions: contents: write`) so updated `data/` files can be committed automatically.
4. Test manually under **Actions > Daily Portfolio Report > Run workflow** with `dry_run: true` first.

### How DST-Safe Scheduling Works

The workflow triggers at both `35 20 * * 1-5` UTC (16:35 EDT) and `35 21 * * 1-5` UTC (16:35 EST):
- Before US market close (+15m buffer), the ET close gate exits cleanly without sending.
- After US market close, the first eligible run sends the report and commits `data/state.json`.
- The second run sees `last_sent_date == today` and exits cleanly without duplicating the message.
- On NYSE holidays, `is_trading_day()` exits cleanly with code `0`.

## Recording Future Rebalances

To record a portfolio weight change, ticker addition, or removal without breaking cumulative NAV history, append a new dated snapshot under `snapshots:` in `config/portfolio.yaml` in chronological order:

```yaml
snapshots:
  - effective_date: "2026-11-24"
    weights:
      GKOS: 7.0
      MU: 4.0
      # ... remaining tickers (unallocated remainder is tracked as cash)

  - effective_date: "2027-01-15"
    weights:
      GKOS: 5.0
      MU: 4.0
      NVDA: 4.0
      # ... remaining tickers
```

When `effective_date` arrives, `PerformanceEngine` rebalances virtual shares at the prior trading day's closing NAV and chains daily returns forward continuously.

## Testing

```bash
pytest -v
```
