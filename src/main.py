"""CLI entrypoint for the daily portfolio performance tracker."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config_loader import load_config
from src.data_fetcher import fetch_prices
from src.formatter import format_report
from src.market_calendar import (
    is_after_market_close_et,
    is_trading_day,
    latest_completed_trading_day,
)
from src.performance import PerformanceEngine
from src.telegram_client import TelegramDeliveryError, redact_secrets, send_telegram_message

logger = logging.getLogger(__name__)
ET_TZ = ZoneInfo("America/New_York")


def _load_dotenv_if_present(dotenv_path: Path | None = None) -> None:
    """Optionally load environment variables from .env without ever logging values."""
    candidate = dotenv_path or Path(".env")
    if not candidate.is_file():
        return

    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]

        load_dotenv(dotenv_path=candidate, override=False)
        return
    except ImportError:
        pass

    try:
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)
    except OSError:
        logger.warning("Could not read local .env file; continuing with process environment.")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute daily & cumulative US equity portfolio performance and send to Telegram."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the report to stdout without sending to Telegram or writing to data/.",
    )
    parser.add_argument(
        "--date",
        dest="target_date",
        type=str,
        default=None,
        help="Explicit target trade date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass market-close time gate and last_sent_date idempotency check.",
    )
    parser.add_argument(
        "--skip-market-time-check",
        action="store_true",
        help="Skip the post-market-close ET time gate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/portfolio.yaml",
        help="Path to portfolio configuration YAML (default: config/portfolio.yaml).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory for persistent NAV/holdings CSVs and state.json (default: data).",
    )
    return parser.parse_args(argv)


def _resolve_target_date(args: argparse.Namespace) -> date | None:
    """Resolve the target trade date and enforce NYSE trading-day & market-close gates."""
    if args.target_date:
        try:
            trade_date = date.fromisoformat(args.target_date.strip())
        except ValueError as err:
            raise ValueError(
                f"Invalid --date format '{args.target_date}'. Expected YYYY-MM-DD."
            ) from err

        if not is_trading_day(trade_date):
            logger.info(
                "Skipping %s: NYSE market is closed (weekend or holiday).",
                trade_date.isoformat(),
            )
            return None

        bypass_time_gate = args.force or args.skip_market_time_check or args.dry_run
        if not bypass_time_gate and not is_after_market_close_et(trade_date):
            logger.info(
                "Skipping %s: US market has not closed yet (ET close buffer active).",
                trade_date.isoformat(),
            )
            return None

        return trade_date

    now_et = datetime.now(ET_TZ)
    today_et = now_et.date()

    if args.skip_market_time_check:
        if is_trading_day(today_et):
            return today_et
        return latest_completed_trading_day(now_et=now_et)

    if not is_trading_day(today_et):
        logger.info(
            "Skipping %s: NYSE market is closed (weekend or holiday).",
            today_et.isoformat(),
        )
        return None

    if not args.force and not is_after_market_close_et(today_et, now_et=now_et):
        logger.info(
            "Skipping %s: US market has not closed yet (ET close buffer active).",
            today_et.isoformat(),
        )
        return None

    return today_et


def _collect_all_tickers(config) -> list[str]:
    tickers: set[str] = set()
    for snapshot in config.snapshots:
        for ticker in snapshot.weights:
            tickers.add(ticker.upper())
    for benchmark in config.benchmarks:
        tickers.add(benchmark.upper())
    return sorted(tickers)


def _safe_print(text: str) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:
        pass
    print(text)


def main(argv: list[str] | None = None) -> int:
    """Execute the portfolio tracker pipeline and return a process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _load_dotenv_if_present()

    try:
        args = _parse_args(argv)
        config = load_config(args.config)
        trade_date = _resolve_target_date(args)
        if trade_date is None:
            return 0

        engine = PerformanceEngine(config, data_dir=args.data_dir)
        trade_date_str = trade_date.isoformat()

        if not args.force and not args.dry_run:
            last_sent = engine.get_last_sent_date()
            if last_sent is not None and last_sent >= trade_date_str:
                logger.info(
                    "Skipping %s: report already sent through %s (idempotency guard).",
                    trade_date_str,
                    last_sent,
                )
                return 0

        tickers = _collect_all_tickers(config)
        start_date = engine.required_start_date(trade_date)
        logger.info(
            "Fetching market data for %d tickers (%s to %s).",
            len(tickers),
            start_date.isoformat(),
            trade_date_str,
        )
        prices = fetch_prices(tickers, start_date, trade_date)

        report, pending = engine.compute(trade_date, prices)
        message = format_report(report)

        if args.dry_run:
            logger.info("Dry-run complete for %s; skipping Telegram send and data save.", trade_date_str)
            _safe_print(message)
            return 0

        send_telegram_message(message)
        engine.save(pending)
        logger.info("Saved updated portfolio state for %s.", trade_date_str)
        return 0

    except TelegramDeliveryError as err:
        logger.error("%s", err)
        return 1
    except Exception as err:
        secrets = (
            os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            os.environ.get("TELEGRAM_CHAT_ID", ""),
        )
        logger.error("Portfolio tracker run failed: %s", redact_secrets(str(err), secrets))
        return 1


if __name__ == "__main__":
    sys.exit(main())
