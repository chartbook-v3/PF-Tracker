"""Tests for NYSE trading calendar (holidays & early close), CLI dry-run, and idempotency."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.main import main
from src.market_calendar import (
    get_market_close_et,
    is_after_market_close_et,
    is_trading_day,
    previous_trading_day,
)
from src.models import PriceData

ET = ZoneInfo("America/New_York")


def _write_minimal_config(path: Path) -> Path:
    cfg_text = """
inception_date: "2026-11-24"
base_nav: 100.0
rebalance_mode: "buy_and_hold"
cash_yield_annual: 0.0
benchmarks:
  - SPY
  - QQQ
snapshots:
  - effective_date: "2026-11-24"
    weights:
      GKOS: 30.0
      MU: 30.6
""".strip()
    path.write_text(cfg_text + "\n", encoding="utf-8")
    return path


def _mock_prices() -> PriceData:
    dates = [date(2026, 11, 20), date(2026, 11, 23), date(2026, 11, 24)]
    df = pd.DataFrame(
        {
            "GKOS": [100.0, 102.0, 107.1],
            "MU": [100.0, 101.0, 106.05],
            "SPY": [500.0, 502.0, 504.51],
            "QQQ": [400.0, 401.0, 405.01],
        },
        index=pd.Index(dates, name="date"),
    )
    return PriceData(close=df.copy(), adj_close=df.copy(), stale_tickers={})


def test_nyse_holidays_and_early_close_detection() -> None:
    # 2026-11-26 is US Thanksgiving (NYSE closed)
    assert is_trading_day(date(2026, 11, 26)) is False
    assert get_market_close_et(date(2026, 11, 26)) is None

    # Weekend closed
    assert is_trading_day(date(2026, 11, 28)) is False

    # Regular session (2026-11-24 closes at 16:00 ET)
    assert is_trading_day(date(2026, 11, 24)) is True
    regular_close = get_market_close_et(date(2026, 11, 24))
    assert regular_close is not None
    assert (regular_close.hour, regular_close.minute) == (16, 0)

    # 2026-11-27 is Black Friday (NYSE early close at 13:00 ET)
    assert is_trading_day(date(2026, 11, 27)) is True
    early_close = get_market_close_et(date(2026, 11, 27))
    assert early_close is not None
    assert (early_close.hour, early_close.minute) == (13, 0)

    # Market close buffer check on Black Friday
    assert is_after_market_close_et(
        date(2026, 11, 27), now_et=datetime(2026, 11, 27, 13, 10, tzinfo=ET), buffer_minutes=15
    ) is False
    assert is_after_market_close_et(
        date(2026, 11, 27), now_et=datetime(2026, 11, 27, 13, 20, tzinfo=ET), buffer_minutes=15
    ) is True

    # Previous trading day before Black Friday skips Thanksgiving -> Wednesday 2026-11-25
    assert previous_trading_day(date(2026, 11, 27)) == date(2026, 11, 25)


def test_cli_skips_nyse_holiday_cleanly(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path / "portfolio.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with (
        patch("src.main.fetch_prices") as mock_fetch,
        patch("src.main.send_telegram_message") as mock_send,
    ):
        exit_code = main(
            [
                "--date",
                "2026-11-26",
                "--skip-market-time-check",
                "--config",
                str(cfg_path),
                "--data-dir",
                str(data_dir),
            ]
        )

    assert exit_code == 0
    mock_fetch.assert_not_called()
    mock_send.assert_not_called()
    assert list(data_dir.iterdir()) == []


def test_cli_dry_run_skips_telegram_and_does_not_modify_data_dir(tmp_path: Path, capsys) -> None:
    cfg_path = _write_minimal_config(tmp_path / "portfolio.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    gitkeep = data_dir / ".gitkeep"
    gitkeep.write_text("", encoding="utf-8")

    with (
        patch("src.main.fetch_prices", return_value=_mock_prices()) as mock_fetch,
        patch("src.main.send_telegram_message") as mock_send,
    ):
        exit_code = main(
            [
                "--dry-run",
                "--date",
                "2026-11-24",
                "--skip-market-time-check",
                "--config",
                str(cfg_path),
                "--data-dir",
                str(data_dir),
            ]
        )

    assert exit_code == 0
    mock_fetch.assert_called_once()
    mock_send.assert_not_called()
    assert [p.name for p in data_dir.iterdir()] == [".gitkeep"]

    stdout = capsys.readouterr().out
    assert "<b>PORTFOLIO DAILY REPORT(11/24)</b>" in stdout
    assert "\u2191" in stdout
    assert "\u2193" in stdout
    assert "\u2191 " not in stdout
    assert "\u2193 " not in stdout
    assert "Winner" not in stdout
    assert "Loser" not in stdout
    assert "Adv/Dec : " in stdout
    assert "Best/Worst : " in stdout
    assert "US Close" not in stdout
    assert "Max DD" not in stdout
    assert "Note:" not in stdout


def test_cli_idempotency_skips_unless_force(tmp_path: Path) -> None:
    cfg_path = _write_minimal_config(tmp_path / "portfolio.yaml")
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # First live run persists state.json with last_trade_date = 2026-11-24
    with (
        patch("src.main.fetch_prices", return_value=_mock_prices()),
        patch("src.main.send_telegram_message", return_value={"ok": True, "message_id": 1}) as mock_send,
    ):
        rc1 = main(
            [
                "--date",
                "2026-11-24",
                "--skip-market-time-check",
                "--config",
                str(cfg_path),
                "--data-dir",
                str(data_dir),
            ]
        )
    assert rc1 == 0
    assert mock_send.call_count == 1
    state = json.loads((data_dir / "state.json").read_text(encoding="utf-8"))
    assert state["last_trade_date"] == "2026-11-24"

    # Second run for same date WITHOUT --force must skip without fetching or sending
    with (
        patch("src.main.fetch_prices") as mock_fetch_2,
        patch("src.main.send_telegram_message") as mock_send_2,
    ):
        rc2 = main(
            [
                "--date",
                "2026-11-24",
                "--skip-market-time-check",
                "--config",
                str(cfg_path),
                "--data-dir",
                str(data_dir),
            ]
        )
    assert rc2 == 0
    mock_fetch_2.assert_not_called()
    mock_send_2.assert_not_called()

    # Third run WITH --force MUST re-run, send, and keep clean single-day state
    with (
        patch("src.main.fetch_prices", return_value=_mock_prices()) as mock_fetch_3,
        patch("src.main.send_telegram_message", return_value={"ok": True, "message_id": 2}) as mock_send_3,
    ):
        rc3 = main(
            [
                "--date",
                "2026-11-24",
                "--force",
                "--config",
                str(cfg_path),
                "--data-dir",
                str(data_dir),
            ]
        )
    assert rc3 == 0
    mock_fetch_3.assert_called_once()
    mock_send_3.assert_called_once()
    state_after = json.loads((data_dir / "state.json").read_text(encoding="utf-8"))
    assert state_after["last_trade_date"] == "2026-11-24"
    assert state_after["day_number"] == 1
