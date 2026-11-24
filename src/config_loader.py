"""YAML configuration loader and validator for portfolio-tracker."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from src.models import PortfolioConfig, WeightSnapshot

VALID_REBALANCE_MODES = {"buy_and_hold"}


def _parse_iso_date(value: Any, field_name: str) -> str:
    """Validate and normalize a YYYY-MM-DD date value from YAML."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        cleaned = value.strip()
        try:
            parsed = date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(
                f"Invalid ISO date for '{field_name}': {value!r} (expected YYYY-MM-DD)"
            ) from exc
        return parsed.isoformat()
    raise ValueError(
        f"Invalid type for '{field_name}': expected YYYY-MM-DD string, got {type(value).__name__}"
    )


def _validate_ticker(raw_ticker: Any, context: str) -> str:
    """Ensure ticker is a non-empty uppercase string with no surrounding whitespace."""
    if not isinstance(raw_ticker, str):
        raise ValueError(f"Ticker in {context} must be a string, got {type(raw_ticker).__name__}")
    if not raw_ticker or raw_ticker != raw_ticker.strip():
        raise ValueError(f"Ticker {raw_ticker!r} in {context} must be non-empty without whitespace")
    if raw_ticker != raw_ticker.upper():
        raise ValueError(f"Ticker {raw_ticker!r} in {context} must be uppercase")
    return raw_ticker


def load_config(path: str | Path = "config/portfolio.yaml") -> PortfolioConfig:
    """Load and validate portfolio configuration from a YAML file.

    YAML weights are specified in readable percentages (e.g., 7.0 for 7.0%).
    Returned WeightSnapshot objects store weights as decimals summing <= 1.0.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")

    if "inception_date" not in raw:
        raise ValueError("Missing required field: 'inception_date'")
    inception_date = _parse_iso_date(raw["inception_date"], "inception_date")

    base_nav_raw = raw.get("base_nav", 100.0)
    if isinstance(base_nav_raw, bool) or not isinstance(base_nav_raw, (int, float)):
        raise ValueError(f"'base_nav' must be a positive number, got {base_nav_raw!r}")
    base_nav = float(base_nav_raw)
    if base_nav <= 0.0:
        raise ValueError(f"'base_nav' must be > 0, got {base_nav}")

    rebalance_mode = raw.get("rebalance_mode", "buy_and_hold")
    if not isinstance(rebalance_mode, str) or rebalance_mode not in VALID_REBALANCE_MODES:
        raise ValueError(
            f"Unsupported 'rebalance_mode': {rebalance_mode!r}. "
            f"Allowed: {sorted(VALID_REBALANCE_MODES)}"
        )

    cash_yield_raw = raw.get("cash_yield_annual", 0.0)
    if isinstance(cash_yield_raw, bool) or not isinstance(cash_yield_raw, (int, float)):
        raise ValueError(f"'cash_yield_annual' must be numeric, got {cash_yield_raw!r}")
    cash_yield_annual = float(cash_yield_raw)
    if cash_yield_annual < 0.0:
        raise ValueError(f"'cash_yield_annual' must be >= 0.0, got {cash_yield_annual}")

    benchmarks_raw = raw.get("benchmarks")
    if not isinstance(benchmarks_raw, list) or not benchmarks_raw:
        raise ValueError("'benchmarks' must be a non-empty list of uppercase tickers")
    benchmarks: list[str] = []
    seen_bench: set[str] = set()
    for b in benchmarks_raw:
        ticker = _validate_ticker(b, "benchmarks")
        if ticker in seen_bench:
            raise ValueError(f"Duplicate benchmark ticker: {ticker}")
        seen_bench.add(ticker)
        benchmarks.append(ticker)

    snapshots_raw = raw.get("snapshots")
    if not isinstance(snapshots_raw, list) or not snapshots_raw:
        raise ValueError("'snapshots' must be a non-empty list of weight snapshots")

    snapshots: list[WeightSnapshot] = []
    prev_eff_date: date | None = None

    for idx, entry in enumerate(snapshots_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Snapshot at index {idx} must be a mapping")
        if "effective_date" not in entry:
            raise ValueError(f"Snapshot at index {idx} is missing 'effective_date'")
        eff_date_str = _parse_iso_date(entry["effective_date"], f"snapshots[{idx}].effective_date")
        eff_date_obj = date.fromisoformat(eff_date_str)

        if prev_eff_date is not None and eff_date_obj <= prev_eff_date:
            raise ValueError(
                f"Snapshot effective_dates must be strictly chronological: "
                f"{eff_date_str} is not after {prev_eff_date.isoformat()}"
            )
        prev_eff_date = eff_date_obj

        weights_raw = entry.get("weights")
        if not isinstance(weights_raw, dict) or not weights_raw:
            raise ValueError(f"Snapshot {eff_date_str} must define a non-empty 'weights' mapping")

        weights_dec: dict[str, float] = {}
        total_pct = 0.0
        for raw_tkr, raw_wt in weights_raw.items():
            ticker = _validate_ticker(raw_tkr, f"snapshots[{idx}].weights")
            if isinstance(raw_wt, bool) or not isinstance(raw_wt, (int, float)):
                raise ValueError(
                    f"Weight for {ticker} in snapshot {eff_date_str} must be numeric, got {raw_wt!r}"
                )
            wt_pct = float(raw_wt)
            if wt_pct <= 0.0:
                raise ValueError(
                    f"Weight for {ticker} in snapshot {eff_date_str} must be > 0, got {wt_pct}"
                )
            total_pct += wt_pct
            weights_dec[ticker] = wt_pct / 100.0

        if total_pct > 100.0 + 1e-9:
            raise ValueError(
                f"Snapshot {eff_date_str} weight sum exceeds 100.0%: {total_pct:.4f}%"
            )

        cash_weight = max(0.0, round(1.0 - sum(weights_dec.values()), 10))
        snapshots.append(
            WeightSnapshot(
                effective_date=eff_date_str,
                weights=weights_dec,
                cash_weight=cash_weight,
            )
        )

    if date.fromisoformat(snapshots[0].effective_date) > date.fromisoformat(inception_date):
        raise ValueError(
            f"First snapshot effective_date ({snapshots[0].effective_date}) "
            f"cannot be after inception_date ({inception_date})"
        )

    return PortfolioConfig(
        inception_date=inception_date,
        base_nav=base_nav,
        rebalance_mode=rebalance_mode,
        cash_yield_annual=cash_yield_annual,
        benchmarks=benchmarks,
        snapshots=snapshots,
    )
