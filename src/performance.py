"""Buy & Hold portfolio performance engine with dual Port/Eqty tracking and rebalance chaining."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.market_calendar import previous_trading_day
from src.models import (
    DailyReportData,
    HoldingSnapshot,
    PendingState,
    PerformanceSummary,
    PortfolioConfig,
    PriceData,
    WeightSnapshot,
)


def _atomic_write_text(target: Path, content: str) -> None:
    """Atomically write UTF-8 text to target path via a same-directory temp file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_csv(target: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Atomically write list of dicts as CSV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _prepare_price_frames(
    prices: PriceData,
    trade_date: date,
    required_tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Copy price DataFrames, validate coverage, detect NaNs/staleness, and forward-fill."""
    close_df = prices.close.copy()
    adj_df = prices.adj_close.copy()
    stale = dict(prices.stale_tickers)

    missing_cols = [t for t in required_tickers if t not in adj_df.columns or t not in close_df.columns]
    if missing_cols:
        raise ValueError(f"Missing price columns for ticker(s): {', '.join(missing_cols)}")

    close_df = close_df.sort_index()
    adj_df = adj_df.sort_index()

    if trade_date not in adj_df.index:
        latest = adj_df.index[-1].isoformat() if len(adj_df.index) > 0 else "none"
        raise ValueError(
            f"Price data for trade_date {trade_date.isoformat()} is not available "
            f"(latest available: {latest})"
        )

    # Detect any un-filled NaNs up through trade_date (e.g. direct PriceData in tests)
    up_to_trade = adj_df.loc[:trade_date]
    for tkr in required_tickers:
        series = up_to_trade[tkr]
        valid = series.dropna()
        if valid.empty:
            raise ValueError(f"No valid prices for ticker {tkr} on or before {trade_date.isoformat()}")
        last_valid = valid.index[-1]
        if last_valid < trade_date and tkr not in stale:
            stale[tkr] = f"frozen at {last_valid.isoformat()}"

    close_df = close_df.ffill()
    adj_df = adj_df.ffill()
    return close_df, adj_df, stale


def _compute_max_dd_pct(nav_series: list[float]) -> float:
    """Compute historical peak-to-trough max drawdown percentage (<= 0.0)."""
    peak = nav_series[0]
    max_dd = 0.0
    for val in nav_series:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (val / peak - 1.0) * 100.0
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _compute_ann_vol_20d_pct(daily_returns_pct: list[float]) -> float | None:
    """Compute annualized volatility (%) of up to last 20 daily returns if >= 5 observations."""
    if len(daily_returns_pct) < 5:
        return None
    window = daily_returns_pct[-20:]
    std_pct = float(np.std(window, ddof=1))
    return std_pct * math.sqrt(252.0)


class PerformanceEngine:
    """Stateful Buy & Hold performance calculation engine."""

    def __init__(self, config: PortfolioConfig, data_dir: str | Path = "data") -> None:
        self.config = config
        self.data_dir = Path(data_dir)
        self.state_path = self.data_dir / "state.json"
        self.nav_history_path = self.data_dir / "nav_history.csv"
        self.holdings_history_path = self.data_dir / "holdings_history.csv"

    def _load_state(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        with self.state_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None

    def _load_nav_history(self) -> list[dict[str, Any]]:
        if not self.nav_history_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.nav_history_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    def _load_holdings_history(self) -> list[dict[str, Any]]:
        if not self.holdings_history_path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with self.holdings_history_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    def get_last_sent_date(self) -> str | None:
        """Return the last persisted trade_date (YYYY-MM-DD), or None if no state exists."""
        state = self._load_state()
        if not state:
            return None
        val = state.get("last_trade_date")
        return str(val) if val else None

    def required_start_date(self, trade_date: date) -> date:
        """Return safe historical start date covering base anchor / last state + buffer."""
        inception = date.fromisoformat(self.config.inception_date)
        state = self._load_state()
        if state and state.get("last_trade_date"):
            last_dt = date.fromisoformat(str(state["last_trade_date"]))
            earliest = min(previous_trading_day(last_dt), previous_trading_day(trade_date))
        else:
            if trade_date >= inception:
                earliest = previous_trading_day(inception)
            else:
                earliest = previous_trading_day(trade_date)

        # Also cover prior month-end in case MTD needs context, plus 10-day buffer for halts/holidays
        month_start_prev = previous_trading_day(date(trade_date.year, trade_date.month, 1))
        anchor_floor = min(earliest, month_start_prev) if state is None and trade_date < inception else earliest
        return anchor_floor - timedelta(days=10)

    def _initial_snapshot_for_date(self, target_date: date) -> WeightSnapshot:
        """Return latest snapshot effective on or before target_date, else first snapshot."""
        chosen = self.config.snapshots[0]
        for snap in self.config.snapshots:
            if date.fromisoformat(snap.effective_date) <= target_date:
                chosen = snap
        return chosen

    def _find_rebalance_snapshot(
        self,
        prev_date: date,
        curr_date: date,
        active_snapshot_date: str,
    ) -> WeightSnapshot | None:
        """Return the latest snapshot whose effective_date falls in (prev_date, curr_date]."""
        active_dt = date.fromisoformat(active_snapshot_date)
        candidate: WeightSnapshot | None = None
        for snap in self.config.snapshots:
            snap_dt = date.fromisoformat(snap.effective_date)
            if prev_date < snap_dt <= curr_date and snap_dt > active_dt:
                candidate = snap
        return candidate

    def _init_units_from_snapshot(
        self,
        snap: WeightSnapshot,
        port_nav: float,
        eqty_nav: float,
        adj_prices: dict[str, float],
        prev_entry_prices: dict[str, float] | None = None,
    ) -> tuple[float, dict[str, float], dict[str, float], dict[str, float]]:
        """Allocate virtual units for Port and normalized Eqty at given closing prices."""
        port_cash = port_nav * snap.cash_weight
        port_units: dict[str, float] = {}
        eqty_units: dict[str, float] = {}
        entry_prices: dict[str, float] = {}

        invested_w = sum(snap.weights.values())
        for ticker, weight in snap.weights.items():
            px = float(adj_prices[ticker])
            if px <= 0.0:
                raise ValueError(f"Non-positive adjusted close for {ticker}: {px}")
            port_units[ticker] = (port_nav * weight) / px
            if invested_w > 0.0:
                norm_w = weight / invested_w
                eqty_units[ticker] = (eqty_nav * norm_w) / px
            if prev_entry_prices and ticker in prev_entry_prices:
                entry_prices[ticker] = float(prev_entry_prices[ticker])
            else:
                entry_prices[ticker] = px

        return port_cash, port_units, eqty_units, entry_prices

    def _snapshot_state_dict(
        self,
        *,
        inception_date: str,
        anchor_date: str,
        base_nav: float,
        last_trade_date: str,
        day_number: int,
        active_snapshot_date: str,
        port_nav: float,
        eqty_nav: float,
        port_cash: float,
        port_units: dict[str, float],
        eqty_units: dict[str, float],
        bench_units: dict[str, float],
        bench_nav: dict[str, float],
        holding_entry_adj_close: dict[str, float],
        last_adj_close: dict[str, float],
        month_start_nav: dict[str, Any],
        peak_port_nav: float,
        max_dd_pct: float,
        prev_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "inception_date": inception_date,
            "anchor_date": anchor_date,
            "base_nav": base_nav,
            "last_trade_date": last_trade_date,
            "day_number": day_number,
            "active_snapshot_date": active_snapshot_date,
            "port_nav": port_nav,
            "eqty_nav": eqty_nav,
            "port_cash": port_cash,
            "port_units": dict(port_units),
            "eqty_units": dict(eqty_units),
            "bench_units": dict(bench_units),
            "bench_nav": dict(bench_nav),
            "holding_entry_adj_close": dict(holding_entry_adj_close),
            "last_adj_close": dict(last_adj_close),
            "month_start_nav": dict(month_start_nav),
            "peak_port_nav": peak_port_nav,
            "max_dd_pct": max_dd_pct,
        }
        if prev_state is not None:
            shallow_prev = {k: v for k, v in prev_state.items() if k != "prev_state"}
            payload["prev_state"] = shallow_prev
        return payload

    def compute(
        self,
        trade_date: date,
        prices: PriceData,
    ) -> tuple[DailyReportData, PendingState]:
        """Compute daily and cumulative performance up through `trade_date`.

        Pure computation: never creates or modifies any files on disk.
        """
        all_tickers = self.config.all_tickers()
        close_df, adj_df, stale_tickers = _prepare_price_frames(prices, trade_date, all_tickers)
        price_dates: list[date] = list(adj_df.index)

        inception_dt = date.fromisoformat(self.config.inception_date)
        base_nav = float(self.config.base_nav)

        existing_state = self._load_state()
        existing_nav_rows = self._load_nav_history()

        # Handle --force re-run of the latest persisted trade_date cleanly
        if existing_state and existing_state.get("last_trade_date") == trade_date.isoformat():
            existing_nav_rows = [
                r for r in existing_nav_rows if r.get("trade_date") != trade_date.isoformat()
            ]
            if existing_state.get("prev_state"):
                existing_state = dict(existing_state["prev_state"])
            else:
                existing_state = None
        elif existing_state and existing_state.get("last_trade_date"):
            last_dt = date.fromisoformat(str(existing_state["last_trade_date"]))
            if trade_date < last_dt:
                raise ValueError(
                    f"trade_date ({trade_date.isoformat()}) is earlier than persisted "
                    f"last_trade_date ({last_dt.isoformat()})"
                )

        if existing_state is None:
            # First run: anchor at prior trading day before inception (or before trade_date for pre-inception dry-run)
            target_anchor = (
                previous_trading_day(inception_dt)
                if trade_date >= inception_dt
                else previous_trading_day(trade_date)
            )
            eligible_anchors = [d for d in price_dates if d <= target_anchor]
            if not eligible_anchors:
                raise ValueError(
                    f"Price data does not include anchor close on or before "
                    f"{target_anchor.isoformat()} (earliest available: {price_dates[0].isoformat()})"
                )
            anchor_date = eligible_anchors[-1]
            anchor_adj = {t: float(adj_df.loc[anchor_date, t]) for t in all_tickers}

            init_snap = self._initial_snapshot_for_date(
                inception_dt if trade_date >= inception_dt else trade_date
            )
            port_cash, port_units, eqty_units, holding_entry_adj_close = self._init_units_from_snapshot(
                snap=init_snap,
                port_nav=base_nav,
                eqty_nav=base_nav,
                adj_prices=anchor_adj,
            )
            bench_units = {b: base_nav / anchor_adj[b] for b in self.config.benchmarks}
            bench_nav = {b: base_nav for b in self.config.benchmarks}

            prev_date = anchor_date
            day_number = 0
            active_snapshot_date = init_snap.effective_date
            port_nav = base_nav
            eqty_nav = base_nav
            last_adj_close = dict(anchor_adj)
            month_start_nav: dict[str, Any] = {
                "month": (
                    inception_dt.strftime("%Y-%m")
                    if trade_date >= inception_dt
                    else trade_date.strftime("%Y-%m")
                ),
                "port": base_nav,
                "eqty": base_nav,
                **{b: base_nav for b in self.config.benchmarks},
            }
            peak_port_nav = base_nav
            max_dd_pct = 0.0
            prior_state_for_save: dict[str, Any] | None = None
            historical_nav_series = [base_nav]
            historical_daily_returns: list[float] = []
        else:
            anchor_date = date.fromisoformat(str(existing_state["anchor_date"]))
            prev_date = date.fromisoformat(str(existing_state["last_trade_date"]))
            day_number = int(existing_state["day_number"])
            active_snapshot_date = str(existing_state["active_snapshot_date"])
            port_nav = float(existing_state["port_nav"])
            eqty_nav = float(existing_state["eqty_nav"])
            port_cash = float(existing_state["port_cash"])
            port_units = {k: float(v) for k, v in existing_state["port_units"].items()}
            eqty_units = {k: float(v) for k, v in existing_state["eqty_units"].items()}
            bench_units = {k: float(v) for k, v in existing_state["bench_units"].items()}
            bench_nav = {k: float(v) for k, v in existing_state["bench_nav"].items()}
            holding_entry_adj_close = {
                k: float(v) for k, v in existing_state["holding_entry_adj_close"].items()
            }
            last_adj_close = {k: float(v) for k, v in existing_state["last_adj_close"].items()}
            month_start_nav = dict(existing_state["month_start_nav"])
            peak_port_nav = float(existing_state.get("peak_port_nav", max(base_nav, port_nav)))
            max_dd_pct = float(existing_state.get("max_dd_pct", 0.0))
            prior_state_for_save = dict(existing_state)

            historical_nav_series = [base_nav] + [
                float(r["port_nav"]) for r in existing_nav_rows if r.get("port_nav") not in (None, "")
            ]
            historical_daily_returns = [
                float(r["port_daily_pct"])
                for r in existing_nav_rows
                if r.get("port_daily_pct") not in (None, "")
            ]

        dates_to_process = [d for d in price_dates if prev_date < d <= trade_date]
        if not dates_to_process:
            raise ValueError(
                f"No trading dates to process between {prev_date.isoformat()} and {trade_date.isoformat()}"
            )

        new_nav_rows: list[dict[str, Any]] = []
        new_holdings_rows: list[dict[str, Any]] = []
        final_report: DailyReportData | None = None

        for curr_date in dates_to_process:
            # Capture pre-step state so final state can support --force re-run
            if day_number > 0:
                prior_state_for_save = self._snapshot_state_dict(
                    inception_date=self.config.inception_date,
                    anchor_date=anchor_date.isoformat(),
                    base_nav=base_nav,
                    last_trade_date=prev_date.isoformat(),
                    day_number=day_number,
                    active_snapshot_date=active_snapshot_date,
                    port_nav=port_nav,
                    eqty_nav=eqty_nav,
                    port_cash=port_cash,
                    port_units=port_units,
                    eqty_units=eqty_units,
                    bench_units=bench_units,
                    bench_nav=bench_nav,
                    holding_entry_adj_close=holding_entry_adj_close,
                    last_adj_close=last_adj_close,
                    month_start_nav=month_start_nav,
                    peak_port_nav=peak_port_nav,
                    max_dd_pct=max_dd_pct,
                )
            else:
                prior_state_for_save = None

            port_nav_prev = port_nav
            eqty_nav_prev = eqty_nav
            bench_nav_prev = dict(bench_nav)

            # Build previous-day adjusted close map (prefer price frame on prev_date, fallback to state)
            prev_adj_map: dict[str, float] = {}
            for tkr in all_tickers:
                if prev_date in adj_df.index and pd.notna(adj_df.loc[prev_date, tkr]):
                    prev_adj_map[tkr] = float(adj_df.loc[prev_date, tkr])
                elif tkr in last_adj_close:
                    prev_adj_map[tkr] = float(last_adj_close[tkr])

            # Check for scheduled rebalance snapshot effective in (prev_date, curr_date]
            rebalance_snap = self._find_rebalance_snapshot(prev_date, curr_date, active_snapshot_date)
            if rebalance_snap is not None:
                port_cash, port_units, eqty_units, holding_entry_adj_close = (
                    self._init_units_from_snapshot(
                        snap=rebalance_snap,
                        port_nav=port_nav_prev,
                        eqty_nav=eqty_nav_prev,
                        adj_prices=prev_adj_map,
                        prev_entry_prices=holding_entry_adj_close,
                    )
                )
                active_snapshot_date = rebalance_snap.effective_date

            # Check month boundary for MTD baseline
            curr_month = curr_date.strftime("%Y-%m")
            if curr_month != month_start_nav.get("month"):
                month_start_nav = {
                    "month": curr_month,
                    "port": port_nav_prev,
                    "eqty": eqty_nav_prev,
                    **{b: bench_nav_prev[b] for b in self.config.benchmarks},
                }

            # Cash yield accrual (0.0% by default -> exact unchanged cash)
            if self.config.cash_yield_annual > 0.0:
                days_elapsed = max(0, (curr_date - prev_date).days)
                port_cash = port_cash * (1.0 + self.config.cash_yield_annual * (days_elapsed / 365.0))

            # Current day's raw Close and Adj Close
            curr_close_map = {t: float(close_df.loc[curr_date, t]) for t in all_tickers}
            curr_adj_map = {t: float(adj_df.loc[curr_date, t]) for t in all_tickers}

            # Portfolio valuation (Buy & Hold virtual units)
            port_equity_val = sum(
                units * curr_adj_map[tkr] for tkr, units in port_units.items()
            )
            port_nav = port_equity_val + port_cash
            eqty_nav = (
                sum(units * curr_adj_map[tkr] for tkr, units in eqty_units.items())
                if eqty_units
                else eqty_nav_prev
            )
            bench_nav = {
                b: bench_units[b] * curr_adj_map[b] for b in self.config.benchmarks
            }

            invested_pct = (port_equity_val / port_nav) * 100.0 if port_nav > 0 else 0.0
            cash_pct = (port_cash / port_nav) * 100.0 if port_nav > 0 else 0.0

            # Returns: 1D, MTD, Cumulative
            port_daily_pct = (port_nav / port_nav_prev - 1.0) * 100.0
            port_mtd_pct = (port_nav / float(month_start_nav["port"]) - 1.0) * 100.0
            port_cum_pct = (port_nav / base_nav - 1.0) * 100.0

            eqty_daily_pct = (eqty_nav / eqty_nav_prev - 1.0) * 100.0
            eqty_mtd_pct = (eqty_nav / float(month_start_nav["eqty"]) - 1.0) * 100.0
            eqty_cum_pct = (eqty_nav / base_nav - 1.0) * 100.0

            port_perf = PerformanceSummary(
                metric=f"Port ({invested_pct:.1f}%)",
                daily_pct=port_daily_pct,
                mtd_pct=port_mtd_pct,
                cum_pct=port_cum_pct,
            )
            eqty_perf = PerformanceSummary(
                metric="Eqty (100%)",
                daily_pct=eqty_daily_pct,
                mtd_pct=eqty_mtd_pct,
                cum_pct=eqty_cum_pct,
            )

            bench_perfs: list[PerformanceSummary] = []
            alpha_perfs: list[PerformanceSummary] = []
            for b in self.config.benchmarks:
                b_daily = (bench_nav[b] / bench_nav_prev[b] - 1.0) * 100.0
                b_mtd = (bench_nav[b] / float(month_start_nav[b]) - 1.0) * 100.0
                b_cum = (bench_nav[b] / base_nav - 1.0) * 100.0
                bench_perfs.append(
                    PerformanceSummary(
                        metric=b,
                        daily_pct=b_daily,
                        mtd_pct=b_mtd,
                        cum_pct=b_cum,
                    )
                )
                alpha_perfs.append(
                    PerformanceSummary(
                        metric=f"Alpha vs {b}",
                        daily_pct=port_daily_pct - b_daily,
                        mtd_pct=port_mtd_pct - b_mtd,
                        cum_pct=port_cum_pct - b_cum,
                    )
                )

            # Holdings snapshots
            holdings_unsorted: list[HoldingSnapshot] = []
            for tkr, units in port_units.items():
                val = units * curr_adj_map[tkr]
                wt_pct = (val / port_nav) * 100.0 if port_nav > 0 else 0.0
                prev_px = prev_adj_map[tkr]
                curr_px = curr_adj_map[tkr]
                entry_px = holding_entry_adj_close[tkr]
                d_ret = (curr_px / prev_px - 1.0) * 100.0 if prev_px > 0 else 0.0
                c_ret = (curr_px / entry_px - 1.0) * 100.0 if entry_px > 0 else 0.0
                status = "FROZEN" if tkr in stale_tickers else "ACTIVE"
                holdings_unsorted.append(
                    HoldingSnapshot(
                        ticker=tkr,
                        weight_pct=wt_pct,
                        close=curr_close_map[tkr],
                        adj_close=curr_px,
                        daily_return_pct=d_ret,
                        cum_return_pct=c_ret,
                        status=status,
                    )
                )

            holdings = sorted(
                holdings_unsorted,
                key=lambda h: (-h.weight_pct, -h.daily_return_pct, h.ticker),
            )
            leaders = sorted(
                holdings_unsorted,
                key=lambda h: (-h.daily_return_pct, -h.weight_pct, h.ticker),
            )[:3]
            laggards = sorted(
                holdings_unsorted,
                key=lambda h: (h.daily_return_pct, -h.weight_pct, h.ticker),
            )[:3]
            best_cum = max(
                holdings_unsorted,
                key=lambda h: (h.cum_return_pct, h.weight_pct, h.ticker),
            )
            worst_cum = min(
                holdings_unsorted,
                key=lambda h: (h.cum_return_pct, -h.weight_pct, h.ticker),
            )

            adv_count = sum(1 for h in holdings if h.daily_return_pct > 1e-9)
            dec_count = sum(1 for h in holdings if h.daily_return_pct < -1e-9)
            flat_count = len(holdings) - adv_count - dec_count

            day_number += 1
            historical_nav_series.append(port_nav)
            historical_daily_returns.append(port_daily_pct)

            if port_nav > peak_port_nav:
                peak_port_nav = port_nav
            max_dd_pct = _compute_max_dd_pct(historical_nav_series)
            ann_vol_20d_pct = _compute_ann_vol_20d_pct(historical_daily_returns)

            # Status notes for stale/frozen portfolio holdings or benchmarks
            status_notes: list[str] = []
            for tkr in list(port_units.keys()) + list(self.config.benchmarks):
                if tkr in stale_tickers:
                    note = f"{tkr} {stale_tickers[tkr]}"
                    if note not in status_notes:
                        status_notes.append(note)

            # Build CSV rows for this trading day
            nav_row: dict[str, Any] = {
                "trade_date": curr_date.isoformat(),
                "day_number": day_number,
                "port_nav": round(port_nav, 6),
                "eqty_nav": round(eqty_nav, 6),
                "invested_pct": round(invested_pct, 4),
                "cash_pct": round(cash_pct, 4),
                "port_daily_pct": round(port_daily_pct, 6),
                "port_mtd_pct": round(port_mtd_pct, 6),
                "port_cum_pct": round(port_cum_pct, 6),
                "eqty_daily_pct": round(eqty_daily_pct, 6),
                "eqty_mtd_pct": round(eqty_mtd_pct, 6),
                "eqty_cum_pct": round(eqty_cum_pct, 6),
            }
            for bp, ap in zip(bench_perfs, alpha_perfs):
                key = bp.metric.lower()
                nav_row[f"{key}_nav"] = round(bench_nav[bp.metric], 6)
                nav_row[f"{key}_daily_pct"] = round(bp.daily_pct, 6)
                nav_row[f"{key}_cum_pct"] = round(bp.cum_pct, 6)
                nav_row[f"alpha_{key}_daily_pct"] = round(ap.daily_pct, 6)
                nav_row[f"alpha_{key}_cum_pct"] = round(ap.cum_pct, 6)
            nav_row["max_dd_pct"] = round(max_dd_pct, 6)
            nav_row["ann_vol_20d_pct"] = (
                round(ann_vol_20d_pct, 6) if ann_vol_20d_pct is not None else ""
            )
            nav_row["adv_count"] = adv_count
            nav_row["dec_count"] = dec_count
            nav_row["flat_count"] = flat_count
            new_nav_rows.append(nav_row)

            for h in holdings:
                new_holdings_rows.append(
                    {
                        "trade_date": curr_date.isoformat(),
                        "ticker": h.ticker,
                        "weight_pct": round(h.weight_pct, 4),
                        "close": round(h.close, 4),
                        "adj_close": round(h.adj_close, 6),
                        "daily_return_pct": round(h.daily_return_pct, 6),
                        "cum_return_pct": round(h.cum_return_pct, 6),
                        "status": h.status,
                    }
                )

            final_report = DailyReportData(
                trade_date=curr_date.isoformat(),
                day_number=day_number,
                num_names=len(holdings),
                port_nav=port_nav,
                base_nav=base_nav,
                invested_pct=invested_pct,
                cash_pct=cash_pct,
                cash_yield_annual_pct=self.config.cash_yield_annual * 100.0,
                port_perf=port_perf,
                eqty_perf=eqty_perf,
                benchmarks=bench_perfs,
                alphas=alpha_perfs,
                leaders=leaders,
                laggards=laggards,
                holdings=holdings,
                adv_count=adv_count,
                dec_count=dec_count,
                flat_count=flat_count,
                max_dd_pct=max_dd_pct,
                ann_vol_20d_pct=ann_vol_20d_pct,
                best_cum=best_cum,
                worst_cum=worst_cum,
                status_notes=status_notes,
            )

            last_adj_close = dict(curr_adj_map)
            prev_date = curr_date

        assert final_report is not None

        final_state = self._snapshot_state_dict(
            inception_date=self.config.inception_date,
            anchor_date=anchor_date.isoformat(),
            base_nav=base_nav,
            last_trade_date=prev_date.isoformat(),
            day_number=day_number,
            active_snapshot_date=active_snapshot_date,
            port_nav=port_nav,
            eqty_nav=eqty_nav,
            port_cash=port_cash,
            port_units=port_units,
            eqty_units=eqty_units,
            bench_units=bench_units,
            bench_nav=bench_nav,
            holding_entry_adj_close=holding_entry_adj_close,
            last_adj_close=last_adj_close,
            month_start_nav=month_start_nav,
            peak_port_nav=peak_port_nav,
            max_dd_pct=max_dd_pct,
            prev_state=prior_state_for_save,
        )

        pending = PendingState(
            nav_rows=new_nav_rows,
            holdings_rows=new_holdings_rows,
            state_dict=final_state,
        )
        return final_report, pending

    def save(self, pending: PendingState) -> None:
        """Atomically persist nav_history.csv, holdings_history.csv, and state.json."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 1. Merge & write nav_history.csv
        existing_nav = self._load_nav_history()
        new_nav_dates = {str(r["trade_date"]) for r in pending.nav_rows}
        merged_nav = [r for r in existing_nav if str(r.get("trade_date")) not in new_nav_dates]
        merged_nav.extend(pending.nav_rows)
        merged_nav.sort(key=lambda r: str(r["trade_date"]))

        if merged_nav:
            nav_fields: list[str] = []
            for row in merged_nav:
                for k in row.keys():
                    if k not in nav_fields:
                        nav_fields.append(k)
            _atomic_write_csv(self.nav_history_path, merged_nav, nav_fields)

        # 2. Merge & write holdings_history.csv
        existing_holdings = self._load_holdings_history()
        new_holding_dates = {str(r["trade_date"]) for r in pending.holdings_rows}
        merged_holdings = [
            r for r in existing_holdings if str(r.get("trade_date")) not in new_holding_dates
        ]
        merged_holdings.extend(pending.holdings_rows)
        merged_holdings.sort(
            key=lambda r: (
                str(r["trade_date"]),
                -float(r.get("weight_pct", 0.0)),
                str(r.get("ticker", "")),
            )
        )

        holdings_fields = [
            "trade_date",
            "ticker",
            "weight_pct",
            "close",
            "adj_close",
            "daily_return_pct",
            "cum_return_pct",
            "status",
        ]
        if merged_holdings:
            _atomic_write_csv(self.holdings_history_path, merged_holdings, holdings_fields)

        # 3. Write state.json last so it acts as the commit marker
        state_json = json.dumps(pending.state_dict, indent=2, sort_keys=False) + "\n"
        _atomic_write_text(self.state_path, state_json)
