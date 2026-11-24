"""Shared data models for the portfolio performance tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class WeightSnapshot:
    """Target weight allocation effective on a given date.

    Attributes:
        effective_date: ISO date string (YYYY-MM-DD) when weights take effect.
        weights: Mapping of uppercase ticker -> decimal weight (e.g. 0.07 for 7.0%).
        cash_weight: Unallocated cash decimal weight (1.0 - sum(weights)).
    """

    effective_date: str
    weights: dict[str, float]
    cash_weight: float


@dataclass(frozen=True)
class PortfolioConfig:
    """Validated portfolio configuration loaded from YAML."""

    inception_date: str
    base_nav: float
    rebalance_mode: str
    cash_yield_annual: float
    benchmarks: list[str]
    snapshots: list[WeightSnapshot]

    def all_tickers(self) -> list[str]:
        """Return deduplicated uppercase list of all portfolio and benchmark tickers."""
        seen: set[str] = set()
        ordered: list[str] = []
        for snap in self.snapshots:
            for ticker in snap.weights:
                if ticker not in seen:
                    seen.add(ticker)
                    ordered.append(ticker)
        for bench in self.benchmarks:
            if bench not in seen:
                seen.add(bench)
                ordered.append(bench)
        return ordered


@dataclass
class PriceData:
    """Historical raw close and adjusted close prices.

    DataFrame index consists of datetime.date objects sorted ascending.
    Columns are uppercase ticker strings.
    stale_tickers maps tickers missing a bar on the latest date to a status note
    such as 'frozen at YYYY-MM-DD'.
    """

    close: pd.DataFrame
    adj_close: pd.DataFrame
    stale_tickers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HoldingSnapshot:
    """Single-holding valuation and return metrics for a trading day."""

    ticker: str
    weight_pct: float
    close: float
    adj_close: float
    daily_return_pct: float
    cum_return_pct: float
    status: str = "ACTIVE"


@dataclass(frozen=True)
class PerformanceSummary:
    """Row in the performance summary table (Port, Eqty, Benchmark, or Alpha)."""

    metric: str
    daily_pct: float
    mtd_pct: float
    cum_pct: float


@dataclass(frozen=True)
class DailyReportData:
    """Complete computed metrics for a single trading day's report."""

    trade_date: str
    day_number: int
    num_names: int
    port_nav: float
    base_nav: float
    invested_pct: float
    cash_pct: float
    cash_yield_annual_pct: float
    port_perf: PerformanceSummary
    eqty_perf: PerformanceSummary
    benchmarks: list[PerformanceSummary]
    alphas: list[PerformanceSummary]
    leaders: list[HoldingSnapshot]
    laggards: list[HoldingSnapshot]
    holdings: list[HoldingSnapshot]
    adv_count: int
    dec_count: int
    flat_count: int
    max_dd_pct: float
    ann_vol_20d_pct: float | None
    best_cum: HoldingSnapshot
    worst_cum: HoldingSnapshot
    status_notes: list[str]


@dataclass
class PendingState:
    """Unsaved state and history rows produced by PerformanceEngine.compute()."""

    nav_rows: list[dict[str, Any]]
    holdings_rows: list[dict[str, Any]]
    state_dict: dict[str, Any]
