"""Unit tests for Buy & Hold performance math, dual tracking, rebalancing, and edge cases."""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src.models import PortfolioConfig, PriceData, WeightSnapshot
from src.performance import PerformanceEngine


def _make_price_data(
    dates: list[date],
    adj_by_ticker: dict[str, list[float]],
    close_by_ticker: dict[str, list[float]] | None = None,
    stale_tickers: dict[str, str] | None = None,
) -> PriceData:
    adj_df = pd.DataFrame(adj_by_ticker, index=pd.Index(dates, name="date"))
    if close_by_ticker is None:
        close_df = adj_df.copy()
    else:
        close_df = pd.DataFrame(close_by_ticker, index=pd.Index(dates, name="date"))
    return PriceData(
        close=close_df,
        adj_close=adj_df,
        stale_tickers=dict(stale_tickers or {}),
    )


def _sample_config(
    inception_date: str = "2026-11-24",
    snapshots: list[WeightSnapshot] | None = None,
) -> PortfolioConfig:
    if snapshots is None:
        snapshots = [
            WeightSnapshot(
                effective_date=inception_date,
                weights={"AAA": 0.406, "BBB": 0.200},
                cash_weight=0.394,
            )
        ]
    return PortfolioConfig(
        inception_date=inception_date,
        base_nav=100.0,
        rebalance_mode="buy_and_hold",
        cash_yield_annual=0.0,
        benchmarks=["SPY", "QQQ"],
        snapshots=snapshots,
    )


def test_real_portfolio_yaml_loads_and_sums_correctly() -> None:
    cfg = load_config(Path("config/portfolio.yaml"))
    assert cfg.inception_date == "2026-11-24"
    assert cfg.base_nav == pytest.approx(100.0)
    assert cfg.rebalance_mode == "buy_and_hold"
    assert cfg.cash_yield_annual == pytest.approx(0.0)
    assert cfg.benchmarks == ["SPY", "QQQ"]
    assert len(cfg.snapshots) >= 1

    snap0 = cfg.snapshots[0]
    assert len(snap0.weights) == 19
    assert snap0.weights["GKOS"] == pytest.approx(0.07)
    assert sum(snap0.weights.values()) == pytest.approx(0.606, abs=1e-9)
    assert snap0.cash_weight == pytest.approx(0.394, abs=1e-9)


def test_buy_and_hold_dual_tracking_drift_and_alpha(tmp_path: Path) -> None:
    """Verify inception base 100.0, natural weight drift, Port vs Eqty, SPY/QQQ, and Alpha."""
    cfg = _sample_config("2026-11-24")
    engine = PerformanceEngine(cfg, data_dir=tmp_path)

    d0 = date(2026, 11, 23)  # Prior trading close anchor (Base = 100.0)
    d1 = date(2026, 11, 24)  # Day 1
    d2 = date(2026, 11, 25)  # Day 2

    prices = _make_price_data(
        dates=[d0, d1, d2],
        adj_by_ticker={
            "AAA": [100.0, 110.0, 121.0],  # +10% Day 1, +10% Day 2 (+21% cum)
            "BBB": [50.0, 45.0, 45.0],     # -10% Day 1,   0% Day 2 (-10% cum)
            "SPY": [500.0, 505.0, 510.05], # +1% Day 1,   +1% Day 2 (+2.01% cum)
            "QQQ": [400.0, 408.0, 403.92], # +2% Day 1,   -1% Day 2 (+0.98% cum)
        },
    )

    # Day 1 computation
    report1, pending1 = engine.compute(d1, prices)
    assert report1.trade_date == "2026-11-24"
    assert report1.day_number == 1
    assert report1.base_nav == pytest.approx(100.0)

    # Port NAV Day 1: 40.6 * 1.10 + 20.0 * 0.90 + 39.4 = 44.66 + 18.0 + 39.4 = 102.06
    expected_port_d1 = 40.6 * 1.10 + 20.0 * 0.90 + 39.4
    assert report1.port_nav == pytest.approx(expected_port_d1, rel=1e-6)
    assert report1.port_perf.daily_pct == pytest.approx(2.06, rel=1e-5)
    assert report1.port_perf.cum_pct == pytest.approx(2.06, rel=1e-5)
    assert report1.port_perf.mtd_pct == pytest.approx(2.06, rel=1e-5)

    # Eqty (100% normalized) Day 1: (40.6/60.6)*10% + (20.0/60.6)*(-10%)
    expected_eqty_d1_ret = ((40.6 * 1.10 + 20.0 * 0.90) / 60.6 - 1.0) * 100.0
    assert report1.eqty_perf.daily_pct == pytest.approx(expected_eqty_d1_ret, rel=1e-6)
    assert report1.eqty_perf.cum_pct == pytest.approx(expected_eqty_d1_ret, rel=1e-6)

    # Natural weight drift on Day 1
    expected_aaa_w1 = (44.66 / expected_port_d1) * 100.0
    expected_bbb_w1 = (18.0 / expected_port_d1) * 100.0
    expected_cash_w1 = (39.4 / expected_port_d1) * 100.0
    holdings_d1 = {h.ticker: h for h in report1.holdings}
    assert holdings_d1["AAA"].weight_pct == pytest.approx(expected_aaa_w1, rel=1e-6)
    assert holdings_d1["BBB"].weight_pct == pytest.approx(expected_bbb_w1, rel=1e-6)
    assert report1.cash_pct == pytest.approx(expected_cash_w1, rel=1e-6)
    assert report1.invested_pct + report1.cash_pct == pytest.approx(100.0, abs=1e-6)

    # Alpha vs SPY and QQQ
    alpha_map = {a.metric: a for a in report1.alphas}
    assert alpha_map["Alpha vs SPY"].daily_pct == pytest.approx(2.06 - 1.0, rel=1e-5)
    assert alpha_map["Alpha vs QQQ"].daily_pct == pytest.approx(2.06 - 2.0, rel=1e-5)

    # Breadth & Vol (< 5 days -> None)
    assert report1.adv_count == 1
    assert report1.dec_count == 1
    assert report1.flat_count == 0
    assert report1.ann_vol_20d_pct is None
    assert report1.max_dd_pct == pytest.approx(0.0)

    # Save Day 1 and run Day 2 to verify Buy & Hold drift persists without unintended rebalancing
    engine.save(pending1)
    report2, pending2 = engine.compute(d2, prices)
    assert report2.day_number == 2

    # Virtual shares held constant: AAA value = 40.6 * 1.21 = 49.126; BBB = 18.0; cash = 39.4
    expected_port_d2 = 40.6 * 1.21 + 20.0 * 0.90 + 39.4
    expected_port_d2_1d = (expected_port_d2 / expected_port_d1 - 1.0) * 100.0
    assert report2.port_nav == pytest.approx(expected_port_d2, rel=1e-6)
    assert report2.port_perf.daily_pct == pytest.approx(expected_port_d2_1d, rel=1e-5)
    assert report2.port_perf.cum_pct == pytest.approx(expected_port_d2 - 100.0, rel=1e-5)

    # BBB flat on Day 2
    assert report2.adv_count == 1
    assert report2.dec_count == 0
    assert report2.flat_count == 1

    # Verify holdings weight drifted further toward AAA
    holdings_d2 = {h.ticker: h for h in report2.holdings}
    assert holdings_d2["AAA"].weight_pct > holdings_d1["AAA"].weight_pct
    assert holdings_d2["BBB"].weight_pct < holdings_d1["BBB"].weight_pct
    assert holdings_d2["AAA"].cum_return_pct == pytest.approx(21.0, rel=1e-5)
    assert holdings_d2["BBB"].cum_return_pct == pytest.approx(-10.0, rel=1e-5)


def test_rebalance_chaining_on_future_snapshot(tmp_path: Path) -> None:
    """Verify future WeightSnapshot rebalances virtual units at prior closing NAV and chains NAV."""
    snaps = [
        WeightSnapshot(
            effective_date="2026-11-24",
            weights={"AAA": 0.50, "BBB": 0.10},
            cash_weight=0.40,
        ),
        WeightSnapshot(
            effective_date="2026-11-27",
            weights={"AAA": 0.30, "CCC": 0.50},
            cash_weight=0.20,
        ),
    ]
    cfg = _sample_config("2026-11-24", snapshots=snaps)
    engine = PerformanceEngine(cfg, data_dir=tmp_path)

    d0 = date(2026, 11, 23)
    d1 = date(2026, 11, 24)
    d2 = date(2026, 11, 25)
    d3 = date(2026, 11, 27)  # Rebalance effective date (Nov 26 is Thanksgiving)

    prices = _make_price_data(
        dates=[d0, d1, d2, d3],
        adj_by_ticker={
            "AAA": [100.0, 120.0, 120.0, 132.0],  # +20% d1, 0% d2, +10% d3
            "BBB": [100.0, 100.0, 100.0, 50.0],   # -50% d3 (removed on d3 rebalance!)
            "CCC": [200.0, 200.0, 200.0, 220.0],  # +10% d3 (added on d3 rebalance!)
            "SPY": [500.0, 500.0, 500.0, 505.0],
            "QQQ": [400.0, 400.0, 400.0, 404.0],
        },
    )

    # Run through Day 2 and save
    r2, p2 = engine.compute(d2, prices)
    engine.save(p2)
    # Day 2 Port NAV: 50 * 1.2 + 10 * 1.0 + 40 = 110.0
    assert r2.port_nav == pytest.approx(110.0, rel=1e-6)

    # On Day 3 (rebalance effective 2026-11-27):
    # Rebalance at prior close NAV = 110.0 -> 30% AAA (33.0), 50% CCC (55.0), 20% Cash (22.0)
    # Day 3 moves: AAA +10% (36.3), CCC +10% (60.5), Cash (22.0) => Port NAV = 118.8
    # BBB's -50% drop MUST NOT affect Day 3 because BBB was removed at rebalance!
    r3, p3 = engine.compute(d3, prices)
    assert r3.port_nav == pytest.approx(118.8, rel=1e-6)
    assert r3.port_perf.daily_pct == pytest.approx((118.8 / 110.0 - 1.0) * 100.0, rel=1e-6)
    assert r3.port_perf.cum_pct == pytest.approx(18.8, rel=1e-6)

    tickers_d3 = {h.ticker for h in r3.holdings}
    assert tickers_d3 == {"AAA", "CCC"}
    h_map = {h.ticker: h for h in r3.holdings}
    # AAA preserves original entry price (100.0 -> 132.0 = +32.0% cum), CCC enters at 200.0 (+10.0% cum)
    assert h_map["AAA"].cum_return_pct == pytest.approx(32.0, rel=1e-5)
    assert h_map["CCC"].cum_return_pct == pytest.approx(10.0, rel=1e-5)


def test_mtd_month_boundary_max_dd_and_ann_vol_threshold(tmp_path: Path) -> None:
    """Verify MTD reset across month boundary, Max DD, and Ann Vol (N/A <5d, numeric >=5d)."""
    cfg = _sample_config("2026-11-24")
    engine = PerformanceEngine(cfg, data_dir=tmp_path)

    # Anchor + 6 trading days crossing Nov -> Dec boundary
    dates = [
        date(2026, 11, 23),  # anchor
        date(2026, 11, 24),  # Day 1 (Nov)
        date(2026, 11, 25),  # Day 2 (Nov)
        date(2026, 11, 27),  # Day 3 (Nov)
        date(2026, 11, 30),  # Day 4 (Nov month-end)
        date(2026, 12, 1),   # Day 5 (Dec 1 -> MTD resets, Ann Vol becomes numeric!)
        date(2026, 12, 2),   # Day 6 (Dec 2)
    ]
    prices = _make_price_data(
        dates=dates,
        adj_by_ticker={
            "AAA": [100.0, 110.0, 105.0, 100.0, 108.0, 112.0, 106.0],
            "BBB": [100.0, 105.0, 95.0, 90.0, 96.0, 98.0, 94.0],
            "SPY": [500.0, 505.0, 500.0, 495.0, 502.0, 507.02, 504.0],
            "QQQ": [400.0, 404.0, 398.0, 392.0, 400.0, 404.0, 400.0],
        },
    )

    # Day 4 (< 5 days): Ann Vol must be None, Max DD must be negative after peak-to-trough drop
    r4, p4 = engine.compute(date(2026, 11, 30), prices)
    engine.save(p4)
    assert r4.day_number == 4
    assert r4.ann_vol_20d_pct is None
    assert r4.max_dd_pct < 0.0

    # Verify exact Max DD on Day 4:
    # Base=100, D1=40.6*1.1+20*1.05+39.4=105.06 (peak), D3=40.6*1.0+20*0.90+39.4=98.0 (trough)
    expected_dd = (98.0 / 105.06 - 1.0) * 100.0
    assert r4.max_dd_pct == pytest.approx(expected_dd, rel=1e-5)

    # Day 5 (2026-12-01, month boundary):
    # - MTD must reset to Day 5's 1D return vs Nov 30 close!
    # - Ann Vol must now be numeric (>= 5 daily return observations)!
    r5, p5 = engine.compute(date(2026, 12, 1), prices)
    engine.save(p5)
    assert r5.day_number == 5
    assert r5.port_perf.mtd_pct == pytest.approx(r5.port_perf.daily_pct, rel=1e-6)
    assert r5.eqty_perf.mtd_pct == pytest.approx(r5.eqty_perf.daily_pct, rel=1e-6)
    assert r5.port_perf.cum_pct != pytest.approx(r5.port_perf.mtd_pct, abs=1e-3)
    assert r5.ann_vol_20d_pct is not None
    assert r5.ann_vol_20d_pct > 0.0

    # Verify exact Ann Vol formula (sample std ddof=1 * sqrt(252))
    nav_rows = p4.nav_rows + p5.nav_rows
    daily_rets = [float(row["port_daily_pct"]) for row in nav_rows]
    expected_vol = float(np.std(daily_rets, ddof=1)) * math.sqrt(252.0)
    assert r5.ann_vol_20d_pct == pytest.approx(expected_vol, rel=1e-5)

    # Day 6 (2026-12-02): MTD compounds Day 5 and Day 6 relative to Nov 30 NAV
    r6, _ = engine.compute(date(2026, 12, 2), prices)
    expected_mtd_d6 = (r6.port_nav / r4.port_nav - 1.0) * 100.0
    assert r6.port_perf.mtd_pct == pytest.approx(expected_mtd_d6, rel=1e-6)


def test_frozen_delisted_ticker_freezes_without_crashing(tmp_path: Path) -> None:
    """Simulated BRZE missing bar on trade_date freezes at last valid close and adds status note."""
    snaps = [
        WeightSnapshot(
            effective_date="2026-11-24",
            weights={"GKOS": 0.30, "BRZE": 0.306},
            cash_weight=0.394,
        )
    ]
    cfg = _sample_config("2026-11-24", snapshots=snaps)
    engine = PerformanceEngine(cfg, data_dir=tmp_path)

    d0 = date(2026, 11, 23)
    d1 = date(2026, 11, 24)
    d2 = date(2026, 11, 25)

    prices = _make_price_data(
        dates=[d0, d1, d2],
        adj_by_ticker={
            "GKOS": [200.0, 210.0, 220.5],
            "BRZE": [40.0, 42.0, float("nan")],  # Missing bar on d2 (delisted/halted)
            "SPY": [500.0, 502.0, 505.0],
            "QQQ": [400.0, 402.0, 404.0],
        },
    )

    report, _ = engine.compute(d2, prices)
    holdings = {h.ticker: h for h in report.holdings}

    assert holdings["BRZE"].status == "FROZEN"
    assert holdings["BRZE"].close == pytest.approx(42.0)
    assert holdings["BRZE"].daily_return_pct == pytest.approx(0.0, abs=1e-9)
    assert holdings["BRZE"].cum_return_pct == pytest.approx(5.0, rel=1e-5)
    assert any("BRZE" in note and "2026-11-24" in note for note in report.status_notes)
