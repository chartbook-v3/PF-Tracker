"""Unit tests for zero-emoji formatting, 1-decimal precision, ASCII <=36 width, and HTML integrity."""

from __future__ import annotations

import re

import pytest

from src.formatter import MAX_PRE_LINE_WIDTH, MAX_TELEGRAM_LENGTH, assert_no_emojis, format_report
from src.models import DailyReportData, HoldingSnapshot, PerformanceSummary


def _make_sample_report(
    *,
    trade_date: str = "2026-11-24",
    ann_vol: float | None = None,
    flat_count: int = 0,
    status_notes: list[str] | None = None,
    extreme_numbers: bool = False,
) -> DailyReportData:
    holdings = [
        HoldingSnapshot(
            ticker="GKOS",
            weight_pct=7.03,
            close=249.64,
            adj_close=249.64,
            daily_return_pct=5.24,
            cum_return_pct=5.24,
        ),
        HoldingSnapshot(
            ticker="MPWR",
            weight_pct=4.18,
            close=1333.92,
            adj_close=1333.92,
            daily_return_pct=16.84 if not extreme_numbers else 188.64,
            cum_return_pct=188.64 if extreme_numbers else 16.84,
        ),
        HoldingSnapshot(
            ticker="MU",
            weight_pct=4.12,
            close=118.47,
            adj_close=118.47,
            daily_return_pct=19.04,
            cum_return_pct=19.04,
        ),
        HoldingSnapshot(
            ticker="MXL",
            weight_pct=4.09,
            close=24.81,
            adj_close=24.81,
            daily_return_pct=21.44,
            cum_return_pct=21.44,
        ),
        HoldingSnapshot(
            ticker="CRWD",
            weight_pct=3.81,
            close=368.19,
            adj_close=368.19,
            daily_return_pct=-8.94,
            cum_return_pct=-8.94,
        ),
        HoldingSnapshot(
            ticker="BRZE",
            weight_pct=3.99,
            close=41.05,
            adj_close=41.05,
            daily_return_pct=-0.01,  # Should normalize to +0.0%, never -0.0%
            cum_return_pct=-0.02,
            status="FROZEN" if status_notes else "ACTIVE",
        ),
    ]

    return DailyReportData(
        trade_date=trade_date,
        day_number=1,
        num_names=19,
        port_nav=104.38,
        base_nav=100.0,
        invested_pct=60.62,
        cash_pct=39.38,
        cash_yield_annual_pct=0.0,
        port_perf=PerformanceSummary("Port (60.6%)", 4.38, 4.38, 188.64 if extreme_numbers else 4.38),
        eqty_perf=PerformanceSummary("Eqty (100%)", 7.23, 7.23, 188.64 if extreme_numbers else 7.23),
        benchmarks=[
            PerformanceSummary("SPY", 0.54, 0.54, 0.54),
            PerformanceSummary("QQQ", 0.87, 0.87, 0.87),
        ],
        alphas=[
            PerformanceSummary("Alpha vs SPY", 3.84, 3.84, 3.84),
            PerformanceSummary("Alpha vs QQQ", 3.51, 3.51, 3.51),
        ],
        leaders=[holdings[3], holdings[2], holdings[1]],
        laggards=[holdings[4], holdings[5], holdings[0]],
        holdings=holdings,
        adv_count=4,
        dec_count=1,
        flat_count=flat_count,
        max_dd_pct=-0.01 if not extreme_numbers else -12.34,
        ann_vol_20d_pct=ann_vol,
        best_cum=holdings[1] if extreme_numbers else holdings[3],
        worst_cum=holdings[4],
        status_notes=status_notes or [],
    )


def test_assert_no_emojis_rejects_emojis_and_pictographs() -> None:
    assert_no_emojis("PORTFOLIO DAILY REPORT(11/24) · \u2191 MXL +21.4% · \u2193 CRWD -8.9%")

    forbidden_samples = [
        "Portfolio 📈 +4.4%",
        "Alert 🚨 drop",
        "Check ✅ ok",
        "Arrow \u2b06 up",
        "Arrow \u2b07 down",
        "Star ⭐ top",
        "Flag 🇺🇸 US",
        "Variation selector \ufe0f",
        "ZWJ \u200d test",
    ]
    for sample in forbidden_samples:
        with pytest.raises(ValueError):
            assert_no_emojis(sample)


def test_format_report_tightened_layout_zero_emojis_no_bp() -> None:
    msg = format_report(_make_sample_report(trade_date="2027-01-05", ann_vol=16.84, flat_count=0))

    # Zero emojis
    assert_no_emojis(msg)

    # Strictly NO "bp" anywhere in report (case-insensitive word boundary)
    assert re.search(r"\bbp\b", msg, flags=re.IGNORECASE) is None
    assert "bps" not in msg.lower()

    # Never emit -0.0%
    assert "-0.0%" not in msg

    # Single-line bold header with zero-padded MM/DD, no sub-header lines
    assert msg.startswith("<b>PORTFOLIO DAILY REPORT(01/05)</b>\n\n<b>PERFORMANCE SUMMARY</b>")
    assert "US Close" not in msg
    assert "NAV " not in msg
    assert "Base 100.0" not in msg

    # Tightened TOP MOVERS (1D): compact no-space ↑/↓ lines joined by commas
    assert (
        "<b>TOP MOVERS (1D)</b>\n"
        "\u2191MXL+21.4%,MU+19.0%,MPWR+16.8%\n"
        "\u2193CRWD-8.9%,BRZE+0.0%,GKOS+5.2%"
    ) in msg
    assert "Winner" not in msg
    assert "Loser" not in msg
    assert "Leaders:" not in msg
    assert "Laggards:" not in msg
    assert "\ufe0f" not in msg

    # Tightened BREADTH & RISK: Adv/Dec & Best/Worst only (no Max DD, Ann. Vol, Note, or Status when empty)
    assert "<b>BREADTH &amp; RISK</b>\nAdv/Dec : 4 / 1 (80%)\nBest/Worst : MXL +21.4% / CRWD -8.9%" in msg
    assert "Max DD" not in msg
    assert "Ann. Vol" not in msg
    assert "Note:" not in msg
    assert "Status:" not in msg
    assert "Flat" not in msg


def test_format_report_flat_count_and_status_notes() -> None:
    msg = format_report(
        _make_sample_report(
            trade_date="2026-11-23",
            flat_count=1,
            status_notes=["BRZE frozen at 2026-11-23"],
        )
    )
    assert msg.startswith("<b>PORTFOLIO DAILY REPORT(11/23)</b>\n\n")
    assert "Adv/Dec : 4 / 1 (67%) \u00b7 Flat 1" in msg
    assert "Status: BRZE frozen at 2026-11-23" in msg


def test_format_report_strict_one_decimal_precision() -> None:
    """All decimal numeric outputs must have strictly 1 decimal place (never 2+ decimals)."""
    msg = format_report(_make_sample_report(ann_vol=18.26))

    # No 2-or-more decimal numbers anywhere in the report
    multi_decimal = re.findall(r"\d+\.\d{2,}", msg)
    assert multi_decimal == [], f"Found numbers with >1 decimal place: {multi_decimal}"

    # Verify key 1-decimal values appear explicitly
    assert "Port (60.6%)" in msg
    assert "Eqty (100%)" in msg
    assert "1333.9" in msg
    assert "249.6" in msg
    assert "\u2191MXL+21.4%,MU+19.0%,MPWR+16.8%" in msg
    assert "\u2193CRWD-8.9%,BRZE+0.0%,GKOS+5.2%" in msg
    assert "Best/Worst : MXL +21.4% / CRWD -8.9%" in msg


def test_pre_blocks_pure_ascii_and_max_36_chars_even_with_extreme_values() -> None:
    """Every line inside <pre> blocks must be pure ASCII and <= 36 chars, even for 1333.9 and +188.6%."""
    msg = format_report(_make_sample_report(ann_vol=22.5, extreme_numbers=True))

    pre_blocks = re.findall(r"<pre>(.*?)</pre>", msg, flags=re.DOTALL)
    assert len(pre_blocks) == 2

    for block in pre_blocks:
        for line in block.splitlines():
            assert all(ord(ch) < 128 for ch in line), f"Non-ASCII in <pre>: {line!r}"
            assert len(line) <= MAX_PRE_LINE_WIDTH, f"Line > {MAX_PRE_LINE_WIDTH} chars ({len(line)}): {line!r}"

    # Verify extreme values actually rendered inside holdings table
    assert "1333.9" in pre_blocks[1]
    assert "+188.6%" in pre_blocks[1]
    assert "+188.6%" in pre_blocks[0]


def test_html_tags_balanced_and_under_telegram_limit() -> None:
    report = _make_sample_report(ann_vol=None)
    # Expand to full 19 holdings to test realistic message length
    full_holdings = [
        HoldingSnapshot(
            ticker=f"T{i:02d}",
            weight_pct=3.19,
            close=123.45 + i,
            adj_close=123.45 + i,
            daily_return_pct=1.23 * (1 if i % 2 == 0 else -1),
            cum_return_pct=10.5 * (1 if i % 2 == 0 else -1),
        )
        for i in range(19)
    ]
    full_report = DailyReportData(
        **{**report.__dict__, "holdings": full_holdings, "num_names": 19}
    )
    msg = format_report(full_report)

    assert len(msg) < MAX_TELEGRAM_LENGTH
    assert msg.count("<b>") == msg.count("</b>")
    assert msg.count("<pre>") == msg.count("</pre>")
    assert msg.count("<b>") == 5
    assert msg.count("<pre>") == 2
