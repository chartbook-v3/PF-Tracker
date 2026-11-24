"""Zero-emoji institutional Telegram HTML formatter for daily portfolio reports."""

from __future__ import annotations

from datetime import date
import html
import re

from src.models import DailyReportData, HoldingSnapshot, PerformanceSummary

MAX_TELEGRAM_LENGTH = 4096
MAX_PRE_LINE_WIDTH = 36

# Comprehensive regex blocking Unicode emojis, pictographs, dingbats,
# symbol arrows, variation selectors, ZWJ, keycaps, and regional/subdivision flags.
# Deliberately excludes standard ASCII, editorial middle dot (U+00B7),
# and plain text arrows U+2191/U+2193 (without variation selectors).
_EMOJI_PATTERN = re.compile(
    "["
    "\u200d"  # Zero Width Joiner
    "\u20e3"  # Combining Enclosing Keycap
    "\u2139"  # Information source
    "\u2194-\u2199"  # Arrows
    "\u21a9-\u21aa"  # Curved arrows
    "\u2300-\u23ff"  # Miscellaneous Technical
    "\u24c2"  # Circled M
    "\u25aa-\u25ab"  # Small squares
    "\u25b6\u25c0"  # Play/reverse triangles
    "\u25fb-\u25fe"  # Medium squares
    "\u2600-\u26ff"  # Miscellaneous Symbols
    "\u2700-\u27bf"  # Dingbats
    "\u2934-\u2935"  # Curving arrows
    "\u2b05-\u2b07"  # Directional arrows
    "\u2b1b-\u2b1c"  # Large squares
    "\u2b50\u2b55"  # Star, heavy large circle
    "\u3030\u303d"  # Wavy dash, part alternation mark
    "\u3297\u3299"  # Circled ideographs
    "\ufe00-\ufe0f"  # Variation Selectors
    "\U0001f000-\U0001faff"  # SMP emojis, pictographs, flags, symbols
    "\U000e0020-\U000e007f"  # Tag characters (subdivision flags)
    "]"
)


def assert_no_emojis(text: str) -> None:
    """Raise ValueError if any emoji, pictograph, dingbat, or variation selector is found."""
    match = _EMOJI_PATTERN.search(text)
    if match is not None:
        ch = match.group(0)
        raise ValueError(
            f"Emoji or pictograph character detected in report output: U+{ord(ch):04X}"
        )


def _normalize_zero(value: float) -> float:
    """Prevent negative zero (-0.0) after rounding to 1 decimal place."""
    return 0.0 if abs(value) < 0.05 else value


def _fmt_num(value: float) -> str:
    """Format a number strictly to 1 decimal place."""
    return f"{_normalize_zero(value):.1f}"


def _fmt_unsigned_pct(value: float) -> str:
    """Format an unsigned percentage strictly to 1 decimal place."""
    return f"{_normalize_zero(value):.1f}%"


def _fmt_signed_pct(value: float) -> str:
    """Format a signed percentage strictly to 1 decimal place (e.g. +4.4%, -0.3%, +0.0%)."""
    return f"{_normalize_zero(value):+.1f}%"


def _format_mmdd(trade_date: str) -> str:
    """Parse ISO trade date (YYYY-MM-DD) into MM/DD."""
    return date.fromisoformat(trade_date.strip()).strftime("%m/%d")


def _assert_pre_block_constraints(lines: list[str], block_name: str) -> None:
    """Ensure every line inside a <pre> block is pure ASCII and <= 36 characters."""
    for idx, line in enumerate(lines, start=1):
        if not all(ord(ch) < 128 for ch in line):
            raise ValueError(
                f"Non-ASCII character found in <pre> block '{block_name}' line {idx}: {line!r}"
            )
        if len(line) > MAX_PRE_LINE_WIDTH:
            raise ValueError(
                f"<pre> block '{block_name}' line {idx} exceeds {MAX_PRE_LINE_WIDTH} chars "
                f"({len(line)}): {line!r}"
            )


def _format_perf_label(perf: PerformanceSummary, invested_pct: float | None = None) -> str:
    """Build a compact <=12-char ASCII label for the performance summary table."""
    if invested_pct is not None:
        if abs(invested_pct - 100.0) < 0.05:
            return "Port (100%)"
        return f"Port ({_fmt_unsigned_pct(invested_pct)})"
    label = perf.metric.strip()
    if label == "Eqty (100.0%)":
        label = "Eqty (100%)"
    return label[:12]


def _render_perf_row(label: str, perf: PerformanceSummary) -> str:
    d1 = _fmt_signed_pct(perf.daily_pct)
    mtd = _fmt_signed_pct(perf.mtd_pct)
    cum = _fmt_signed_pct(perf.cum_pct)
    return f"{label:<12} {d1:>7} {mtd:>7} {cum:>7}"


def _build_performance_table(report: DailyReportData) -> str:
    header = f"{'Metric':<12} {'1D':>7} {'MTD':>7} {'Cumul':>7}"
    divider = "-" * 36
    lines: list[str] = [
        header,
        divider,
        _render_perf_row(
            _format_perf_label(report.port_perf, invested_pct=report.invested_pct),
            report.port_perf,
        ),
        _render_perf_row(_format_perf_label(report.eqty_perf), report.eqty_perf),
    ]
    for bm in report.benchmarks:
        lines.append(_render_perf_row(_format_perf_label(bm), bm))

    if report.alphas:
        lines.append(divider)
        for alpha in report.alphas:
            lines.append(_render_perf_row(_format_perf_label(alpha), alpha))

    _assert_pre_block_constraints(lines, "PERFORMANCE SUMMARY")
    return "\n".join(lines)


def _render_holding_row(
    ticker: str,
    weight_pct: float,
    close_value: float | None,
    daily_pct: float,
    cum_pct: float,
) -> str:
    tkr = ticker.upper().strip()[:5]
    wt = _fmt_unsigned_pct(weight_pct)
    close_str = "-" if close_value is None else _fmt_num(close_value)
    d1 = _fmt_signed_pct(daily_pct)
    cum = _fmt_signed_pct(cum_pct)
    return f"{tkr:<5} {wt:>5} {close_str:>7} {d1:>7} {cum:>7}"


def _build_holdings_table(report: DailyReportData) -> str:
    header = f"{'Tkr':<5} {'Wt':>5} {'Close':>7} {'1D':>7} {'Cum':>7}"
    divider = "-" * 35
    lines: list[str] = [header, divider]

    cash_entry: HoldingSnapshot | None = None
    for holding in report.holdings:
        if holding.ticker.upper() == "CASH":
            cash_entry = holding
            continue
        lines.append(
            _render_holding_row(
                ticker=holding.ticker,
                weight_pct=holding.weight_pct,
                close_value=holding.close,
                daily_pct=holding.daily_return_pct,
                cum_pct=holding.cum_return_pct,
            )
        )

    if cash_entry is not None:
        lines.append(
            _render_holding_row(
                ticker="CASH",
                weight_pct=cash_entry.weight_pct,
                close_value=None,
                daily_pct=cash_entry.daily_return_pct,
                cum_pct=cash_entry.cum_return_pct,
            )
        )
    elif report.cash_pct >= 0.05:
        lines.append(
            _render_holding_row(
                ticker="CASH",
                weight_pct=report.cash_pct,
                close_value=None,
                daily_pct=0.0,
                cum_pct=0.0,
            )
        )

    _assert_pre_block_constraints(lines, "HOLDINGS SNAPSHOT")
    return "\n".join(lines)


def _format_movers(items: list[HoldingSnapshot]) -> str:
    if not items:
        return "N/A"
    parts = [f"{item.ticker.upper()}{_fmt_signed_pct(item.daily_return_pct)}" for item in items]
    return ",".join(parts)


def format_report(report: DailyReportData) -> str:
    """Render DailyReportData into a clean, zero-emoji Telegram HTML message."""
    sep = " \u00b7 "
    mmdd = _format_mmdd(report.trade_date)

    header_section = f"<b>{html.escape(f'PORTFOLIO DAILY REPORT({mmdd})')}</b>"

    perf_table = _build_performance_table(report)
    perf_section = (
        f"<b>{html.escape('PERFORMANCE SUMMARY')}</b>\n"
        f"<pre>{html.escape(perf_table)}</pre>"
    )

    movers_section = "\n".join(
        [
            f"<b>{html.escape('TOP MOVERS (1D)')}</b>",
            html.escape(f"\u2191{_format_movers(report.leaders)}"),
            html.escape(f"\u2193{_format_movers(report.laggards)}"),
        ]
    )

    holdings_table = _build_holdings_table(report)
    holdings_section = (
        f"<b>{html.escape('HOLDINGS SNAPSHOT (BY WEIGHT)')}</b>\n"
        f"<pre>{html.escape(holdings_table)}</pre>"
    )

    total_breadth = report.adv_count + report.dec_count + report.flat_count
    pos_pct = round((100.0 * report.adv_count) / total_breadth) if total_breadth > 0 else 0
    adv_dec_line = f"Adv/Dec : {report.adv_count} / {report.dec_count} ({pos_pct}%)"
    if report.flat_count > 0:
        adv_dec_line += f"{sep}Flat {report.flat_count}"

    best_worst_line = (
        f"Best/Worst : "
        f"{report.best_cum.ticker.upper()} {_fmt_signed_pct(report.best_cum.cum_return_pct)} / "
        f"{report.worst_cum.ticker.upper()} {_fmt_signed_pct(report.worst_cum.cum_return_pct)}"
    )

    breadth_lines = [
        f"<b>{html.escape('BREADTH & RISK')}</b>",
        html.escape(adv_dec_line),
        html.escape(best_worst_line),
    ]
    if report.status_notes:
        status_text = "; ".join(note.strip() for note in report.status_notes if note.strip())
        if status_text:
            breadth_lines.append(html.escape(f"Status: {status_text}"))

    message = "\n\n".join(
        [
            header_section,
            perf_section,
            movers_section,
            holdings_section,
            "\n".join(breadth_lines),
        ]
    )

    assert_no_emojis(message)
    if len(message) > MAX_TELEGRAM_LENGTH:
        raise ValueError(
            f"Formatted Telegram message exceeds {MAX_TELEGRAM_LENGTH} chars ({len(message)})."
        )
    return message
