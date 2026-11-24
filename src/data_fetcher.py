"""Yahoo Finance market data fetcher with retry, normalization, and stale-bar handling."""

from __future__ import annotations

import atexit
import logging
import os
import ssl
import sys
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.models import PriceData

logger = logging.getLogger(__name__)

_WINDOWS_CA_BUNDLE_PATH: str | None = None
_SERVER_AUTH_OID = "1.3.6.1.5.5.7.3.1"


def _cleanup_ca_bundle(path_str: str) -> None:
    try:
        Path(path_str).unlink(missing_ok=True)
    except OSError:
        pass


def _ensure_windows_curl_ca_bundle() -> None:
    """Populate CURL_CA_BUNDLE on Windows with certifi + Windows trusted ROOT/CA certs.

    curl_cffi (used by yfinance) reads certifi only by default and ignores the
    Windows certificate store. Combining certifi with Windows trusted ROOT/CA
    certificates preserves full TLS verification in corporate/proxy Windows
    environments without ever disabling SSL checks.
    """
    global _WINDOWS_CA_BUNDLE_PATH

    if sys.platform != "win32" or os.environ.get("CURL_CA_BUNDLE"):
        return
    if _WINDOWS_CA_BUNDLE_PATH and Path(_WINDOWS_CA_BUNDLE_PATH).is_file():
        os.environ.setdefault("CURL_CA_BUNDLE", _WINDOWS_CA_BUNDLE_PATH)
        return

    try:
        import certifi

        pem_blocks: list[str] = [Path(certifi.where()).read_text(encoding="utf-8")]
        seen_pems: set[str] = set()

        for store_name in ("ROOT", "CA"):
            for cert_bytes, encoding, trust in ssl.enum_certificates(store_name):
                if encoding != "x509_asn":
                    continue
                if trust is not True and not (
                    isinstance(trust, (set, frozenset)) and _SERVER_AUTH_OID in trust
                ):
                    continue
                pem = ssl.DER_cert_to_PEM_cert(cert_bytes)
                if pem not in seen_pems:
                    seen_pems.add(pem)
                    pem_blocks.append(pem)

        fd, tmp_path = tempfile.mkstemp(prefix="portfolio_tracker_ca_", suffix=".pem")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(pem_blocks))

        _WINDOWS_CA_BUNDLE_PATH = tmp_path
        os.environ.setdefault("CURL_CA_BUNDLE", tmp_path)
        atexit.register(_cleanup_ca_bundle, tmp_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skipping Windows CA bundle augmentation: %s", exc)


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tickers:
        tkr = str(raw).strip().upper()
        if not tkr:
            continue
        if tkr not in seen:
            seen.add(tkr)
            normalized.append(tkr)
    if not normalized:
        raise ValueError("tickers must be a non-empty sequence")
    return normalized


def _extract_price_frames(
    raw: pd.DataFrame,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract (close_df, adj_close_df) cleanly from single- or multi-ticker downloads."""
    if raw is None or raw.empty:
        raise ValueError("Downloaded price DataFrame is empty")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        lvl1 = set(raw.columns.get_level_values(1))
        if "Close" in lvl0:
            close_obj = raw["Close"]
            adj_obj = raw["Adj Close"] if "Adj Close" in lvl0 else close_obj
        elif "Close" in lvl1:
            close_obj = raw.xs("Close", axis=1, level=1)
            adj_obj = raw.xs("Adj Close", axis=1, level=1) if "Adj Close" in lvl1 else close_obj
        else:
            raise ValueError(f"Expected 'Close' column in downloaded MultiIndex data: {raw.columns}")

        close_df = close_obj.to_frame(name=tickers[0]) if isinstance(close_obj, pd.Series) else close_obj.copy()
        adj_df = adj_obj.to_frame(name=tickers[0]) if isinstance(adj_obj, pd.Series) else adj_obj.copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError(f"Expected 'Close' column in downloaded data: {list(raw.columns)}")
        col_name = tickers[0]
        close_df = raw[["Close"]].rename(columns={"Close": col_name}).copy()
        if "Adj Close" in raw.columns:
            adj_df = raw[["Adj Close"]].rename(columns={"Adj Close": col_name}).copy()
        else:
            adj_df = close_df.copy()

    close_df.columns = [str(c).strip().upper() for c in close_df.columns]
    adj_df.columns = [str(c).strip().upper() for c in adj_df.columns]

    # Ensure all requested tickers exist as columns in consistent order
    close_df = close_df.reindex(columns=tickers)
    adj_df = adj_df.reindex(columns=tickers)
    return close_df, adj_df


def _normalize_date_index(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    """Convert index to sorted unique datetime.date objects bounded to [start_date, end_date]."""
    out = df.copy()
    dates: list[date] = []
    for idx in out.index:
        if isinstance(idx, datetime):
            dates.append(idx.date())
        elif isinstance(idx, date):
            dates.append(idx)
        else:
            dates.append(pd.Timestamp(idx).date())
    out.index = pd.Index(dates, name="date")
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    mask = [(start_date <= d <= end_date) for d in out.index]
    out = out.loc[mask]
    return out.apply(pd.to_numeric, errors="coerce")


def _repair_missing_adj_close(close_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    """Fill isolated NaN Adj Close cells when raw Close is present on that date."""
    repaired = adj_df.copy()
    for tkr in repaired.columns:
        c_series = close_df[tkr]
        a_series = repaired[tkr]
        missing_adj = a_series.isna() & c_series.notna()
        if not missing_adj.any():
            continue
        prev_c: float | None = None
        prev_a: float | None = None
        for dt in repaired.index:
            c_val = c_series.loc[dt]
            a_val = a_series.loc[dt]
            if pd.notna(a_val) and pd.notna(c_val):
                prev_c = float(c_val)
                prev_a = float(a_val)
            elif pd.isna(a_val) and pd.notna(c_val):
                if prev_c is not None and prev_c > 0 and prev_a is not None:
                    est = prev_a * (float(c_val) / prev_c)
                else:
                    est = float(c_val)
                repaired.loc[dt, tkr] = est
                prev_c = float(c_val)
                prev_a = est
    return repaired


def fetch_prices(
    tickers: list[str],
    start_date: date,
    end_date: date,
    retries: int = 3,
) -> PriceData:
    """Download raw Close and Adj Close prices via yfinance with retry and stale-ticker tracking.

    Args:
        tickers: List of ticker symbols.
        start_date: Inclusive start date.
        end_date: Inclusive end date.
        retries: Number of download attempts with exponential backoff.

    Returns:
        PriceData with ascending datetime.date index, uppercase ticker columns,
        forward-filled missing trailing bars, and stale_tickers annotations.
    """
    clean_tickers = _normalize_tickers(tickers)
    if end_date < start_date:
        raise ValueError(f"end_date ({end_date}) cannot be earlier than start_date ({start_date})")
    if retries < 1:
        retries = 1

    _ensure_windows_curl_ca_bundle()

    # yfinance `end` is exclusive; add 1 day so `end_date` is included
    yf_start = start_date.isoformat()
    yf_end = (end_date + timedelta(days=1)).isoformat()

    last_exc: Exception | None = None
    raw: pd.DataFrame | None = None

    for attempt in range(retries):
        try:
            downloaded = yf.download(
                tickers=clean_tickers if len(clean_tickers) > 1 else clean_tickers[0],
                start=yf_start,
                end=yf_end,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if downloaded is not None and not downloaded.empty:
                raw = downloaded
                break
            last_exc = ValueError("yfinance returned empty DataFrame")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        if attempt < retries - 1:
            sleep_sec = 2**attempt
            logger.warning(
                "Price download attempt %d/%d failed (%s); retrying in %ds",
                attempt + 1,
                retries,
                last_exc,
                sleep_sec,
            )
            time.sleep(sleep_sec)

    if raw is None or raw.empty:
        raise RuntimeError(
            f"Failed to fetch market prices after {retries} attempt(s): {last_exc}"
        )

    close_df, adj_df = _extract_price_frames(raw, clean_tickers)
    close_df = _normalize_date_index(close_df, start_date, end_date)
    adj_df = _normalize_date_index(adj_df, start_date, end_date)

    # Align indices between close and adj_close
    common_index = close_df.index.union(adj_df.index).sort_values()
    if len(common_index) == 0:
        raise RuntimeError(
            f"No market bars found in range {start_date.isoformat()}..{end_date.isoformat()}"
        )
    close_df = close_df.reindex(common_index)
    adj_df = adj_df.reindex(common_index)

    # Repair rare cases where Close exists on latest bar but Adj Close is NaN
    adj_df = _repair_missing_adj_close(close_df, adj_df)
    # Also fill missing Close from Adj Close if Close was somehow NaN while Adj Close existed
    close_df = close_df.fillna(adj_df)

    # Verify every requested ticker has at least one valid observation
    missing_all = [t for t in clean_tickers if adj_df[t].dropna().empty]
    if missing_all:
        raise ValueError(
            f"No historical price data available for ticker(s): {', '.join(missing_all)}"
        )

    # Identify tickers missing a bar on the latest available market date (or end_date)
    latest_bar_date: date = adj_df.index[-1]
    stale_tickers: dict[str, str] = {}
    for tkr in clean_tickers:
        valid_dates = adj_df[tkr].dropna().index
        last_valid: date = valid_dates[-1]
        if last_valid < latest_bar_date:
            stale_tickers[tkr] = f"frozen at {last_valid.isoformat()}"

    # Forward-fill missing bars (e.g., halts, delistings, acquisitions)
    close_df = close_df.ffill()
    adj_df = adj_df.ffill()

    return PriceData(
        close=close_df,
        adj_close=adj_df,
        stale_tickers=stale_tickers,
    )
