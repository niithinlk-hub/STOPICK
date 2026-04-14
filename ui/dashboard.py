from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stopick_app.alerts import build_alert_message, load_alert_state, save_alert_state, send_email, send_telegram, send_webhook, should_alert
from stopick_app.workstation import ScanBundle, scan_market
from backtest.engine import run_backtest
from config import load_app_config
from journal.storage import JournalStore


def _setup_to_dict_map(bundle: ScanBundle) -> dict[str, object]:
    return {f"{setup.ticker}|{setup.setup_family}": setup for setup in bundle.setups}


def _chart_for_setup(bundle: ScanBundle, setup) -> go.Figure:
    symbol_key = f"{setup.ticker}.NS" if setup.market == "NSE" else setup.ticker
    frame_bundle = bundle.frame_cache.get(symbol_key) or bundle.frame_cache.get(setup.ticker) or {}
    frame = frame_bundle.get(setup.timeframe)
    if frame is None or frame.empty:
        frame = frame_bundle.get("1d")
    if frame is None or frame.empty:
        return go.Figure()

    chart_frame = frame.copy().sort_values("datetime").tail(180)
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=chart_frame["datetime"],
            open=chart_frame["open"],
            high=chart_frame["high"],
            low=chart_frame["low"],
            close=chart_frame["close"],
            name="Price",
        ),
    )
    fig.add_hline(y=setup.breakout.breakout_level, line_dash="dot", line_color="#ff4b4b", annotation_text="Breakout")
    if setup.structure.fvg_zone:
        fig.add_hrect(y0=setup.structure.fvg_zone[0], y1=setup.structure.fvg_zone[1], fillcolor="rgba(0, 200, 120, 0.12)", line_width=0)
    if setup.structure.retest_zone:
        fig.add_hrect(y0=setup.structure.retest_zone[0], y1=setup.structure.retest_zone[1], fillcolor="rgba(255, 165, 0, 0.10)", line_width=0)
    fig.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20), xaxis_rangeslider_visible=False, title=f"{setup.ticker} {setup.setup_family} context")
    return fig


def _export_chart_png(fig: go.Figure) -> bytes | None:
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def _render_scanner(bundle: ScanBundle | None) -> None:
    st.subheader("Scanner")
    if bundle is None:
        st.info("Run a scan to populate high-conviction setups.")
        return
    summary_cols = st.columns(4)
    summary_cols[0].metric("Symbols requested", bundle.scanned_symbols)
    summary_cols[1].metric("Data loaded", bundle.successful_symbols)
    summary_cols[2].metric("Failures", len(bundle.failures))
    summary_cols[3].metric("Qualified setups", 0 if bundle.results.empty else len(bundle.results))
    for note in bundle.notes:
        st.caption(note)
    if bundle.results.empty:
        if bundle.failures:
            st.warning("The scan finished, but no symbols produced qualifying rows. The failure log below explains what data calls failed.")
            failure_df = pd.DataFrame({"symbol": list(bundle.failures.keys()), "error": list(bundle.failures.values())})
            st.dataframe(failure_df, width="stretch")
        else:
            st.info("The scan completed successfully, but nothing met the current minimum-score threshold.")
        return
    results = bundle.results.copy()
    col1, col2, col3 = st.columns(3)
    markets = sorted(results["market"].dropna().unique().tolist())
    exchanges = sorted(results["exchange"].dropna().unique().tolist())
    sectors = sorted(results["sector"].dropna().unique().tolist())
    selected_markets = col1.multiselect("Country / market", markets, default=markets)
    selected_exchanges = col2.multiselect("Exchange", exchanges, default=exchanges)
    selected_sectors = col3.multiselect("Sector", sectors, default=sectors)
    price_min, price_max = float(results["current_price"].min()), float(results["current_price"].max())
    volume_floor = st.slider("Minimum average volume", 0.0, float(results["avg_volume_20"].fillna(0.0).max()), 0.0)
    atr_floor = st.slider("Minimum ATR percentile", 0.0, 100.0, 0.0)
    event_cap = st.slider("Maximum event proximity days", 0, 30, 30, help="Filters earnings/results proximity when event data is available.")
    price_range = st.slider("Price range", min_value=price_min, max_value=price_max, value=(price_min, price_max))

    filtered = results.copy()
    if selected_markets:
        filtered = filtered.loc[filtered["market"].isin(selected_markets)]
    if selected_exchanges:
        filtered = filtered.loc[filtered["exchange"].isin(selected_exchanges)]
    if selected_sectors:
        filtered = filtered.loc[filtered["sector"].isin(selected_sectors)]
    filtered = filtered.loc[filtered["avg_volume_20"].fillna(0.0) >= volume_floor]
    filtered = filtered.loc[filtered["atr_percentile"].fillna(0.0) >= atr_floor]
    filtered = filtered.loc[filtered["current_price"].between(price_range[0], price_range[1])]
    filtered = filtered.loc[filtered["event_risk_days"].isna() | (filtered["event_risk_days"] <= event_cap)]

    st.dataframe(filtered, width="stretch")
    st.download_button("Export setups CSV", filtered.to_csv(index=False), "stopick_setups.csv", "text/csv")
    if bundle.failures:
        with st.expander("Show symbol download failures"):
            failure_df = pd.DataFrame({"symbol": list(bundle.failures.keys()), "error": list(bundle.failures.values())})
            st.dataframe(failure_df, width="stretch")


def _render_top_setups(bundle: ScanBundle | None) -> None:
    st.subheader("Top setups")
    if bundle is None or bundle.results.empty:
        st.info("No setups available.")
        return
    top = bundle.results.head(10)
    cols = st.columns(4)
    cols[0].metric("Setups", len(bundle.results))
    cols[1].metric("A+ / A", int(bundle.results["grade"].isin(["A+", "A"]).sum()))
    cols[2].metric("NSE", int((bundle.results["market"] == "NSE").sum()))
    cols[3].metric("US", int((bundle.results["market"] == "US").sum()))
    st.dataframe(top, width="stretch")


def _render_structure_charts(bundle: ScanBundle | None) -> None:
    st.subheader("Market structure charts")
    if bundle is None or bundle.results.empty:
        st.info("Run a scan to visualize structure and FVG overlays.")
        return
    setup_map = _setup_to_dict_map(bundle)
    option = st.selectbox("Select setup", list(setup_map))
    setup = setup_map[option]
    st.write("Why this qualified:", " ".join(setup.reasons_for))
    fig = _chart_for_setup(bundle, setup)
    st.plotly_chart(fig, width="stretch")
    png = _export_chart_png(fig)
    if png:
        st.download_button("Download chart PNG", png, f"{setup.ticker}_chart.png", "image/png")


def _render_rs_leaderboard(bundle: ScanBundle | None) -> None:
    st.subheader("Relative strength leaderboard")
    if bundle is None or bundle.results.empty:
        st.info("No scan results yet.")
        return
    leaderboard = bundle.results.sort_values(["relative_strength_score", "score"], ascending=[False, False]).reset_index(drop=True)
    st.dataframe(leaderboard[["ticker", "market", "pattern", "relative_strength_score", "score", "grade"]], width="stretch")


def _render_backtest(bundle: ScanBundle | None) -> None:
    st.subheader("Backtest analytics")
    if bundle is None or not bundle.frame_cache:
        st.info("Scan first so the app has symbol history to backtest.")
        return
    options = sorted(bundle.frame_cache)
    symbol_key = st.selectbox("Backtest symbol", options)
    frame = bundle.frame_cache[symbol_key].get("1d")
    if frame is None or frame.empty:
        st.warning("Daily history is unavailable for that symbol.")
        return
    if st.button("Run walk-forward backtest", width="stretch"):
        result = run_backtest(frame)
        st.dataframe(result.summary, width="stretch")
        st.dataframe(result.walk_forward, width="stretch")
        st.dataframe(result.monte_carlo.head(30), width="stretch")
        st.dataframe(result.trades.head(50), width="stretch")


def _render_journal(config) -> None:
    st.subheader("Trade journal")
    store = JournalStore(config.journal_db_path)
    with st.form("journal_form"):
        ticker = st.text_input("Ticker")
        market = st.selectbox("Market", ["NSE", "US"])
        setup_family = st.selectbox("Setup family", ["breakout", "pullback"])
        grade = st.selectbox("Grade", ["A+", "A", "B", "C", "Reject"])
        note = st.text_area("Review note")
        submitted = st.form_submit_button("Save journal entry")
        if submitted and ticker and note:
            store.add_entry(datetime.now(timezone.utc).isoformat(), ticker.upper(), market, setup_family, grade, note)
            st.success("Journal entry saved.")
    entries = store.load_entries()
    if not entries.empty:
        st.dataframe(entries, width="stretch")


def _render_watchlist(config) -> None:
    st.subheader("Watchlist")
    nse_text = (config.project_root / "config" / "watchlist_nse_sample.txt").read_text(encoding="utf-8")
    us_text = (config.project_root / "config" / "watchlist_us_sample.txt").read_text(encoding="utf-8")
    col1, col2 = st.columns(2)
    col1.text_area("NSE sample watchlist", nse_text, height=300)
    col2.text_area("US sample watchlist", us_text, height=300)


def _render_alerts(bundle: ScanBundle | None, config) -> None:
    st.subheader("Alerts")
    if bundle is None or not bundle.setups:
        st.info("Scan first to generate alert candidates.")
        return
    min_alert_score = st.slider("Alert threshold", 65, 100, 85)
    setup_map = _setup_to_dict_map(bundle)
    option = st.selectbox("Alert candidate", list(setup_map), key="alert_setup")
    setup = setup_map[option]
    message = build_alert_message(setup)
    st.code(message)
    state = load_alert_state(config.alert_state_path)
    st.caption("Duplicate prevention is keyed by ticker, setup family, and breakout level.")
    if st.button("Send configured alerts", width="stretch"):
        if should_alert(setup, state, min_alert_score):
            logs = [send_telegram(message, config), send_email(message, f"STOPICK {setup.ticker}", config), send_webhook(setup.model_dump(), config)]
            state[f"{setup.ticker}|{setup.setup_family}|{setup.breakout.breakout_level}"] = round(setup.score, 2)
            save_alert_state(config.alert_state_path, state)
            st.success(" | ".join(logs))
        else:
            st.warning("This setup already triggered at the same breakout level or is below threshold.")


def _render_regime(bundle: ScanBundle | None) -> None:
    st.subheader("Market regime dashboard")
    if bundle is None or not bundle.setups:
        st.info("Scan first to summarize market regime.")
        return
    rows = []
    for setup in bundle.setups:
        if setup.regime:
            rows.append(
                {
                    "market": setup.market,
                    "benchmark": setup.regime.benchmark_symbol,
                    "direction": setup.regime.direction,
                    "trend_strength": setup.regime.trend_strength,
                    "volatility_state": setup.regime.volatility_state,
                    "breadth_like_proxy": setup.regime.breadth_like_proxy,
                },
            )
    regime_df = pd.DataFrame(rows).drop_duplicates()
    st.dataframe(regime_df, width="stretch")


def run_dashboard() -> None:
    config = load_app_config(Path(__file__).resolve().parent.parent)
    st.set_page_config(page_title=config.runtime.app_name, layout="wide")
    st.title(config.runtime.app_name)
    st.caption("Dual-universe technical workstation for only high-conviction setups. No lookahead assumptions. Confidence scores are probabilistic heuristics, not guarantees.")

    page = st.sidebar.radio(
        "Page",
        [
            "Scanner",
            "Top setups",
            "Market structure charts",
            "Relative strength leaderboard",
            "Backtest analytics",
            "Trade journal",
            "Watchlist",
            "Alerts",
            "Market regime dashboard",
        ],
    )

    st.sidebar.subheader("Scan controls")
    country = st.sidebar.selectbox("Country", ["BOTH", "NSE", "US"])
    source = st.sidebar.selectbox("Watchlist source", ["sample", "manual"])
    timeframe = st.sidebar.selectbox("Timeframe", ["1d", "4h", "1h", "15m"])
    setup_mode = st.sidebar.selectbox("Setup type", ["both", "breakout", "pullback"])
    min_score = st.sidebar.slider("Minimum score", 50, 100, config.runtime.min_score_default)
    refresh = st.sidebar.checkbox("Refresh market data", value=False)
    manual_watchlist = st.sidebar.text_area("Manual watchlist", value="RELIANCE, TCS, AAPL, NVDA", disabled=source != "manual")
    uploaded_file = st.sidebar.file_uploader("Upload watchlist CSV or TXT", type=["csv", "txt"])

    uploaded_text = None
    uploaded_frame = None
    if uploaded_file is not None:
        uploaded_text = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.lower().endswith(".csv"):
            uploaded_frame = pd.read_csv(BytesIO(uploaded_text.encode("utf-8")))
            uploaded_text = uploaded_frame.iloc[:, 0].astype(str).str.cat(sep=",")

    if st.sidebar.button("Run STOPICK scan", type="primary", width="stretch"):
        with st.spinner("Scanning both markets for high-conviction setups..."):
            st.session_state["stopick_bundle"] = scan_market(
                config,
                country=country,
                source=source,
                scan_timeframe=timeframe,
                minimum_score=min_score,
                setup_mode=setup_mode,
                manual_watchlist=manual_watchlist,
                uploaded_watchlist_text=uploaded_text,
                uploaded_watchlist_frame=uploaded_frame,
                refresh_data=refresh,
            )
        st.rerun()

    bundle: ScanBundle | None = st.session_state.get("stopick_bundle")
    if page == "Scanner":
        _render_scanner(bundle)
    elif page == "Top setups":
        _render_top_setups(bundle)
    elif page == "Market structure charts":
        _render_structure_charts(bundle)
    elif page == "Relative strength leaderboard":
        _render_rs_leaderboard(bundle)
    elif page == "Backtest analytics":
        _render_backtest(bundle)
    elif page == "Trade journal":
        _render_journal(config)
    elif page == "Watchlist":
        _render_watchlist(config)
    elif page == "Alerts":
        _render_alerts(bundle, config)
    elif page == "Market regime dashboard":
        _render_regime(bundle)
