"""
Vega Engine v1 — Live Volatility Dashboard
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

from volatility_trading_algorithm import (
    BlackScholesModel,
    VolatilitySurface,
    VolatilityTradingStrategy,
)
from data_fetcher import (
    get_spot_price,
    get_risk_free_rate,
    get_price_history,
    get_options_chain,
    compute_realized_vols,
    get_atm_iv_series,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Vega Engine v1",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: #0e1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e6edf3; font-size: 26px; font-weight: 700; margin-top: 4px; }
    .metric-delta-pos { color: #3fb950; font-size: 13px; }
    .metric-delta-neg { color: #f85149; font-size: 13px; }
    .signal-buy  { background:#1a3a2a; color:#3fb950; padding:6px 14px; border-radius:6px; font-weight:700; }
    .signal-sell { background:#3a1a1a; color:#f85149; padding:6px 14px; border-radius:6px; font-weight:700; }
    .signal-neutral { background:#1c2128; color:#8b949e; padding:6px 14px; border-radius:6px; font-weight:700; }
    [data-testid="stSidebar"] { background-color: #0d1117; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

BS = BlackScholesModel()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Vega Engine v1")
    st.markdown("*Live Volatility Analytics*")
    st.divider()

    ticker = st.text_input("Ticker", value="SPY", max_chars=10).upper().strip()
    rv_window = st.slider("RV Window (days)", 5, 60, 20)
    iv_threshold = st.slider("IV Z-Score Threshold", 0.5, 3.0, 1.5, step=0.25)
    iv_rv_threshold = st.slider("IV-RV Spread Threshold", 0.01, 0.15, 0.05, step=0.01)

    st.divider()
    run = st.button("🔄 Load / Refresh", use_container_width=True, type="primary")

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Pulls live options chains via yfinance. "
        "Builds a real volatility surface, computes Greeks, "
        "and generates IV mean-reversion signals."
    )
    st.markdown("[GitHub](https://github.com/aniXC12/vega-engine-v1)")

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, rv_window: int):
    spot = get_spot_price(ticker)
    rfr = get_risk_free_rate()
    hist = get_price_history(ticker, period="1y")
    hist = compute_realized_vols(hist, window=rv_window)
    options_df, options_list = get_options_chain(ticker, spot, rfr)
    return spot, rfr, hist, options_df, options_list


if "loaded" not in st.session_state:
    st.session_state.loaded = False

if run or not st.session_state.loaded:
    with st.spinner(f"Fetching live data for **{ticker}**…"):
        try:
            spot, rfr, hist, options_df, options_list = load_data(ticker, rv_window)
            st.session_state.update(
                loaded=True, spot=spot, rfr=rfr, hist=hist,
                options_df=options_df, options_list=options_list,
                ticker=ticker,
            )
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.stop()

if not st.session_state.loaded:
    st.info("Click **Load / Refresh** in the sidebar to begin.")
    st.stop()

spot       = st.session_state.spot
rfr        = st.session_state.rfr
hist       = st.session_state.hist
options_df = st.session_state.options_df
options_list = st.session_state.options_list
loaded_ticker = st.session_state.ticker

# ── Header metrics ─────────────────────────────────────────────────────────────

daily_ret = hist["close"].pct_change().iloc[-1]
yz_rv = hist["rv_yz"].dropna().iloc[-1] if not hist["rv_yz"].dropna().empty else np.nan
atm_iv = options_df[
    (options_df["moneyness"].between(0.98, 1.02)) &
    (options_df["expiry_years"] < 0.15)
]["implied_vol"].median() if not options_df.empty else np.nan

iv_rv_spread = (atm_iv - yz_rv) if (not np.isnan(atm_iv) and not np.isnan(yz_rv)) else np.nan

st.markdown(f"## {loaded_ticker} — Volatility Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, label, value, delta=None, delta_pos_good=True):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if (delta >= 0) == delta_pos_good else "metric-delta-neg"
        sign = "+" if delta >= 0 else ""
        delta_html = f'<div class="{cls}">{sign}{delta:.2%}</div>'
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

metric_card(c1, "Spot Price",  f"${spot:,.2f}", daily_ret)
metric_card(c2, "ATM IV (1M)", f"{atm_iv:.1%}" if not np.isnan(atm_iv) else "N/A")
metric_card(c3, "Realized Vol (YZ)", f"{yz_rv:.1%}" if not np.isnan(yz_rv) else "N/A")
metric_card(c4, "IV−RV Spread", f"{iv_rv_spread:+.1%}" if not np.isnan(iv_rv_spread) else "N/A",
            delta_pos_good=False)
metric_card(c5, "Risk-Free Rate", f"{rfr:.2%}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_surface, tab_greeks, tab_signals, tab_risk = st.tabs([
    "🌋 Vol Surface", "🔢 Greeks", "📡 Signals", "⚖️ Risk"
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — VOLATILITY SURFACE
# ═══════════════════════════════════════════════════════════════════

with tab_surface:
    if options_df.empty:
        st.warning("No options data available for this ticker.")
    else:
        st.markdown("### Live Implied Volatility Surface")

        # Pivot to grid for surface plot
        calls_df = options_df[options_df["option_type"] == "call"].copy()
        calls_df = calls_df[(calls_df["moneyness"] > 0.7) & (calls_df["moneyness"] < 1.4)]
        calls_df = calls_df[calls_df["expiry_years"] <= 1.5]

        if len(calls_df) >= 6:
            mono_vals = np.linspace(calls_df["moneyness"].min(), calls_df["moneyness"].max(), 60)
            exp_vals  = np.linspace(calls_df["expiry_years"].min(), calls_df["expiry_years"].max(), 40)
            mono_grid, exp_grid = np.meshgrid(mono_vals, exp_vals)

            iv_grid = griddata(
                calls_df[["moneyness", "expiry_years"]].values,
                calls_df["implied_vol"].values,
                (mono_grid, exp_grid),
                method="cubic",
            )
            # Fill NaN holes with linear
            iv_grid_lin = griddata(
                calls_df[["moneyness", "expiry_years"]].values,
                calls_df["implied_vol"].values,
                (mono_grid, exp_grid),
                method="linear",
            )
            iv_grid = np.where(np.isnan(iv_grid), iv_grid_lin, iv_grid)

            exp_days = exp_vals * 365

            fig_surface = go.Figure(data=[go.Surface(
                x=mono_vals,
                y=exp_days,
                z=iv_grid * 100,
                colorscale="Plasma",
                colorbar=dict(title="IV (%)", tickformat=".0f"),
                hovertemplate=(
                    "Moneyness: %{x:.3f}<br>"
                    "Expiry: %{y:.0f}d<br>"
                    "IV: %{z:.1f}%<extra></extra>"
                ),
            )])

            fig_surface.add_shape(
                type="line", line=dict(color="white", width=2, dash="dot"),
                x0=1.0, x1=1.0, y0=exp_days.min(), y1=exp_days.max(),
            )

            fig_surface.update_layout(
                scene=dict(
                    xaxis_title="Moneyness (K/S)",
                    yaxis_title="Days to Expiry",
                    zaxis_title="Implied Vol (%)",
                    bgcolor="#0e1117",
                    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
                    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
                    zaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
                ),
                paper_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
                height=600,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_surface, use_container_width=True)

        # Skew charts per expiry
        st.markdown("### Volatility Smile by Expiry")
        exp_options = sorted(calls_df["expiry_date"].unique())[:6]
        selected_exps = st.multiselect(
            "Select expiries to display",
            exp_options,
            default=exp_options[:min(3, len(exp_options))],
        )

        if selected_exps:
            fig_smile = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, exp in enumerate(selected_exps):
                subset = calls_df[calls_df["expiry_date"] == exp].sort_values("moneyness")
                days = int(subset["expiry_years"].iloc[0] * 365)
                fig_smile.add_trace(go.Scatter(
                    x=subset["moneyness"],
                    y=subset["implied_vol"] * 100,
                    mode="lines+markers",
                    name=f"{exp} ({days}d)",
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=5),
                    hovertemplate="Moneyness: %{x:.3f}<br>IV: %{y:.1f}%<extra></extra>",
                ))
            fig_smile.add_vline(x=1.0, line=dict(color="white", dash="dot", width=1.5))
            fig_smile.update_layout(
                xaxis_title="Moneyness (K/S)",
                yaxis_title="Implied Vol (%)",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="#e6edf3"),
                legend=dict(bgcolor="#0e1117"),
                xaxis=dict(gridcolor="#21262d"),
                yaxis=dict(gridcolor="#21262d"),
                height=380,
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_smile, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — GREEKS CALCULATOR
# ═══════════════════════════════════════════════════════════════════

with tab_greeks:
    st.markdown("### Interactive Black-Scholes Greeks")

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        g_strike   = st.number_input("Strike (K)", value=round(spot), step=1.0)
        g_expiry   = st.slider("Days to Expiry", 1, 730, 30)
        g_vol      = st.slider("Implied Vol (%)", 5, 150, int(atm_iv * 100) if not np.isnan(atm_iv) else 25)
        g_type     = st.radio("Option Type", ["call", "put"], horizontal=True)
        g_qty      = st.number_input("Contracts (qty)", value=1, step=1)
        g_rfr      = st.number_input("Risk-Free Rate (%)", value=round(rfr * 100, 2), step=0.25)

    T = g_expiry / 365
    sig = g_vol / 100
    r_g = g_rfr / 100

    if g_type == "call":
        price = BS.call_price(spot, g_strike, T, r_g, sig)
    else:
        price = BS.put_price(spot, g_strike, T, r_g, sig)

    delta_v  = BS.delta(spot, g_strike, T, r_g, sig, g_type)
    gamma_v  = BS.gamma(spot, g_strike, T, r_g, sig)
    vega_v   = BS.vega(spot, g_strike, T, r_g, sig)
    theta_v  = BS.theta(spot, g_strike, T, r_g, sig, g_type)

    with col_output:
        st.markdown(f"#### {g_type.upper()} — ${g_strike} — {g_expiry}d")

        def greek_row(name, value, unit="", fmt=".4f"):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:8px 0;'
                f'border-bottom:1px solid #21262d">'
                f'<span style="color:#8b949e">{name}</span>'
                f'<span style="color:#e6edf3;font-weight:600">{value:{fmt}}{unit}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        greek_row("Option Price",  price * g_qty,   " / contract", ".2f")
        greek_row("Total Premium", price * g_qty * 100, "",          ".2f")
        greek_row("Delta",         delta_v * g_qty)
        greek_row("Gamma",         gamma_v * g_qty)
        greek_row("Vega",          vega_v * g_qty,  " / 1% vol")
        greek_row("Theta",         theta_v * g_qty, " / day",       ".4f")

        hedge_shares = -delta_v * g_qty * 100
        st.markdown(f"""
        <div style="margin-top:16px;padding:12px;background:#161b22;border-radius:8px;
                    border:1px solid #21262d">
            <div style="color:#8b949e;font-size:12px;text-transform:uppercase">
                Delta Hedge
            </div>
            <div style="color:#e6edf3;font-size:18px;font-weight:700;margin-top:4px">
                {"BUY" if hedge_shares > 0 else "SELL"} {abs(hedge_shares):.1f} shares of {loaded_ticker}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Greeks vs Spot chart
    st.markdown("### Greeks vs Spot Price")
    spot_range = np.linspace(spot * 0.7, spot * 1.3, 200)

    deltas = [BS.delta(s, g_strike, T, r_g, sig, g_type) for s in spot_range]
    gammas = [BS.gamma(s, g_strike, T, r_g, sig) for s in spot_range]
    prices_range = [
        BS.call_price(s, g_strike, T, r_g, sig) if g_type == "call"
        else BS.put_price(s, g_strike, T, r_g, sig)
        for s in spot_range
    ]

    fig_greeks = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Option Price", "Delta", "Gamma"),
    )
    kw = dict(mode="lines", showlegend=False)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=prices_range, line=dict(color="#58a6ff", width=2), **kw), row=1, col=1)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=deltas,       line=dict(color="#3fb950", width=2), **kw), row=1, col=2)
    fig_greeks.add_trace(go.Scatter(x=spot_range, y=gammas,       line=dict(color="#d2a8ff", width=2), **kw), row=1, col=3)

    for col in range(1, 4):
        fig_greeks.add_vline(x=spot,     line=dict(color="#e6edf3", dash="dot", width=1), row=1, col=col)
        fig_greeks.add_vline(x=g_strike, line=dict(color="#f0883e", dash="dash", width=1), row=1, col=col)

    fig_greeks.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e6edf3"), height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_greeks.update_xaxes(gridcolor="#21262d", title_text="Spot Price")
    fig_greeks.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig_greeks, use_container_width=True)
    st.caption("White dotted = current spot · Orange dashed = strike")

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — SIGNALS
# ═══════════════════════════════════════════════════════════════════

with tab_signals:
    st.markdown("### Volatility Signal Dashboard")

    strategy = VolatilityTradingStrategy(risk_free_rate=rfr)
    hist_clean = hist.dropna(subset=["rv_yz"])

    # Build synthetic IV proxy from options if available
    if not options_df.empty:
        # Use near-dated ATM calls to proxy current IV
        near = options_df[
            (options_df["option_type"] == "call") &
            (options_df["moneyness"].between(0.97, 1.03)) &
            (options_df["expiry_years"] < 0.15)
        ]
        current_iv = near["implied_vol"].median() if not near.empty else atm_iv
    else:
        current_iv = atm_iv

    current_rv = hist_clean["rv_yz"].iloc[-1] if not hist_clean.empty else np.nan

    # Rolling 30-day IV proxy using historical close-to-close vol as stand-in
    # (Real IV time series would need a premium data provider)
    hist_iv_proxy = hist_clean["rv_yz"].rolling(5).mean() * 1.15  # stylized IV ~ RV + premium

    if not np.isnan(current_iv):
        hist_iv_array = hist_iv_proxy.dropna().values[-60:]
        mr_signal = strategy.volatility_mean_reversion_signal(
            current_iv, hist_iv_array, threshold=iv_threshold
        )
        ivrv_signal = strategy.iv_rv_spread_signal(
            current_iv, current_rv, threshold=iv_rv_threshold
        )
    else:
        mr_signal = ivrv_signal = "NEUTRAL"

    # Signal badges
    def signal_badge(sig: str) -> str:
        if sig == "BUY_VOL":
            return f'<span class="signal-buy">BUY VOL</span>'
        elif sig == "SELL_VOL":
            return f'<span class="signal-sell">SELL VOL</span>'
        return f'<span class="signal-neutral">NEUTRAL</span>'

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**Mean Reversion Signal**")
        st.markdown(signal_badge(mr_signal), unsafe_allow_html=True)
        if not np.isnan(current_iv):
            iv_mean = np.mean(hist_iv_array)
            iv_std  = np.std(hist_iv_array)
            z = (current_iv - iv_mean) / iv_std if iv_std > 0 else 0
            st.markdown(f"Z-score: **{z:+.2f}** (threshold ±{iv_threshold})")

    with s2:
        st.markdown("**IV−RV Spread Signal**")
        st.markdown(signal_badge(ivrv_signal), unsafe_allow_html=True)
        if not np.isnan(current_iv) and not np.isnan(current_rv):
            spread = current_iv - current_rv
            st.markdown(f"Spread: **{spread:+.1%}** (threshold ±{iv_rv_threshold:.0%})")

    with s3:
        combined = "NEUTRAL"
        if mr_signal == ivrv_signal and mr_signal != "NEUTRAL":
            combined = mr_signal
        st.markdown("**Combined Signal**")
        st.markdown(signal_badge(combined), unsafe_allow_html=True)
        if combined == "SELL_VOL":
            st.markdown("→ Consider short straddle / strangle")
        elif combined == "BUY_VOL":
            st.markdown("→ Consider long straddle / strangle")
        else:
            st.markdown("→ Mixed signals, stay observant")

    st.divider()

    # RV vs IV chart
    st.markdown("### Realized Volatility vs IV Proxy")
    plot_hist = hist_clean.tail(252).copy()
    plot_hist["iv_proxy"] = hist_iv_proxy.reindex(plot_hist.index)

    fig_rv = go.Figure()
    for col, color, name in [
        ("rv_parkinson", "#58a6ff", "Parkinson RV"),
        ("rv_gk",        "#3fb950", "Garman-Klass RV"),
        ("rv_yz",        "#d2a8ff", "Yang-Zhang RV"),
        ("iv_proxy",     "#f0883e", "IV Proxy (stylized)"),
    ]:
        if col in plot_hist.columns:
            fig_rv.add_trace(go.Scatter(
                x=plot_hist.index, y=plot_hist[col] * 100,
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
                hovertemplate="%{y:.1f}%<extra>" + name + "</extra>",
            ))

    if not np.isnan(current_iv):
        fig_rv.add_hline(
            y=current_iv * 100,
            line=dict(color="#f85149", width=1.5, dash="dash"),
            annotation_text=f"Current ATM IV: {current_iv:.1%}",
            annotation_position="top left",
            annotation_font_color="#f85149",
        )

    fig_rv.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e6edf3"),
        legend=dict(bgcolor="#0e1117", bordercolor="#21262d", borderwidth=1),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Annualized Vol (%)"),
        height=380,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig_rv, use_container_width=True)

    # Price chart with vol overlay
    st.markdown("### Price History")
    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               row_heights=[0.65, 0.35], vertical_spacing=0.04)
    fig_price.add_trace(go.Scatter(
        x=plot_hist.index, y=plot_hist["close"],
        mode="lines", name="Close",
        line=dict(color="#58a6ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.05)",
    ), row=1, col=1)
    fig_price.add_trace(go.Bar(
        x=plot_hist.index, y=plot_hist["volume"],
        name="Volume", marker_color="#21262d",
    ), row=2, col=1)

    fig_price.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e6edf3"),
        showlegend=False,
        xaxis2=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Price ($)"),
        yaxis2=dict(gridcolor="#21262d", title="Volume"),
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — RISK
# ═══════════════════════════════════════════════════════════════════

with tab_risk:
    st.markdown("### Portfolio Risk & Arbitrage Scanner")

    col_arb, col_skew = st.columns(2)

    with col_arb:
        st.markdown("#### Calendar Spread Arbitrage")
        if not options_df.empty and len(options_list) >= 4:
            with st.spinner("Running arbitrage scan…"):
                try:
                    # Use a subset for performance
                    sample = options_list[:min(len(options_list), 200)]
                    vol_surface = VolatilitySurface(sample, spot, rfr)
                    arb_opps = vol_surface.detect_arbitrage_opportunities()
                except Exception:
                    arb_opps = []
            if arb_opps:
                arb_df = pd.DataFrame(arb_opps)
                arb_df["iv_difference"] = arb_df["iv_difference"].map("{:.2%}".format)
                arb_df["moneyness"] = arb_df["moneyness"].map("{:.3f}".format)
                st.dataframe(
                    arb_df[["type", "moneyness", "near_expiry", "far_expiry",
                             "iv_difference", "signal"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success("No calendar spread arbitrage detected.")
        else:
            st.info("Insufficient options data for arbitrage scan.")

    with col_skew:
        st.markdown("#### Volatility Skew by Expiry")
        if not options_df.empty:
            calls_sub = options_df[options_df["option_type"] == "call"]
            exp_list = sorted(calls_sub["expiry_date"].unique())[:6]
            skew_rows = []
            for exp in exp_list:
                sub = calls_sub[calls_sub["expiry_date"] == exp]
                if len(sub) < 3:
                    continue
                atm_idx = (sub["moneyness"] - 1.0).abs().idxmin()
                atm_v = sub.loc[atm_idx, "implied_vol"]
                otm_p = sub[sub["moneyness"] < 0.95]["implied_vol"].mean()
                otm_c = sub[sub["moneyness"] > 1.05]["implied_vol"].mean()
                days = int(sub["expiry_years"].iloc[0] * 365)
                skew_rows.append({
                    "Expiry": exp,
                    "Days": days,
                    "ATM IV": f"{atm_v:.1%}",
                    "Put Skew": f"{(otm_p - atm_v):+.1%}" if not np.isnan(otm_p) else "N/A",
                    "Call Skew": f"{(otm_c - atm_v):+.1%}" if not np.isnan(otm_c) else "N/A",
                })
            if skew_rows:
                st.dataframe(pd.DataFrame(skew_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No options data available.")

    st.divider()
    st.markdown("### Straddle P&L Simulator")

    p1, p2, p3 = st.columns(3)
    with p1:
        sim_strike = st.number_input("Strike", value=round(spot), step=1.0, key="sim_k")
        sim_expiry = st.slider("Days to Expiry", 1, 180, 30, key="sim_t")
    with p2:
        sim_vol = st.slider("Entry IV (%)", 5, 100,
                            int(atm_iv * 100) if not np.isnan(atm_iv) else 25,
                            key="sim_v")
        sim_qty = st.number_input("Contracts", value=1, min_value=1, key="sim_q")
    with p3:
        sim_position = st.radio("Position", ["Long Straddle", "Short Straddle"], key="sim_pos")

    T_sim = sim_expiry / 365
    sig_sim = sim_vol / 100
    call_px = BS.call_price(spot, sim_strike, T_sim, rfr, sig_sim)
    put_px  = BS.put_price(spot, sim_strike, T_sim, rfr, sig_sim)
    premium = (call_px + put_px) * sim_qty * 100

    spot_scenarios = np.linspace(spot * 0.7, spot * 1.3, 300)
    pnl_at_exp = []
    for s in spot_scenarios:
        c_val = max(s - sim_strike, 0)
        p_val = max(sim_strike - s, 0)
        payoff = (c_val + p_val) * sim_qty * 100
        if sim_position == "Long Straddle":
            pnl_at_exp.append(payoff - premium)
        else:
            pnl_at_exp.append(premium - payoff)

    fig_pnl = go.Figure()
    colors_pnl = ["#3fb950" if p >= 0 else "#f85149" for p in pnl_at_exp]
    fig_pnl.add_trace(go.Scatter(
        x=spot_scenarios, y=pnl_at_exp,
        mode="lines", name="P&L at Expiry",
        line=dict(color="#58a6ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(88,166,255,0.08)",
        hovertemplate="Spot: $%{x:.2f}<br>P&L: $%{y:,.0f}<extra></extra>",
    ))
    fig_pnl.add_hline(y=0, line=dict(color="#8b949e", width=1))
    fig_pnl.add_vline(x=spot,       line=dict(color="white", dash="dot", width=1.5),
                      annotation_text="Current Spot", annotation_font_color="white")
    fig_pnl.add_vline(x=sim_strike, line=dict(color="#f0883e", dash="dash", width=1.5),
                      annotation_text="Strike", annotation_font_color="#f0883e")

    be_up   = sim_strike + (premium / (sim_qty * 100))
    be_down = sim_strike - (premium / (sim_qty * 100))
    if sim_position == "Long Straddle":
        fig_pnl.add_vline(x=be_up,   line=dict(color="#3fb950", dash="dot", width=1))
        fig_pnl.add_vline(x=be_down, line=dict(color="#3fb950", dash="dot", width=1))

    fig_pnl.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e6edf3"),
        xaxis=dict(gridcolor="#21262d", title="Spot at Expiry ($)"),
        yaxis=dict(gridcolor="#21262d", title="P&L ($)"),
        height=360,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Premium", f"${premium:,.0f}")
    m2.metric("Max Profit" if sim_position == "Long Straddle" else "Max Profit",
              "Unlimited" if sim_position == "Long Straddle" else f"${premium:,.0f}")
    m3.metric("Max Loss",
              f"${premium:,.0f}" if sim_position == "Long Straddle" else "Unlimited")
    m4.metric("Breakeven", f"${be_down:,.1f} / ${be_up:,.1f}")
