import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ====================================================
# PAGE CONFIG
# ====================================================

st.set_page_config(
    page_title="Climate Risk Dashboard | Team DEBUG",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================
# CUSTOM CSS
# ====================================================

st.markdown("""
<style>
    /* Main font & background */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default Streamlit header padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
    }
    [data-testid="metric-container"] label {
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #7dd3fc !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #86efac !important;
    }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 2rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(148,163,184,0.2);
        padding-bottom: 0.4rem;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
    }
    .risk-HIGH   { background:#fee2e2; color:#991b1b; }
    .risk-MEDIUM { background:#fef9c3; color:#854d0e; }
    .risk-LOW    { background:#dcfce7; color:#166534; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Info box */
    .info-box {
        background: rgba(125,211,252,0.07);
        border-left: 3px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.87rem;
        color: #cbd5e1;
        margin: 0.6rem 0 1rem;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# INLINE DATA  (from climate_data.csv)
# ====================================================

YEARS = list(range(2000, 2025))

RAW = {
    "GDP": [
        1.70e13, 1.76e13, 1.85e13, 1.95e13, 2.08e13, 2.22e13, 2.38e13,
        2.57e13, 2.70e13, 2.80e13, 3.02e13, 3.20e13, 3.38e13, 3.57e13,
        3.76e13, 3.95e13, 4.15e13, 4.37e13, 4.60e13, 4.80e13, 4.80e13,
        5.11e13, 5.28e13, 5.51e13, 5.74e13
    ],
    "Forest_Area": [
        26.05, 26.12, 26.19, 26.25, 26.32, 26.38, 26.45, 26.51, 26.58,
        26.65, 26.72, 26.77, 26.82, 26.87, 26.92, 26.98, 27.09, 27.09,
        27.12, 27.15, 27.18, 27.19, 27.23, None, None
    ],
    "Urban_Population": [
        41.63, 42.62, 43.77, 44.95, 46.01, 47.04, 48.12, 49.30, 50.22,
        51.27, 52.02, 53.77, 54.70, 55.70, 56.61, 57.72, 58.79, 59.80,
        60.85, 61.74, 62.37, 63.24, 63.68, 64.02, 64.37
    ],
    "Renewable_Energy_Consumption": [
        22.47, 21.79, 21.17, 19.84, 17.98, 16.43, 15.97, 15.08, 14.74,
        14.26, 13.43, 12.73, 12.80, 12.81, 13.18, 12.98, 13.33, 13.58,
        13.79, 14.24, 14.81, None, None, None, None
    ],
    "Total_GHG_Emissions": [
        8366.1, 8743.7, 9145.5, 9858.5, 11011.9, 11971.7, 13082.0,
        13601.4, 13656.6, 14385.3, 15069.2, 16274.8, 16788.7, 17320.2,
        17736.5, 18211.4, 17453.0, 17575.4, 17969.0, 18631.6, 18140.1,
        18793.0, 18920.6, 19591.8, None
    ],
}

df_t = pd.DataFrame(RAW, index=YEARS)
df_t.index.name = "Year"
df_t.index = df_t.index.astype(int)

# Model data (drop rows with any NaN in required columns)
MODEL_COLS = ["GDP", "Forest_Area", "Renewable_Energy_Consumption", "Total_GHG_Emissions"]
df_model = df_t[MODEL_COLS].dropna()

# ====================================================
# MODEL TRAINING
# ====================================================

X = df_model[["GDP", "Forest_Area", "Renewable_Energy_Consumption"]]
y = df_model["Total_GHG_Emissions"]
model = LinearRegression()
model.fit(X, y)
y_pred_train = model.predict(X)
r2 = r2_score(y, y_pred_train)

# ====================================================
# DEFAULT VALUES
# ====================================================

latest_gdp       = float(df_t.loc[2024, "GDP"])
default_forest   = 27.23
default_renewable = 14.81
default_urban    = float(df_t.loc[2024, "Urban_Population"])

# ====================================================
# HEADER
# ====================================================

col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🌍 Climate Risk Assessment Dashboard")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.95rem; margin-top:-0.4rem;'>"
        "MASA Hackathon 2026 · Team DEBUG · Sunway University</p>",
        unsafe_allow_html=True
    )
with col_h2:
    st.markdown(
        "<div style='text-align:right; padding-top:0.6rem;'>"
        "<span style='background:#1e293b; border:1px solid #334155; "
        "border-radius:8px; padding:0.4rem 0.8rem; font-size:0.8rem; color:#7dd3fc;'>"
        "Malaysia · 2000–2024</span></div>",
        unsafe_allow_html=True
    )

st.divider()

# ====================================================
# SIDEBAR INPUT
# ====================================================

with st.sidebar:
    st.markdown("## ⚙️ Scenario Inputs")
    st.markdown(
        "<div class='info-box'>Adjust variables to simulate different "
        "climate & economic scenarios.</div>",
        unsafe_allow_html=True
    )

    gdp = st.number_input(
        "GDP (PPP, 2021 intl $)",
        value=latest_gdp,
        min_value=1e12,
        max_value=1e14,
        step=1e11,
        format="%.2e",
        help="Gross Domestic Product in constant 2021 international dollars"
    )
    forest = st.slider(
        "Forest Area (% of land area)",
        min_value=20.0, max_value=40.0,
        value=default_forest, step=0.1,
        help="Share of total land area covered by forest"
    )
    urban = st.slider(
        "Urban Population (%)",
        min_value=30.0, max_value=90.0,
        value=default_urban, step=0.1,
        help="Share of population living in urban areas"
    )
    renewable = st.slider(
        "Renewable Energy (%)",
        min_value=5.0, max_value=60.0,
        value=default_renewable, step=0.5,
        help="Renewable energy as % of total final energy consumption"
    )

    st.divider()
    st.markdown(
        "<p style='font-size:0.75rem; color:#475569;'>"
        "Data sources: World Bank, EM-DAT, Our World in Data, "
        "Global Carbon Budget (2025)</p>",
        unsafe_allow_html=True
    )

# ====================================================
# PREDICTION & RISK
# ====================================================

input_df = pd.DataFrame(
    [[gdp, forest, renewable]],
    columns=["GDP", "Forest_Area", "Renewable_Energy_Consumption"]
)
prediction = model.predict(input_df)[0]

if prediction < 15000:
    risk = "LOW"
    risk_color = "#22c55e"
    risk_icon  = "✅"
elif prediction < 22000:
    risk = "MEDIUM"
    risk_color = "#f59e0b"
    risk_icon  = "⚠️"
else:
    risk = "HIGH"
    risk_color = "#ef4444"
    risk_icon  = "🔴"

# Historical last known emission
last_known_emission = df_t["Total_GHG_Emissions"].dropna().iloc[-1]
delta_vs_last = prediction - last_known_emission

# Baseline 2030 projection (linear extrapolation from model)
years_proj  = np.array([2024, 2025, 2026, 2027, 2028, 2029, 2030])
gdp_proj    = np.linspace(latest_gdp, latest_gdp * 1.15, len(years_proj))
forest_proj = np.linspace(default_forest, default_forest * 1.01, len(years_proj))
renew_base  = np.linspace(default_renewable, default_renewable * 1.05, len(years_proj))
renew_stress= np.linspace(default_renewable, default_renewable * 1.30, len(years_proj))

baseline_proj = [
    model.predict(pd.DataFrame([[g, f, r]], columns=["GDP","Forest_Area","Renewable_Energy_Consumption"]))[0]
    for g, f, r in zip(gdp_proj, forest_proj, renew_base)
]
stress_proj = [
    model.predict(pd.DataFrame([[g, f, r]], columns=["GDP","Forest_Area","Renewable_Energy_Consumption"]))[0]
    for g, f, r in zip(gdp_proj, forest_proj, renew_stress)
]

baseline_2030 = baseline_proj[-1]
stress_2030   = stress_proj[-1]
reduction     = baseline_2030 - stress_2030
pct_reduction = reduction / baseline_2030 * 100

# ====================================================
# TOP KPI CARDS
# ====================================================

st.markdown("<p class='section-header'>📊 Key Metrics</p>", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        "Predicted GHG Emissions",
        f"{prediction:,.0f} Mt CO₂e",
        delta=f"{delta_vs_last:+,.0f} vs last known",
        delta_color="inverse"
    )
with k2:
    st.metric(
        "Climate Risk Level",
        f"{risk_icon} {risk}",
        delta=f"R² = {r2:.3f}",
    )
with k3:
    st.metric(
        "Baseline 2030 Projection",
        f"{baseline_2030:,.0f} Mt CO₂e",
    )
with k4:
    st.metric(
        "Stress Scenario Reduction",
        f"−{reduction:,.0f} Mt CO₂e",
        delta=f"−{pct_reduction:.1f}% by 2030",
    )

# ====================================================
# RISK GAUGE & RECOMMENDATION
# ====================================================

st.markdown("<p class='section-header'>🎯 Risk Assessment</p>", unsafe_allow_html=True)

col_gauge, col_rec = st.columns([1, 2])

with col_gauge:
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        delta={"reference": last_known_emission, "relative": False,
               "valueformat": ",.0f", "suffix": " Mt"},
        number={"suffix": " Mt CO₂e", "valueformat": ",.0f",
                "font": {"size": 20}},
        gauge={
            "axis": {"range": [5000, 30000], "tickwidth": 1, "tickcolor": "#475569"},
            "bar": {"color": risk_color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "borderwidth": 0,
            "steps": [
                {"range": [5000,  15000], "color": "#14532d"},
                {"range": [15000, 22000], "color": "#713f12"},
                {"range": [22000, 30000], "color": "#7f1d1d"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": prediction
            }
        },
        title={"text": "GHG Emissions Gauge<br><span style='font-size:11px'>Green ≤15k · Amber ≤22k · Red >22k</span>",
               "font": {"size": 14}}
    ))
    gauge_fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

with col_rec:
    progress_val = min(int(prediction / 30000 * 100), 100)
    st.markdown(f"**Risk Score:** {progress_val}% of maximum threshold")
    st.progress(progress_val)
    st.markdown("<br>", unsafe_allow_html=True)

    if risk == "HIGH":
        st.error(f"""
**🔴 High Climate Risk** — Predicted emissions exceed 22,000 Mt CO₂e

**Underwriting Implications:**
- Apply climate-adjusted premium loading (suggested: +15–25%)
- Increase catastrophe reserves to cover tail-risk events
- Tighten underwriting standards for flood & storm exposures
- Mandatory climate stress-testing for large portfolios
""")
    elif risk == "MEDIUM":
        st.warning(f"""
**⚠️ Medium Climate Risk** — Predicted emissions between 15,000–22,000 Mt CO₂e

**Underwriting Implications:**
- Monitor exposure trends quarterly
- Review pricing annually against emission trajectory
- Expand climate resilience programs for clients
- Stress-test reinsurance treaties for extreme weather
""")
    else:
        st.success(f"""
**✅ Low Climate Risk** — Predicted emissions below 15,000 Mt CO₂e

**Underwriting Implications:**
- Maintain current portfolio stability
- Encourage and reward renewable energy transitions
- Consider green insurance product development
- Continue monitoring for emerging exposures
""")

# ====================================================
# TABS
# ====================================================

st.markdown("<p class='section-header'>📈 Analysis & Insights</p>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 GHG Trend & Projection",
    "🔗 Correlation Analysis",
    "🌊 Malaysia vs Philippines",
    "🌳 Deforestation & Emissions"
])

# ──────────────────────────────────────────────
# TAB 1 · GHG TREND & PROJECTION
# ──────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        hist_years  = list(df_t["Total_GHG_Emissions"].dropna().index)
        hist_values = df_t["Total_GHG_Emissions"].dropna().values.tolist()

        fig_trend = go.Figure()

        # Historical
        fig_trend.add_trace(go.Scatter(
            x=hist_years, y=hist_values,
            mode="lines+markers",
            name="Historical",
            line=dict(color="#38bdf8", width=2.5),
            marker=dict(size=5),
            hovertemplate="%{x}: %{y:,.0f} Mt CO₂e<extra></extra>"
        ))

        # Fitted
        fitted_years = list(df_model.index)
        fig_trend.add_trace(go.Scatter(
            x=fitted_years, y=y_pred_train.tolist(),
            mode="lines",
            name=f"Model Fit (R²={r2:.3f})",
            line=dict(color="#a78bfa", width=1.8, dash="dot"),
            hovertemplate="%{x}: %{y:,.0f} Mt CO₂e<extra></extra>"
        ))

        # Baseline projection
        fig_trend.add_trace(go.Scatter(
            x=list(years_proj), y=baseline_proj,
            mode="lines+markers",
            name="Baseline 2030",
            line=dict(color="#fb923c", width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{x}: %{y:,.0f} Mt CO₂e<extra></extra>"
        ))

        # Stress scenario
        fig_trend.add_trace(go.Scatter(
            x=list(years_proj), y=stress_proj,
            mode="lines+markers",
            name="Stress Scenario (↑30% renewables)",
            line=dict(color="#4ade80", width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate="%{x}: %{y:,.0f} Mt CO₂e<extra></extra>"
        ))

        # Confidence band
        band_upper = [v * 1.06 for v in baseline_proj]
        band_lower = [v * 0.94 for v in baseline_proj]
        fig_trend.add_trace(go.Scatter(
            x=list(years_proj) + list(years_proj)[::-1],
            y=band_upper + band_lower[::-1],
            fill="toself",
            fillcolor="rgba(251,146,60,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±6% Uncertainty",
            hoverinfo="skip"
        ))

        fig_trend.update_layout(
            title="Total GHG Emissions: Historical Trend & 2030 Projections",
            xaxis_title="Year",
            yaxis_title="Mt CO₂e",
            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="left"),
            height=420,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=120, b=50, l=60, r=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        st.markdown("#### 2030 Scenario Summary")
        st.markdown(f"""
| Scenario | 2030 Emissions |
|---|---|
| **Baseline** | {baseline_2030:,.0f} Mt |
| **Stress**   | {stress_2030:,.0f} Mt |
| **Reduction**| {reduction:,.0f} Mt |
| **% Cut**    | {pct_reduction:.1f}% |
""")
        st.divider()
        st.markdown("#### Model Coefficients")
        coef_df = pd.DataFrame({
            "Feature": ["GDP", "Forest Area", "Renewables"],
            "Coefficient": [f"{c:.3e}" for c in model.coef_]
        })
        st.dataframe(coef_df, hide_index=True, use_container_width=True)
        st.markdown(
            f"<div class='info-box'>Model R² = <strong>{r2:.4f}</strong> — "
            f"{r2*100:.1f}% of emission variance explained by GDP, "
            "forest area, and renewable energy.</div>",
            unsafe_allow_html=True
        )

        # Renewable energy indicator trend
        renew_vals = df_t["Renewable_Energy_Consumption"].dropna()
        fig_re = go.Figure(go.Scatter(
            x=list(renew_vals.index), y=list(renew_vals.values),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.15)",
            line=dict(color="#4ade80", width=2),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>"
        ))
        fig_re.update_layout(
            title="Renewable Energy (%)",
            height=220, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            margin=dict(t=40, b=30, l=40, r=10),
            xaxis_title="", yaxis_title="%"
        )
        st.plotly_chart(fig_re, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 2 · CORRELATION ANALYSIS
# ──────────────────────────────────────────────
with tab2:
    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        corr_cols = ["GDP", "Forest_Area", "Urban_Population",
                     "Renewable_Energy_Consumption", "Total_GHG_Emissions"]
        corr_data = df_t[corr_cols].dropna()
        corr_matrix = corr_data.corr()

        labels = ["GDP", "Forest\nArea", "Urban\nPop", "Renewable\nEnergy", "Total\nGHG"]
        z = corr_matrix.values
        text_vals = [[f"{v:.2f}" for v in row] for row in z]

        fig_hm = go.Figure(go.Heatmap(
            z=z, x=labels, y=labels,
            text=text_vals, texttemplate="%{text}",
            colorscale=[
                [0.0,  "#ef4444"],
                [0.5,  "#1e293b"],
                [1.0,  "#22c55e"]
            ],
            zmin=-1, zmax=1,
            showscale=True,
            hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>"
        ))
        fig_hm.update_layout(
            title="Correlation Matrix (2000–2022)",
            height=420, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=50, b=60, l=80, r=20),
            xaxis=dict(tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=11))
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with col_c2:
        st.markdown("#### Key Correlation Findings")
        st.markdown("""
| Pair | r | Direction |
|---|---|---|
| GDP → GHG | **+0.94** | 📈 Strong positive |
| Forest Area → GHG | **+0.98** | 📈 Strong positive |
| Urban Pop → GHG | **+0.97** | 📈 Strong positive |
| Renewables → GHG | **−0.92** | 📉 Strong negative |
| Renewables → GDP | **−0.75** | 📉 Moderate negative |
""")
        st.divider()
        st.markdown("""
**Interpretation for Insurers**

- Economic growth is the primary driver of rising emissions.
- Counterintuitively, forest area grows alongside GDP — likely due to reforestation policies — but GHG still rises because fossil fuel consumption outpaces forest sequestration.
- Renewable energy is the **strongest mitigating lever** (r = −0.92), highlighting its critical role in decarbonisation strategies.
- Rising urbanisation concentrates risk in cities, increasing insured asset exposure.
""")

        # Scatter: renewables vs GHG
        fig_sc = px.scatter(
            corr_data, x="Renewable_Energy_Consumption", y="Total_GHG_Emissions",
            trendline="ols",
            labels={"Renewable_Energy_Consumption": "Renewables (%)",
                    "Total_GHG_Emissions": "GHG (Mt CO₂e)"},
            title="Renewables vs GHG Emissions",
            template="plotly_dark",
            color_discrete_sequence=["#4ade80"]
        )
        fig_sc.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=50, b=40, l=60, r=10)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# ──────────────────────────────────────────────
# TAB 3 · MALAYSIA vs PHILIPPINES
# ──────────────────────────────────────────────
with tab3:
    st.markdown("""
<div class='info-box'>
Insurance risk comparison using EM-DAT disaster data (1980–2024).
Malaysia and the Philippines represent contrasting climate risk profiles within Southeast Asia.
</div>
""", unsafe_allow_html=True)

    # ----- Disaster data -----
    malaysia_data = {
        "Type":   ["Flood", "Earthquake", "Wildfire", "Drought",
                   "Storm", "Mass movement (wet)", "Mass movement (dry)", "Epidemic"],
        "Count":  [68, 6, 4, 4, 10, 7, 2, 14],
        "Damages_000USD": [3_200_000, 820_000, 510_000, 12_000,
                           45_000, 8_000, 2_000, 0],
    }

    philippines_data = {
        "Type":   ["Storm", "Flood", "Earthquake", "Mass movement (wet)",
                   "Epidemic", "Volcanic activity", "Wildfire", "Drought",
                   "Extreme temperature", "Infestation", "Mass movement (dry)"],
        "Count":  [430, 225, 55, 62, 42, 40, 8, 18, 6, 4, 3],
        "Damages_000USD": [85_000_000, 12_000_000, 3_200_000, 800_000,
                           200_000, 950_000, 30_000, 400_000,
                           50_000, 5_000, 20_000],
    }

    df_my = pd.DataFrame(malaysia_data)
    df_ph = pd.DataFrame(philippines_data)

    # CO₂ per capita comparison
    co2_years = list(range(2000, 2025))
    malaysia_co2 = [
        5.1, 5.3, 5.4, 5.7, 6.0, 6.2, 6.5, 6.8, 6.6, 6.5,
        7.0, 7.4, 7.6, 7.9, 8.0, 7.5, 7.8, 7.9, 8.1, 7.8,
        6.9, 7.6, 7.9, 7.7, 7.8
    ]
    philippines_co2 = [
        0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0,
        0.9, 1.0, 1.1, 1.1, 1.1
    ]

    # CO₂ per capita chart
    fig_co2 = go.Figure()
    fig_co2.add_trace(go.Scatter(
        x=co2_years, y=malaysia_co2,
        name="Malaysia", mode="lines+markers",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=4),
        hovertemplate="Malaysia %{x}: %{y:.1f} t<extra></extra>"
    ))
    fig_co2.add_trace(go.Scatter(
        x=co2_years, y=philippines_co2,
        name="Philippines", mode="lines+markers",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=4),
        hovertemplate="Philippines %{x}: %{y:.2f} t<extra></extra>"
    ))
    fig_co2.update_layout(
        title="CO₂ Emissions per Capita (t, 2000–2024)",
        xaxis_title="Year", yaxis_title="Tonnes per person",
        height=320, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=40, l=60, r=20)
    )

    col_co2, col_risk_scores = st.columns([2, 1])
    with col_co2:
        st.plotly_chart(fig_co2, use_container_width=True)
    with col_risk_scores:
        st.markdown("#### Insurance Risk Scores")
        risk_scores_df = pd.DataFrame({
            "Country": ["Malaysia", "Philippines"],
            "Composite Risk Score": [45, 90],
            "Dominant Hazard": ["Flood", "Storm / Typhoon"],
            "Avg Annual Disasters": ["3–4", "15–20"],
        })
        st.dataframe(risk_scores_df, hide_index=True, use_container_width=True)

        fig_rs = go.Figure(go.Bar(
            x=["Malaysia", "Philippines"],
            y=[45, 90],
            marker_color=["#38bdf8", "#ef4444"],
            text=[45, 90], textposition="outside",
            hovertemplate="%{x}: %{y}/100<extra></extra>"
        ))
        fig_rs.update_layout(
            title="Composite Risk Score (/100)",
            height=230, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            yaxis=dict(range=[0, 110]),
            margin=dict(t=50, b=30, l=40, r=10)
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    # Disaster breakdown side by side
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("#### 🇲🇾 Malaysia — Disaster Breakdown (1980–2024)")
        fig_my_bar = go.Figure(go.Bar(
            x=df_my["Count"], y=df_my["Type"],
            orientation="h",
            marker_color="#38bdf8",
            hovertemplate="%{y}: %{x} events<extra></extra>"
        ))
        fig_my_bar.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=20, b=30, l=160, r=20),
            xaxis_title="Number of Events"
        )
        st.plotly_chart(fig_my_bar, use_container_width=True)

        st.markdown("""
**Key Observations:**
- Floods dominate both frequency and economic damage (~60% of total damages)
- Earthquake damage is second-largest (~25%)
- Low storm frequency, but cost is notable
- Disaster count has increased post-2015 — driven by floods
""")

    with col_d2:
        st.markdown("#### 🇵🇭 Philippines — Disaster Breakdown (1980–2024)")
        fig_ph_bar = go.Figure(go.Bar(
            x=df_ph["Count"], y=df_ph["Type"],
            orientation="h",
            marker_color="#f97316",
            hovertemplate="%{y}: %{x} events<extra></extra>"
        ))
        fig_ph_bar.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=20, b=30, l=160, r=20),
            xaxis_title="Number of Events"
        )
        st.plotly_chart(fig_ph_bar, use_container_width=True)

        st.markdown("""
**Key Observations:**
- Typhoons (storms) dominate damage — ~80% of total economic losses
- Annual disaster count is 4–5× higher than Malaysia
- Volcanic activity adds unique peril layer absent in Malaysia
- 2013 Typhoon Haiyan was single largest loss event (~10M+ 000 USD)
""")

    # Implication for reinsurers
    st.divider()
    col_imp1, col_imp2, col_imp3 = st.columns(3)
    with col_imp1:
        st.info("**Premium Loading**\n\nPhilippines requires 2–3× higher catastrophe premium loading versus Malaysia due to typhoon frequency and severity.")
    with col_imp2:
        st.warning("**Reinsurance Structure**\n\nPhilippines portfolios need higher excess-of-loss reinsurance attachment points and broader peril coverage (storm + volcanic).")
    with col_imp3:
        st.error("**Capital Reserves**\n\nPhilippines underwriters should hold materially higher catastrophe reserves — at least 1.5× the Malaysia benchmark.")

# ──────────────────────────────────────────────
# TAB 4 · DEFORESTATION & EMISSIONS
# ──────────────────────────────────────────────
with tab4:
    st.markdown("""
<div class='info-box'>
Tree cover loss data from Global Forest Watch (2025).
Urbanisation data from World Bank / UN Population Division (2026).
</div>
""", unsafe_allow_html=True)

    tree_years = list(range(2015, 2025))
    malaysia_tree = [450, 570, 480, 440, 400, 268, 278, 248, 310, 279]
    philippines_tree = [63, 128, 111, 68, 62, 48, 84, 38, 56, 56]  # approx from chart

    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(
            x=tree_years, y=malaysia_tree,
            name="Malaysia", mode="lines+markers",
            line=dict(color="#4ade80", width=2.5), marker=dict(size=6),
            hovertemplate="Malaysia %{x}: %{y:,} ha<extra></extra>"
        ))
        fig_tree.add_trace(go.Scatter(
            x=tree_years, y=philippines_tree,
            name="Philippines", mode="lines+markers",
            line=dict(color="#fbbf24", width=2.5), marker=dict(size=6),
            hovertemplate="Philippines %{x}: %{y:,} ha<extra></extra>"
        ))
        fig_tree.update_layout(
            title="Tree Cover Loss (ha, 2015–2024)",
            xaxis_title="Year", yaxis_title="Hectares",
            height=340, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=60, b=40, l=70, r=20)
        )
        st.plotly_chart(fig_tree, use_container_width=True)

    with col_t2:
        # Urbanisation bar race — use Malaysia 2024 vs 2000
        urb_years = [2000, 2005, 2010, 2015, 2020, 2024]
        my_urb   = [41.6, 47.0, 52.0, 57.7, 62.4, 64.4]
        ph_urb   = [46.1, 49.5, 51.5, 53.4, 55.5, 57.0]

        fig_urb = go.Figure()
        fig_urb.add_trace(go.Bar(
            name="Malaysia", x=urb_years, y=my_urb,
            marker_color="#38bdf8",
            hovertemplate="Malaysia %{x}: %{y:.1f}%<extra></extra>"
        ))
        fig_urb.add_trace(go.Bar(
            name="Philippines", x=urb_years, y=ph_urb,
            marker_color="#f97316",
            hovertemplate="Philippines %{x}: %{y:.1f}%<extra></extra>"
        ))
        fig_urb.update_layout(
            title="Urban Population (%) 2000–2024",
            barmode="group", height=310, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=50, b=40, l=50, r=10),
            legend=dict(orientation="h", y=1.08)
        )
        st.plotly_chart(fig_urb, use_container_width=True)

    st.divider()
    col_t3, col_t4 = st.columns(2)

    with col_t3:
        # Forest area trend
        forest_s = df_t["Forest_Area"].dropna()
        fig_fa = go.Figure(go.Scatter(
            x=list(forest_s.index), y=list(forest_s.values),
            fill="tozeroy", fillcolor="rgba(74,222,128,0.12)",
            line=dict(color="#4ade80", width=2.5),
            mode="lines+markers", marker=dict(size=5),
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>"
        ))
        fig_fa.update_layout(
            title="Malaysia Forest Area (% of Land Area)",
            xaxis_title="Year", yaxis_title="%",
            height=280, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(t=50, b=40, l=60, r=20)
        )
        st.plotly_chart(fig_fa, use_container_width=True)

    with col_t4:
        st.markdown("#### 🌿 Land Use & Emissions Nexus")
        st.markdown("""
Malaysia's forest area has **increased** from 26.0% to 27.2% (2000–2022),
driven by afforestation policies and secondary regrowth.

Yet tree **cover loss** remains high — peaking at 570,000 ha in 2016 — 
reflecting selective logging and plantation expansion rather than 
wholesale clearance. Loss has trended downward since 2016.

**Insurance Implications:**

- Continued deforestation elevates flood risk by reducing natural
  water retention — increasing frequency of flash floods.
- Loss of peatland increases wildfire risk (El Niño years especially).
- Malaysia's forest cover correlation with GHG (+0.98) reflects that
  afforestation policies are insufficient alone to offset fossil emissions.
- For property & engineering underwriters: monitor watershed loss in
  Peninsular Malaysia and Sabah/Sarawak.
""")

# ====================================================
# FOOTER
# ====================================================

st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.markdown(
        "**Team DEBUG** · Sunway University\n\n"
        "Lee Earn Hui · Grace Wong Xin En · Lee Min Ming"
    )
with col_f2:
    st.markdown(
        "**Data Sources**\n\n"
        "World Bank · EM-DAT · Our World in Data\n\n"
        "Global Carbon Budget 2025 · Global Forest Watch"
    )
with col_f3:
    st.markdown(
        "**Model**\n\n"
        f"Linear Regression · R² = {r2:.4f}\n\n"
        "Features: GDP, Forest Area, Renewable Energy"
    )