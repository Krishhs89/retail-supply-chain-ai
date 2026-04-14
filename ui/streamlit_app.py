"""
Retail Supply Chain Optimization — Streamlit UI

Sidebar version selector controls the layout:
  v1 — Core    : Dashboard + Chat only (2 tabs)
  v2 — Full    : All 15 original tabs with right-side panels, Flow Map, Scenario Builder
  v3 — Simplified : 7 merged tabs (default)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import streamlit as st


def _get_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
            return key
    except Exception:
        pass
    return ""


import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config.settings import settings
from tools import mock_executor

logging.basicConfig(level=logging.INFO)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Retail Supply Chain AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #21262d; border: 1px solid #30363d;
    color: #e6edf3 !important; border-radius: 6px; font-size: 13px;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #0071ce; border-color: #0071ce; }
div[data-testid="metric-container"] {
    background: #f8f9fa; border: 1px solid #e9ecef;
    border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="stAlert"] { border-radius: 8px; }
[data-testid="stChatMessage"] { border-radius: 10px; margin-bottom: 8px; }
button[data-baseweb="tab"] { font-size: 13px; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0071ce !important; border-bottom-color: #0071ce !important;
}
.chip-green  { background:#d4edda; color:#155724; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.chip-red    { background:#f8d7da; color:#721c24; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.chip-yellow { background:#fff3cd; color:#856404; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.chip-blue   { background:#cce5ff; color:#004085; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
details summary { font-size: 13px; color: #6c757d; }
.js-plotly-plot { border-radius: 10px; overflow: hidden; }
.hist-panel { background:#f8f9fa; border:1px solid #e9ecef; border-radius:10px; padding:14px 16px; height:100%; }
.hist-panel h4 { font-size:13px; font-weight:700; color:#495057; margin:0 0 10px 0; letter-spacing:0.3px; text-transform:uppercase; }
.hist-item { background:white; border:1px solid #dee2e6; border-radius:8px; padding:10px 12px; margin-bottom:8px; font-size:12px; line-height:1.5; }
.hist-item-header { font-weight:600; color:#212529; font-size:12px; margin-bottom:4px; }
.hist-item-meta { color:#6c757d; font-size:11px; }
.hist-item-result { color:#0071ce; font-weight:600; font-size:12px; margin-top:4px; }
.hist-empty { color:#adb5bd; font-size:12px; font-style:italic; text-align:center; padding:20px 0; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "conversation_history": [], "last_tool_calls": [], "freshness_warnings": [],
        "max_iterations_override": 10,
        "session_queries": 0, "session_tool_calls": 0, "session_iterations": 0,
        "pipeline_version": "V2 — LangGraph",
        "app_version": "v3 — Simplified (7 tabs)",
        "hist_price": [], "hist_supply": [], "hist_forecast": [],
        "hist_scenario": [], "hist_shelf": [], "hist_finance": [],
        "hist_workflow": [], "hist_flowmap": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Data Dicts ───────────────────────────────────────────────────────────────

_DATA_SOURCES = {
    "price": {
        "title": "Price Cascade Data Sources",
        "sources": [
            ("Snowflake (PROD)", "POS transactions · 15-min refresh", "green"),
            ("SAP S/4HANA", "Pricing engine · Order-to-cash", "green"),
            ("Azure Synapse", "Historical price-elasticity model", "blue"),
            ("Nielsen IQ", "Syndicated retail price benchmarks", "blue"),
            ("Palantir Foundry", "Cascade simulation output store", "purple"),
        ],
        "cloud_note": "Integrates with: AWS Redshift, GCP BigQuery, Databricks Delta Lake"
    },
    "supply": {
        "title": "Supply Disruption Data Sources",
        "sources": [
            ("Oracle Transportation Mgmt", "Carrier status · Route tracking", "green"),
            ("Blue Yonder WMS", "DC inventory snapshots · 30-min refresh", "green"),
            ("Kafka Streams", "Real-time carrier event bus", "orange"),
            ("FreightWaves SONAR", "Market disruption signals", "blue"),
            ("FedEx / UPS APIs", "Carrier capacity + lead time", "blue"),
        ],
        "cloud_note": "Integrates with: AWS EventBridge, Azure Service Bus, GCP Pub/Sub"
    },
    "forecast": {
        "title": "Demand Forecast Data Sources",
        "sources": [
            ("Circana (IRI)", "Syndicated scanner + panel data", "green"),
            ("NOAA Weather API", "7-day regional weather signals", "green"),
            ("Brandwatch", "Social sentiment index (CPG)", "blue"),
            ("Google Trends", "Search demand proxy", "blue"),
            ("Snowflake ML", "15-variable model feature store", "purple"),
        ],
        "cloud_note": "Integrates with: AWS SageMaker, Azure ML, GCP Vertex AI, Databricks MLflow"
    },
    "scenario": {
        "title": "Scenario Planner Data Sources",
        "sources": [
            ("Pros Holdings RGM", "Revenue management baselines", "green"),
            ("Blacksmith TPM", "Trade promotion calendar", "green"),
            ("FactSet", "Competitor pricing intelligence", "blue"),
            ("S&P Global Market Intel", "Tariff + FX risk signals", "blue"),
            ("Anaplan", "Financial scenario models", "purple"),
        ],
        "cloud_note": "Integrates with: Snowflake Marketplace, AWS Data Exchange, Azure Open Datasets"
    },
    "shelf": {
        "title": "Shelf & Store Data Sources",
        "sources": [
            ("JDA Blue Yonder", "Space planning + planogram data", "green"),
            ("Trax AI", "Computer-vision shelf execution", "green"),
            ("SAP EWM", "Extended Warehouse Mgmt (DC layer)", "green"),
            ("Relex Solutions", "Store-level replenishment engine", "blue"),
            ("Retail Link (Walmart)", "Store inventory + velocity data", "blue"),
        ],
        "cloud_note": "Integrates with: AWS IoT Core (shelf sensors), Azure Digital Twins, GCP Retail API"
    },
    "finance": {
        "title": "Financial Impact Data Sources",
        "sources": [
            ("SAP S/4HANA Finance", "P&L · Cost of goods · GL entries", "green"),
            ("Kyriba", "Treasury + FX carrying cost", "green"),
            ("BlackLine", "Trade promotion accruals", "blue"),
            ("Coupa", "Vendor invoice + trade dollar tracking", "blue"),
            ("Tableau / Power BI", "CFO dashboard push", "purple"),
        ],
        "cloud_note": "Integrates with: AWS Redshift (data warehouse), Azure Cost Mgmt, GCP Looker"
    },
}

_FORMULAS = {
    "price": {
        "title": "Price Cascade Formula",
        "formula": r"""
**Demand Change:** `Demand_delta% = Elasticity × Price_delta%`

**Asymmetric rule** (tobacco, alcohol, diapers, infant formula):
- Price ↑: `Elasticity_up = base_elasticity × 1.3`
- Price ↓: `Elasticity_dn = base_elasticity × 0.7`

**Financial cascade:**
```
Gross_Revenue_delta = delta_Price × New_Volume + Old_Price × delta_Volume
Net_Revenue         = Gross_Revenue_delta + Vendor_Trade_Offset
Bottom_Line         = Net_Revenue + Margin_delta - Carrying_Cost_delta
```
""",
        "params": [
            ("Base elasticity", "elasticity", -1.4, -0.5, -3.0, 0.1),
            ("Safety stock (weeks)", "safety_wks", 2.0, 1.0, 6.0, 0.5),
            ("Vendor trade offset %", "trade_pct", 3.0, 0.0, 10.0, 0.5),
        ]
    },
    "supply": {
        "title": "Supply Risk Formula",
        "formula": r"""
**Days to Stockout:** `DTS = Current_Inventory / (Avg_Daily_Demand × Safety_Factor)`

**Revenue at Risk:** `Rev_at_Risk = max(0, Replenishment_Lag - DTS) × Daily_Revenue`

**Severity thresholds:**
- Critical: `DTS < Replenishment_Lag`
- Warning:  `DTS < Replenishment_Lag × 1.5`
""",
        "params": [
            ("Replenishment lag (days)", "lag_days", 4.0, 1.0, 14.0, 1.0),
            ("Safety factor multiplier", "safety_f", 1.5, 1.0, 3.0, 0.1),
            ("Daily demand base (units)", "daily_dem", 48.0, 10.0, 200.0, 5.0),
        ]
    },
    "forecast": {
        "title": "15-Variable Demand Model",
        "formula": r"""
**Multiplicative demand function:** `D(t) = Base_Demand × Π(1 + fᵢ × wᵢ)`

Where fᵢ = factor signal, wᵢ = learned weight for:
`price, promo, tariff, weather, seasonality, trend, competitor_price,
social_sentiment, days_supply, channel_mix, planogram_compliance,
regional_income, household_penetration, repeat_rate, new_item_velocity`

**Confidence interval:** `CI_width = Base_MAPE × (1 + 0.05 × Horizon_weeks)`
""",
        "params": [
            ("Promo weight", "promo_w", 0.18, 0.0, 0.5, 0.01),
            ("Weather weight", "weather_w", 0.08, 0.0, 0.3, 0.01),
            ("Seasonality weight", "season_w", 0.12, 0.0, 0.4, 0.01),
        ]
    },
    "scenario": {
        "title": "Scenario Comparison Formula",
        "formula": r"""
**Per-scenario revenue:** `Rev_s = Price_s × Demand_s(price, promo, supply_reduction, tariff) × Horizon`

**Compound execution penalty** (3+ scenarios): `Penalty = (N_scenarios - 2) × 2%`

**Conflict score:** `Conflict_severity = Σ(block_weight_i × block_weight_j) for conflicting pairs`
""",
        "params": [
            ("Compound penalty per extra block %", "comp_pen", 2.0, 0.0, 5.0, 0.5),
            ("Conflict threshold (score)", "conf_thresh", 0.6, 0.1, 1.0, 0.05),
        ]
    },
    "shelf": {
        "title": "Replenishment Formula",
        "formula": r"""
**Reorder Point:** `ROP = Lead_Time_days × Avg_Daily_Demand + Safety_Stock`

**Safety Stock:** `SS = Z_score × σ_demand × √Lead_Time`  (Z=1.65 for 95% service level)

**Perishable cap:** `Max_Order = (Shelf_Life_days - Lead_Time) × Avg_Daily_Demand × Planogram_Cap`

**Freight:** Standard 1×  ·  Expedited 2×  ·  Emergency 3×
""",
        "params": [
            ("Service level Z-score", "z_score", 1.65, 1.0, 2.58, 0.1),
            ("Demand std-dev (units/day)", "sigma_d", 8.0, 1.0, 30.0, 1.0),
            ("Base freight $/unit", "base_fr", 0.18, 0.05, 1.0, 0.01),
        ]
    },
    "finance": {
        "title": "P&L Impact Formula",
        "formula": r"""
```
Gross_Revenue_delta = (New_Price - Old_Price)×New_Vol + (New_Vol - Old_Vol)×Old_Price
Vendor_Trade_Offset = Gross_Revenue_delta × trade_rate
Net_Revenue_delta   = Gross_Revenue_delta + Vendor_Trade_Offset
Gross_Margin_delta  = Net_Revenue_delta - COGS_delta
Carrying_Cost_delta = delta_Inventory × holding_rate × days
Bottom_Line         = Gross_Margin_delta - Carrying_Cost_delta - Tax_approx
```
""",
        "params": [
            ("Holding cost rate ($/unit/day)", "hold_r", 0.02, 0.005, 0.1, 0.005),
            ("Vendor trade rate %", "trade_r", 3.0, 0.0, 15.0, 0.5),
            ("Tax rate (approx) %", "tax_r", 21.0, 0.0, 35.0, 1.0),
        ]
    },
}

# ─── Helper functions ─────────────────────────────────────────────────────────

def _render_history(items: list, empty_msg: str = "No runs yet."):
    """v2-style HTML right-panel history."""
    st.markdown('<div class="hist-panel"><h4>📋 Run History</h4>', unsafe_allow_html=True)
    if not items:
        st.markdown(f'<div class="hist-empty">{empty_msg}</div>', unsafe_allow_html=True)
    else:
        for item in reversed(items):
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-item-header">{item.get("header","")}</div>'
                f'<div class="hist-item-meta">{item.get("meta","")}</div>'
                f'<div class="hist-item-result">{item.get("result","")}</div>'
                f'<div class="hist-item-meta" style="margin-top:4px;color:#ced4da;">{item.get("ts","")}</div>'
                f'</div>', unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)


def _render_formula_panel(tab_key: str):
    """v2-style formula + parameter sliders expander."""
    fml = _FORMULAS.get(tab_key, {})
    if not fml:
        return
    with st.expander("Formula & Parameters", expanded=False):
        st.markdown(f"**{fml['title']}**")
        st.markdown(fml["formula"])
        if fml.get("params"):
            st.markdown("---")
            st.markdown("**Adjust parameters to test:**")
            for label, key, default, mn, mx, step in fml["params"]:
                st.slider(label, min_value=float(mn), max_value=float(mx),
                          value=float(default), step=float(step),
                          key=f"fml_{tab_key}_{key}")


def _render_data_sources(tab_key: str):
    """v2-style cloud data source expander."""
    src = _DATA_SOURCES.get(tab_key, {})
    if not src:
        return
    with st.expander("Data Sources", expanded=False):
        st.markdown(f"**{src['title']}**")
        color_map = {"green": "#2ea043", "blue": "#388bfd", "orange": "#d29922", "purple": "#bc8cff"}
        for name, desc, color in src["sources"]:
            dot = f'<span style="color:{color_map[color]};font-size:10px;">&#9679;</span>'
            st.markdown(f'{dot} **{name}**  \n<span style="color:#8b949e;font-size:11px;">{desc}</span>',
                        unsafe_allow_html=True)
        st.caption(src["cloud_note"])


def _render_details(tab_key: str):
    """v3-style combined formula + data sources in single expander."""
    with st.expander("Model details & data sources", expanded=False):
        fml = _FORMULAS.get(tab_key, {})
        if fml:
            st.markdown(f"**{fml['title']}**")
            st.markdown(fml["formula"])
            if fml.get("params"):
                st.markdown("**Adjust parameters:**")
                for label, key, default, mn, mx, step in fml["params"]:
                    st.slider(label, min_value=float(mn), max_value=float(mx),
                              value=float(default), step=float(step),
                              key=f"fml3_{tab_key}_{key}")
        src = _DATA_SOURCES.get(tab_key, {})
        if src:
            st.divider()
            st.markdown(f"**{src['title']}**")
            color_map = {"green": "#2ea043", "blue": "#388bfd", "orange": "#d29922", "purple": "#bc8cff"}
            for name, desc, color in src["sources"]:
                dot = f'<span style="color:{color_map[color]};font-size:10px;">&#9679;</span>'
                st.markdown(f'{dot} **{name}** — {desc}', unsafe_allow_html=True)
            st.caption(src["cloud_note"])


def _render_run_history(items: list, empty_msg: str = "No runs yet."):
    """v3-style compact run history expander — auto-expanded when runs exist."""
    with st.expander(f"📋 Run History ({len(items)})", expanded=bool(items)):
        if not items:
            st.caption(empty_msg)
        else:
            for item in reversed(items[-5:]):
                st.markdown(
                    f"**{item.get('header','')}** — {item.get('result','')}  \n"
                    f"*{item.get('meta','')}* · {item.get('ts','')}"
                )
                st.divider()


# ─── Orchestrator (lazy, cached) ─────────────────────────────────────────────

@st.cache_resource
def _get_orchestrator(_key: str = ""):
    from agents.orchestrator import Orchestrator
    return Orchestrator()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏪 Retail Supply Chain AI")
    st.markdown("*Powered by Claude claude-sonnet-4-6*")
    st.caption("build 2026-04-13.e")
    st.divider()

    # ── App Version Selector ──
    st.markdown("### 🗂 App Version")
    app_version = st.radio(
        "Select layout:",
        ["v1 — Core (2 tabs)", "v2 — Full (15 tabs)", "v3 — Simplified (7 tabs)"],
        index=2,
        key="app_version",
        help="Switch between UI versions. All versions use the same AI backend and data.",
    )

    ver_info = {
        "v1 — Core (2 tabs)":        ("🟡", "Dashboard + Chat only. The original MVP — pure AI interface."),
        "v2 — Full (15 tabs)":        ("🟠", "Every tool in its own tab. Full right-side panels, Flow Map, Scenario Builder."),
        "v3 — Simplified (7 tabs)":   ("🟢", "Merged tabs, cleaner layout. Recommended for demos."),
    }
    badge, blurb = ver_info.get(app_version, ("🟢", "Merged tabs, cleaner layout. Recommended for demos."))
    st.caption(f"{badge} {blurb}")
    st.divider()

    # ── API Key Status ──
    st.markdown("### 🔑 API Status")
    key_ok = bool(_get_api_key())
    if key_ok:
        st.markdown('<span class="chip-green">✓ Anthropic API Key loaded</span>', unsafe_allow_html=True)
        st.caption(f"Model: `{settings.MODEL_ID}`")
    else:
        st.markdown('<span class="chip-red">✗ API Key missing — add to .env</span>', unsafe_allow_html=True)

    st.divider()

    # ── Advanced (pipeline toggle + iteration cap) ──
    with st.expander("⚙️ Advanced", expanded=False):
        _pipeline_default_idx = 0  # LangGraph V2 is the default for all versions
        pipeline_ver = st.radio(
            "Chat agent backend:",
            ["V2 — LangGraph", "V1 — Agentic Loop"],
            index=0,
            key="pipeline_radio",
        )
        if pipeline_ver != st.session_state.pipeline_version:
            st.session_state.pipeline_version = pipeline_ver
            st.session_state.conversation_history = []
        if "V1" in pipeline_ver:
            max_iter = st.slider("Max iterations (V1 only)", min_value=3, max_value=25, value=10)
            st.session_state.max_iterations_override = max_iter
        else:
            st.caption("LangGraph routing is automatic — no iteration cap needed.")

    st.divider()

    # ── Session Stats ──
    st.markdown("### 📊 Session Stats")
    s1, s2, s3 = st.columns(3)
    s1.metric("Queries", st.session_state.session_queries)
    s2.metric("Tool calls", st.session_state.session_tool_calls)
    s3.metric("Iterations", st.session_state.session_iterations)

    st.divider()

    # ── Quick Scenario Presets ──
    st.markdown("### 🚀 Quick Scenarios")
    PRESETS = {
        "🔺 Diaper Price Hike": (
            "HUG48-3 diaper price is being raised from $12.99 to $14.49. "
            "Simulate the full cascade: demand impact, PO adjustments, inventory implications, "
            "and financial effect across all DCs and 30 stores."
        ),
        "🚛 TruckCo B Strike": (
            "TruckCo B — our diaper carrier for SE and MW regions — has gone on strike. "
            "Expected duration: 14 days. What is the stockout risk, what alternate carriers "
            "are available regionally, and what is the revenue at risk?"
        ),
        "📉 Forecast Accuracy Gap": (
            "Analyze demand forecast accuracy for diapers (HUG48-3) at 8 weeks out. "
            "What is the gap to industry benchmark? What is the dollar revenue impact of "
            "improving by 7 percentage points?"
        ),
        "⚠️ Promo + Strike Conflict": (
            "We are planning a 10% promotional price cut on HUG48-3 starting 2026-05-01 "
            "for 30 days. TruckCo B is still on strike. Detect scenario conflicts and advise."
        ),
        "🥛 Milk Shelf Replenishment": (
            "STR-005 reports critically low milk (MLK-GAL) inventory. Check stockout risk, "
            "perishable 3-day cap, planogram capacity, and recommend replenishment."
        ),
        "📊 3-Scenario Comparison": (
            "Compare three scenarios for HUG48-3 over 8 weeks: "
            "A) Hold price at $12.99, B) Raise to $14.49, C) Drop to $11.99 with 15% promo. "
            "Which maximizes revenue? Which maximizes margin?"
        ),
    }
    for label, query in PRESETS.items():
        if st.button(label, use_container_width=True, key=f"preset_{label}"):
            st.session_state["_pending_query"] = query
            st.session_state["_goto_chat"] = True

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.last_tool_calls = []
            st.session_state.freshness_warnings = []
            st.rerun()
    with col_b:
        if st.button("↺ Reset All", use_container_width=True):
            for k in ["conversation_history", "last_tool_calls", "freshness_warnings",
                      "session_queries", "session_tool_calls", "session_iterations"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else 0
            st.rerun()

# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown(
    "# 🏪 Retail Supply Chain Optimization AI\n"
    "Multi-agent system connecting **pricing → demand → inventory → supply chain → finance** "
    "at Walmart scale. Built on Claude with agentic tool use."
)

# ─── Shared content blocks (reused across versions) ──────────────────────────

def _dashboard_content():
    st.subheader("Network Status Dashboard")
    st.caption("Live snapshot of the supply chain network. TruckCo B is currently on strike.")
    st.markdown("#### Carrier Status")
    cc = st.columns(4)
    carrier_info = [
        ("TruckCo A", "Active", "NW + MW", "Tableware / Linen", "green"),
        ("TruckCo B", "ON STRIKE", "SE + MW", "Diapers / Formula", "red"),
        ("TruckCo C", "Active", "NW + SE", "Dairy / Produce", "green"),
        ("TruckCo D", "Active", "All Regions", "General / Mixed", "green"),
    ]
    for col, (name, status, regions, cargo, color) in zip(cc, carrier_info):
        with col:
            chip_class = "chip-red" if color == "red" else "chip-green"
            st.markdown(f"""
            <div style="border:1px solid {'#f8d7da' if color=='red' else '#d4edda'};
                        border-radius:10px;padding:14px;text-align:center;
                        background:{'#fff5f5' if color=='red' else '#f8fff8'};">
              <div style="font-weight:700;font-size:15px;margin-bottom:6px;">{name}</div>
              <span class="{chip_class}">{status}</span>
              <div style="font-size:12px;color:#6c757d;margin-top:8px;">{regions}</div>
              <div style="font-size:11px;color:#adb5bd;">{cargo}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### DC Inventory Snapshot — Diapers (HUG48-3)")
    dc_cols = st.columns(3)
    dc_summary = []
    for dc_id, dc in mock_executor.DCS.items():
        inv = dc["inventory"].get("HUG48-3", {})
        wms = inv.get("wms", 0)
        prod = mock_executor.PRODUCTS["HUG48-3"]
        dc_daily = (prod["base_demand_per_store_week"] / 7) * len(dc["stores_served"])
        dos = round(wms / dc_daily, 1) if dc_daily else 0
        dc_summary.append((dc_id, dc["name"].split("—")[1].strip(), wms, dos, dc["region"]))
    for col, (dc_id, city, qty, dos, region) in zip(dc_cols, dc_summary):
        with col:
            st.metric(f"{dc_id} — {city}", f"{qty:,} units", f"{dos}d on-hand")
            chip = "chip-red" if dos < 7 else ("chip-yellow" if dos < 14 else "chip-blue")
            st.markdown(f'<span class="{chip}">Region: {region}</span>', unsafe_allow_html=True)
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    total_dc_inv = sum(dc["inventory"].get("HUG48-3", {}).get("wms", 0) for dc in mock_executor.DCS.values())
    base_weekly_rev = (mock_executor.PRODUCTS["HUG48-3"]["base_price"]
                       * mock_executor.PRODUCTS["HUG48-3"]["base_demand_per_store_week"] * 30)
    k1.metric("Total DC Inventory", f"{total_dc_inv:,} units")
    k2.metric("Weekly Network Revenue", f"${base_weekly_rev:,.0f}")
    k3.metric("Active Carriers", "3 / 4", delta="-1 on strike", delta_color="inverse")
    k4.metric("Forecast Accuracy (8W)", "78%", delta="-7pts vs benchmark", delta_color="inverse")
    k5.metric("Stores at Risk", "10", delta="SE+MW regions", delta_color="inverse")
    st.markdown("---")
    inv_data = []
    for dc_id, dc in mock_executor.DCS.items():
        for sku in ["HUG48-3", "PAM72-5", "TAB-DIN", "BLK-THR"]:
            inv_data.append({"DC": dc_id, "SKU": sku, "Units (WMS)": dc["inventory"].get(sku, {}).get("wms", 0)})
    fig_inv = px.bar(pd.DataFrame(inv_data), x="DC", y="Units (WMS)", color="SKU", barmode="group",
                     title="DC Inventory by SKU (WMS — 15min lag)",
                     color_discrete_sequence=px.colors.qualitative.Set2, height=320)
    fig_inv.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig_inv, use_container_width=True)
    st.markdown("#### TruckCo B Strike Impact Timeline — Diapers (SE Region)")
    days = list(range(0, 18))
    prod_daily = mock_executor.PRODUCTS["HUG48-3"]["base_demand_per_store_week"] / 7
    si, di, store_inv_line, dc_inv_line = 22, 3200, [], []
    for d in days:
        store_inv_line.append(max(0, si)); dc_inv_line.append(max(0, di))
        si -= prod_daily; di -= prod_daily * 10
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=days, y=store_inv_line, name="Store Inventory",
                                 line=dict(color="#e74c3c", width=2), fill="tozeroy",
                                 fillcolor="rgba(231,76,60,0.08)"))
    fig_tl.add_trace(go.Scatter(x=days, y=[v/100 for v in dc_inv_line], name="DC-SE (/100)",
                                 line=dict(color="#3498db", width=2, dash="dot")))
    fig_tl.add_vline(x=3, line_dash="dash", line_color="red",
                     annotation_text="Store stockout", annotation_position="top right")
    fig_tl.add_vline(x=14, line_dash="dash", line_color="orange",
                     annotation_text="Strike ends (est.)", annotation_position="top left")
    fig_tl.update_layout(title="Projected Inventory Depletion During Strike",
                          xaxis_title="Days from Strike Start", yaxis_title="Units",
                          height=320, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig_tl, use_container_width=True)


def _chat_content(full_width: bool = True):
    use_v2 = "V2" in st.session_state.get("pipeline_version", "V2")
    st.subheader("Multi-Agent Chat — " + ("LangGraph (V2)" if use_v2 else "Agentic Loop (V1)"))
    st.caption(
        "**LangGraph V2:** router → domain nodes → synthesizer."
        if use_v2 else
        "**Agentic Loop V1:** single Claude agent with all 17 tools."
    )
    if not key_ok:
        st.warning("API key not detected. Add `ANTHROPIC_API_KEY` to Streamlit secrets or `.env`.")

    # ── v2 right-column history panel (only in v2 full layout) ──────────────────
    if not full_width:
        chat_col, hist_col = st.columns([2, 1])
        with hist_col:
            st.markdown('<div class="hist-panel"><h4>📋 Conversation History</h4>', unsafe_allow_html=True)
            chat_msgs = [m for m in st.session_state.conversation_history if isinstance(m.get("content"), str)]
            if not chat_msgs:
                st.markdown('<div class="hist-empty">No messages yet.</div>', unsafe_allow_html=True)
            else:
                pairs = []
                i = 0
                while i < len(chat_msgs):
                    if chat_msgs[i]["role"] == "user":
                        q = chat_msgs[i]["content"]
                        a = chat_msgs[i+1]["content"] if i+1 < len(chat_msgs) else ""
                        pairs.append((q, a)); i += 2
                    else:
                        i += 1
                for idx, (q, a) in enumerate(reversed(pairs), 1):
                    st.markdown(
                        f'<div class="hist-item">'
                        f'<div class="hist-item-header">Q{len(pairs)-idx+1}: {q[:80]}{"…" if len(q)>80 else ""}</div>'
                        f'<div class="hist-item-meta" style="color:#212529;margin-top:4px;">{a[:120]}{"…" if len(a)>120 else ""}</div>'
                        f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── render chat messages + input in the left column ──
        with chat_col:
            for msg in st.session_state.conversation_history:
                if isinstance(msg.get("content"), str):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            pending = st.session_state.pop("_pending_query", None)
            st.session_state.pop("_goto_chat", None)
            prompt = st.chat_input("Ask anything — or load a preset from the sidebar →") or pending
    else:
        # ── full-width: render directly, no container wrapper ──
        for msg in st.session_state.conversation_history:
            if isinstance(msg.get("content"), str):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        pending = st.session_state.pop("_pending_query", None)
        st.session_state.pop("_goto_chat", None)
        prompt = st.chat_input("Ask anything — or load a preset from the sidebar →") or pending

    if prompt:
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        _msg_ctx = chat_col if not full_width else st
        with _msg_ctx.chat_message("user"):
            st.markdown(prompt)

        with _msg_ctx.chat_message("assistant"):
            if not key_ok:
                st.error("ANTHROPIC_API_KEY not found.")
            else:
                result = None
                response_placeholder = st.empty()
                if use_v2:
                    with st.status("LangGraph agent working...", expanded=True) as status_box:
                        try:
                            from agents.langgraph_flow import run_langgraph
                            t0 = time.time()
                            result = run_langgraph(prompt)
                            elapsed_total = round(time.time() - t0, 1)
                            if result.get("error"):
                                status_box.update(label="Error", state="error", expanded=True)
                                st.error(result["error"]); result = None
                            else:
                                nodes = list(result.get("node_outputs", {}).keys())
                                n_tools = len(result.get("tool_calls_made", []))
                                st.write(f"**Intent:** `{result.get('intent','?')}`  |  **SKU:** `{result.get('sku','?')}`")
                                st.write(f"**Nodes:** {' → '.join(nodes)}  |  **Tools:** {n_tools}  |  **Time:** {elapsed_total}s")
                                status_box.update(label=f"Done — {len(nodes)} nodes, {n_tools} tools, {elapsed_total}s",
                                                  state="complete", expanded=False)
                                result["updated_history"] = [
                                    *st.session_state.conversation_history[:-1],
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": result["response_text"]},
                                ]
                                if result.get("node_outputs"):
                                    with st.expander("🗂 Node outputs (LangGraph trace)", expanded=False):
                                        for node_name, output in result["node_outputs"].items():
                                            st.markdown(f"**Node: `{node_name}`**")
                                            for k, v in output.items():
                                                if k.endswith("_summary") and isinstance(v, str):
                                                    st.markdown(v[:500])
                                                elif not k.endswith("_summary"):
                                                    st.json({k: v} if not isinstance(v, dict) else v, expanded=False)
                                            st.divider()
                        except Exception as exc:
                            status_box.update(label="Error", state="error", expanded=True)
                            st.error(f"LangGraph error: {exc}"); result = None
                else:
                    with st.status("Agent working...", expanded=True) as status_box:
                        try:
                            from agents.orchestrator import _detect_max_iterations, SYSTEM_PROMPT, _summarize_history
                            import anthropic as _anthropic
                            from tools.tool_definitions import ALL_TOOLS
                            client = _anthropic.Anthropic(api_key=_get_api_key())
                            messages = list(st.session_state.conversation_history[:-1])
                            if len(messages) >= settings.HISTORY_SUMMARIZE_THRESHOLD:
                                messages = _summarize_history(client, messages)
                            messages.append({"role": "user", "content": prompt})
                            max_iter = st.session_state.max_iterations_override or _detect_max_iterations(prompt)
                            tool_calls_made, freshness_warnings, iteration = [], [], 0
                            response = client.messages.create(model=settings.MODEL_ID, max_tokens=settings.MAX_TOKENS,
                                                              system=SYSTEM_PROMPT, tools=ALL_TOOLS, messages=messages)
                            while response.stop_reason == "tool_use" and iteration < max_iter:
                                iteration += 1
                                st.write(f"**Iteration {iteration}** — processing tool calls...")
                                tool_results = []
                                for block in response.content:
                                    if block.type != "tool_use": continue
                                    t0 = time.time()
                                    res = mock_executor.execute(block.name, block.input)
                                    elapsed = round(time.time() - t0, 2)
                                    had_error = bool(res.get("error"))
                                    prov = res.get("provenance", "OLTP"); fresh = res.get("freshness_minutes", 5)
                                    st.write(f"  {'✗' if had_error else '✓'} `{block.name}` — {prov} ({fresh}min) — {elapsed}s")
                                    if res.get("is_stale") or prov == "OLAP":
                                        freshness_warnings.append(f"'{block.name}' uses {prov} ({fresh}min lag).")
                                    data = res.get("data", {})
                                    if isinstance(data, dict):
                                        for fw in data.get("freshness_warnings", []): freshness_warnings.append(fw)
                                    tool_calls_made.append({"tool": block.name, "input": block.input,
                                        "result_summary": res.get("error") or str(data)[:200],
                                        "provenance": prov, "freshness_minutes": fresh,
                                        "had_error": had_error, "elapsed_s": elapsed})
                                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(res)})
                                messages.append({"role": "assistant", "content": response.content})
                                messages.append({"role": "user", "content": tool_results})
                                response = client.messages.create(model=settings.MODEL_ID, max_tokens=settings.MAX_TOKENS,
                                                                  system=SYSTEM_PROMPT, tools=ALL_TOOLS, messages=messages)
                            final_text = " ".join(b.text for b in response.content if hasattr(b, "text"))
                            if iteration >= max_iter and response.stop_reason == "tool_use":
                                final_text += f"\n\n⚠ Reached max iterations ({max_iter})."
                            status_box.update(label=f"Done — {iteration} iter(s), {len(tool_calls_made)} tool(s)",
                                              state="complete", expanded=False)
                            result = {"response_text": final_text, "tool_calls_made": tool_calls_made,
                                      "iterations_used": iteration, "data_freshness_warnings": freshness_warnings,
                                      "updated_history": messages + [{"role": "assistant", "content": final_text}], "error": None}
                        except Exception as exc:
                            status_box.update(label="Error", state="error", expanded=True)
                            st.error(f"Agent error: {exc}"); result = None

                if result:
                    def _chunks(text):
                        for i in range(0, len(text), 40):
                            yield text[i:i+40]; time.sleep(0.005)
                    response_placeholder.write_stream(_chunks(result["response_text"]))
                    if result["tool_calls_made"]:
                        with st.expander(f"🔧 {len(result['tool_calls_made'])} tool call(s)", expanded=False):
                            st.dataframe(pd.DataFrame([{"Tool": tc["tool"], "Provenance": tc["provenance"],
                                "Lag (min)": tc["freshness_minutes"], "Time (s)": tc.get("elapsed_s","—"),
                                "Status": "ERROR" if tc["had_error"] else "OK"} for tc in result["tool_calls_made"]]),
                                use_container_width=True, hide_index=True)
                    shown = set()
                    for w in result["data_freshness_warnings"]:
                        if w not in shown: st.warning(w, icon="⏱"); shown.add(w)
                    st.session_state.session_queries += 1
                    st.session_state.session_tool_calls += len(result["tool_calls_made"])
                    st.session_state.session_iterations += result.get("iterations_used", 0)
                    st.session_state.last_tool_calls = result["tool_calls_made"]
                    st.session_state.freshness_warnings = result["data_freshness_warnings"]
                    st.session_state.conversation_history = result.get("updated_history", st.session_state.conversation_history)


def _network_graph_content():
    """Full Network Graph (LangGraph DAG with step-through)."""
    import datetime as _dt
    st.subheader("🕸 Network Graph — LangGraph Agent Architecture")
    st.caption("Logically layered multi-agent DAG. Run a query in Chat, then use Step-Through to replay data flow.")

    _last_tcs = st.session_state.get("last_tool_calls", [])
    _TOOL_TO_NODE = {
        "simulate_price_change": "price_cascade", "adjust_promotional_price": "price_cascade",
        "get_competitive_pricing": "price_cascade",
        "get_carrier_status": "supply_disruption", "get_supply_disruption_impact": "supply_disruption",
        "find_alternate_carriers": "carrier_node",
        "get_demand_forecast": "demand_forecast", "analyze_demand_variables": "demand_forecast",
        "get_forecast_accuracy": "accuracy_node",
        "detect_scenario_conflicts": "scenario_planning", "run_scenario_comparison": "scenario_planning",
        "check_shelf_capacity": "shelf_replenishment", "trigger_replenishment": "shelf_replenishment",
        "check_perishable_status": "perishable_check",
        "get_inventory_levels": "inventory_node", "get_reorder_recommendations": "inventory_node",
        "calculate_stockout_risk": "inventory_node",
        "calculate_revenue_impact": "financial_impact", "calculate_carrying_cost": "financial_impact",
    }
    _exec_nodes = set()
    if _last_tcs:
        _exec_nodes.update(["router", "synthesizer"])
        for tc in _last_tcs:
            nd = _TOOL_TO_NODE.get(tc.get("tool", ""))
            if nd:
                _exec_nodes.add(nd)
                if nd == "price_cascade":        _exec_nodes.update(["inventory_node", "financial_impact"])
                if nd == "supply_disruption":    _exec_nodes.add("carrier_node")
                if nd == "demand_forecast":      _exec_nodes.add("accuracy_node")
                if nd == "shelf_replenishment":  _exec_nodes.add("perishable_check")

    _LAYERS = {
        "user_input": (5.0,10.0), "router": (5.0,8.5),
        "price_cascade": (1.5,7.0), "supply_disruption": (3.2,7.0),
        "demand_forecast": (5.0,7.0), "scenario_planning": (6.8,7.0),
        "shelf_replenishment": (8.5,7.0), "inventory_node": (1.5,5.5),
        "carrier_node": (3.2,5.5), "accuracy_node": (5.0,5.5),
        "perishable_check": (8.5,5.5), "financial_impact": (2.3,4.0),
        "synthesizer": (5.0,2.5), "output": (5.0,1.0),
    }
    _LABELS = {
        "user_input": "👤 User Query", "router": "🔀 Router\n(intent + entity)",
        "price_cascade": "💰 Price\nCascade", "supply_disruption": "🚛 Supply\nDisruption",
        "demand_forecast": "📈 Demand\nForecast", "scenario_planning": "⚖️ Scenario\nPlanning",
        "shelf_replenishment": "🏪 Shelf\nReplenishment", "inventory_node": "📦 Inventory\nNode",
        "carrier_node": "🛤 Carrier\nNode", "accuracy_node": "🎯 Accuracy\nGate",
        "perishable_check": "🥛 Perishable\nCheck", "financial_impact": "💵 Financial\nImpact",
        "synthesizer": "🔮 Synthesizer", "output": "📋 Response",
    }
    _DOMAIN_COLOR = {
        "user_input": "#6c757d", "router": "#0071ce",
        "price_cascade": "#e67e22", "supply_disruption": "#e74c3c",
        "demand_forecast": "#27ae60", "scenario_planning": "#8e44ad",
        "shelf_replenishment": "#16a085", "inventory_node": "#d35400",
        "carrier_node": "#c0392b", "accuracy_node": "#1e8449",
        "perishable_check": "#117a65", "financial_impact": "#1a5276",
        "synthesizer": "#0071ce", "output": "#6c757d",
    }
    _TOOLS_HINT = {
        "user_input": "Natural language query", "router": "Classifies intent → routes to domain",
        "price_cascade": "simulate_price_change\nadjust_promotional_price\nget_competitive_pricing",
        "supply_disruption": "get_carrier_status\nget_supply_disruption_impact",
        "demand_forecast": "get_demand_forecast\nanalyze_demand_variables",
        "scenario_planning": "run_scenario_comparison\ndetect_scenario_conflicts",
        "shelf_replenishment": "check_shelf_capacity\ntrigger_replenishment",
        "inventory_node": "get_inventory_levels\ncalculate_stockout_risk",
        "carrier_node": "find_alternate_carriers", "accuracy_node": "get_forecast_accuracy",
        "perishable_check": "check_perishable_status",
        "financial_impact": "calculate_revenue_impact\ncalculate_carrying_cost",
        "synthesizer": "Merges all node outputs → final response",
        "output": "Structured recommendation delivered",
    }
    _EDGES_NG = [
        ("user_input","router"), ("router","price_cascade"), ("router","supply_disruption"),
        ("router","demand_forecast"), ("router","scenario_planning"), ("router","shelf_replenishment"),
        ("price_cascade","inventory_node"), ("price_cascade","financial_impact"),
        ("supply_disruption","carrier_node"), ("demand_forecast","accuracy_node"),
        ("shelf_replenishment","perishable_check"), ("inventory_node","financial_impact"),
        ("carrier_node","synthesizer"), ("accuracy_node","synthesizer"),
        ("perishable_check","synthesizer"), ("financial_impact","synthesizer"),
        ("scenario_planning","synthesizer"), ("synthesizer","output"),
    ]

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2,1,1,1])
    with ctrl_c1:
        view_mode = st.radio("View mode", ["Full graph","Execution path only","Step-through"], horizontal=True, key="ng_view_mode")
    with ctrl_c2:
        show_labels = st.toggle("Tool hints", value=True, key="ng_show_tools")
    with ctrl_c3:
        show_layers = st.toggle("Layer bands", value=True, key="ng_layers")
    with ctrl_c4:
        if st.button("🔄 Reset step", key="ng_reset"):
            st.session_state["ng_step"] = "all"

    step_node = None
    if view_mode == "Step-through" and _exec_nodes:
        exec_order = ["user_input","router"] + [n for n in
            ["price_cascade","supply_disruption","demand_forecast","scenario_planning","shelf_replenishment",
             "inventory_node","carrier_node","accuracy_node","perishable_check","financial_impact","synthesizer","output"]
            if n in _exec_nodes or n in ("synthesizer","output")]
        step_idx = st.slider("Execution step", 0, len(exec_order)-1,
                             st.session_state.get("ng_step_idx",0), format="Step %d", key="ng_step_slider")
        st.session_state["ng_step_idx"] = step_idx
        step_node = exec_order[step_idx] if step_idx < len(exec_order) else None
        if step_node:
            st.info(f"**Step {step_idx}: `{step_node}`** — {_TOOLS_HINT.get(step_node,'').replace(chr(10),', ')}")

    fig_ng = go.Figure()
    if show_layers:
        for y0, y1, color, label in [
            (9.3,10.7,"rgba(108,117,125,0.05)","Input"), (7.7,9.2,"rgba(0,113,206,0.06)","Router"),
            (6.2,7.6,"rgba(230,126,34,0.06)","Domain Nodes"), (4.7,6.1,"rgba(39,174,96,0.06)","Support Nodes"),
            (3.2,4.6,"rgba(26,82,118,0.06)","Financial"), (1.7,3.1,"rgba(0,113,206,0.06)","Synthesizer"),
            (0.2,1.6,"rgba(108,117,125,0.05)","Output"),
        ]:
            fig_ng.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0,
                             annotation_text=label, annotation_position="right",
                             annotation_font=dict(size=10, color="#adb5bd"))

    visible_nodes = (_exec_nodes | {"user_input","output"}) if view_mode=="Execution path only" and _exec_nodes else set(_LAYERS.keys())
    visible_edges = [(s,d) for s,d in _EDGES_NG if s in visible_nodes and d in visible_nodes] if view_mode=="Execution path only" and _exec_nodes else _EDGES_NG

    for src, dst in visible_edges:
        x0,y0 = _LAYERS[src]; x1,y1 = _LAYERS[dst]
        both_exec = src in _exec_nodes and dst in _exec_nodes
        step_active = step_node and (src==step_node or dst==step_node)
        if view_mode=="Step-through":
            ec,ew = ("#f39c12",3.5) if step_active else (("#27ae60",2.5) if both_exec else ("#dee2e6",1.0))
        else:
            ec,ew = ("#27ae60",2.5) if both_exec else ("#dee2e6",1.2)
        fig_ng.add_annotation(x=x1,y=y1,ax=x0,ay=y0, xref="x",yref="y",axref="x",ayref="y",
                              showarrow=True,arrowhead=3,arrowsize=1.3,arrowwidth=ew,arrowcolor=ec)

    for node in visible_nodes:
        x,y = _LAYERS[node]
        base_color = _DOMAIN_COLOR.get(node,"#6c757d")
        is_exec = node in _exec_nodes; is_step = (step_node==node)
        if view_mode=="Step-through":
            fc,bc,bw,op = ("#f39c12","#e67e22",4,1.0) if is_step else ((base_color,"white",2,1.0) if is_exec else ("#ecf0f1","#bdc3c7",1,0.5))
        else:
            fc,bc,bw,op = (base_color,"white",2.5,1.0) if is_exec else ("#ecf0f1","#bdc3c7",1,0.85)
        tc = "white" if (is_exec or is_step) else "#495057"
        ns = 72 if is_step else (60 if is_exec else 48)
        hint = _TOOLS_HINT.get(node,"").replace("\n","<br>") if show_labels else ""
        status_txt = "⚡ Active step" if is_step else ("✅ Executed" if is_exec else "⬜ Available")
        fig_ng.add_trace(go.Scatter(x=[x],y=[y], mode="markers+text",
            marker=dict(size=ns,color=fc,opacity=op,line=dict(color=bc,width=bw),symbol="square"),
            text=[_LABELS.get(node,node)], textposition="middle center",
            textfont=dict(size=9 if ns<60 else 10, color=tc, family="Arial"),
            name=node,
            hovertemplate=f"<b>{_LABELS.get(node,node)}</b><br>Status: {status_txt}<br>" + (f"Tools:<br>{hint}<br>" if hint else "") + "<extra></extra>",
            showlegend=False))

    for label, color in [("Executed","#27ae60"),("Active Step","#f39c12"),("Available","#bdc3c7")]:
        fig_ng.add_trace(go.Scatter(x=[None],y=[None], mode="markers",
            marker=dict(size=12,color=color,symbol="square"), name=label, showlegend=True))

    fig_ng.update_layout(
        title=dict(text="LangGraph Multi-Agent Network — Layered Architecture", font=dict(size=15)),
        xaxis=dict(range=[0,10.5],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(range=[0.2,11.2],showgrid=False,zeroline=False,showticklabels=False),
        height=640, plot_bgcolor="white", paper_bgcolor="#f8f9fa",
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="right",x=1),
        margin=dict(l=10,r=120,t=60,b=20), hoverlabel=dict(bgcolor="white",font_size=12))
    st.plotly_chart(fig_ng, use_container_width=True)

    if _exec_nodes:
        ex_c1,ex_c2,ex_c3,ex_c4 = st.columns(4)
        domain_nodes = [n for n in _exec_nodes if n in ("price_cascade","supply_disruption","demand_forecast","scenario_planning","shelf_replenishment")]
        support_nodes = [n for n in _exec_nodes if n in ("inventory_node","carrier_node","accuracy_node","perishable_check","financial_impact")]
        ex_c1.metric("Nodes executed", len(_exec_nodes)); ex_c2.metric("Tool calls", len(_last_tcs))
        ex_c3.metric("Domain nodes", len(domain_nodes)); ex_c4.metric("Support nodes", len(support_nodes))
        with st.expander("📋 Node-level tool breakdown", expanded=False):
            rows_ng = []
            for nd in sorted(_exec_nodes):
                tcs_for_node = [tc for tc in _last_tcs if _TOOL_TO_NODE.get(tc.get("tool",""))==nd]
                rows_ng.append({"Node":nd,"Domain":_LABELS.get(nd,nd).replace("\n"," "),
                    "Tools called":len(tcs_for_node),
                    "Tool names":", ".join(tc["tool"] for tc in tcs_for_node) or "—",
                    "Provenance":", ".join(set(tc.get("provenance","?") for tc in tcs_for_node)) or "—"})
            if rows_ng: st.dataframe(pd.DataFrame(rows_ng), use_container_width=True, hide_index=True)
    else:
        st.info("No query run yet. Run a question in the **Chat** tab, then return here.")

    st.divider()
    st.markdown("### How the Graph Works")
    arch_col1, arch_col2 = st.columns(2)
    with arch_col1:
        st.markdown("""
**Entry Point: Router** — classifies intent, extracts SKU/region, sets route in state.

**Domain Nodes:** `price_cascade` · `supply_disruption` · `demand_forecast` · `scenario_planning` · `shelf_replenishment`

**Supporting Nodes:** `inventory_node` · `carrier_node` · `accuracy_node` · `perishable_check` · `financial_impact`
        """)
    with arch_col2:
        st.markdown("""
**Exit Node: Synthesizer** — merges domain findings, applies provenance warnings, returns final recommendation.

**Execution Paths:**
| Intent | Path |
|--------|------|
| Price change | router → price_cascade → inventory → financial → synthesizer |
| Supply disruption | router → supply_disruption → carrier → synthesizer |
| Demand forecast | router → demand_forecast → accuracy → synthesizer |
| Shelf replenishment | router → shelf_replenishment → perishable → synthesizer |
| Scenario planning | router → scenario_planning → synthesizer |
        """)


def _workflow_content():
    """Workflow builder (Step 1–6)."""
    import datetime as _dt
    st.subheader("Integrated Scenario Workflow Builder")
    st.caption("Walk through a structured decision — pick trigger, context, objective, then run the full AI analysis.")

    trigger_options = {
        "🔺 Price Change": "price_change", "🚛 Supply Disruption": "supply_disruption",
        "📉 Demand Shift": "demand_shift", "📦 Tariff / Cost Increase": "tariff",
        "🎄 Seasonal Event": "seasonal", "⚔️ Competitive Move": "competitive",
    }
    sku_labels = {
        "HUG48-3": "Huggies Size 3 (Diapers)", "MLK-GAL": "Whole Milk Gallon",
        "CIG-PKT": "Marlboro Cigarettes", "OJ-64": "Tropicana OJ 64oz",
        "FORMULA-24": "Similac Formula 24pk",
    }

    st.markdown("### Step 1 — Business Trigger")
    wf_trigger_sel = st.selectbox("Select trigger:", list(trigger_options.keys()), key="wf_trigger_sel")
    trigger_key = trigger_options[wf_trigger_sel]
    st.divider()

    st.markdown("### Step 2 — Context")
    wf_col1, wf_col2, wf_col3 = st.columns(3)
    with wf_col1:
        wf_sku = st.selectbox("SKU / Product", list(sku_labels.keys()), format_func=lambda x: sku_labels[x], key="wf_sku")
    with wf_col2:
        wf_region = st.selectbox("Region", ["All Regions","SE — Southeast","NW — Northwest","NE — Northeast","SW — Southwest"], key="wf_region")
    with wf_col3:
        wf_horizon = st.selectbox("Horizon", ["2 weeks (operational)","4 weeks (tactical)","8 weeks (strategic)"], key="wf_horizon")

    st.markdown("#### Trigger Parameters")
    if trigger_key == "price_change":
        base_price_wf = mock_executor.PRODUCTS.get(wf_sku, {}).get("base_price", 10.0)
        p_col1, p_col2 = st.columns(2)
        with p_col1: wf_cur_price = st.number_input("Current Price ($)", value=float(base_price_wf), step=0.01, key="wf_cur_price")
        with p_col2: wf_new_price = st.number_input("New Price ($)", value=float(base_price_wf*1.10), step=0.01, key="wf_new_price")
        pct_change = round((wf_new_price - wf_cur_price) / wf_cur_price * 100, 1) if wf_cur_price else 0
        st.caption(f"→ {abs(pct_change):.1f}% price {'increase' if pct_change>0 else 'decrease'}")
    elif trigger_key == "supply_disruption":
        d_col1, d_col2 = st.columns(2)
        with d_col1: wf_carrier = st.selectbox("Affected Carrier", ["TruckCo A","TruckCo B","TruckCo C","TruckCo D"], index=1, key="wf_carrier")
        with d_col2: wf_disruption_days = st.slider("Expected Duration (days)", 1, 30, 14, key="wf_disrupt_days")
    elif trigger_key == "demand_shift":
        wf_demand_pct = st.slider("Demand Change (%)", -30, 30, -10, key="wf_demand_pct")
        wf_demand_driver = st.selectbox("Primary Driver", ["Weather event","Competitor promotion","Demographic shift","Media coverage","Seasonal"], key="wf_demand_driver")
    elif trigger_key == "tariff":
        wf_tariff_pct = st.slider("Cost Increase (%)", 1, 50, 25, key="wf_tariff_pct")
        st.caption("The AI will model how much of this cost can be passed to consumers vs. absorbed in margin.")
    elif trigger_key == "seasonal":
        wf_season = st.selectbox("Seasonal Event", ["Back-to-School (Aug)","Thanksgiving (Nov)","Christmas (Dec)","Super Bowl (Feb)","Summer (Jun-Aug)"], key="wf_season")
        wf_uplift_pct = st.slider("Expected Demand Uplift (%)", 5, 50, 20, key="wf_uplift_pct")
    elif trigger_key == "competitive":
        wf_competitor = st.selectbox("Competitor", ["Costco","Target","Amazon","Kroger","Aldi"], key="wf_competitor")
        wf_comp_pct = st.slider("Competitor Price Change (%)", -30, 0, -15, key="wf_comp_pct")

    st.divider()
    st.markdown("### Step 3 — Decision Objective")
    obj_col1, obj_col2, obj_col3, obj_col4 = st.columns(4)
    obj_revenue = obj_col1.checkbox("Maximize Revenue", value=True, key="wf_obj_rev")
    obj_margin  = obj_col2.checkbox("Maximize Margin",  value=True, key="wf_obj_margin")
    obj_service = obj_col3.checkbox("Service Level",    value=False, key="wf_obj_service")
    obj_cost    = obj_col4.checkbox("Minimize Cost",    value=False, key="wf_obj_cost")
    objectives = [o for flag, o in [(obj_revenue,"revenue maximization"),(obj_margin,"margin maximization"),
                                     (obj_service,"service level / stockout prevention"),(obj_cost,"cost minimization")] if flag]
    if not objectives: objectives = ["revenue maximization"]

    st.divider()
    st.markdown("### Step 4 — Run AI Analysis")

    def _build_wf_query():
        sku_name = sku_labels.get(wf_sku, wf_sku)
        region_str = wf_region.split(" — ")[0] if " — " in wf_region else wf_region
        horizon_str = " ".join(wf_horizon.split(" ")[:2])
        obj_str = " and ".join(objectives)
        if trigger_key == "price_change":
            pc = round((wf_new_price - wf_cur_price) / wf_cur_price * 100, 1) if wf_cur_price else 0
            return (f"Analyze a price change for {sku_name} ({wf_sku}) from ${wf_cur_price:.2f} to ${wf_new_price:.2f} "
                    f"({'+' if pc>0 else ''}{pc:.1f}%) in {region_str} over {horizon_str}. Optimize for: {obj_str}. "
                    f"Cover: demand elasticity, inventory impact, carrier capacity, financial margin with vendor trade netting, conflict detection. 3 ranked options.")
        elif trigger_key == "supply_disruption":
            dur = st.session_state.get("wf_disrupt_days",14); carrier = st.session_state.get("wf_carrier","TruckCo B")
            return (f"{carrier} disruption expected to last {dur} days, affecting {sku_name} in {region_str}. "
                    f"Optimize for: {obj_str} over {horizon_str}. Analyze: stockout risk by DC, alternate carriers, revenue at risk, mitigation plan.")
        elif trigger_key == "demand_shift":
            change = st.session_state.get("wf_demand_pct",-10); driver = st.session_state.get("wf_demand_driver","external event")
            return (f"Demand for {sku_name} in {region_str} expected to {'increase' if change>0 else 'decrease'} "
                    f"by {abs(change)}% due to {driver} over {horizon_str}. Optimize for: {obj_str}. "
                    f"Analyze: inventory adequacy, replenishment adjustments, financial impact, 3 ranked options.")
        elif trigger_key == "tariff":
            tariff = st.session_state.get("wf_tariff_pct",25)
            return (f"A {tariff}% cost increase is being applied to {sku_name} in {region_str} over {horizon_str}. "
                    f"Optimize for: {obj_str}. Model pass-through scenarios, demand elasticity, margin erosion. 3 strategies.")
        elif trigger_key == "seasonal":
            return (f"Plan for {st.session_state.get('wf_season','Seasonal')} demand uplift of "
                    f"{st.session_state.get('wf_uplift_pct',20)}% for {sku_name} in {region_str} over {horizon_str}. "
                    f"Optimize for: {obj_str}. Analyze inventory build, replenishment, carrier capacity. 3 strategies.")
        elif trigger_key == "competitive":
            return (f"{st.session_state.get('wf_competitor','Competitor')} cut price by "
                    f"{abs(st.session_state.get('wf_comp_pct',-15))}% for product comparable to {sku_name} in {region_str}. "
                    f"Horizon: {horizon_str}. Optimize for: {obj_str}. Analyze cannibalization, match vs differentiate, 3 strategies.")
        return f"Analyze supply chain situation for {sku_name} in {region_str} over {horizon_str}."

    wf_query = _build_wf_query()
    with st.expander("Preview AI Query", expanded=False): st.text(wf_query)

    run_col1, run_col2 = st.columns([2,1])
    with run_col1: run_workflow = st.button("Run Analysis", type="primary", use_container_width=True, key="wf_run_btn")
    with run_col2: wf_pipeline = st.selectbox("Pipeline", ["V2 — LangGraph","V1 — Agentic Loop"], key="wf_pipeline_sel", label_visibility="collapsed")

    if run_workflow:
        wf_result = None
        with st.status("Running workflow analysis...", expanded=True) as wf_status:
            try:
                if "V2" in wf_pipeline:
                    st.write("→ Routing through LangGraph nodes...")
                    from agents.langgraph_flow import run_langgraph
                    wf_result = run_langgraph(wf_query, [])
                    st.write(f"→ Nodes: {', '.join(wf_result.get('node_outputs',{}).keys())}")
                    wf_status.update(label="Analysis complete", state="complete")
                else:
                    st.write("→ Running V1 agentic loop...")
                    orch = _get_orchestrator(_get_api_key())
                    wf_result = orch.run(wf_query, history=[], max_iterations=10)
                    wf_status.update(label="Analysis complete", state="complete")
            except Exception as exc:
                wf_status.update(label=f"Error: {exc}", state="error"); st.error(str(exc))
        if wf_result:
            st.session_state["wf_last_result"] = wf_result
            st.session_state["wf_last_trigger"] = trigger_key
            st.session_state["wf_last_sku"] = wf_sku
            resp_preview = (wf_result.get("response_text") or wf_result.get("response") or "")[:100]
            n_nodes = len(wf_result.get("node_outputs",{})); n_tools_wf = len(wf_result.get("tool_calls_made") or wf_result.get("tool_calls") or [])
            st.session_state.hist_workflow.append({
                "header": f"{wf_trigger_sel}  |  {wf_sku}", "meta": f"Region: {wf_region.split(' — ')[0]}  |  {wf_pipeline}",
                "result": f"{n_nodes} nodes, {n_tools_wf} tools" if n_nodes else f"{n_tools_wf} tools",
                "ts": _dt.datetime.now().strftime("%H:%M"),
            })

    wf_last = st.session_state.get("wf_last_result")
    if wf_last:
        st.divider()
        st.markdown("### Step 5 — Analysis Results")
        response_text = wf_last.get("response_text") or wf_last.get("response","")
        if response_text: st.markdown(response_text)
        tool_calls = wf_last.get("tool_calls",[])
        if tool_calls:
            with st.expander(f"Tools Used ({len(tool_calls)})", expanded=False):
                st.dataframe(pd.DataFrame([{"Tool":tc.get("tool",""),"Input":str(tc.get("input",""))[:80],"Result":str(tc.get("result",""))[:80]} for tc in tool_calls]), use_container_width=True, hide_index=True)
        if wf_last.get("node_outputs"):
            with st.expander("LangGraph Node Trace", expanded=False):
                for nn, no in wf_last["node_outputs"].items(): st.markdown(f"**{nn}** → {str(no)[:200]}")
        st.divider()
        st.markdown("### Step 6 — Decision Ripple Effect (Day 0 → 30)")
        last_trigger = st.session_state.get("wf_last_trigger","price_change")
        last_sku = st.session_state.get("wf_last_sku","HUG48-3")
        prod = mock_executor.PRODUCTS.get(last_sku,{})
        if last_trigger == "price_change":
            elasticity = prod.get("elasticity",-1.2)
            new_p = st.session_state.get("wf_new_price", prod.get("base_price",10)*1.10)
            cur_p = st.session_state.get("wf_cur_price", prod.get("base_price",10))
            pct = (new_p-cur_p)/cur_p if cur_p else 0
            demand_impact = pct*elasticity*100
            if prod.get("asymmetric") and pct<0: demand_impact *= prod.get("recovery_factor",0.70)
            timeline_events = [
                {"Day":0,"Domain":"Pricing","Event":f"Price changes: ${cur_p:.2f} → ${new_p:.2f}"},
                {"Day":1,"Domain":"Demand","Event":f"Demand model updates: {demand_impact:+.1f}% volume shift"},
                {"Day":1,"Domain":"Finance","Event":"Revenue forecast recalculated"},
                {"Day":2,"Domain":"Inventory","Event":"Reorder point recalculated"},
                {"Day":3,"Domain":"Supply Chain","Event":"Replenishment PO adjusted and transmitted"},
                {"Day":7,"Domain":"Inventory","Event":"DC receives adjusted stock; WMS updated"},
                {"Day":10,"Domain":"Inventory","Event":"Store shelves reflect new inventory level"},
                {"Day":14,"Domain":"Finance","Event":"14-day margin vs. forecast reconciliation"},
                {"Day":30,"Domain":"Finance","Event":"Month-end P&L close"},
            ]
        elif last_trigger == "supply_disruption":
            dur = st.session_state.get("wf_disrupt_days",14)
            timeline_events = [
                {"Day":0,"Domain":"Supply Chain","Event":"Disruption confirmed — carrier unavailable"},
                {"Day":1,"Domain":"Inventory","Event":"DC days-of-supply clock starts"},
                {"Day":2,"Domain":"Finance","Event":"Revenue-at-risk quantified per region"},
                {"Day":2,"Domain":"Supply Chain","Event":"Alternate carrier contracted"},
                {"Day":4,"Domain":"Inventory","Event":"First DC hits safety stock threshold"},
                {"Day":7,"Domain":"Inventory","Event":"Alternate carrier delivers first shipment (+45% cost)"},
                {"Day":dur,"Domain":"Supply Chain","Event":"Disruption resolved — primary carrier resumes"},
                {"Day":30,"Domain":"Finance","Event":"Total disruption cost reconciled"},
            ]
        else:
            timeline_events = [
                {"Day":0,"Domain":"Strategy","Event":"Decision triggered"},
                {"Day":1,"Domain":"Demand","Event":"Demand model receives new signal"},
                {"Day":2,"Domain":"Inventory","Event":"Inventory plan adjusted"},
                {"Day":3,"Domain":"Supply Chain","Event":"Replenishment updated"},
                {"Day":7,"Domain":"Finance","Event":"7-day financial impact measurement"},
                {"Day":30,"Domain":"Finance","Event":"Month-end reconciliation"},
            ]
        timeline_df = pd.DataFrame(timeline_events)
        domain_colors = {"Pricing":"#0071ce","Demand":"#f39c12","Inventory":"#27ae60","Supply Chain":"#e74c3c","Finance":"#8e44ad","Strategy":"#2980b9"}
        fig_tl = go.Figure()
        for _, row in timeline_df.iterrows():
            color = domain_colors.get(row["Domain"],"#95a5a6")
            fig_tl.add_trace(go.Scatter(x=[row["Day"]],y=[row["Domain"]], mode="markers+text",
                marker=dict(size=16,color=color,symbol="circle"), text=[f"Day {row['Day']}"],
                textposition="top center", hovertemplate=f"<b>Day {row['Day']} — {row['Domain']}</b><br>{row['Event']}<extra></extra>", showlegend=False))
        for domain, grp in timeline_df.groupby("Domain"):
            if len(grp)>1:
                fig_tl.add_trace(go.Scatter(x=grp["Day"].tolist(), y=[domain]*len(grp), mode="lines",
                    line=dict(color=domain_colors.get(domain,"#95a5a6"),width=2,dash="dot"), showlegend=False, hoverinfo="skip"))
        fig_tl.update_layout(title="Decision Ripple Effect — Day 0 to 30", xaxis_title="Day", yaxis_title="Domain",
                              height=380, xaxis=dict(range=[-1,32]), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_tl, use_container_width=True)
        with st.expander("Full Event Timeline", expanded=False):
            st.dataframe(timeline_df, use_container_width=True, hide_index=True)


def _guide_content(compact: bool = False):
    """Guide tab. compact=True shows v1 minimal version."""
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0071ce 0%,#004a8f 100%);
                border-radius:14px;padding:32px 36px;margin-bottom:24px;">
      <h1 style="color:white;margin:0;font-size:2em;">🏪 Welcome to Retail Supply Chain Optimization AI</h1>
      <p style="color:#cce5ff;margin:10px 0 0 0;font-size:1.05em;">
        A multi-agent AI system that simulates how a single retail decision cascades across
        pricing, inventory, supply chain, and finance <strong style="color:white;">simultaneously</strong>.
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("## What Does This App Do?")
    st.markdown("""
In retail, no decision lives in isolation. When you raise the price of diapers by 10%:
- Demand drops by ~14% (price elasticity)
- Replenishment orders to the DC get adjusted
- Carrier load requirements shift across regions
- A planned promotion may conflict with a supply disruption
- Net margin changes after vendor trade dollar netting

**This AI reasons through all of those connections at once** — using 17 specialist tools and two pipelines.
    """)
    st.info("**No spreadsheet expertise needed.** Ask questions in plain English in the Chat tab.", icon="💡")
    st.divider()

    st.markdown("## Quick Start — 3 Ways to Use This App")
    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown("""<div style="background:#f0f4ff;border:1px solid #0071ce;border-radius:10px;padding:18px;">
        <h4 style="color:#0071ce;margin-top:0;">💬 Option 1: Just Chat</h4>
        Go to the <strong>Chat</strong> tab and ask anything in plain English. The AI picks the right tools automatically.
        </div>""", unsafe_allow_html=True)
    with qs2:
        st.markdown("""<div style="background:#f0fff4;border:1px solid #28a745;border-radius:10px;padding:18px;">
        <h4 style="color:#28a745;margin-top:0;">🚀 Option 2: Use a Preset</h4>
        Click any of the <strong>6 Quick Scenario</strong> buttons in the sidebar to load a pre-written expert query into Chat.
        </div>""", unsafe_allow_html=True)
    with qs3:
        st.markdown("""<div style="background:#fff8f0;border:1px solid #f39c12;border-radius:10px;padding:18px;">
        <h4 style="color:#e67e22;margin-top:0;">🔄 Option 3: Workflow</h4>
        Use the Workflow builder to run a structured 6-step analysis with a Day 0→30 ripple timeline.
        </div>""", unsafe_allow_html=True)

    if compact:
        st.divider()
        st.success("**Ready to start?** Click the **Chat** tab and type your first question.", icon="🚀")
        return

    st.divider()
    st.markdown("## Glossary — All Terms Used in This App")
    st.caption("Expand any category below. Covers every acronym, business term, functional feature, technical concept, and mock data object used across the app.")

    with st.expander("📌 Acronyms & Short Forms", expanded=False):
        st.markdown("""
| Acronym | Full Form | Context in this app |
|---------|-----------|---------------------|
| **AI** | Artificial Intelligence | The reasoning engine powering all simulations and chat |
| **API** | Application Programming Interface | How this app calls Claude (Anthropic API) |
| **CI** | Confidence Interval | Forecast uncertainty band — widens 15% per 4-week horizon |
| **COGS** | Cost of Goods Sold | Direct cost of merchandise sold; used in P&L waterfall |
| **DAG** | Directed Acyclic Graph | The visual flow diagram of the LangGraph agent pipeline |
| **DC** | Distribution Center | Regional warehouse — DC-SE, DC-MW, DC-NE, DC-SW |
| **DoS** | Days of Supply | Current inventory ÷ daily demand rate |
| **EDLP** | Everyday Low Price | Pricing strategy: consistently low price, no promotions |
| **ERP** | Enterprise Resource Planning | Back-office system (e.g., SAP) for POs, finance, HR |
| **Hi-Lo** | High-Low Pricing | Regular price with periodic deep promotional discounts |
| **HQ** | Headquarters | Central buying / planning office that issues store directives |
| **KPI** | Key Performance Indicator | Headline metric shown in dashboard tiles |
| **LLM** | Large Language Model | The type of AI (Claude claude-sonnet-4-6) running the chat and analysis |
| **ML** | Machine Learning | Statistical models underlying demand forecasting |
| **MW** | Midwest (region) | One of 4 carrier/store regions — affected by TruckCo B strike |
| **NE** | Northeast (region) | One of 4 DC/store regions |
| **OLAP** | Online Analytical Processing | Aggregated analytics data warehouse — 24-hour refresh lag |
| **OLTP** | Online Transaction Processing | Point-of-sale and transactional database — 5-minute refresh |
| **P&L** | Profit & Loss | Income statement view: revenue → gross margin → net income |
| **PO** | Purchase Order | Replenishment order from DC to supplier |
| **SE** | Southeast (region) | One of 4 DC/store regions — affected by TruckCo B strike |
| **SKU** | Stock Keeping Unit | Unique product identifier (e.g., HUG48-3 = Huggies Size 3, 48ct) |
| **SW** | Southwest (region) | One of 4 DC/store regions |
| **UI** | User Interface | The Streamlit app you are using right now |
| **VMI** | Vendor-Managed Inventory | Manufacturer-owned stock at DC — excluded from retailer carrying cost |
| **WMS** | Warehouse Management System | Inventory and warehouse movement data — 15-minute refresh |
        """)

    with st.expander("🏪 Business & Retail Terms", expanded=False):
        st.markdown("""
| Term | Definition |
|------|------------|
| **Asymmetric Elasticity** | Price increases reduce demand more than equivalent price decreases recover it. Applies to diapers, tobacco, alcohol, and baby formula. |
| **Carrying Cost** | Annual cost of holding inventory — modeled at **25% of inventory value per year** (includes capital, storage, shrink, obsolescence). |
| **Chargeback / Deduction** | Penalty fee from retailer to supplier for service level failures (late PO, short fill). |
| **Co-op Advertising** | Promotional spend partially funded by the manufacturer as part of trade terms. |
| **Days of Supply (DoS)** | How long current on-hand inventory will last at current daily demand. Formula: `Inventory ÷ Daily Demand`. |
| **EDLP vs Hi-Lo** | Two pricing strategies. EDLP = stable low price. Hi-Lo = normal + frequent sales. This app models Hi-Lo with asymmetric recovery. |
| **Forecast Accuracy** | % of weeks where forecast is within 10% of actual demand. Below **60%** triggers the forecast gate (POs blocked). |
| **Forecast Gate** | Minimum accuracy threshold: if accuracy < 60%, automatic replenishment is paused pending manual review. |
| **In-Stock Rate** | % of SKU-store combinations with positive inventory at any given time. |
| **Lead Time** | Total elapsed time from PO creation → DC receipt → store shelf. Modeled as 3–4 days + 30% variance. |
| **Lost Sales** | Revenue forfeited during a stockout — units demanded but not available. |
| **Markdown** | Permanent price reduction, typically for clearance (vs a temporary promotion). |
| **Perishable Cap** | Hard supply ceiling for refrigerated / perishable items. Milk (MLK-GAL) = **3-day maximum**. Prevents over-ordering spoilage. |
| **Planogram** | Prescribed shelf layout plan. If a replenishment order exceeds shelf capacity, excess goes to back-of-store. |
| **Price Elasticity** | % change in unit demand per 1% change in price. Huggies diapers = **-1.4** (inelastic). Milk ≈ -0.5 (very inelastic). |
| **Promo Lift** | Additional demand generated by a promotional event (price cut, feature, display). |
| **Replenishment Lag** | Delay between placing a PO and receiving goods. Base: 3–4 days. 30% probability of 3 additional days. |
| **Safety Stock** | Buffer inventory held above cycle stock to cover demand spikes and late deliveries. |
| **Scenario Conflict** | Two simultaneous decisions pulling supply in opposite directions — e.g., diaper promotion (needs extra supply) + TruckCo B strike (reduces supply). |
| **Service Level** | % of customer demand fulfilled without stockout. Target: 95–98% for essentials. |
| **Stockout** | On-hand inventory reaches zero; sales are lost until the next replenishment arrives. |
| **Trade Spend / Vendor Trade Dollars** | Manufacturer subsidies paid to the retailer for running promotions. Net cost = promo cost − trade dollars. |
| **VMI (Vendor-Managed Inventory)** | Inventory physically at the DC but owned by the manufacturer. Does NOT appear in the retailer's carrying cost. |
| **Waterfall Chart** | Step-by-step financial chart showing how gross revenue flows through deductions to net income. Used in Financial Impact tab. |
        """)

    with st.expander("⚙️ Functional Features (App Tabs)", expanded=False):
        st.markdown("""
| Feature / Tab | What it does |
|---------------|-------------|
| **Dashboard** | Live network KPI tiles: total demand, at-risk revenue, supply nodes, stockout probability. Refreshes on each run. |
| **Chat** | Streaming multi-agent conversational interface. Ask any retail question in plain English — the AI picks the right tools. |
| **Price Cascade** | Simulate a retail price change → see ripple effect on demand, replenishment POs, inventory levels, carrier load, and P&L simultaneously. |
| **Supply Alert** | Model carrier strikes, port delays, or supplier bankruptcies. Shows inventory depletion timeline, at-risk revenue, and alternate routing. |
| **Demand Forecast** | 15-variable statistical demand model with configurable horizon (1–16 weeks). Outputs weekly forecast + confidence interval fan chart. |
| **Scenario Planner** | Compare up to 3 simultaneous business decisions. Detects conflicts, ranks scenarios by revenue impact, shows resolution options. |
| **Shelf & Store** | End-to-end replenishment simulation: HQ order → DC processing → last-mile store delivery → shelf capacity vs planogram check. |
| **Financial Impact** | Full P&L waterfall: gross revenue → vendor trade → net revenue → margin → carrying cost → bottom-line impact. |
| **Data Sources** | Provenance and freshness tracker — shows which OLTP/WMS/OLAP sources fed each result and how stale the data is. |
| **Workflow** | 6-step guided scenario builder: define event → assess impact → identify risk → choose action → model outcome → sign off. |
| **Flow Map** | Static Plotly DAG of the LangGraph multi-agent graph. Shows nodes and edges at a glance (v2 only). |
| **Network Graph** | Interactive step-through DAG with layer bands, execution path highlighting, and per-node tool/output detail. |
| **Scenario Builder** | HTML5 drag-and-drop canvas with 30 decision blocks across 6 categories. Wire blocks together → Run Analysis for compound KPI impact. |
| **Strategy Canvas** | Drag-and-drop strategic planning workspace — place and connect strategy elements visually. |
| **Guide** | This tab — onboarding, sample queries, glossary, system facts, and AI architecture walkthrough. |
        """)

    with st.expander("🤖 Technical & AI Terms", expanded=False):
        st.markdown("""
| Term | Definition |
|------|------------|
| **Agentic Loop (V1)** | Single Claude model repeatedly calls tools, evaluates results, and decides the next tool — until it reaches a final answer. |
| **Claude claude-sonnet-4-6** | Anthropic's AI model powering this app. 200K context window, natively supports tool use. |
| **DAG (Directed Acyclic Graph)** | A flowchart where nodes are agents/tools and edges are data flows. No cycles — execution is always forward. |
| **Function Calling / Tool Use** | The mechanism by which the AI calls external Python functions (tools) to retrieve data rather than hallucinating it. |
| **LangGraph** | Python framework for building multi-agent AI systems as explicit state graphs. Used for the V2 pipeline. |
| **LangGraph Node** | A single processing step in the graph — e.g., `price_cascade`, `inventory_check`, `synthesizer`. |
| **LangGraph Router** | The entry node that reads the user's query and decides which domain nodes to activate. |
| **LangGraph Synthesizer** | The exit node that receives outputs from all activated domain nodes and produces one coherent final answer. |
| **Mock Executor** | Python module (`tools/mock_executor.py`) that simulates all 17 AI tools with realistic retail data — no live database needed. |
| **Orchestrator (V1)** | `agents/orchestrator.py` — manages the V1 agentic loop, history summarization, and provenance tracking. |
| **Pipeline Toggle** | Sidebar Advanced setting: switch between V1 (agentic loop) and V2 (LangGraph) at runtime. |
| **Plotly** | Python visualization library used for waterfall charts, bar/fan charts, gauge charts, and DAG diagrams. |
| **Pydantic** | Python data validation library. All tool inputs and outputs are typed with Pydantic models (`data/schemas.py`). |
| **Session State** | Streamlit's per-user in-memory dictionary (`st.session_state`) that persists data across interactions within one browser session. |
| **Streamlit** | Python framework for building interactive data web apps. Runs server-side; reruns the entire script on each user action. |
| **Streaming** | Chat responses appear word-by-word (token-by-token) as the model generates them, rather than waiting for the full response. |
| **Tool Schema** | JSON definition of a tool's name, description, and input parameters — what the AI reads to decide which tool to call. |
| **V1 Pipeline** | Single-agent agentic loop. Best for exploratory, multi-turn conversation. Default for v1 app version. |
| **V2 Pipeline** | LangGraph 12-node multi-agent graph. Best for structured, repeatable decisions. Default for v2 and v3 app versions. |
        """)

    with st.expander("🗄️ Data Sources & Systems", expanded=False):
        st.markdown("""
| System | Type | Refresh Rate | What it provides in this app |
|--------|------|-------------|------------------------------|
| **OLTP** | Transactional DB (e.g., SAP) | 5 minutes | POS sales, PO status, real-time inventory movements |
| **WMS** | Warehouse Mgmt System | 15 minutes | DC inventory positions, inbound shipments, pick/pack status |
| **OLAP** | Analytics Data Warehouse (e.g., Snowflake) | 24 hours | Historical demand, forecast accuracy, financial aggregates |
| **SAP** | ERP System | ~5 min (OLTP) | Purchase orders, vendor master, financial postings |
| **Oracle SCM** | Supply Chain Mgmt | ~15 min (WMS) | Carrier assignments, shipment tracking, logistics routing |
| **Snowflake** | Cloud Data Warehouse | 24 hours (OLAP) | Historical analytics, forecast model training data |
| **Circana (IRI)** | Third-party Retail Analytics | Daily | Market share, category trends, competitive pricing |
| **Nielsen** | Consumer Research | Weekly | Consumer panel data, demand elasticity benchmarks |
| **Freshness Warning** | App concept | — | Alert shown when a data source is beyond its expected refresh window |
| **Provenance** | App concept | — | Which source(s) fed a specific calculation — shown in Data Sources tab |
        """)

    with st.expander("📦 Mock Data — SKUs, Carriers & Network", expanded=False):
        st.markdown("""
**Products (SKUs)**

| SKU ID | Name | Base Price | Elasticity | Notes |
|--------|------|-----------|------------|-------|
| **HUG48-3** | Huggies Diapers, Size 3, 48ct | $12.99 | -1.4 | Asymmetric elasticity; subject to TruckCo B strike |
| **PAM72-5** | Pampers Diapers, Size 5, 72ct | $22.99 | -1.2 | Premium tier; also affected by strike |
| **MLK-GAL** | Whole Milk, 1 Gallon | $4.29 | -0.5 | Perishable — 3-day supply cap enforced |
| **TAB-DIN** | Tablet / Dinner (shelf-stable) | $3.49 | -0.9 | General grocery; no special constraints |
| **BLK-THR** | Black & Mild (tobacco) | $6.99 | -0.3 | Highly inelastic; age-restricted; asymmetric |
| **CIG-PKT** | Cigarette Pack | $9.49 | -0.25 | Most inelastic SKU in the model |

**Carriers**

| Carrier | Status | Regions Covered | Notes |
|---------|--------|----------------|-------|
| **TruckCo_A** | ✅ Active | All regions | Primary backup during TruckCo B strike |
| **TruckCo_B** | 🔴 On Strike | SE, MW | Carries diapers (HUG48-3, PAM72-5) in Southeast + Midwest |
| **RailFreight_C** | ✅ Active | NE, SW | Long-haul; higher lead time |
| **AirCargo_D** | ✅ Active | All regions | Expedited; 3× cost — only for critical stockout prevention |

**Distribution Network**

| Node | Type | Region | Capacity |
|------|------|--------|---------|
| **DC-SE** | Distribution Center | Southeast | 50,000 units |
| **DC-MW** | Distribution Center | Midwest | 45,000 units |
| **DC-NE** | Distribution Center | Northeast | 40,000 units |
| **DC-SW** | Distribution Center | Southwest | 35,000 units |
| **STR-001 → STR-030** | Store | All regions | 30 stores, 7–8 per region |
        """)

    with st.expander("📐 Formulas & Model Parameters", expanded=False):
        st.markdown("""
**Price Cascade**
```
Demand Δ% = Elasticity × Price Δ%
  (with asymmetric cap: recovery is max 85% of the uplift for inelastic categories)
Weekly Unit Delta = Base Weekly Units × Demand Δ%
Net Revenue Δ = (New Price × New Units) − (Old Price × Old Units) − Vendor Trade Offset
Carrying Cost Δ = ΔInventory × (Unit Cost × 25% annual rate / 52 weeks)
```

**Demand Forecast**
```
Forecast = Base Demand × (1 + Trend) × Seasonality × Promo Lift × Elasticity Adjustment
Confidence Interval width = ±(Base CI) × (1 + 0.15 × floor(Horizon Weeks / 4))
Forecast Gate: if Accuracy < 60% → block automated POs
```

**Replenishment / Inventory**
```
Reorder Point = (Daily Demand × Lead Time) + Safety Stock
Safety Stock = Z-score(95%) × StdDev(Demand) × √Lead Time
Days of Supply = On-Hand Units / Daily Demand Rate
Perishable Max Order = min(Reorder Qty, 3 × Daily Demand)   [for MLK-GAL]
```

**Financial Impact**
```
Gross Revenue Δ = ΔPrice × New Volume
Vendor Trade Offset = Promo subsidy from manufacturer (reduces net cost)
Net Revenue Δ = Gross Revenue Δ − Vendor Trade Offset
Margin Δ = Net Revenue Δ × Margin %
Bottom-Line Impact = Margin Δ − Carrying Cost Δ − Lost Sales cost
```

**Scenario Conflict Score**
```
Conflict = 1 if (Scenario A requires supply increase) AND (Scenario B causes supply decrease)
        for the same SKU-region combination within the same time window
```
        """)

    st.divider()
    st.markdown("## Sample Questions to Ask in the Chat Tab")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**Pricing**")
        for q in ["Raise HUG48-3 price from $12.99 to $14.49 — full cascade",
                   "Revenue impact of dropping diaper price to $11.99 for 30 days?",
                   "Compare holding vs raising vs promoting HUG48-3 over 8 weeks"]:
            st.markdown(f"- *{q}*")
        st.markdown("**Supply Chain**")
        for q in ["TruckCo B is on strike — stockout risk in SE and MW?",
                   "Find alternate carriers for diapers in the Southeast",
                   "Revenue at risk if TruckCo B strike lasts 21 days?"]:
            st.markdown(f"- *{q}*")
    with q2:
        st.markdown("**Inventory & Demand**")
        for q in ["Replenishment status for milk (MLK-GAL) at STR-005",
                   "Demand forecast for diapers at 8 weeks with confidence intervals",
                   "Forecast accuracy for HUG48-3 is 67% — what's the revenue risk?"]:
            st.markdown(f"- *{q}*")
        st.markdown("**Finance**")
        for q in ["Promo on diapers AND TruckCo B strike — is this safe?",
                   "P&L waterfall for a 10% diaper price increase",
                   "Carrying cost for DC-SE if we pre-build 10 days of inventory"]:
            st.markdown(f"- *{q}*")

    st.divider()
    st.markdown("## System Facts at a Glance")
    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("AI Tools", "17", "across 5 domains")
    fc2.metric("LangGraph Nodes", "12", "router + specialists + synthesizer")
    fc3.metric("Edge Cases Handled", "17", "production-grade")
    fc4.metric("App Tabs", "7 (v3) / 15 (v2) / 2 (v1)")
    kn1, kn2, kn3, kn4 = st.columns(4)
    kn1.metric("Diaper Elasticity", "-1.4", "10% up → 14% vol down")
    kn2.metric("Dairy Max Supply", "3 days", "hard perishable cap")
    kn3.metric("Replenishment Lag", "3–4 days", "+30% chance of +3 more")
    kn4.metric("Forecast Gate", "60%", "below this: PO blocked")

    st.divider()
    st.markdown("## How the AI Works")
    arch1, arch2 = st.columns([2,3])
    with arch1:
        st.markdown("""
**V2 — LangGraph (default)**
12-node graph: Router → domain specialists → Synthesizer. Best for structured, repeatable decisions.

**V1 — Agentic Loop**
Single Claude claude-sonnet-4-6 agent with all 17 tools. Best for exploratory multi-turn conversation.

Switch in the **⚙️ Advanced** sidebar expander.
        """)
    with arch2:
        st.code("""
V2: LangGraph Multi-Agent Graph
    ROUTER → price_cascade → inventory → financial → SYNTHESIZER
    ROUTER → supply_disruption → carrier → SYNTHESIZER
    ROUTER → demand_forecast → accuracy → SYNTHESIZER
    ROUTER → shelf_replenishment → perishable → SYNTHESIZER
    ROUTER → scenario_planning → SYNTHESIZER

V1: Single Agent
    Claude claude-sonnet-4-6 → 17 tools → autonomous decision

[Mock Data] 5 SKUs · 4 Carriers · 4 DCs · 30 Stores
           OLTP (5min) · WMS (15min) · OLAP (24h)
        """, language="")
    st.success("**Ready to start?** Click the **Chat** tab and type your first question.", icon="🚀")


# ─── Price Cascade content (shared between v2 tab and v3 simulate) ───────────

def _price_cascade_content(with_right_panel: bool = False):
    import datetime as _dt
    st.subheader("Price Change Cascade Simulator")
    st.caption("Change a retail price → see the full ripple effect on demand, POs, inventory, and financials.")

    def _body():
        col1,col2,col3,col4 = st.columns(4)
        with col1: pc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="pc_sku")
        prod_info = mock_executor.PRODUCTS.get(pc_sku, {})
        with col2: pc_old_price = st.number_input("Current Price ($)", value=float(prod_info.get("base_price",12.99)), min_value=0.01, step=0.10, key="pc_old")
        with col3: pc_new_price = st.number_input("New Price ($)", value=float(prod_info.get("base_price",12.99))+1.50, min_value=0.01, step=0.10, key="pc_new")
        with col4: pc_horizon = st.slider("Horizon (weeks)", 1, 32, 8, key="pc_horizon")

        prod_class = prod_info.get("product_class","general")
        if prod_class in ("diaper","tobacco","alcohol","formula"):
            st.info(f"**Asymmetric elasticity** active for `{prod_class}` category.")

        if st.button("Simulate Price Cascade", type="primary", key="pc_run"):
            with st.spinner("Running cascade model..."):
                res = mock_executor.execute("simulate_price_change", {"sku":pc_sku,"old_price":pc_old_price,"new_price":pc_new_price,"horizon_weeks":pc_horizon})
            if res.get("error"):
                st.error(res["error"])
            else:
                d = res["data"]; di,fi,ii = d["demand_impact"],d["financial_impact"],d["inventory_impact"]
                k1,k2,k3,k4,k5 = st.columns(5)
                k1.metric("Price Δ", f"{d['price_change_pct']:+.1f}%")
                k2.metric("Demand Δ", f"{di['demand_change_pct']:+.1f}%", f"{di['weekly_unit_delta']:+.0f} units/wk")
                k3.metric("Net Revenue Δ", f"${fi['net_revenue_change_usd']:+,.0f}", delta_color="inverse" if fi["net_revenue_change_usd"]<0 else "normal")
                k4.metric("Margin Δ", f"${fi['margin_change_usd']:+,.0f}")
                k5.metric("Carrying Cost Δ", f"${ii['carrying_cost_increase_usd']:+,.0f}")
                st.markdown(f"**Affected nodes:** {', '.join(d['affected_nodes'])}")
                fig_wf = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["relative","relative","total","relative","relative","total"],
                    x=["Gross Revenue Δ","Vendor Trade","Net Revenue","Margin Δ","Carrying Cost","Bottom Line"],
                    y=[fi["gross_revenue_change_usd"],fi["vendor_trade_offset_usd"],fi["net_revenue_change_usd"],
                       fi["margin_change_usd"],-ii["carrying_cost_increase_usd"],fi["combined_bottom_line_impact_usd"]],
                    connector={"line":{"color":"#dee2e6"}},
                    increasing={"marker":{"color":"#2ecc71"}}, decreasing={"marker":{"color":"#e74c3c"}},
                    totals={"marker":{"color":"#3498db"}},
                    text=[f"${abs(fi['gross_revenue_change_usd']):,.0f}",f"${fi['vendor_trade_offset_usd']:,.0f}",
                          f"${abs(fi['net_revenue_change_usd']):,.0f}",f"${abs(fi['margin_change_usd']):,.0f}",
                          f"${ii['carrying_cost_increase_usd']:,.0f}",f"${abs(fi['combined_bottom_line_impact_usd']):,.0f}"],
                    textposition="outside"))
                fig_wf.update_layout(title=f"Financial Impact Waterfall — {d['sku_name']} ({pc_horizon}W)", height=380, margin=dict(l=40,r=40,t=55,b=40))
                st.plotly_chart(fig_wf, use_container_width=True)
                col_l,col_r = st.columns(2)
                with col_l:
                    fig_dem = go.Figure(data=[
                        go.Bar(name="Before", x=["Per Store/Wk","Total Network/Wk"],
                               y=[di["old_weekly_demand_per_store"],di["old_total_weekly"]], marker_color="#3498db"),
                        go.Bar(name="After", x=["Per Store/Wk","Total Network/Wk"],
                               y=[di["new_weekly_demand_per_store"],di["new_total_weekly"]],
                               marker_color="#e74c3c" if d["price_direction"]=="increase" else "#2ecc71"),
                    ])
                    fig_dem.update_layout(barmode="group",title="Demand Before vs After",height=300,margin=dict(l=30,r=20,t=40,b=30))
                    st.plotly_chart(fig_dem, use_container_width=True)
                with col_r:
                    pos = d.get("po_adjustments",[])
                    if pos:
                        st.markdown("**Open PO Adjustments**")
                        po_df = pd.DataFrame(pos)
                        po_df["Recommendation"] = po_df["recommended_adjustment_units"].apply(lambda x: f"{'Reduce' if x<0 else 'Increase'} {abs(int(x))} units")
                        po_df["Adjustable"] = po_df["adjustable"].map({True:"Yes",False:"Too close"})
                        st.dataframe(po_df[["po_id","current_units","Recommendation","eta_days","Adjustable"]], use_container_width=True, hide_index=True)
                    else:
                        st.info("No open POs for this SKU.")
                st.subheader("Recommended Actions")
                for i, rec in enumerate(d.get("recommendations",[]),1): st.markdown(f"**{i}.** {rec}")
                if d.get("asymmetric_elasticity_note") and "N/A" not in d.get("asymmetric_elasticity_note",""):
                    st.warning(d["asymmetric_elasticity_note"])
                st.session_state.hist_price.append({
                    "header":f"{pc_sku}  ${pc_old_price:.2f} → ${pc_new_price:.2f}",
                    "meta":f"Horizon: {pc_horizon}W  |  Demand Δ: {di['demand_change_pct']:+.1f}%",
                    "result":f"Revenue Δ: ${fi['net_revenue_change_usd']:+,.0f}  |  Margin Δ: ${fi['margin_change_usd']:+,.0f}",
                    "ts":_dt.datetime.now().strftime("%H:%M")})

    if with_right_panel:
        pc_main, pc_hist = st.columns([2,1])
        with pc_hist:
            _render_history(st.session_state.hist_price, "Run a simulation to see history.")
            _render_formula_panel("price"); _render_data_sources("price")
        with pc_main:
            _body()
    else:
        _body()


def _supply_alert_content(with_right_panel: bool = False):
    import datetime as _dt
    st.subheader("Supply Disruption Analyzer")
    st.caption("Model carrier strikes, port delays, and supplier bankruptcies.")

    if with_right_panel:
        sa_main, sa_hist = st.columns([2,1])
        with sa_hist:
            _render_history(st.session_state.hist_supply, "Run a disruption analysis to see history.")
            _render_formula_panel("supply"); _render_data_sources("supply")
    else:
        sa_main = st.container()

    with sa_main:
        col1,col2,col3 = st.columns([1,1,1])
        with col1:
            sa_type = st.selectbox("Disruption Type", ["carrier_strike","port_delay","supplier_bankruptcy"], key="sa_type")
            sa_entity = st.selectbox("Affected Entity", ["TruckCo_B","TruckCo_A","TruckCo_C","PORT-LA","SUP-KIMBERLY"], key="sa_entity")
        with col2:
            defaults_dur = {"carrier_strike":14,"port_delay":45,"supplier_bankruptcy":0}
            sa_duration = st.number_input("Duration (days)", value=defaults_dur[sa_type], min_value=0, max_value=180, key="sa_dur")
            sa_region = st.selectbox("Focus Region", ["all","SE","NW","MW"], key="sa_region")
        with col3:
            sa_skus_all = st.multiselect("Override affected SKUs (blank = auto-detect)", list(mock_executor.PRODUCTS.keys()), key="sa_skus")
            st.markdown(""); st.markdown("")
            run_sa = st.button("Analyze Disruption", type="primary", key="sa_run")
        st.info({"carrier_strike":"Carrier strikes: typically resolve in **2-3 weeks**.",
                 "port_delay":"Port delays: typically **6-10 weeks**.",
                 "supplier_bankruptcy":"Supplier bankruptcy: **permanent** until new supplier qualified (8-16 weeks)."}[sa_type])
        if run_sa:
            with st.spinner("Analyzing disruption impact..."):
                res = mock_executor.execute("get_supply_disruption_impact", {"disruption_type":sa_type,"affected_entity":sa_entity,"duration_days":sa_duration,"affected_skus":sa_skus_all or None})
            if res.get("error"): st.error(res["error"])
            else:
                d = res["data"]; crit = d.get("critical_count",0); warn = d.get("warning_count",0)
                if crit>0: st.error(f"CRITICAL: {crit} location(s) hit stockout before replenishment arrives.")
                elif warn>0: st.warning(f"WARNING: {warn} location(s) in danger zone.")
                else: st.success("All monitored locations within safety stock parameters.")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("SKUs Affected",len(d.get("affected_skus",[])))
                m2.metric("Critical Locations",crit)
                m3.metric("Warning Locations",warn)
                m4.metric("Revenue at Risk",f"${d.get('total_revenue_at_risk_usd',0):,.0f}")
                risks = d.get("stockout_risks",[])
                if risks:
                    risk_df = pd.DataFrame(risks)
                    fig_r = px.bar(risk_df,x="store_id",y="days_to_store_stockout",color="severity",
                        color_discrete_map={"critical":"#e74c3c","warning":"#f39c12","ok":"#2ecc71"},
                        title="Days to Store Stockout",labels={"days_to_store_stockout":"Days on Hand"},height=320)
                    fig_r.add_hline(y=4,line_dash="dash",line_color="orange",annotation_text="Replenishment lag (4d)",annotation_position="top right")
                    st.plotly_chart(fig_r, use_container_width=True)
                alts = d.get("alternates_by_region",{})
                if alts:
                    st.subheader("Alternate Carriers by Region")
                    alt_cols = st.columns(len(alts))
                    for col,(region,options) in zip(alt_cols,alts.items()):
                        with col:
                            st.markdown(f"**Region: {region}**")
                            if not options: st.markdown('<span class="chip-red">No viable alternates</span>',unsafe_allow_html=True)
                            for opt in options:
                                if opt.get("available"):
                                    st.markdown(f'<span class="chip-green">✓ {opt["carrier_id"]}</span> +{opt.get("additional_lead_time_days",0)}d, {opt.get("capacity_pct",0)}% cap, +{opt.get("cost_premium_pct",0):.0f}% cost',unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<span class="chip-red">✗ {opt["carrier_id"]}</span> {opt.get("reason_unavailable","")}',unsafe_allow_html=True)
                st.subheader("Mitigation Plan")
                for i,step in enumerate(d.get("mitigation_plan",[]),1):
                    (st.error if "IMMEDIATE" in step else st.info if "STRATEGIC" in step else st.markdown)(f"**{i}.** {step}")
                severity_label = "CRITICAL" if crit>0 else ("WARNING" if warn>0 else "OK")
                st.session_state.hist_supply.append({
                    "header":f"{sa_entity} — {sa_type.replace('_',' ').title()}",
                    "meta":f"{sa_duration}d  |  {len(d.get('affected_skus',[]))} SKU(s)",
                    "result":f"{severity_label}: {crit} critical, {warn} warning  |  Rev@Risk: ${d.get('total_revenue_at_risk_usd',0):,.0f}",
                    "ts":_dt.datetime.now().strftime("%H:%M")})


def _demand_forecast_content(with_right_panel: bool = False):
    import datetime as _dt
    st.subheader("Demand Forecast — 15-Variable Model")
    st.caption("Demand is a class function. Price, promo, tariffs, weather, and 11 more variables all move the signal.")

    if with_right_panel:
        fc_main, fc_hist = st.columns([2,1])
        with fc_hist:
            _render_history(st.session_state.hist_forecast, "Run a forecast to see history.")
            _render_formula_panel("forecast"); _render_data_sources("forecast")
    else:
        fc_main = st.container()

    with fc_main:
        col1,col2 = st.columns([1,1])
        with col1:
            fc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="fc_sku")
            fc_horizon = st.slider("Forecast Horizon (weeks)", 1, 32, 8, key="fc_horizon")
        with col2:
            st.markdown("**Override demand variables**")
            fc_c1,fc_c2 = st.columns(2)
            fc_price  = fc_c1.number_input("Price change %", value=0.0, step=1.0, key="fc_price")
            fc_promo  = fc_c2.number_input("Promo intensity %", value=0.0, step=5.0, key="fc_promo")
            fc_tariff = fc_c1.number_input("Tariff increase %", value=0.0, step=1.0, key="fc_tariff")
            fc_weather= fc_c2.number_input("Weather impact (0=none)", value=0.0, step=1.0, key="fc_weather")
        if st.button("Run Forecast", type="primary", key="fc_run"):
            overrides = {k:v for k,v in {"price":fc_price,"promo":fc_promo,"tariff":fc_tariff,"weather":fc_weather}.items() if v!=0}
            with st.spinner("Running 15-variable demand model..."):
                res = mock_executor.execute("get_demand_forecast",{"sku":fc_sku,"horizon_weeks":fc_horizon,"variable_overrides":overrides or None})
                acc_res = mock_executor.execute("get_forecast_accuracy",{"sku":fc_sku,"horizon_weeks":fc_horizon})
            if res.get("error"): st.error(res["error"])
            else:
                d = res["data"]; acc = d["forecast_accuracy_mape"]
                fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta",value=acc*100,
                    title={"text":"Forecast Accuracy (MAPE)","font":{"size":16}},
                    delta={"reference":85,"valueformat":".1f","suffix":"%"},
                    gauge={"axis":{"range":[0,100]},"bar":{"color":"#2ecc71" if acc>=0.85 else "#f39c12" if acc>=0.70 else "#e74c3c"},
                           "steps":[{"range":[0,60],"color":"#fadbd8"},{"range":[60,70],"color":"#fdebd0"},
                                    {"range":[70,85],"color":"#fef9e7"},{"range":[85,100],"color":"#eafaf1"}],
                           "threshold":{"line":{"color":"#2c3e50","width":3},"value":85}},
                    number={"suffix":"%","valueformat":".1f"}))
                fig_gauge.update_layout(height=280,margin=dict(l=20,r=20,t=40,b=20))
                col_g,col_note = st.columns([1,2])
                with col_g: st.plotly_chart(fig_gauge, use_container_width=True)
                with col_note:
                    st.markdown(f"**SKU:** `{fc_sku}` — {d['sku_name']}")
                    st.markdown(f"**Horizon:** {fc_horizon} weeks")
                    if not d["is_reliable"]: st.error(f"UNRELIABLE ({acc*100:.1f}%). DO NOT pass to PO system.")
                    elif acc<0.70: st.warning(f"Below 70% — use with caution.")
                    else: st.success("Acceptable accuracy. Suitable as planning input.")
                    st.markdown(f"**Gap to benchmark (85%):** {round((0.85-acc)*100,1)}pts  \n**Revenue impact:** ~{round((0.85-acc)*100,1)}% efficiency loss")
                    st.caption(d.get("ci_note",""))
                points = d.get("weekly_forecast",[])
                if points:
                    wks=[p["week"] for p in points]; pts=[p["point_estimate"] for p in points]
                    l80=[p["ci_80_lower"] for p in points]; u80=[p["ci_80_upper"] for p in points]
                    l95=[p["ci_95_lower"] for p in points]; u95=[p["ci_95_upper"] for p in points]
                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=wks+wks[::-1],y=u95+l95[::-1],fill="toself",fillcolor="rgba(52,152,219,0.08)",line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
                    fig_fc.add_trace(go.Scatter(x=wks+wks[::-1],y=u80+l80[::-1],fill="toself",fillcolor="rgba(52,152,219,0.22)",line=dict(color="rgba(0,0,0,0)"),name="80% CI"))
                    fig_fc.add_trace(go.Scatter(x=wks,y=pts,mode="lines+markers",line=dict(color="#2980b9",width=2.5),marker=dict(size=5),name="Point Estimate"))
                    if fc_horizon>4: fig_fc.add_vline(x=4,line_dash="dot",line_color="#adb5bd",annotation_text="CI widens past week 4",annotation_position="top right")
                    fig_fc.update_layout(title=f"Demand Forecast — {d['sku_name']} ({fc_horizon}W)",xaxis_title="Week",yaxis_title="Units / Store / Week",height=360,margin=dict(l=40,r=30,t=55,b=40))
                    st.plotly_chart(fig_fc, use_container_width=True)
                contribs = {k:v for k,v in d.get("variable_contributions_units_per_week",{}).items() if v!=0}
                if contribs:
                    cdf = pd.DataFrame(list(contribs.items()),columns=["Variable","Impact"]).sort_values("Impact")
                    fig_c = px.bar(cdf,x="Impact",y="Variable",orientation="h",color="Impact",
                        color_continuous_scale=["#e74c3c","#ecf0f1","#2ecc71"],color_continuous_midpoint=0,
                        title="Demand Variable Contributions (units/store/week)",height=max(300,len(contribs)*35))
                    st.plotly_chart(fig_c, use_container_width=True)
                st.info(d.get("accuracy_note",""))
                st.session_state.hist_forecast.append({
                    "header":f"{fc_sku}  |  {fc_horizon}W","meta":f"Accuracy: {acc*100:.1f}%  |  {'Reliable' if d.get('is_reliable') else 'UNRELIABLE'}",
                    "result":f"Point est: {d['weekly_forecast'][0]['point_estimate']:.0f} units/store/wk",
                    "ts":_dt.datetime.now().strftime("%H:%M")})


def _scenario_planner_content(with_right_panel: bool = False):
    import datetime as _dt
    st.subheader("Scenario Planner")
    st.caption("Compare up to 4 scenarios and detect conflicts.")

    if with_right_panel:
        sc_main, sc_hist = st.columns([2,1])
        with sc_hist:
            _render_history(st.session_state.hist_scenario, "Run a comparison to see history.")
            _render_formula_panel("scenario"); _render_data_sources("scenario")
    else:
        sc_main = st.container()

    with sc_main:
        sc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="sc_sku")
        sc_horizon = st.slider("Horizon (weeks)", 4, 32, 8, key="sc_horizon")
        base_price = mock_executor.PRODUCTS.get(sc_sku,{}).get("base_price",12.99)
        st.markdown("**Define Scenarios**")
        cols = st.columns(4)
        labels = ["Baseline","Scenario B","Scenario C","Scenario D"]
        defaults = [
            {"price":base_price,"promo":0.0,"sup_red":0.0,"tariff":0.0,"ddir":"neutral","sdir":"neutral"},
            {"price":base_price+1.5,"promo":0.0,"sup_red":0.0,"tariff":0.0,"ddir":"decrease","sdir":"neutral"},
            {"price":base_price-1.0,"promo":15.0,"sup_red":0.0,"tariff":0.0,"ddir":"increase","sdir":"neutral"},
            {"price":base_price,"promo":0.0,"sup_red":30.0,"tariff":5.0,"ddir":"decrease","sdir":"constrain"},
        ]
        scenarios, scenario_meta = [], []
        for i,(col,label,df) in enumerate(zip(cols,labels,defaults)):
            with col:
                st.markdown(f"**{label}**")
                price   = st.number_input("Price ($)",value=float(df["price"]),min_value=0.01,step=0.10,key=f"sc_p{i}")
                promo   = st.number_input("Promo %",value=float(df["promo"]),step=5.0,key=f"sc_pr{i}")
                sup_red = st.number_input("Supply cut %",value=float(df["sup_red"]),min_value=0.0,max_value=100.0,step=5.0,key=f"sc_sr{i}")
                tariff  = st.number_input("Tariff cost %",value=float(df["tariff"]),step=1.0,key=f"sc_ta{i}")
                sdate   = st.date_input("Start date",key=f"sc_sd{i}")
                scenarios.append({"name":label,"price":price,"promo_uplift_pct":promo,"supply_reduction_pct":sup_red,"tariff_additional_cost_pct":tariff})
                scenario_meta.append({"name":label,"scenario_type":"supply_disruption" if sup_red>0 else "promotion" if promo>0 else "price_change",
                    "start_date":sdate.isoformat(),"horizon_days":sc_horizon*7,"demand_direction":df["ddir"],"supply_direction":df["sdir"]})
        if st.button("Compare + Detect Conflicts", type="primary", key="sc_run"):
            with st.spinner("Running scenario comparison and conflict detection..."):
                comp_res = mock_executor.execute("run_scenario_comparison",{"sku":sc_sku,"scenarios":scenarios,"horizon_weeks":sc_horizon})
                conf_res = mock_executor.execute("detect_scenario_conflicts",{"scenarios":scenario_meta})
            for conflict in conf_res.get("data",{}).get("conflicts",[]):
                sev = conflict["severity"]
                msg = f"**{'CRITICAL' if sev=='critical' else 'WARNING'}:** '{conflict['scenario_a']}' + '{conflict['scenario_b']}' — {conflict['description']}\n\n**Rec:** {conflict['recommendation']}"
                (st.error if sev=="critical" else st.warning)(msg)
            if not conf_res.get("data",{}).get("conflicts"): st.success("No conflicts detected.")
            cd = comp_res.get("data",{}); rows = cd.get("comparison",[])
            if rows:
                sc_df = pd.DataFrame(rows)
                col_l,col_r = st.columns(2)
                with col_l:
                    st.plotly_chart(px.bar(sc_df,x="scenario",y="total_revenue_usd",color="scenario",title="Total Revenue by Scenario",height=300), use_container_width=True)
                with col_r:
                    st.plotly_chart(px.bar(sc_df,x="scenario",y="total_margin_usd",color="scenario",title="Total Margin by Scenario",height=300), use_container_width=True)
                st.dataframe(sc_df[["scenario","price","demand_per_store_week","weekly_revenue_usd","total_revenue_usd","total_margin_usd"]], use_container_width=True, hide_index=True)
                st.success(cd.get("recommendation",""))
                best = max(rows,key=lambda r:r["total_revenue_usd"])
                n_conflicts = len(conf_res.get("data",{}).get("conflicts",[]))
                st.session_state.hist_scenario.append({
                    "header":f"{sc_sku}  |  {sc_horizon}W  |  {len(rows)} scenarios","meta":f"Conflicts: {n_conflicts}",
                    "result":f"Best rev: {best['scenario']}  ${best['total_revenue_usd']:,.0f}","ts":_dt.datetime.now().strftime("%H:%M")})


def _shelf_replenishment_content(with_right_panel: bool = False):
    import datetime as _dt
    st.subheader("Shelf & Store Replenishment")
    st.caption("HQ → DC → Store with 3-4d lag, lead-time variability, perishable caps, and planogram constraints.")

    if with_right_panel:
        sh_main, sh_hist = st.columns([2,1])
        with sh_hist:
            _render_history(st.session_state.hist_shelf, "Run a replenishment analysis to see history.")
            _render_formula_panel("shelf"); _render_data_sources("shelf")
    else:
        sh_main = st.container()

    with sh_main:
        col1,col2,col3 = st.columns(3)
        with col1:
            sh_sku   = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="sh_sku")
            sh_store = st.selectbox("Store", [f"STR-{i:03d}" for i in range(1,31)], key="sh_store")
        with col2:
            sh_qty      = st.number_input("Proposed Replenishment Units", value=48, min_value=1, max_value=500, key="sh_qty")
            sh_priority = st.selectbox("Priority", ["standard","expedited","emergency"], key="sh_priority")
        with col3:
            sh_extra = st.number_input("Supply disruption extra delay (days)", value=0, min_value=0, key="sh_extra")
            st.markdown(""); run_sh = st.button("Analyze & Recommend", type="primary", key="sh_run")
        if run_sh:
            store_meta = mock_executor.STORES.get(sh_store, mock_executor.STORES["STR-001"]); dc_id = store_meta["dc"]
            prod_p = mock_executor.PRODUCTS.get(sh_sku,{})
            with st.spinner("Checking stockout risk, shelf capacity, perishable status..."):
                risk_res = mock_executor.execute("calculate_stockout_risk",{"sku":sh_sku,"location_id":sh_store,"supply_disruption_additional_days":sh_extra})
                cap_res  = mock_executor.execute("check_shelf_capacity",{"sku":sh_sku,"store_id":sh_store,"proposed_replenishment_units":sh_qty})
                per_res  = mock_executor.execute("check_perishable_status",{"sku":sh_sku,"location_id":sh_store,"proposed_replenishment_units":sh_qty}) if prod_p.get("perishable") else None
                rep_res  = mock_executor.execute("trigger_replenishment",{"sku":sh_sku,"from_location":dc_id,"to_location":sh_store,"quantity":sh_qty,"priority":sh_priority})
            rd = risk_res.get("data",{}); sev = rd.get("severity","ok")
            (st.error if sev=="critical" else st.warning if sev=="warning" else st.success)(rd.get("recommendation",""))
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Current Inventory",f"{rd.get('current_inventory_units','?')} units")
            m2.metric("Days on Hand",f"{rd.get('days_on_hand','?')}d")
            m3.metric("Effective Lag",f"{rd.get('effective_lag_days','?')}d")
            m4.metric("Buffer Above Lag",f"{rd.get('days_above_lag','?')}d",delta_color="inverse" if (rd.get("days_above_lag",1) or 1)<0 else "normal")
            chain = [{"Step":"HQ Signal Generated","Start":0,"Duration":1},{"Step":"DC Pick & Ship","Start":1,"Duration":2},{"Step":"Store Receive & Shelf","Start":3,"Duration":1}]
            if sh_extra>0: chain.append({"Step":f"Disruption Delay (+{sh_extra}d)","Start":4,"Duration":sh_extra})
            cost_mult = {"standard":1.0,"expedited":2.0,"emergency":3.0}[sh_priority]
            base_d = pd.Timestamp("2026-01-01")
            chain_df = pd.DataFrame([{"Task":c["Step"],"Start":base_d+pd.Timedelta(days=c["Start"]),"Finish":base_d+pd.Timedelta(days=c["Start"]+c["Duration"])} for c in chain])
            fig_g = px.timeline(chain_df,x_start="Start",x_end="Finish",y="Task",color="Task",title=f"Replenishment Chain — {sh_store} ({sh_priority})",height=280)
            fig_g.update_yaxes(autorange="reversed"); fig_g.update_layout(margin=dict(l=10,r=10,t=50,b=20))
            st.plotly_chart(fig_g, use_container_width=True)
            col_c,col_p = st.columns(2)
            cd = cap_res.get("data",{})
            with col_c:
                st.markdown("**Planogram Check**"); st.markdown(cd.get("recommendation",""))
                if cd.get("overflow_to_back_storage_units",0)>0: st.warning(f"{cd['overflow_to_back_storage_units']} units → back-of-store.")
            with col_p:
                if per_res:
                    pd_data = per_res.get("data",{}); st.markdown("**Perishable Cap Check**")
                    (st.error if pd_data.get("exceeds_perishable_cap") else st.success)(pd_data.get("recommendation",""))
                    if pd_data.get("exceeds_perishable_cap"): st.metric("Write-off Risk",f"${pd_data.get('write_off_risk_usd',0):,.2f}")
                else:
                    st.markdown("**Perishable Cap Check**"); st.info("Not applicable — non-perishable SKU.")
            rep_d = rep_res.get("data",{}); st.info(rep_d.get("confirmation",""))
            st.metric("Freight Cost",f"${rep_d.get('freight_cost_usd',0):,.2f}",delta=f"{cost_mult}× standard rate")
            st.session_state.hist_shelf.append({"header":f"{sh_sku}  →  {sh_store}  |  {sh_qty} units","meta":f"Priority: {sh_priority}  |  Severity: {sev}",
                "result":f"DoH: {rd.get('days_on_hand','?')}d  |  Freight: ${rep_d.get('freight_cost_usd',0):,.0f}","ts":_dt.datetime.now().strftime("%H:%M")})


def _financial_impact_content(with_right_panel: bool = False):
    import datetime as _dt
    st.subheader("Financial Impact Calculator")
    st.caption("Revenue, margin, vendor trade dollars, VMI split, carrying cost.")

    if with_right_panel:
        fi_main, fi_hist = st.columns([2,1])
        with fi_hist:
            _render_history(st.session_state.hist_finance, "Run a P&L calculation to see history.")
            _render_formula_panel("finance"); _render_data_sources("finance")
    else:
        fi_main = st.container()

    with fi_main:
        col1,col2 = st.columns(2)
        with col1:
            fi_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="fi_sku")
            fi_prod = mock_executor.PRODUCTS.get(fi_sku,{})
            fi_old_p = st.number_input("Old Price ($)",value=float(fi_prod.get("base_price",12.99)),min_value=0.01,step=0.10,key="fi_op")
            fi_new_p = st.number_input("New Price ($)",value=float(fi_prod.get("base_price",12.99))+1.50,min_value=0.01,step=0.10,key="fi_np")
        with col2:
            fi_old_v = st.number_input("Old Volume (units, total horizon)",value=12000.0,step=100.0,key="fi_ov")
            fi_new_v = st.number_input("New Volume (units, total horizon)",value=10056.0,step=100.0,key="fi_nv")
            fi_jx    = st.selectbox("Tax Jurisdiction",["US","US-WA","US-GA","US-IL"],key="fi_jx")
            fi_trade = st.checkbox("Include vendor trade dollars",value=True,key="fi_trade")
        if st.button("Calculate P&L Impact", type="primary", key="fi_run"):
            with st.spinner("Calculating..."):
                rev_res = mock_executor.execute("calculate_revenue_impact",{"sku":fi_sku,"old_price":fi_old_p,"new_price":fi_new_p,"old_volume_units":fi_old_v,"new_volume_units":fi_new_v,"include_trade_dollars":fi_trade,"jurisdiction":fi_jx})
                excess  = max(0.0,fi_old_v-fi_new_v)
                car_res = mock_executor.execute("calculate_carrying_cost",{"sku":fi_sku,"excess_units":excess,"carrying_weeks":4.0})
            rd = rev_res.get("data",{}); cd = car_res.get("data",{})
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Gross Revenue Δ",f"${rd.get('gross_revenue_change_usd',0):+,.0f}")
            k2.metric("Vendor Trade Offset",f"${rd.get('vendor_trade_offset_usd',0):+,.0f}")
            k3.metric("Net Revenue Δ",f"${rd.get('net_revenue_change_usd',0):+,.0f}")
            k4.metric("Margin Δ",f"${rd.get('margin_change_usd',0):+,.0f}")
            fig_pl = go.Figure(go.Waterfall(orientation="v",
                measure=["relative","relative","total","relative","relative","total"],
                x=["Gross Revenue","Trade Dollars","Net Revenue","Margin","Tax (est.)","Carrying Cost"],
                y=[rd.get("gross_revenue_change_usd",0),rd.get("vendor_trade_offset_usd",0),rd.get("net_revenue_change_usd",0),
                   rd.get("margin_change_usd",0),-rd.get("tax_on_margin_usd",0),-cd.get("total_carrying_cost_usd",0)],
                connector={"line":{"color":"#dee2e6"}},increasing={"marker":{"color":"#2ecc71"}},
                decreasing={"marker":{"color":"#e74c3c"}},totals={"marker":{"color":"#3498db"}},
                texttemplate="%{y:+,.0f}",textposition="outside"))
            fig_pl.update_layout(title="P&L Waterfall",height=380,margin=dict(l=40,r=40,t=55,b=40))
            st.plotly_chart(fig_pl, use_container_width=True)
            col_n1,col_n2 = st.columns(2)
            with col_n1: st.caption(f"Tax: {rd.get('tax_note','')}"); st.info(rd.get("trade_dollar_note",""))
            with col_n2:
                st.caption(rd.get("vmi_carrying_note",""))
                if excess>0: st.metric("Inventory Carrying Cost (4W)",f"${cd.get('total_carrying_cost_usd',0):,.2f}")
            st.session_state.hist_finance.append({"header":f"{fi_sku}  ${fi_old_p:.2f} → ${fi_new_p:.2f}",
                "meta":f"Vol: {fi_old_v:.0f} → {fi_new_v:.0f}","result":f"Net Rev Δ: ${rd.get('net_revenue_change_usd',0):+,.0f}  |  Margin Δ: ${rd.get('margin_change_usd',0):+,.0f}",
                "ts":_dt.datetime.now().strftime("%H:%M")})


def _data_sources_content():
    st.subheader("Data Sources & Provenance")
    st.caption("Know which data is real-time vs batch before making operational decisions.")
    sources = [
        {"System":"OLTP (Transactional DB)","Lag":"~5 min","Use For":"Pricing, PO creation, financial posting","Risk":"Low","Tools Used":"simulate_price_change, calculate_revenue_impact"},
        {"System":"WMS (Warehouse Mgmt)","Lag":"~15 min","Use For":"Operational inventory, stockout, replenishment","Risk":"Low-Medium","Tools Used":"get_inventory_levels, calculate_stockout_risk"},
        {"System":"OLAP (Analytics DW)","Lag":"24 hours","Use For":"Trend analysis, financial reporting","Risk":"HIGH for ops — use WMS instead","Tools Used":"get_demand_forecast, get_forecast_accuracy"},
        {"System":"Carrier API","Lag":"15–30 min","Use For":"Carrier status, alternate availability","Risk":"Medium — strikes may be delayed","Tools Used":"get_carrier_status, find_alternate_carriers"},
        {"System":"Competitor Feed","Lag":"4–6 hr","Use For":"Competitive pricing context","Risk":"Medium — treat as indicative","Tools Used":"get_competitive_pricing"},
    ]
    st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Session Freshness Warnings")
    warnings = st.session_state.get("freshness_warnings",[])
    if warnings:
        for w in set(warnings): st.warning(w, icon="⏱")
    else: st.info("No freshness warnings yet. Run a query in the Chat tab.")
    st.divider()
    st.subheader("WMS vs OLAP Inventory Check")
    disc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="disc_sku")
    rows = []
    for dc_id,dc in mock_executor.DCS.items():
        inv = dc["inventory"].get(disc_sku,{"wms":0,"olap":0}); diff = inv["wms"]-inv["olap"]
        diff_pct = round(diff/inv["olap"]*100,1) if inv["olap"]>0 else 0
        rows.append({"DC":dc_id,"Region":dc["region"],"Location":dc["name"].split("—")[1].strip(),
            "WMS (15min)":inv["wms"],"OLAP (24h)":inv["olap"],"Diff (units)":diff,"Diff %":diff_pct,
            "Recommendation":"Use WMS" if abs(diff_pct)>2 else "Either"})
    disc_df = pd.DataFrame(rows); st.dataframe(disc_df, use_container_width=True, hide_index=True)
    fig_d = px.bar(disc_df.melt(id_vars="DC",value_vars=["WMS (15min)","OLAP (24h)"],var_name="Source",value_name="Inventory"),
        x="DC",y="Inventory",color="Source",barmode="group",title=f"WMS vs OLAP — {disc_sku}",
        color_discrete_map={"WMS (15min)":"#2ecc71","OLAP (24h)":"#3498db"},height=300)
    st.plotly_chart(fig_d, use_container_width=True)


def _strategy_canvas_content():
    """Strategy Canvas (drag-and-drop scenario builder)."""
    st.subheader("Strategy Canvas")
    st.caption("Click **+ Add** on any scenario block. Adjust impact sliders live and run the analysis.")
    _SC_BLOCKS = {
        "PRICING":[
            {"id":"price-up","label":"Price Increase","rev":8.0,"mgn":5.0,"dem":-14.0,"risk":"Med"},
            {"id":"price-dn","label":"Price Cut","rev":-3.0,"mgn":-8.0,"dem":12.0,"risk":"Low"},
            {"id":"edlp","label":"EDLP Rollback","rev":1.0,"mgn":-3.0,"dem":8.0,"risk":"Low"},
            {"id":"markdown","label":"Markdown Clearance","rev":-5.0,"mgn":-10.0,"dem":22.0,"risk":"Med"},
            {"id":"price-match","label":"Competitor Match","rev":-2.0,"mgn":-5.0,"dem":6.0,"risk":"Med"},
        ],
        "PROMOTIONS":[
            {"id":"tpr","label":"Temp Price Reduction","rev":4.0,"mgn":-6.0,"dem":20.0,"risk":"Low"},
            {"id":"display","label":"Display / End Cap","rev":6.0,"mgn":-2.0,"dem":15.0,"risk":"Low"},
            {"id":"bogo","label":"BOGO Event","rev":3.0,"mgn":-12.0,"dem":30.0,"risk":"Low"},
            {"id":"coupon","label":"Digital Coupon Drop","rev":2.0,"mgn":-8.0,"dem":18.0,"risk":"Low"},
            {"id":"promo","label":"Shopper Marketing","rev":5.0,"mgn":-3.0,"dem":12.0,"risk":"Low"},
        ],
        "SUPPLY CHAIN":[
            {"id":"strike","label":"Carrier Strike","rev":-12.0,"mgn":-9.0,"dem":0.0,"risk":"Crit"},
            {"id":"port-delay","label":"Port Delay","rev":-7.0,"mgn":-6.0,"dem":0.0,"risk":"High"},
            {"id":"dc-closure","label":"DC Closure","rev":-10.0,"mgn":-8.0,"dem":0.0,"risk":"Crit"},
            {"id":"supplier-bk","label":"Supplier Bankruptcy","rev":-15.0,"mgn":-12.0,"dem":0.0,"risk":"Crit"},
            {"id":"prestock","label":"Pre-Build Inventory","rev":3.0,"mgn":-2.0,"dem":5.0,"risk":"Low"},
            {"id":"alt-carrier","label":"Alt Carrier Switch","rev":-2.0,"mgn":-3.0,"dem":0.0,"risk":"Med"},
        ],
        "DEMAND SIGNALS":[
            {"id":"surge","label":"Demand Surge","rev":15.0,"mgn":10.0,"dem":20.0,"risk":"Low"},
            {"id":"drop","label":"Demand Drop","rev":-10.0,"mgn":-7.0,"dem":-15.0,"risk":"High"},
            {"id":"seasonal-pk","label":"Seasonal Peak","rev":12.0,"mgn":8.0,"dem":18.0,"risk":"Med"},
            {"id":"weather-ev","label":"Weather Event","rev":-6.0,"mgn":-5.0,"dem":-10.0,"risk":"High"},
            {"id":"new-item","label":"New Item Launch","rev":20.0,"mgn":5.0,"dem":25.0,"risk":"Med"},
            {"id":"comp-oos","label":"Competitor OOS","rev":8.0,"mgn":6.0,"dem":10.0,"risk":"Low"},
        ],
        "TRADE & EXTERNAL":[
            {"id":"tariff","label":"Tariff Increase","rev":-4.0,"mgn":-6.0,"dem":-8.0,"risk":"High"},
            {"id":"fx-impact","label":"FX / Currency Move","rev":-3.0,"mgn":-4.0,"dem":-5.0,"risk":"Med"},
            {"id":"regulatory","label":"Regulatory Change","rev":-5.0,"mgn":-4.0,"dem":0.0,"risk":"High"},
            {"id":"comp-move","label":"Competitor Promo","rev":-3.0,"mgn":-3.0,"dem":-8.0,"risk":"Med"},
            {"id":"inflation","label":"Input Cost Inflation","rev":-2.0,"mgn":-8.0,"dem":-4.0,"risk":"High"},
        ],
        "OPERATIONS":[
            {"id":"safety-stk","label":"Safety Stock Raise","rev":-1.0,"mgn":-2.0,"dem":3.0,"risk":"Med"},
            {"id":"vmi-switch","label":"VMI Transition","rev":2.0,"mgn":3.0,"dem":1.0,"risk":"Med"},
            {"id":"dc-optim","label":"DC Optimization","rev":3.0,"mgn":4.0,"dem":2.0,"risk":"Low"},
            {"id":"planogram","label":"Planogram Reset","rev":4.0,"mgn":2.0,"dem":5.0,"risk":"Med"},
            {"id":"trade-mgt","label":"Trade Spend Realloc","rev":3.0,"mgn":-1.0,"dem":8.0,"risk":"Med"},
        ],
    }
    _SC_CONFLICTS = [("tpr","strike"),("bogo","port-delay"),("bogo","strike"),("display","strike"),
                     ("price-up","price-dn"),("edlp","price-up"),("promo","supplier-bk"),("surge","dc-closure"),
                     ("seasonal-pk","strike"),("new-item","dc-closure")]
    _CAT_COLORS = {"PRICING":"#388bfd","PROMOTIONS":"#3fb950","SUPPLY CHAIN":"#f85149",
                   "DEMAND SIGNALS":"#3fb950","TRADE & EXTERNAL":"#d29922","OPERATIONS":"#58a6ff"}
    if "canvas_blocks" not in st.session_state: st.session_state.canvas_blocks = []

    left_col, right_col = st.columns([2,3])
    with left_col:
        st.markdown("### Scenario Palette")
        base_rev = st.number_input("Base Annual Revenue ($M)",min_value=1.0,max_value=9999.0,value=100.0,step=10.0,key="sc_base_rev")
        for cat,blocks in _SC_BLOCKS.items():
            cat_color = _CAT_COLORS.get(cat,"#8b949e")
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{cat_color};text-transform:uppercase;letter-spacing:.08em;margin-top:12px;border-top:1px solid {cat_color}44;padding-top:6px;">{cat}</div>',unsafe_allow_html=True)
            for blk in blocks:
                blk_id = blk["id"]; already = any(b["id"]==blk_id for b in st.session_state.canvas_blocks)
                rc = {"Crit":"#f85149","High":"#d29922","Med":"#388bfd","Low":"#3fb950"}.get(blk["risk"],"#8b949e")
                col_a,col_b = st.columns([3,1])
                with col_a:
                    st.markdown(f'<div style="font-size:12px;font-weight:600;color:#cdd9e5;">{blk["label"]} <span style="font-size:10px;color:{rc}">({blk["risk"]})</span></div>'
                                f'<div style="font-size:10px;color:#8b949e;">Rev {blk["rev"]:+.0f}% | Mgn {blk["mgn"]:+.0f}% | Dem {blk["dem"]:+.0f}%</div>',unsafe_allow_html=True)
                with col_b:
                    if already:
                        if st.button("✕ Remove",key=f"rm_{blk_id}"):
                            st.session_state.canvas_blocks=[b for b in st.session_state.canvas_blocks if b["id"]!=blk_id]; st.rerun()
                    else:
                        if st.button("+ Add",key=f"add_{blk_id}"):
                            st.session_state.canvas_blocks.append({"id":blk["id"],"label":blk["label"],"rev":blk["rev"],"mgn":blk["mgn"],"dem":blk["dem"],"risk":blk["risk"]}); st.rerun()

    with right_col:
        st.markdown("### Active Canvas")
        blocks = st.session_state.canvas_blocks
        if not blocks: st.info("No blocks added yet. Click **+ Add** on any scenario in the palette.")
        else:
            for i,blk in enumerate(blocks):
                with st.expander(f"**{blk['label']}**",expanded=True):
                    c1,c2,c3,c4 = st.columns([3,3,3,1])
                    with c1:
                        nrev=st.slider("Rev %",-30.0,30.0,float(blk["rev"]),0.5,key=f"cv_rev_{blk['id']}_{i}")
                        st.session_state.canvas_blocks[i]["rev"]=nrev
                    with c2:
                        nmgn=st.slider("Mgn %",-20.0,20.0,float(blk["mgn"]),0.5,key=f"cv_mgn_{blk['id']}_{i}")
                        st.session_state.canvas_blocks[i]["mgn"]=nmgn
                    with c3:
                        ndem=st.slider("Dem %",-40.0,40.0,float(blk["dem"]),0.5,key=f"cv_dem_{blk['id']}_{i}")
                        st.session_state.canvas_blocks[i]["dem"]=ndem
                    with c4:
                        if st.button("✕",key=f"del_{blk['id']}_{i}"):
                            st.session_state.canvas_blocks.pop(i); st.rerun()
            st.markdown("---")
            if st.button("▶ Run Analysis",type="primary",use_container_width=True,key="cv_run"):
                base=base_rev*1_000_000; ids=[b["id"] for b in blocks]; n=len(blocks)
                tot_rev=sum(b["rev"] for b in blocks); tot_mgn=sum(b["mgn"] for b in blocks); tot_dem=sum(b["dem"] for b in blocks)
                penalty=max(0,(n-2)*2.0); adj_rev=tot_rev-penalty; adj_mgn=tot_mgn-penalty
                def fmt_d(v):
                    s="+" if v>=0 else "-"; a=abs(v)
                    if a>=1e9: return f"{s}${a/1e9:.1f}B"
                    if a>=1e6: return f"{s}${a/1e6:.1f}M"
                    if a>=1e3: return f"{s}${a/1e3:.0f}K"
                    return f"{s}${a:.0f}"
                def fmt_p(v): return f"+{v:.1f}%" if v>=0 else f"{v:.1f}%"
                conflicts=[]
                for c1_id,c2_id in _SC_CONFLICTS:
                    if c1_id in ids and c2_id in ids:
                        conflicts.append(f"{next((b['label'] for b in blocks if b['id']==c1_id),c1_id)} ↔ {next((b['label'] for b in blocks if b['id']==c2_id),c2_id)}")
                k1,k2,k3,k4=st.columns(4)
                k1.metric("Revenue Impact",fmt_p(adj_rev),fmt_d(base*adj_rev/100))
                k2.metric("Margin Impact",fmt_p(adj_mgn),fmt_d(base*adj_mgn/100))
                k3.metric("Demand Shift",fmt_p(tot_dem)); k4.metric("Blocks Active",str(n))
                rows2=[]
                for b in blocks:
                    rows2.append({"Scenario":b["label"],"Rev %":fmt_p(b["rev"]),"Rev $":fmt_d(base*b["rev"]/100),"Mgn %":fmt_p(b["mgn"]),"Mgn $":fmt_d(base*b["mgn"]/100),"Demand %":fmt_p(b["dem"]),"Risk":b["risk"]})
                if penalty>0: rows2.append({"Scenario":f"⚠ Penalty ({n-2} extra)","Rev %":fmt_p(-penalty),"Rev $":fmt_d(-base*penalty/100),"Mgn %":fmt_p(-penalty),"Mgn $":fmt_d(-base*penalty/100),"Demand %":"—","Risk":"—"})
                rows2.append({"Scenario":"NET IMPACT","Rev %":fmt_p(adj_rev),"Rev $":fmt_d(base*adj_rev/100),"Mgn %":fmt_p(adj_mgn),"Mgn $":fmt_d(base*adj_mgn/100),"Demand %":fmt_p(tot_dem),"Risk":"—"})
                st.dataframe(pd.DataFrame(rows2),use_container_width=True,hide_index=True)
                if conflicts:
                    for cf in conflicts: st.error(f"⚡ Conflict: **{cf}**")
                else: st.success("✓ No conflicts detected.")
                recs=[]
                if conflicts: recs.append("🔴 Resolve conflicts before activating simultaneously.")
                if "price-up" in ids and "prestock" in ids: recs.append("✅ Execute Pre-Build Inventory before Price Increase.")
                if any(x in ids for x in ["tpr","display","bogo"]): recs.append("📦 Align promotions with DC replenishment cycle.")
                if any(x in ids for x in ["strike","port-delay","dc-closure"]): recs.append("🚚 Activate alternate carriers within 48h.")
                if adj_rev<0 and adj_mgn<0: recs.append("⚠️ Net negative — consider removing high-risk blocks.")
                if adj_rev>10: recs.append(f"📈 Strong upside ({fmt_p(adj_rev)}) — ensure DC stock covers demand uplift.")
                if not recs: recs.append("✔ Chain balanced. Review timing with merchant and supply teams.")
                st.markdown("**Recommendations**")
                for r in recs: st.markdown(f"- {r}")
            if blocks:
                if st.button("🗑 Clear Canvas",key="cv_clear"):
                    st.session_state.canvas_blocks=[]; st.rerun()


def _flow_map_content():
    """v2-style basic LangGraph Flow Map."""
    st.subheader("LangGraph Flow Map")
    st.caption("Visual DAG of the V2 LangGraph pipeline. Green nodes = executed in last Chat query.")

    fm_main, fm_hist = st.columns([3,1])

    with fm_hist:
        st.markdown('<div class="hist-panel"><h4>📋 Execution Log</h4>', unsafe_allow_html=True)
        chat_msgs_fm = [m for m in st.session_state.conversation_history if isinstance(m.get("content"),str)]
        user_msgs = [m["content"] for m in chat_msgs_fm if m["role"]=="user"]
        all_entries = [("Chat",q[:70]+("…" if len(q)>70 else "")) for q in reversed(user_msgs[-8:])]
        all_entries += [("Workflow",w.get("header","")) for w in list(reversed(st.session_state.hist_workflow))[:5]]
        if not all_entries:
            st.markdown('<div class="hist-empty">Run queries in Chat or Workflow tabs.</div>', unsafe_allow_html=True)
        else:
            for source,label in all_entries[:8]:
                bc="#0071ce" if source=="Chat" else "#28a745"
                st.markdown(f'<div class="hist-item"><div class="hist-item-header" style="display:flex;gap:6px;align-items:center;"><span style="background:{bc};color:white;font-size:10px;padding:1px 6px;border-radius:4px;">{source[:2]}</span>{label}</div></div>',unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with fm_main:
        NODE_POSITIONS = {
            "router":(5.0,9.5),"price_cascade":(1.0,7.5),"supply_disruption":(3.0,7.5),
            "demand_forecast":(5.0,7.5),"scenario_planning":(7.0,7.5),"shelf_replenishment":(9.0,7.5),
            "inventory_node":(1.0,5.5),"carrier_node":(3.0,5.5),"accuracy_node":(5.0,5.5),
            "perishable_check":(9.0,5.5),"financial_impact":(2.0,3.5),"synthesizer":(5.0,1.5),
        }
        NODE_LABELS = {
            "router":"ROUTER\n(intent+entity)","price_cascade":"Price\nCascade","supply_disruption":"Supply\nDisruption",
            "demand_forecast":"Demand\nForecast","scenario_planning":"Scenario\nPlanning","shelf_replenishment":"Shelf\nReplenishment",
            "inventory_node":"Inventory\nNode","carrier_node":"Carrier\nNode","accuracy_node":"Accuracy\nGate",
            "perishable_check":"Perishable\nCheck","financial_impact":"Financial\nImpact","synthesizer":"SYNTHESIZER\n(final response)",
        }
        NODE_TOOLS = {
            "router":"—","price_cascade":"simulate_price_change\nget_competitive_pricing\nadjust_promotional_price",
            "supply_disruption":"get_carrier_status\nget_dc_inventory","demand_forecast":"get_demand_forecast\nanalyze_demand_variables",
            "scenario_planning":"detect_scenario_conflicts\ncompare_scenarios","shelf_replenishment":"get_replenishment_schedule\ncalculate_stockout_risk",
            "inventory_node":"get_inventory_levels\nget_reorder_recommendations","carrier_node":"find_alternate_carriers\ncalculate_revenue_impact",
            "accuracy_node":"get_forecast_accuracy","perishable_check":"get_inventory_levels\ncalculate_stockout_risk",
            "financial_impact":"calculate_revenue_impact\ncalculate_carrying_cost","synthesizer":"—",
        }
        EDGES = [("router","price_cascade"),("router","supply_disruption"),("router","demand_forecast"),("router","scenario_planning"),("router","shelf_replenishment"),
                 ("price_cascade","inventory_node"),("supply_disruption","carrier_node"),("demand_forecast","accuracy_node"),("shelf_replenishment","perishable_check"),
                 ("inventory_node","financial_impact"),("carrier_node","synthesizer"),("accuracy_node","synthesizer"),("perishable_check","synthesizer"),
                 ("financial_impact","synthesizer"),("scenario_planning","synthesizer"),("price_cascade","financial_impact")]
        _TOOL_TO_NODE_FM = {
            "simulate_price_change":"price_cascade","adjust_promotional_price":"price_cascade","get_competitive_pricing":"price_cascade",
            "get_carrier_status":"supply_disruption","find_alternate_carriers":"carrier_node","get_demand_forecast":"demand_forecast",
            "get_forecast_accuracy":"accuracy_node","detect_scenario_conflicts":"scenario_planning","run_scenario_comparison":"scenario_planning",
            "get_inventory_levels":"inventory_node","calculate_stockout_risk":"inventory_node","calculate_revenue_impact":"financial_impact","calculate_carrying_cost":"financial_impact",
        }
        executed_nodes = set()
        last_tool_calls_fm = st.session_state.get("last_tool_calls",[])
        if last_tool_calls_fm:
            executed_nodes.update(["router","synthesizer"])
            for tc in last_tool_calls_fm:
                nd = _TOOL_TO_NODE_FM.get(tc.get("tool",""))
                if nd:
                    executed_nodes.add(nd)
                    if nd=="price_cascade": executed_nodes.update(["inventory_node","financial_impact"])
                    elif nd=="supply_disruption": executed_nodes.add("carrier_node")
                    elif nd=="demand_forecast": executed_nodes.add("accuracy_node")
                    elif nd=="shelf_replenishment": executed_nodes.add("perishable_check")

        fig_flow = go.Figure()
        for src,dst in EDGES:
            x0,y0=NODE_POSITIONS[src]; x1,y1=NODE_POSITIONS[dst]
            both_exec = src in executed_nodes and dst in executed_nodes
            fig_flow.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref="x",yref="y",axref="x",ayref="y",
                showarrow=True,arrowhead=3,arrowsize=1.2,arrowwidth=2.5 if both_exec else 1.5,
                arrowcolor="#27ae60" if both_exec else "#dee2e6")
        for node,(x,y) in NODE_POSITIONS.items():
            is_exec = node in executed_nodes
            fig_flow.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",
                marker=dict(size=60,color="#27ae60" if is_exec else "#bdc3c7",symbol="square",
                            line=dict(color="#2c3e50" if is_exec else "#adb5bd",width=2)),
                text=[NODE_LABELS[node]],textposition="middle center",
                textfont=dict(size=10,color="white" if is_exec else "#2c3e50",family="monospace"),
                name=node,hovertemplate=f"<b>{node}</b><br>{'✅ Executed' if is_exec else '⬜ Available'}<br>Tools:<br>{NODE_TOOLS[node].replace(chr(10),'<br>')}<extra></extra>",
                showlegend=False))
        fig_flow.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(size=12,color="#27ae60",symbol="square"),name="Executed",showlegend=True))
        fig_flow.add_trace(go.Scatter(x=[None],y=[None],mode="markers",marker=dict(size=12,color="#bdc3c7",symbol="square"),name="Not invoked",showlegend=True))
        fig_flow.update_layout(title="LangGraph Multi-Agent Graph — V2 Pipeline",
            xaxis=dict(range=[-0.5,10.5],showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(range=[0.5,11.0],showgrid=False,zeroline=False,showticklabels=False),
            height=600,plot_bgcolor="white",paper_bgcolor="#f8f9fa",
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
            margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig_flow, use_container_width=True)

        if executed_nodes:
            st.success(f"**Last query executed {len(executed_nodes)} nodes:** " + ", ".join(f"`{n}`" for n in sorted(executed_nodes)))
        else:
            st.info("No query has been run yet. Run a question in **Chat** tab, then return here.")

        st.divider()
        st.markdown("### How the Graph Works")
        a1,a2 = st.columns(2)
        with a1:
            st.markdown("""
**Entry Point: Router** — classifies intent, extracts SKU/region, sets route.

**Domain Nodes** — `price_cascade` · `supply_disruption` · `demand_forecast` · `scenario_planning` · `shelf_replenishment`

**Supporting Nodes** — `inventory_node` · `carrier_node` · `accuracy_node` · `perishable_check` · `financial_impact`
            """)
        with a2:
            st.markdown("""
**Exit: Synthesizer** — merges all node outputs into one final response.

**Execution Paths:**
| Intent | Path |
|--------|------|
| Price change | router → price_cascade → inventory → financial → synthesizer |
| Supply disruption | router → supply_disruption → carrier → synthesizer |
| Demand forecast | router → demand_forecast → accuracy → synthesizer |
| Scenario planning | router → scenario_planning → synthesizer |
            """)
        st.divider()
        st.markdown("### Node Tool Assignments")
        tool_rows = [{"Node":node,"Status":"Executed" if node in executed_nodes else "Available",
                      "Tools":NODE_TOOLS[node].replace("\n",", ")} for node in NODE_TOOLS]
        st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# VERSION RENDERING  (build: 2026-04-13.e)
# ═══════════════════════════════════════════════════════════════════════════

# ── v1: Core (Dashboard + Chat) ──────────────────────────────────────────────
if "v1" in (app_version or ""):
    tab_dash_v1, tab_chat_v1 = st.tabs(["Dashboard", "Chat"])
    with tab_dash_v1:
        try: _dashboard_content()
        except Exception as _e: st.exception(_e)
    with tab_chat_v1:
        try: _chat_content(full_width=True)
        except Exception as _e: st.exception(_e)

# ── v2: Full (15 tabs) ────────────────────────────────────────────────────────
elif "v2" in (app_version or ""):
    (tab_guide, tab_dash, tab_chat, tab_price, tab_supply, tab_forecast,
     tab_scenario, tab_shelf, tab_finance, tab_data,
     tab_workflow, tab_flowmap, tab_netgraph, tab_builder, tab_canvas) = st.tabs([
        "⭐ Guide", "Dashboard", "Chat",
        "Price Cascade", "Supply Alert", "Demand Forecast",
        "Scenario Planner", "Shelf & Store", "Financial Impact", "Data Sources",
        "Workflow", "Flow Map", "🕸 Network Graph", "🧩 Scenario Builder", "🎯 Strategy Canvas",
    ])

    with tab_guide:
        try: _guide_content(compact=False)
        except Exception as _e: st.exception(_e)

    with tab_dash:
        try: _dashboard_content()
        except Exception as _e: st.exception(_e)

    with tab_chat:
        try: _chat_content(full_width=False)
        except Exception as _e: st.exception(_e)

    with tab_price:
        try: _price_cascade_content(with_right_panel=True)
        except Exception as _e: st.exception(_e)

    with tab_supply:
        try: _supply_alert_content(with_right_panel=True)
        except Exception as _e: st.exception(_e)

    with tab_forecast:
        try: _demand_forecast_content(with_right_panel=True)
        except Exception as _e: st.exception(_e)

    with tab_scenario:
        try: _scenario_planner_content(with_right_panel=True)
        except Exception as _e: st.exception(_e)

    with tab_shelf:
        try: _shelf_replenishment_content(with_right_panel=True)
        except Exception as _e: st.exception(_e)

    with tab_finance:
        try: _financial_impact_content(with_right_panel=True)
        except Exception as _e: st.exception(_e)

    with tab_data:
        try: _data_sources_content()
        except Exception as _e: st.exception(_e)

    with tab_workflow:
        try:
            wf_main2, wf_hist2 = st.columns([2,1])
            with wf_hist2:
                _render_history(st.session_state.hist_workflow, "Run a workflow analysis to see history.")
            with wf_main2:
                _workflow_content()
        except Exception as _e: st.exception(_e)

    with tab_flowmap:
        try: _flow_map_content()
        except Exception as _e: st.exception(_e)

    with tab_netgraph:
        try: _network_graph_content()
        except Exception as _e: st.exception(_e)

    with tab_builder:
        st.subheader("🧩 Scenario Builder — Palette + Canvas")
        st.caption("Quick scenario builder. Click **＋** to place blocks on the canvas, adjust sliders, and run analysis.")
        _BLK_DEF = {
            "PRICING":[("price-up","Price Increase",8.0,5.0,-14.0,"Med","#388bfd"),("price-dn","Price Cut",-3.0,-8.0,12.0,"Low","#388bfd"),
                       ("edlp","EDLP Rollback",1.0,-3.0,8.0,"Low","#388bfd"),("markdown","Markdown Clearance",-5.0,-10.0,22.0,"Med","#388bfd"),("price-match","Competitor Match",-2.0,-5.0,6.0,"Med","#388bfd")],
            "PROMOTIONS":[("tpr","Temp Price Reduction",4.0,-6.0,20.0,"Low","#3fb950"),("display","Display / End Cap",6.0,-2.0,15.0,"Low","#3fb950"),
                          ("bogo","BOGO Event",3.0,-12.0,30.0,"Low","#3fb950"),("coupon","Digital Coupon Drop",2.0,-8.0,18.0,"Low","#3fb950"),("promo","Shopper Marketing",5.0,-3.0,12.0,"Low","#3fb950")],
            "SUPPLY CHAIN":[("strike","Carrier Strike",-12.0,-9.0,0.0,"Crit","#f85149"),("port-delay","Port Delay",-7.0,-6.0,0.0,"High","#f85149"),
                            ("dc-closure","DC Closure",-10.0,-8.0,0.0,"Crit","#f85149"),("supplier-bk","Supplier Bankruptcy",-15.0,-12.0,0.0,"Crit","#f85149"),
                            ("prestock","Pre-Build Inventory",3.0,-2.0,5.0,"Low","#f85149"),("alt-carrier","Alt Carrier Switch",-2.0,-3.0,0.0,"Med","#f85149")],
            "DEMAND SIGNALS":[("surge","Demand Surge",15.0,10.0,20.0,"Low","#2ea043"),("demand-drop","Demand Drop",-10.0,-7.0,-15.0,"High","#2ea043"),
                              ("seasonal-pk","Seasonal Peak",12.0,8.0,18.0,"Med","#2ea043"),("weather-ev","Weather Event",-6.0,-5.0,-10.0,"High","#2ea043"),
                              ("new-item","New Item Launch",20.0,5.0,25.0,"Med","#2ea043"),("comp-oos","Competitor OOS",8.0,6.0,10.0,"Low","#2ea043")],
            "TRADE & EXTERNAL":[("tariff","Tariff Increase",-4.0,-6.0,-8.0,"High","#d29922"),("fx-impact","FX / Currency Move",-3.0,-4.0,-5.0,"Med","#d29922"),
                                ("regulatory","Regulatory Change",-5.0,-4.0,0.0,"High","#d29922"),("comp-move","Competitor Promo",-3.0,-3.0,-8.0,"Med","#d29922"),("inflation","Input Cost Inflation",-2.0,-8.0,-4.0,"High","#d29922")],
            "OPERATIONS":[("safety-stk","Safety Stock Raise",-1.0,-2.0,3.0,"Med","#58a6ff"),("vmi-switch","VMI Transition",2.0,3.0,1.0,"Med","#58a6ff"),
                          ("dc-optim","DC Optimization",3.0,4.0,2.0,"Low","#58a6ff"),("planogram","Planogram Reset",4.0,2.0,5.0,"Med","#58a6ff"),("trade-mgt","Trade Spend Realloc",3.0,-1.0,8.0,"Med","#58a6ff")],
        }
        _CONF_PAIRS = [("tpr","strike"),("bogo","port-delay"),("bogo","strike"),("display","strike"),("price-up","price-dn"),
                       ("edlp","price-up"),("promo","supplier-bk"),("surge","dc-closure"),("seasonal-pk","strike"),("new-item","dc-closure")]
        _RISK_COLOR = {"Crit":"#f85149","High":"#d29922","Med":"#388bfd","Low":"#3fb950"}
        if "builder_blocks" not in st.session_state: st.session_state.builder_blocks = []

        pal_col, canvas_col, kpi_col = st.columns([2,4,3])
        with pal_col:
            st.markdown("#### 📦 Scenario Palette")
            base_rev_b = st.number_input("Base Revenue ($M)",1.0,9999.0,100.0,10.0,key="bldr_base")
            for cat,blocks in _BLK_DEF.items():
                cat_color = blocks[0][6]
                st.markdown(f'<div style="font-size:10px;font-weight:700;color:{cat_color};letter-spacing:.08em;text-transform:uppercase;margin-top:10px;padding:3px 0;border-bottom:1px solid {cat_color}44;">{cat}</div>',unsafe_allow_html=True)
                for bid,bname,brev,bmgn,bdem,brisk,bclr in blocks:
                    already = any(b["id"]==bid for b in st.session_state.builder_blocks)
                    rc = _RISK_COLOR.get(brisk,"#8b949e")
                    ca,cb = st.columns([3,1])
                    with ca:
                        st.markdown(f'<div style="font-size:11px;font-weight:600;color:#cdd9e5;">{bname} <span style="font-size:9px;color:{rc};font-weight:700;">({brisk})</span></div><div style="font-size:9px;color:#8b949e;">Rev {brev:+.0f}% | Mgn {bmgn:+.0f}% | Dem {bdem:+.0f}%</div>',unsafe_allow_html=True)
                    with cb:
                        if already:
                            if st.button("✕",key=f"brm_{bid}"):
                                st.session_state.builder_blocks=[b for b in st.session_state.builder_blocks if b["id"]!=bid]; st.rerun()
                        else:
                            if st.button("＋",key=f"bad_{bid}"):
                                st.session_state.builder_blocks.append({"id":bid,"label":bname,"rev":brev,"mgn":bmgn,"dem":bdem,"risk":brisk}); st.rerun()

        with canvas_col:
            st.markdown("#### 🎯 Decision Canvas")
            blocks = st.session_state.builder_blocks
            if not blocks: st.info("👈 Click **＋** on any scenario block to add it here.")
            else:
                for i,blk in enumerate(blocks):
                    clr = _RISK_COLOR.get(blk["risk"],"#8b949e")
                    with st.container():
                        st.markdown(f'<div style="border-left:3px solid {clr};padding-left:8px;margin-bottom:2px;"><span style="font-size:13px;font-weight:700;color:#cdd9e5;">{blk["label"]}</span> <span style="font-size:10px;color:{clr};">({blk["risk"]})</span></div>',unsafe_allow_html=True)
                        c1,c2,c3,c4=st.columns([3,3,3,1])
                        with c1:
                            nrev=st.slider("Rev %",-30.0,30.0,float(blk["rev"]),0.5,key=f"br_{blk['id']}_{i}")
                            st.session_state.builder_blocks[i]["rev"]=nrev
                        with c2:
                            nmgn=st.slider("Mgn %",-20.0,20.0,float(blk["mgn"]),0.5,key=f"bm_{blk['id']}_{i}")
                            st.session_state.builder_blocks[i]["mgn"]=nmgn
                        with c3:
                            ndem=st.slider("Dem %",-40.0,40.0,float(blk["dem"]),0.5,key=f"bd_{blk['id']}_{i}")
                            st.session_state.builder_blocks[i]["dem"]=ndem
                        with c4:
                            st.write("")
                            if st.button("✕",key=f"bx_{blk['id']}_{i}"):
                                st.session_state.builder_blocks.pop(i); st.rerun()
                        st.divider()
                if st.button("🗑 Clear Canvas",key="bldr_clear"):
                    st.session_state.builder_blocks=[]; st.rerun()

        with kpi_col:
            st.markdown("#### 📊 Live Impact")
            blocks = st.session_state.builder_blocks; base=base_rev_b*1_000_000; n=len(blocks)
            def _fp(v): return f"+{v:.1f}%" if v>=0 else f"{v:.1f}%"
            def _fd(v):
                s="+" if v>=0 else "-"; a=abs(v)
                if a>=1e9: return f"{s}${a/1e9:.1f}B"
                if a>=1e6: return f"{s}${a/1e6:.1f}M"
                if a>=1e3: return f"{s}${a/1e3:.0f}K"
                return f"{s}${a:.0f}"
            if n==0: st.info("Add blocks to see live KPIs")
            else:
                tot_r=sum(b["rev"] for b in blocks); tot_m=sum(b["mgn"] for b in blocks); tot_d=sum(b["dem"] for b in blocks)
                pen=max(0,(n-2)*2.0); adj_r=tot_r-pen; adj_m=tot_m-pen
                ids=[b["id"] for b in blocks]
                conflicts_b=[(a,b2) for a,b2 in _CONF_PAIRS if a in ids and b2 in ids]
                k1,k2=st.columns(2)
                k1.metric("Revenue",_fp(adj_r),_fd(base*adj_r/100)); k2.metric("Margin",_fp(adj_m),_fd(base*adj_m/100))
                k3,k4=st.columns(2); k3.metric("Demand",_fp(tot_d)); k4.metric("Blocks",str(n))
                if conflicts_b:
                    for a,b2 in conflicts_b:
                        an=next((x["label"] for x in blocks if x["id"]==a),a); bn=next((x["label"] for x in blocks if x["id"]==b2),b2)
                        st.error(f"⚡ Conflict: **{an}** ↔ **{bn}**")
                else: st.success("✓ No conflicts")
                if pen>0: st.warning(f"⚠ Compound penalty: −{pen:.0f}%")
                st.markdown("---")
                if st.button("▶ Run Analysis",type="primary",use_container_width=True,key="bldr_run"):
                    st.markdown("**Block Breakdown**")
                    rows_b=[]
                    for b in blocks: rows_b.append({"Scenario":b["label"],"Rev %":_fp(b["rev"]),"Rev $":_fd(base*b["rev"]/100),"Mgn %":_fp(b["mgn"]),"Mgn $":_fd(base*b["mgn"]/100),"Demand %":_fp(b["dem"]),"Risk":b["risk"]})
                    if pen>0: rows_b.append({"Scenario":f"⚠ Penalty ({n-2} extra)","Rev %":_fp(-pen),"Rev $":_fd(-base*pen/100),"Mgn %":_fp(-pen),"Mgn $":_fd(-base*pen/100),"Demand %":"—","Risk":"—"})
                    rows_b.append({"Scenario":"🔵 NET IMPACT","Rev %":_fp(adj_r),"Rev $":_fd(base*adj_r/100),"Mgn %":_fp(adj_m),"Mgn $":_fd(base*adj_m/100),"Demand %":_fp(tot_d),"Risk":"—"})
                    st.dataframe(pd.DataFrame(rows_b),use_container_width=True,hide_index=True)
                    recs_b=[]
                    if conflicts_b: recs_b.append("🔴 Resolve conflicts before activating both blocks.")
                    if "price-up" in ids and "prestock" in ids: recs_b.append("✅ Pre-Build Inventory before Price Increase.")
                    if any(x in ids for x in ["tpr","display","bogo"]): recs_b.append("📦 Align promotions with DC replenishment.")
                    if any(x in ids for x in ["strike","port-delay","dc-closure"]): recs_b.append("🚚 Activate alternate carriers within 48h.")
                    if adj_r<0 and adj_m<0: recs_b.append("⚠️ Net negative — consider removing high-risk blocks.")
                    if adj_r>10: recs_b.append(f"📈 Strong upside ({_fp(adj_r)}) — confirm DC stock covers demand uplift.")
                    if not recs_b: recs_b.append("✔ Chain balanced.")
                    st.markdown("**Recommendations**")
                    for r in recs_b: st.markdown(f"- {r}")

    with tab_canvas:
        _strategy_canvas_content()

# ── v3: Simplified (7 tabs) ───────────────────────────────────────────────────
else:
    (tab_guide, tab_dash, tab_chat,
     tab_simulate, tab_scenarios, tab_ops, tab_arch) = st.tabs([
        "⭐ Guide", "Dashboard", "Chat",
        "Simulate", "Scenarios", "Operations", "Architecture",
    ])

    with tab_guide:
        try: _guide_content(compact=False)
        except Exception as _e: st.exception(_e)

    with tab_dash:
        try: _dashboard_content()
        except Exception as _e: st.exception(_e)

    with tab_chat:
        try: _chat_content(full_width=True)
        except Exception as _e: st.exception(_e)

    with tab_simulate:
        try:
            sim_choice = st.selectbox("Analysis type:", ["Price Cascade","Supply Alert","Demand Forecast"], key="sim_choice")
            st.divider()
            if sim_choice == "Price Cascade":
                _price_cascade_content(with_right_panel=False)
                _render_details("price")
            elif sim_choice == "Supply Alert":
                _supply_alert_content(with_right_panel=False)
                _render_details("supply")
            else:
                _demand_forecast_content(with_right_panel=False)
                _render_details("forecast")
            # Always show history for all 3 simulation types below the active form
            st.divider()
            st.markdown("#### 📋 Simulation History")
            hc1, hc2, hc3 = st.columns(3)
            with hc1:
                _render_run_history(st.session_state.hist_price, "No Price Cascade runs yet.")
            with hc2:
                _render_run_history(st.session_state.hist_supply, "No Supply Alert runs yet.")
            with hc3:
                _render_run_history(st.session_state.hist_forecast, "No Demand Forecast runs yet.")
        except Exception as _e: st.exception(_e)

    with tab_scenarios:
        try:
            scen_choice = st.radio("Mode:", ["Scenario Planner","Strategy Canvas"], horizontal=True, key="scen_choice")
            st.divider()
            if scen_choice == "Scenario Planner":
                _scenario_planner_content(with_right_panel=False)
                _render_details("scenario")
                st.divider()
                _render_run_history(st.session_state.hist_scenario)
            else:
                _strategy_canvas_content()
        except Exception as _e: st.exception(_e)

    with tab_ops:
        try:
            ops_choice = st.selectbox("View:", ["Shelf & Store","Financial Impact","Data Sources"], key="ops_choice")
            st.divider()
            if ops_choice == "Shelf & Store":
                _shelf_replenishment_content(with_right_panel=False)
                _render_details("shelf")
            elif ops_choice == "Financial Impact":
                _financial_impact_content(with_right_panel=False)
                _render_details("finance")
            else:
                _data_sources_content()
            if ops_choice != "Data Sources":
                # Always show history for both ops tools below the active form
                st.divider()
                st.markdown("#### 📋 Operations History")
                oh1, oh2 = st.columns(2)
                with oh1:
                    _render_run_history(st.session_state.hist_shelf, "No Shelf & Store runs yet.")
                with oh2:
                    _render_run_history(st.session_state.hist_finance, "No Financial Impact runs yet.")
        except Exception as _e: st.exception(_e)

    with tab_arch:
        try:
            _network_graph_content()
            st.divider()
            with st.expander("🔄 Workflow Builder — Step-by-Step Scenario Analysis", expanded=False):
                _workflow_content()
        except Exception as _e: st.exception(_e)
