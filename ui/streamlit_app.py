"""
Retail Supply Chain Optimization — Streamlit UI (Full)

Tabs:
  0. Guide         — new user onboarding, tab walkthrough, sample queries, glossary
  1. Dashboard     — live network status + KPI tiles
  2. Chat          — streaming multi-agent conversational interface
  3. Price Cascade — price change → full downstream simulation
  4. Supply Alert  — carrier strike / port delay / supplier bankruptcy
  5. Demand Forecast — 15-variable model with CI fan chart
  6. Scenario Planner — multi-scenario comparison + conflict detection
  7. Shelf & Store — HQ→DC→Store replenishment chain
  8. Financial Impact — P&L waterfall, trade dollars, carrying cost
  9. Data Sources  — provenance and freshness tracker
 10. Workflow      — integrated step-by-step scenario builder
 11. Flow Map      — LangGraph DAG visualization with execution trace
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import streamlit as st


def _get_api_key() -> str:
    """Read the Anthropic API key at render time from every possible source."""
    # 1. Already in environment (local .env via dotenv, or previously injected)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # 2. Streamlit Cloud secrets dashboard
    try:
        key = st.secrets["ANTHROPIC_API_KEY"]
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key  # cache for non-Streamlit callers
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
/* ── Global fonts & background ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar header ── */
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3 !important;
    border-radius: 6px;
    font-size: 13px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #0071ce;
    border-color: #0071ce;
}

/* ── KPI cards ── */
div[data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── Critical/warning banners ── */
div[data-testid="stAlert"] { border-radius: 8px; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 8px;
}

/* ── Tab styling ── */
button[data-baseweb="tab"] {
    font-size: 13px;
    font-weight: 500;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0071ce !important;
    border-bottom-color: #0071ce !important;
}

/* ── Status chip ── */
.chip-green  { background:#d4edda; color:#155724; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.chip-red    { background:#f8d7da; color:#721c24; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.chip-yellow { background:#fff3cd; color:#856404; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.chip-blue   { background:#cce5ff; color:#004085; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }

/* ── Tool call expander ── */
details summary { font-size: 13px; color: #6c757d; }

/* ── Plotly chart border ── */
.js-plotly-plot { border-radius: 10px; overflow: hidden; }

/* ── History panel (right-side) ── */
.hist-panel {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
}
.hist-panel h4 {
    font-size: 13px;
    font-weight: 700;
    color: #495057;
    margin: 0 0 10px 0;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.hist-item {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 12px;
    line-height: 1.5;
}
.hist-item-header {
    font-weight: 600;
    color: #212529;
    font-size: 12px;
    margin-bottom: 4px;
}
.hist-item-meta {
    color: #6c757d;
    font-size: 11px;
}
.hist-item-result {
    color: #0071ce;
    font-weight: 600;
    font-size: 12px;
    margin-top: 4px;
}
.hist-empty {
    color: #adb5bd;
    font-size: 12px;
    font-style: italic;
    text-align: center;
    padding: 20px 0;
}

/* ── Chat layout ── */
.chat-messages-container {
    overflow-y: auto;
    padding-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "conversation_history": [],
        "last_tool_calls": [],
        "freshness_warnings": [],
        "max_iterations_override": 10,
        "session_queries": 0,
        "session_tool_calls": 0,
        "session_iterations": 0,
        "pipeline_version": "V1 — Agentic Loop",
        # Per-tab run history
        "hist_price": [],
        "hist_supply": [],
        "hist_forecast": [],
        "hist_scenario": [],
        "hist_shelf": [],
        "hist_finance": [],
        "hist_workflow": [],
        "hist_flowmap": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_history(items: list, empty_msg: str = "No runs yet."):
    """Render a right-side history panel from a list of history dicts."""
    st.markdown(
        '<div class="hist-panel"><h4>📋 Run History</h4>',
        unsafe_allow_html=True,
    )
    if not items:
        st.markdown(
            f'<div class="hist-empty">{empty_msg}</div>',
            unsafe_allow_html=True,
        )
    else:
        for item in reversed(items):
            header = item.get("header", "")
            meta   = item.get("meta", "")
            result = item.get("result", "")
            ts     = item.get("ts", "")
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-item-header">{header}</div>'
                f'<div class="hist-item-meta">{meta}</div>'
                f'<div class="hist-item-result">{result}</div>'
                f'<div class="hist-item-meta" style="margin-top:4px;color:#ced4da;">{ts}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

_init_state()

# ─── Data Source & Formula helpers ───────────────────────────────────────────

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
**Demand Change:**
`Demand_delta% = Elasticity × Price_delta%`

**Asymmetric rule** (tobacco, alcohol, diapers, infant formula):
- Price ↑: `Elasticity_up = base_elasticity × 1.3` (demand loss amplified)
- Price ↓: `Elasticity_dn = base_elasticity × 0.7` (demand recovery muted)

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
**Days to Stockout:**
`DTS = Current_Inventory / (Avg_Daily_Demand × Safety_Factor)`

**Revenue at Risk:**
`Rev_at_Risk = max(0, Replenishment_Lag - DTS) × Daily_Revenue`

**Severity thresholds:**
- Critical: `DTS < Replenishment_Lag` (will stockout before truck arrives)
- Warning:  `DTS < Replenishment_Lag × 1.5`
- OK:       `DTS ≥ Replenishment_Lag × 1.5`
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
**Multiplicative demand function:**
`D(t) = Base_Demand × Π(1 + fᵢ × wᵢ)`

Where fᵢ = factor signal and wᵢ = learned weight for each of:
`price, promo, tariff, weather, seasonality, trend, competitor_price,
social_sentiment, days_supply, channel_mix, planogram_compliance,
regional_income, household_penetration, repeat_rate, new_item_velocity`

**Confidence interval:**
`CI_width = Base_MAPE × (1 + 0.05 × Horizon_weeks)`
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
**Per-scenario revenue:**
`Rev_s = Price_s × Demand_s(price, promo, supply_reduction, tariff) × Horizon`

**Compound execution penalty** (3+ scenarios simultaneously):
`Penalty = (N_scenarios - 2) × 2%  [applied to both revenue and margin]`

**Conflict score:**
`Conflict_severity = Σ(block_weight_i × block_weight_j) for conflicting pairs`
""",
        "params": [
            ("Compound penalty per extra block %", "comp_pen", 2.0, 0.0, 5.0, 0.5),
            ("Conflict threshold (score)", "conf_thresh", 0.6, 0.1, 1.0, 0.05),
        ]
    },
    "shelf": {
        "title": "Replenishment Formula",
        "formula": r"""
**Reorder Point:**
`ROP = Lead_Time_days × Avg_Daily_Demand + Safety_Stock`

**Safety Stock:**
`SS = Z_score × σ_demand × √Lead_Time`   (Z=1.65 for 95% service level)

**Perishable cap:**
`Max_Order = (Shelf_Life_days - Lead_Time) × Avg_Daily_Demand × Planogram_Cap_units`

**Freight cost:**
`Freight = Base_rate × Units × Priority_multiplier`
- Standard: 1×, Expedited: 2×, Emergency: 3×
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
**Bottom-line waterfall:**
```
Gross_Revenue_delta   = (New_Price - Old_Price) × New_Vol
                      + (New_Vol  - Old_Vol)  × Old_Price
Vendor_Trade_Offset   = Gross_Revenue_delta × trade_rate
Net_Revenue_delta     = Gross_Revenue_delta + Vendor_Trade_Offset
COGS_delta            = (New_Vol - Old_Vol) × Unit_Cost
Gross_Margin_delta    = Net_Revenue_delta - COGS_delta
Carrying_Cost_delta   = delta_Inventory_units × holding_rate × days
Bottom_Line           = Gross_Margin_delta - Carrying_Cost_delta - Tax_approx
```
""",
        "params": [
            ("Holding cost rate ($/unit/day)", "hold_r", 0.02, 0.005, 0.1, 0.005),
            ("Vendor trade rate %", "trade_r", 3.0, 0.0, 15.0, 0.5),
            ("Tax rate (approx) %", "tax_r", 21.0, 0.0, 35.0, 1.0),
        ]
    },
}


def _render_data_sources(tab_key: str):
    """Render cloud data source panel as expander in right column."""
    src = _DATA_SOURCES.get(tab_key, {})
    if not src:
        return
    with st.expander("Data Sources", expanded=False):
        st.markdown(f"**{src['title']}**")
        color_map = {"green": "#2ea043", "blue": "#388bfd", "orange": "#d29922", "purple": "#bc8cff"}
        for name, desc, color in src["sources"]:
            dot = f'<span style="color:{color_map[color]};font-size:10px;">&#9679;</span>'
            st.markdown(
                f'{dot} **{name}**  \n<span style="color:#8b949e;font-size:11px;">{desc}</span>',
                unsafe_allow_html=True,
            )
        st.caption(src["cloud_note"])


def _render_formula_panel(tab_key: str):
    """Render formula + editable parameters as expander in right column."""
    fml = _FORMULAS.get(tab_key, {})
    if not fml:
        return
    with st.expander("Formula & Parameters", expanded=False):
        st.markdown(f"**{fml['title']}**")
        st.markdown(fml["formula"], unsafe_allow_html=False)
        if fml.get("params"):
            st.markdown("---")
            st.markdown("**Adjust parameters to test:**")
            for label, key, default, mn, mx, step in fml["params"]:
                st.slider(label, min_value=float(mn), max_value=float(mx),
                          value=float(default), step=float(step),
                          key=f"fml_{tab_key}_{key}",
                          help=f"Override default: {default}")


# ─── Orchestrator (lazy, cached) ─────────────────────────────────────────────

@st.cache_resource
def _get_orchestrator(_key: str = ""):
    """Cached per unique API key so a new key always gets a fresh client."""
    from agents.orchestrator import Orchestrator
    return Orchestrator()

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏪 Retail Supply Chain AI")
    st.markdown("*Powered by Claude claude-sonnet-4-6*")
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

    # ── Pipeline Version ──
    st.markdown("### 🔀 Pipeline Version")
    pipeline_ver = st.radio(
        "Chat agent backend:",
        ["V1 — Agentic Loop", "V2 — LangGraph"],
        index=0,
        help=(
            "V1: Single Claude agent, all 17 tools, configurable iterations.\n"
            "V2: LangGraph multi-agent graph — router + 10 domain nodes + synthesizer."
        ),
    )
    if pipeline_ver != st.session_state.pipeline_version:
        st.session_state.pipeline_version = pipeline_ver
        st.session_state.conversation_history = []  # reset on pipeline switch

    # ── Agentic Loop Config (V1 only) ──
    st.markdown("### ⚙️ Agent Config")
    if "V1" in pipeline_ver:
        max_iter = st.slider(
            "Max iterations (V1 only)",
            min_value=3, max_value=25, value=10,
            help="Simple queries: 3-6. Complex (multi-SKU, tariff): 15-20."
        )
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
            "are available regionally, and what is the revenue at risk? Give me a full mitigation plan."
        ),
        "📉 Forecast Accuracy Gap": (
            "Analyze demand forecast accuracy for diapers (HUG48-3) at 8 weeks out. "
            "What is the gap to industry benchmark? What is the dollar revenue impact of "
            "improving by 7 percentage points? Which demand variables are the biggest drivers?"
        ),
        "⚠️ Promo + Strike Conflict": (
            "We are planning a 10% promotional price cut on HUG48-3 starting 2026-05-01 "
            "for 30 days. TruckCo B is still on strike affecting diaper supply in SE and MW. "
            "Detect scenario conflicts and advise whether to proceed."
        ),
        "🥛 Milk Shelf Replenishment": (
            "STR-005 reports critically low milk (MLK-GAL) inventory. Check stockout risk, "
            "perishable 3-day cap, planogram capacity, and recommend replenishment quantity "
            "and priority from DC-NW."
        ),
        "📊 3-Scenario Comparison": (
            "Compare three scenarios for HUG48-3 over 8 weeks: "
            "A) Hold price at $12.99 (baseline), "
            "B) Raise price to $14.49, "
            "C) Drop price to $11.99 with 15% promo uplift. "
            "Which maximizes revenue? Which maximizes margin? Factor in asymmetric elasticity."
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

# ─── Tabs ─────────────────────────────────────────────────────────────────────

(tab_guide, tab_dash, tab_chat, tab_price, tab_supply, tab_forecast,
 tab_scenario, tab_shelf, tab_finance, tab_data,
 tab_workflow, tab_flowmap, tab_netgraph, tab_builder, tab_canvas) = st.tabs([
    "⭐ Guide",
    "Dashboard",
    "Chat",
    "Price Cascade",
    "Supply Alert",
    "Demand Forecast",
    "Scenario Planner",
    "Shelf & Store",
    "Financial Impact",
    "Data Sources",
    "Workflow",
    "Flow Map",
    "🕸 Network Graph",
    "🧩 Scenario Builder",
    "🎯 Strategy Canvas",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — GUIDE (New User Onboarding)
# ════════════════════════════════════════════════════════════════════════════
with tab_guide:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0071ce 0%,#004a8f 100%);
                border-radius:14px;padding:32px 36px;margin-bottom:24px;">
      <h1 style="color:white;margin:0;font-size:2em;">
        🏪 Welcome to Retail Supply Chain Optimization AI
      </h1>
      <p style="color:#cce5ff;margin:10px 0 0 0;font-size:1.05em;">
        A multi-agent AI system that simulates how a single retail decision —
        a price change, a carrier strike, a demand shift — cascades across
        pricing, inventory, supply chain, and finance <strong style="color:white;">simultaneously</strong>.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── What this app does ──
    st.markdown("## What Does This App Do?")
    st.markdown("""
In retail, no decision lives in isolation. When you raise the price of diapers by 10%:
- Demand drops by ~14% (price elasticity)
- Replenishment orders to the distribution center get adjusted
- Carrier load requirements shift across regions
- A planned promotion may suddenly conflict with a supply disruption
- Net margin changes after vendor trade dollar netting

**This AI reasons through all of those connections at once** — in seconds — using 17 specialist tools and two AI pipelines (a direct Claude agent and a LangGraph multi-agent graph).
    """)

    st.info(
        "**No spreadsheet expertise needed.** Ask questions in plain English in the Chat tab, "
        "or use any of the pre-built tabs to explore specific scenarios.",
        icon="💡",
    )

    st.divider()

    # ── Quick Start ──
    st.markdown("## Quick Start — 3 Ways to Use This App")

    qs1, qs2, qs3 = st.columns(3)
    with qs1:
        st.markdown("""
        <div style="background:#f0f4ff;border:1px solid #0071ce;border-radius:10px;padding:18px;">
        <h4 style="color:#0071ce;margin-top:0;">💬 Option 1: Just Chat</h4>
        Go to the <strong>Chat</strong> tab and ask anything:<br><br>
        <em>"What happens if we raise diaper prices by 10%?"</em><br><br>
        <em>"TruckCo B is on strike — what's our stockout risk?"</em><br><br>
        The AI picks the right tools and walks you through the full cascade.
        </div>
        """, unsafe_allow_html=True)

    with qs2:
        st.markdown("""
        <div style="background:#f0fff4;border:1px solid #28a745;border-radius:10px;padding:18px;">
        <h4 style="color:#28a745;margin-top:0;">🚀 Option 2: Use a Preset</h4>
        In the left sidebar, click any of the <strong>6 Quick Scenario</strong> buttons.<br><br>
        These load a pre-written expert query directly into the Chat tab — perfect for seeing the AI's full reasoning on a real scenario.
        </div>
        """, unsafe_allow_html=True)

    with qs3:
        st.markdown("""
        <div style="background:#fff8f0;border:1px solid #f39c12;border-radius:10px;padding:18px;">
        <h4 style="color:#e67e22;margin-top:0;">🔄 Option 3: Build a Workflow</h4>
        Go to the <strong>Workflow</strong> tab and pick your trigger (price change, carrier strike, demand shift, etc.), set context, choose objectives, and let the AI run the full structured analysis with a Day 0→30 ripple timeline.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Tab-by-Tab Navigation ──
    st.markdown("## Tab-by-Tab Navigation Guide")

    tab_guide_data = [
        {
            "icon": "📊",
            "name": "Dashboard",
            "summary": "Your network at a glance",
            "desc": (
                "Start here to understand the current state of the supply chain. "
                "You'll see **carrier status chips** (TruckCo B is currently on strike — shown in red), "
                "**DC inventory KPI cards** for all 4 distribution centers, and a **strike impact timeline** "
                "showing how inventory is projected to deplete over 14 days. "
                "This tab needs no API key."
            ),
            "tip": "Check this tab first before running any analysis to understand what's already broken in the network.",
            "bg": "#f0f4ff",
            "border": "#0071ce",
        },
        {
            "icon": "💬",
            "name": "Chat",
            "summary": "The main AI interface",
            "desc": (
                "This is the core of the app. Type any supply chain question and the AI agent reasons through it, "
                "calling the right tools in sequence. You can see **every tool call live** as it happens in the status panel. "
                "Switch between **V1 (Agentic Loop)** and **V2 (LangGraph)** in the sidebar — V1 gives you a single "
                "all-capable agent, V2 routes your query through specialist nodes and shows a node trace."
            ),
            "tip": "Use V2 (LangGraph) for structured decisions. Use V1 for exploratory multi-turn conversation.",
            "bg": "#f0fff4",
            "border": "#28a745",
        },
        {
            "icon": "🔺",
            "name": "Price Cascade",
            "summary": "Simulate a price change end-to-end",
            "desc": (
                "Select a SKU and enter a new price. The tab instantly shows: the demand volume change "
                "(using real price elasticity), the revenue and margin impact, the adjusted replenishment "
                "order quantity, and a **waterfall chart** tracing revenue → gross margin → net margin. "
                "Asymmetric elasticity is applied automatically — sticky products like diapers and tobacco "
                "don't recover volume proportionally on price cuts."
            ),
            "tip": "Try raising Huggies (HUG48-3) from $12.99 to $14.49 and watch the full waterfall.",
            "bg": "#fff8f0",
            "border": "#e67e22",
        },
        {
            "icon": "🚛",
            "name": "Supply Alert",
            "summary": "Carrier strike and supply disruption analysis",
            "desc": (
                "Enter a disruption event (carrier strike, port delay, supplier issue) and see: "
                "**days-of-supply remaining** per DC as a bar chart, **alternate carrier options** "
                "with availability, cost premium, and regional coverage gaps. "
                "TruckCo B is currently on strike — the SE region has a critical gap because "
                "TruckCo C can't handle diapers (refrigerated trucks only) and TruckCo D charges +45%."
            ),
            "tip": "SE is the most vulnerable region. Always check regional coverage before choosing an alternate carrier.",
            "bg": "#fff0f0",
            "border": "#dc3545",
        },
        {
            "icon": "📈",
            "name": "Demand Forecast",
            "summary": "8-week demand forecast with uncertainty",
            "desc": (
                "View the AI-generated demand forecast for any SKU across a configurable horizon. "
                "The **fan chart** shows confidence intervals that widen 15% per 4-week period — "
                "so at 8 weeks you're looking at ±30% uncertainty. "
                "A **forecast accuracy gauge** shows current model accuracy — if it falls below 60%, "
                "the system flags the forecast as unreliable and won't pass it to the PO system. "
                "A **variable contribution chart** shows which of the 15 demand drivers matter most."
            ),
            "tip": "If accuracy is below 70%, treat any PO decision based on this forecast as high-risk.",
            "bg": "#f0fff4",
            "border": "#28a745",
        },
        {
            "icon": "⚖️",
            "name": "Scenario Planner",
            "summary": "Compare 3 options side by side",
            "desc": (
                "Build and compare up to 3 simultaneous scenarios for the same SKU: "
                "Hold price / Raise price / Promote. The system runs all three through "
                "the full cascade and shows a **side-by-side bar chart** of revenue and margin outcomes. "
                "Critically, it also runs **conflict detection** — if two scenarios overlap in a way that "
                "creates supply risk (e.g., promotion during carrier strike), it flags CRITICAL."
            ),
            "tip": "Always run conflict detection before committing to a promo during any active supply disruption.",
            "bg": "#f8f0ff",
            "border": "#8e44ad",
        },
        {
            "icon": "🏬",
            "name": "Shelf & Store",
            "summary": "Store-level replenishment and perishables",
            "desc": (
                "Enter a store and SKU to see the full replenishment chain: HQ → DC → Store, "
                "with each leg's timing. A **Gantt chart** shows the timeline. "
                "For perishable items (dairy), the system enforces a hard **3-day max days-of-supply cap** — "
                "it will never recommend ordering more than 3 days of milk regardless of demand signal. "
                "Planogram capacity is also checked — orders exceeding shelf space go to back-of-store."
            ),
            "tip": "Replenishment takes 3–4 days minimum. There's a 30% chance of +3 extra days. Plan 5–7 days ahead.",
            "bg": "#f0f8ff",
            "border": "#2980b9",
        },
        {
            "icon": "💰",
            "name": "Financial Impact",
            "summary": "P&L waterfall and net margin",
            "desc": (
                "See the full financial picture of any pricing decision: a **waterfall chart** "
                "stepping from revenue → gross margin → vendor trade dollar netting → "
                "VMI inventory adjustment → carrying cost → net margin. "
                "Vendor trade dollars (the manufacturer subsidy on promotions) are automatically "
                "netted against promo cost. VMI-owned inventory is excluded from your carrying cost. "
                "This is the tab to use when the CFO asks what a decision actually cost."
            ),
            "tip": "Vendor trade dollars can flip a loss-making promo into a profitable one. Always check the net.",
            "bg": "#fffff0",
            "border": "#f39c12",
        },
        {
            "icon": "🗄️",
            "name": "Data Sources",
            "summary": "Know how fresh your data is",
            "desc": (
                "Every tool call in this system pulls from one of three data sources with different freshness: "
                "**OLTP** (5-min lag — pricing, POs), **WMS** (15-min lag — operational inventory), "
                "**OLAP** (24-hour lag — analytics). "
                "This tab shows which source each tool uses, flags where OLAP data is being used for "
                "an operational decision (high risk), and lets you compare WMS vs OLAP inventory "
                "to see how much the 24h batch has drifted from reality."
            ),
            "tip": "Never use OLAP inventory numbers for same-day replenishment decisions. Use WMS.",
            "bg": "#f5f5f5",
            "border": "#6c757d",
        },
        {
            "icon": "🔄",
            "name": "Workflow",
            "summary": "Step-by-step guided decision builder",
            "desc": (
                "A structured 6-step workflow for specialist decisions. Pick your **business trigger** "
                "(price change, supply disruption, demand shift, tariff, seasonal, competitive move), "
                "set the **context** (SKU, region, horizon), choose your **objectives** "
                "(revenue / margin / service level / cost), then run the AI analysis. "
                "Results show ranked options and a **Day 0→30 ripple effect timeline** "
                "showing how the decision propagates across all domains over a month."
            ),
            "tip": "Use this tab when you need a structured, auditable analysis — not just a chat answer.",
            "bg": "#f0fff4",
            "border": "#27ae60",
        },
        {
            "icon": "🗺️",
            "name": "Flow Map",
            "summary": "See the LangGraph agent graph live",
            "desc": (
                "Visual representation of the V2 LangGraph pipeline. All 12 nodes are shown as a DAG "
                "(directed acyclic graph). After you run a query in the Chat tab with V2 selected, "
                "come here to see which nodes were executed (shown in **green**) and which were "
                "available but not invoked (shown in **gray**). Hover over any node to see which "
                "tools it has access to. Includes an architecture explanation and tool assignment table."
            ),
            "tip": "Run a carrier strike query in Chat (V2), then switch here to see the supply_disruption → carrier_node path light up green.",
            "bg": "#f0f4ff",
            "border": "#0071ce",
        },
    ]

    for tab_item in tab_guide_data:
        with st.expander(
            f"{tab_item['icon']}  **{tab_item['name']}** — {tab_item['summary']}",
            expanded=False,
        ):
            col_desc, col_tip = st.columns([3, 2])
            with col_desc:
                st.markdown(tab_item["desc"])
            with col_tip:
                st.markdown(
                    f"""<div style="background:{tab_item['bg']};border-left:4px solid {tab_item['border']};
                    border-radius:6px;padding:14px 16px;">
                    <strong style="color:{tab_item['border']};">💡 Pro Tip</strong><br>
                    <span style="font-size:0.93em;">{tab_item['tip']}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.divider()

    # ── Sidebar Guide ──
    st.markdown("## Left Sidebar — What Each Section Does")

    sb1, sb2 = st.columns(2)
    with sb1:
        st.markdown("""
**🔑 API Status**
Shows whether the Anthropic API key is loaded. Green = AI chat is active. Red = Chat tab won't work but all other tabs do.

**🔀 Pipeline Version**
- **V1 — Agentic Loop:** Single Claude agent with all 17 tools. Best for exploratory, conversational analysis.
- **V2 — LangGraph:** Structured multi-agent graph. Best for repeatable, auditable decisions. Switching resets chat history.

**⚙️ Agent Config**
Max iterations slider (V1 only). Simple price queries need 3–6 iterations. Complex multi-SKU scenarios need 15–20.
        """)
    with sb2:
        st.markdown("""
**📊 Session Stats**
Live counter of queries run, total tool calls made, and total agent iterations this session.

**🚀 Quick Scenarios**
Six pre-written expert queries. Click any button to instantly load it into the Chat tab and run it. Great for demos or learning what the system can do.

**🗑 Clear Chat / ↺ Reset All**
Clear Chat removes conversation history. Reset All also zeroes session stats. Use Reset when switching between scenarios to keep context clean.
        """)

    st.divider()

    # ── Glossary ──
    st.markdown("## Key Concepts Glossary")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Price Elasticity** | How much demand changes per 1% price change. Diapers = -1.4 (10% price rise → 14% volume drop) |
| **Asymmetric Elasticity** | Price cuts don't recover demand proportionally. Diapers recover only 70%, tobacco only 40% |
| **Replenishment Lag** | Time from order to shelf: 3–4 days base + 30% chance of +3 extra days |
| **Days of Supply (DoS)** | How many days current inventory can cover at current demand rate |
| **Safety Stock** | Minimum buffer inventory kept to absorb demand spikes and late deliveries |
| **VMI** | Vendor-Managed Inventory — stock owned by the manufacturer, not the retailer. Excludes from your carrying cost |
        """)
    with g2:
        st.markdown("""
| Term | Meaning |
|------|---------|
| **Vendor Trade Dollars** | Manufacturer subsidies on promotions. Netted against promo cost before margin is reported |
| **Carrying Cost** | Cost of holding inventory: 25% of inventory value per year |
| **OLTP / WMS / OLAP** | Data sources with 5-min / 15-min / 24-hour freshness lag respectively |
| **Planogram** | Shelf layout plan. Orders that exceed shelf capacity go to back-of-store |
| **Scenario Conflict** | When two simultaneous decisions create opposing pressure (e.g., promo + supply disruption) |
| **Confidence Interval** | Forecast uncertainty band. Widens 15% per 4-week horizon — ±30% at 8 weeks |
        """)

    st.divider()

    # ── Sample Queries ──
    st.markdown("## Sample Questions to Ask in the Chat Tab")

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown("**Pricing**")
        for q in [
            "Raise HUG48-3 price from $12.99 to $14.49 — show me the full cascade",
            "What's the revenue impact of dropping diaper price to $11.99 for 30 days?",
            "Costco just dropped their diaper price by 15% — should we match it?",
            "Compare holding vs raising vs promoting HUG48-3 over 8 weeks",
        ]:
            st.markdown(f"- *{q}*")

        st.markdown("**Supply Chain**")
        for q in [
            "TruckCo B is on strike — what's our stockout risk in SE and MW?",
            "Find alternate carriers for diapers in the Southeast region",
            "What is our revenue at risk if the TruckCo B strike lasts 21 days?",
        ]:
            st.markdown(f"- *{q}*")

    with q_col2:
        st.markdown("**Inventory & Demand**")
        for q in [
            "Check replenishment status for milk (MLK-GAL) at STR-005",
            "What's the demand forecast for diapers at 8 weeks with confidence intervals?",
            "Our forecast accuracy for HUG48-3 is 67% — what's the revenue risk?",
            "How much inventory should DC-SE hold given the current carrier situation?",
        ]:
            st.markdown(f"- *{q}*")

        st.markdown("**Scenarios & Finance**")
        for q in [
            "We're running a promo on diapers AND TruckCo B is on strike — is this safe?",
            "Show me the P&L waterfall for a 10% diaper price increase",
            "Calculate carrying cost for DC-SE if we pre-build 10 days of inventory",
            "What's the financial impact of improving forecast accuracy by 7 points?",
        ]:
            st.markdown(f"- *{q}*")

    st.divider()

    # ── System Facts ──
    st.markdown("## System Facts at a Glance")

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("AI Tools", "17", "across 5 domains")
    fc2.metric("LangGraph Nodes", "12", "router + specialists + synthesizer")
    fc3.metric("Edge Cases Handled", "17", "production-grade")
    fc4.metric("App Tabs", "12", "Dashboard to Flow Map")

    st.markdown("")

    kn1, kn2, kn3, kn4 = st.columns(4)
    kn1.metric("Diaper Elasticity", "-1.4", "10% up → 14% vol down")
    kn2.metric("Dairy Max Supply", "3 days", "hard perishable cap")
    kn3.metric("Replenishment Lag", "3–4 days", "+30% chance of +3 more")
    kn4.metric("Forecast Gate", "60%", "below this: PO blocked")

    st.divider()

    # ── Architecture summary ──
    st.markdown("## How the AI Works")

    arch1, arch2 = st.columns([2, 3])
    with arch1:
        st.markdown("""
**Two Pipelines, Same Tools**

The sidebar lets you switch between two AI backends:

**V1 — Agentic Loop**
One Claude claude-sonnet-4-6 agent. It sees all 17 tools and decides which ones to call and in what order. Like a senior analyst who knows every system.

**V2 — LangGraph**
A graph of 12 specialist agents. A Router classifies your query, routes it to the right domain node (Price Cascade, Supply Disruption, Demand Forecast, etc.), runs supporting nodes (Inventory, Carrier, Accuracy), then a Synthesizer combines everything into one response.

**Both hit the same mock data layer** — a realistic simulation of a Walmart-scale operation with 5 SKUs, 4 carriers, 4 DCs, 10 stores.
        """)
    with arch2:
        st.code("""
User Query
    │
    ▼
[PIPELINE TOGGLE — sidebar]
    │
    ├── V1: Single Agent (Claude claude-sonnet-4-6)
    │       → 17 tools available
    │       → Picks tools autonomously
    │       → Up to 20 iterations
    │
    └── V2: LangGraph Multi-Agent Graph
            → ROUTER  (classifies intent)
            │
            ├── price_cascade  → inventory_node → financial_impact
            ├── supply_disruption → carrier_node
            ├── demand_forecast   → accuracy_node
            ├── shelf_replenishment → perishable_check
            └── scenario_planning
            │
            └── SYNTHESIZER (merges all outputs)

[Mock Data Layer]
  5 SKUs · 4 Carriers · 4 DCs · 10 Stores
  OLTP (5min) · WMS (15min) · OLAP (24h)
        """, language="")

    st.success(
        "**Ready to start?** Click the **Chat** tab and type your first question, "
        "or try a **Quick Scenario** from the sidebar. The Dashboard tab shows the live network state.",
        icon="🚀",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader("Network Status Dashboard")
    st.caption("Live snapshot of the supply chain network. TruckCo B is currently on strike.")

    # ── Carrier Status Row ──
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
            chip_class = f"chip-{'red' if color == 'red' else 'green'}"
            st.markdown(f"""
            <div style="border:1px solid {'#f8d7da' if color=='red' else '#d4edda'};
                        border-radius:10px; padding:14px; text-align:center;
                        background:{'#fff5f5' if color=='red' else '#f8fff8'};">
              <div style="font-weight:700; font-size:15px; margin-bottom:6px;">{name}</div>
              <span class="{chip_class}">{status}</span>
              <div style="font-size:12px; color:#6c757d; margin-top:8px;">{regions}</div>
              <div style="font-size:11px; color:#adb5bd;">{cargo}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── DC Inventory Snapshot ──
    st.markdown("#### DC Inventory Snapshot — Diapers (HUG48-3)")
    dc_cols = st.columns(3)
    dc_summary = []
    for dc_id, dc in mock_executor.DCS.items():
        inv = dc["inventory"].get("HUG48-3", {})
        wms = inv.get("wms", 0)
        prod = mock_executor.PRODUCTS["HUG48-3"]
        dc_daily = (prod["base_demand_per_store_week"] / 7) * len(dc["stores_served"])
        dos = wms / dc_daily if dc_daily else 0
        dc_summary.append((dc_id, dc["name"].split("—")[1].strip(), wms, round(dos, 1), dc["region"]))

    for col, (dc_id, city, qty, dos, region) in zip(dc_cols, dc_summary):
        color = "red" if dos < 7 else ("yellow" if dos < 14 else "green")
        with col:
            st.metric(f"{dc_id} — {city}", f"{qty:,} units", f"{dos}d on-hand")
            chip = "chip-red" if dos < 7 else ("chip-yellow" if dos < 14 else "chip-blue")
            st.markdown(f'<span class="{chip}">Region: {region}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Network KPI Bar ──
    st.markdown("#### Network KPIs — Diapers")
    k1, k2, k3, k4, k5 = st.columns(5)
    total_dc_inv = sum(dc["inventory"].get("HUG48-3", {}).get("wms", 0) for dc in mock_executor.DCS.values())
    total_stores = 30
    base_weekly_rev = mock_executor.PRODUCTS["HUG48-3"]["base_price"] * mock_executor.PRODUCTS["HUG48-3"]["base_demand_per_store_week"] * total_stores
    k1.metric("Total DC Inventory", f"{total_dc_inv:,} units")
    k2.metric("Weekly Network Revenue", f"${base_weekly_rev:,.0f}")
    k3.metric("Active Carriers", "3 / 4", delta="-1 on strike", delta_color="inverse")
    k4.metric("Forecast Accuracy (8W)", "78%", delta="-7pts vs benchmark", delta_color="inverse")
    k5.metric("Stores at Risk", "10", delta="SE+MW regions", delta_color="inverse")

    st.markdown("---")

    # ── Inventory chart across all DCs ──
    inv_data = []
    for dc_id, dc in mock_executor.DCS.items():
        for sku in ["HUG48-3", "PAM72-5", "TAB-DIN", "BLK-THR"]:
            wms = dc["inventory"].get(sku, {}).get("wms", 0)
            inv_data.append({"DC": dc_id, "SKU": sku, "Units (WMS)": wms})

    inv_df = pd.DataFrame(inv_data)
    fig_inv = px.bar(
        inv_df, x="DC", y="Units (WMS)", color="SKU", barmode="group",
        title="DC Inventory by SKU (WMS — 15min lag)",
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=320,
    )
    fig_inv.update_layout(margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig_inv, use_container_width=True)

    # ── Strike impact timeline ──
    st.markdown("#### TruckCo B Strike Impact Timeline — Diapers (SE Region)")
    days = list(range(0, 18))
    store_inv = [22] * 18
    dc_inv = [3200] * 18
    prod_daily = mock_executor.PRODUCTS["HUG48-3"]["base_demand_per_store_week"] / 7

    store_inv_line, dc_inv_line = [], []
    si, di = 22, 3200
    dc_stores = 10
    for d in days:
        store_inv_line.append(max(0, si))
        dc_inv_line.append(max(0, di))
        si -= prod_daily
        di -= prod_daily * dc_stores

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=days, y=store_inv_line, name="Store Inventory (avg STR-01x)",
                                 line=dict(color="#e74c3c", width=2), fill="tozeroy",
                                 fillcolor="rgba(231,76,60,0.08)"))
    fig_tl.add_trace(go.Scatter(x=days, y=[v / 100 for v in dc_inv_line], name="DC-SE Inventory (/100)",
                                 line=dict(color="#3498db", width=2, dash="dot")))
    fig_tl.add_vline(x=3, line_dash="dash", line_color="red",
                     annotation_text="Store stockout", annotation_position="top right")
    fig_tl.add_vline(x=14, line_dash="dash", line_color="orange",
                     annotation_text="Strike ends (est.)", annotation_position="top left")
    fig_tl.add_hrect(y0=0, y1=4, fillcolor="rgba(231,76,60,0.05)", line_width=0)
    fig_tl.update_layout(
        title="Projected Inventory Depletion During Strike",
        xaxis_title="Days from Strike Start",
        yaxis_title="Units",
        height=320,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    st.plotly_chart(fig_tl, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT  (V1 Agentic Loop  |  V2 LangGraph)
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:
    use_v2 = "V2" in st.session_state.get("pipeline_version", "V1")

    st.subheader(
        "Multi-Agent Chat — "
        + ("LangGraph (V2)" if use_v2 else "Agentic Loop (V1)")
    )
    st.caption(
        ("**LangGraph V2:** router → domain nodes → synthesizer. "
         "Intent is classified first; only relevant tools run per node."
         if use_v2
         else
         "**Agentic Loop V1:** single Claude agent with all 17 tools. "
         "Claude decides which tools to call at each iteration.")
    )

    if not key_ok:
        st.warning(
            "API key not detected. Add `ANTHROPIC_API_KEY` to Streamlit secrets (cloud) "
            "or your `.env` file (local). All other tabs work without an API key."
        )

    # ── Two-column layout: chat (left) + history (right) ──
    chat_col, hist_col = st.columns([2, 1])

    with hist_col:
        st.markdown('<div class="hist-panel"><h4>📋 Conversation History</h4>', unsafe_allow_html=True)
        chat_msgs = [m for m in st.session_state.conversation_history if isinstance(m.get("content"), str)]
        if not chat_msgs:
            st.markdown('<div class="hist-empty">No messages yet. Start chatting!</div>', unsafe_allow_html=True)
        else:
            # Group into Q&A pairs for display
            pairs = []
            i = 0
            while i < len(chat_msgs):
                if chat_msgs[i]["role"] == "user":
                    q = chat_msgs[i]["content"]
                    a = chat_msgs[i + 1]["content"] if i + 1 < len(chat_msgs) else ""
                    pairs.append((q, a))
                    i += 2
                else:
                    i += 1
            for idx, (q, a) in enumerate(reversed(pairs), 1):
                q_short = q[:80] + ("…" if len(q) > 80 else "")
                a_short = a[:120] + ("…" if len(a) > 120 else "")
                st.markdown(
                    f'<div class="hist-item">'
                    f'<div class="hist-item-header">Q{len(pairs)-idx+1}: {q_short}</div>'
                    f'<div class="hist-item-meta" style="color:#212529;margin-top:4px;">{a_short}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with chat_col:
        # ── Render existing messages (above the input) ──
        for msg in st.session_state.conversation_history:
            if isinstance(msg.get("content"), str):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # ── Inject preset ──
        pending = st.session_state.pop("_pending_query", None)
        st.session_state.pop("_goto_chat", None)

        prompt = st.chat_input(
            "Ask anything — or load a preset from the sidebar →"
        ) or pending

    if prompt:
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        with chat_col:
            with st.chat_message("user"):
                st.markdown(prompt)

        with chat_col:
            with st.chat_message("assistant"):
                if not key_ok:
                    st.error("ANTHROPIC_API_KEY not found. Add it to Streamlit secrets (cloud) or .env (local).")
                else:
                    result = None
                    response_placeholder = st.empty()

                    # ──────────────────────────────────────────────────────────
                    # V2 — LangGraph path
                    # ──────────────────────────────────────────────────────────
                    if use_v2:
                        with st.status("LangGraph agent working...", expanded=True) as status_box:
                            try:
                                st.write("Routing query through LangGraph nodes...")
                                from agents.langgraph_flow import run_langgraph
                                t0 = time.time()
                                result = run_langgraph(prompt)
                                elapsed_total = round(time.time() - t0, 1)

                                if result.get("error"):
                                    status_box.update(label="Error", state="error", expanded=True)
                                    st.error(result["error"])
                                    result = None
                                else:
                                    intent = result.get("intent", "?")
                                    sku    = result.get("sku", "?")
                                    nodes  = list(result.get("node_outputs", {}).keys())
                                    n_tools = len(result.get("tool_calls_made", []))
                                    st.write(f"**Intent:** `{intent}`  |  **SKU:** `{sku}`")
                                    st.write(f"**Nodes executed:** {' → '.join(nodes)}")
                                    st.write(f"**Tools called:** {n_tools}  |  **Time:** {elapsed_total}s")
                                    status_box.update(
                                        label=f"Done — {len(nodes)} nodes, {n_tools} tools, {elapsed_total}s",
                                        state="complete", expanded=False,
                                    )
                                    # Patch updated_history for session continuity
                                    result["updated_history"] = [
                                        *st.session_state.conversation_history[:-1],
                                        {"role": "user", "content": prompt},
                                        {"role": "assistant", "content": result["response_text"]},
                                    ]

                                    # ── Node output explorer ──
                                    node_outputs = result.get("node_outputs", {})
                                    if node_outputs:
                                        with st.expander("🗂 Node outputs (LangGraph trace)", expanded=False):
                                            for node_name, output in node_outputs.items():
                                                st.markdown(f"**Node: `{node_name}`**")
                                                for k, v in output.items():
                                                    if k.endswith("_summary") and isinstance(v, str):
                                                        st.markdown(f"*{k}:*")
                                                        st.markdown(v[:500])
                                                    elif not k.endswith("_summary"):
                                                        st.json({k: v} if not isinstance(v, dict) else v,
                                                                expanded=False)
                                                st.divider()

                            except Exception as exc:
                                status_box.update(label="Error", state="error", expanded=True)
                                st.error(f"LangGraph error: {exc}")
                                result = None

                    # ──────────────────────────────────────────────────────────
                    # V1 — Direct Agentic Loop path
                    # ──────────────────────────────────────────────────────────
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
                                tool_calls_made, freshness_warnings = [], []
                                iteration = 0

                                response = client.messages.create(
                                    model=settings.MODEL_ID,
                                    max_tokens=settings.MAX_TOKENS,
                                    system=SYSTEM_PROMPT,
                                    tools=ALL_TOOLS,
                                    messages=messages,
                                )

                                while response.stop_reason == "tool_use" and iteration < max_iter:
                                    iteration += 1
                                    st.write(f"**Iteration {iteration}** — processing tool calls...")
                                    tool_results = []
                                    for block in response.content:
                                        if block.type != "tool_use":
                                            continue
                                        t0 = time.time()
                                        res = mock_executor.execute(block.name, block.input)
                                        elapsed = round(time.time() - t0, 2)
                                        had_error = bool(res.get("error"))
                                        prov  = res.get("provenance", "OLTP")
                                        fresh = res.get("freshness_minutes", 5)
                                        st.write(
                                            f"  {'✗' if had_error else '✓'} `{block.name}` "
                                            f"— {prov} ({fresh}min) — {elapsed}s"
                                        )
                                        if res.get("is_stale") or prov == "OLAP":
                                            freshness_warnings.append(f"'{block.name}' uses {prov} ({fresh}min lag).")
                                        data = res.get("data", {})
                                        if isinstance(data, dict):
                                            for fw in data.get("freshness_warnings", []):
                                                freshness_warnings.append(fw)
                                        tool_calls_made.append({
                                            "tool": block.name, "input": block.input,
                                            "result_summary": res.get("error") or str(data)[:200],
                                            "provenance": prov, "freshness_minutes": fresh,
                                            "had_error": had_error, "elapsed_s": elapsed,
                                        })
                                        tool_results.append({
                                            "type": "tool_result",
                                            "tool_use_id": block.id,
                                            "content": json.dumps(res),
                                        })

                                    messages.append({"role": "assistant", "content": response.content})
                                    messages.append({"role": "user", "content": tool_results})
                                    response = client.messages.create(
                                        model=settings.MODEL_ID,
                                        max_tokens=settings.MAX_TOKENS,
                                        system=SYSTEM_PROMPT,
                                        tools=ALL_TOOLS,
                                        messages=messages,
                                    )

                                final_text = " ".join(b.text for b in response.content if hasattr(b, "text"))
                                if iteration >= max_iter and response.stop_reason == "tool_use":
                                    final_text += f"\n\n⚠ Reached max iterations ({max_iter}). Increase slider for deeper analysis."

                                status_box.update(
                                    label=f"Done — {iteration} iteration(s), {len(tool_calls_made)} tool call(s)",
                                    state="complete", expanded=False,
                                )
                                result = {
                                    "response_text": final_text,
                                    "tool_calls_made": tool_calls_made,
                                    "iterations_used": iteration,
                                    "data_freshness_warnings": freshness_warnings,
                                    "updated_history": messages + [{"role": "assistant", "content": final_text}],
                                    "error": None,
                                }

                            except Exception as exc:
                                status_box.update(label="Error", state="error", expanded=True)
                                st.error(f"Agent error: {exc}")
                                result = None

                    # ── Shared output rendering ──────────────────────────────
                    if result:
                        def _chunks(text):
                            for i in range(0, len(text), 40):
                                yield text[i:i + 40]
                                time.sleep(0.005)
                        response_placeholder.write_stream(_chunks(result["response_text"]))

                        if result["tool_calls_made"]:
                            with st.expander(
                                f"🔧 {len(result['tool_calls_made'])} tool call(s) detail", expanded=False
                            ):
                                tc_df = pd.DataFrame([{
                                    "Tool": tc["tool"],
                                    "Provenance": tc["provenance"],
                                    "Lag (min)": tc["freshness_minutes"],
                                    "Time (s)": tc.get("elapsed_s", "—"),
                                    "Status": "ERROR" if tc["had_error"] else "OK",
                                } for tc in result["tool_calls_made"]])
                                st.dataframe(tc_df, use_container_width=True, hide_index=True)

                        shown = set()
                        for w in result["data_freshness_warnings"]:
                            if w not in shown:
                                st.warning(w, icon="⏱")
                                shown.add(w)

                        st.session_state.session_queries += 1
                        st.session_state.session_tool_calls += len(result["tool_calls_made"])
                        st.session_state.session_iterations += result.get("iterations_used", 0)
                        st.session_state.last_tool_calls = result["tool_calls_made"]
                        st.session_state.freshness_warnings = result["data_freshness_warnings"]
                        st.session_state.conversation_history = result.get("updated_history",
                            st.session_state.conversation_history)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE CASCADE
# ════════════════════════════════════════════════════════════════════════════
with tab_price:
    st.subheader("Price Change Cascade Simulator")
    st.caption("Change a retail price → see the full ripple effect on demand, POs, inventory, and financials.")

    pc_main, pc_hist = st.columns([2, 1])

    with pc_hist:
        _render_history(st.session_state.hist_price, "Run a simulation to see history.")
        _render_formula_panel("price")
        _render_data_sources("price")

    with pc_main:
      col1, col2, col3, col4 = st.columns(4)
      with col1:
        pc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="pc_sku")
      prod_info = mock_executor.PRODUCTS.get(pc_sku, {})
      with col2:
        pc_old_price = st.number_input("Current Price ($)", value=float(prod_info.get("base_price", 12.99)),
                                       min_value=0.01, step=0.10, key="pc_old")
      with col3:
        pc_new_price = st.number_input("New Price ($)", value=float(prod_info.get("base_price", 12.99)) + 1.50,
                                       min_value=0.01, step=0.10, key="pc_new")
      with col4:
        pc_horizon = st.slider("Horizon (weeks)", 1, 32, 8, key="pc_horizon")

    with pc_main:
        prod_class = prod_info.get("product_class", "general")
        if prod_class in ("diaper", "tobacco", "alcohol", "formula"):
            st.info(
                f"**Asymmetric elasticity** active for `{prod_class}` category. "
                "A price increase suppresses demand more than a price decrease recovers it."
            )

        if st.button("Simulate Price Cascade", type="primary", key="pc_run"):
            with st.spinner("Running cascade model..."):
                res = mock_executor.execute("simulate_price_change", {
                    "sku": pc_sku, "old_price": pc_old_price,
                    "new_price": pc_new_price, "horizon_weeks": pc_horizon,
                })

            if res.get("error"):
                st.error(res["error"])
            else:
                d = res["data"]
                di, fi, ii = d["demand_impact"], d["financial_impact"], d["inventory_impact"]

                # KPI row
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Price Δ", f"{d['price_change_pct']:+.1f}%")
                k2.metric("Demand Δ", f"{di['demand_change_pct']:+.1f}%",
                          f"{di['weekly_unit_delta']:+.0f} units/wk (network)")
                k3.metric("Net Revenue Δ", f"${fi['net_revenue_change_usd']:+,.0f}",
                          delta_color="inverse" if fi["net_revenue_change_usd"] < 0 else "normal")
                k4.metric("Margin Δ", f"${fi['margin_change_usd']:+,.0f}")
                k5.metric("Carrying Cost Δ", f"${ii['carrying_cost_increase_usd']:+,.0f}")

                st.markdown(f"**Affected nodes:** {', '.join(d['affected_nodes'])}")

                # Waterfall chart
                fig_wf = go.Figure(go.Waterfall(
                    orientation="v",
                    measure=["relative", "relative", "total", "relative", "relative", "total"],
                    x=["Gross Revenue Δ", "Vendor Trade", "Net Revenue",
                       "Margin Δ", "Carrying Cost", "Bottom Line"],
                    y=[fi["gross_revenue_change_usd"], fi["vendor_trade_offset_usd"],
                       fi["net_revenue_change_usd"], fi["margin_change_usd"],
                       -ii["carrying_cost_increase_usd"], fi["combined_bottom_line_impact_usd"]],
                    connector={"line": {"color": "#dee2e6"}},
                    increasing={"marker": {"color": "#2ecc71"}},
                    decreasing={"marker": {"color": "#e74c3c"}},
                    totals={"marker": {"color": "#3498db"}},
                    text=[
                        f"${abs(fi['gross_revenue_change_usd']):,.0f}",
                        f"${fi['vendor_trade_offset_usd']:,.0f}",
                        f"${abs(fi['net_revenue_change_usd']):,.0f}",
                        f"${abs(fi['margin_change_usd']):,.0f}",
                        f"${ii['carrying_cost_increase_usd']:,.0f}",
                        f"${abs(fi['combined_bottom_line_impact_usd']):,.0f}",
                    ],
                    textposition="outside",
                ))
                fig_wf.update_layout(
                    title=f"Financial Impact Waterfall — {d['sku_name']} ({pc_horizon}W)",
                    height=380, margin=dict(l=40, r=40, t=55, b=40),
                )
                st.plotly_chart(fig_wf, use_container_width=True)

                col_l, col_r = st.columns(2)
                with col_l:
                    fig_dem = go.Figure(data=[
                        go.Bar(name="Before", x=["Per Store/Wk", "Total Network/Wk"],
                               y=[di["old_weekly_demand_per_store"], di["old_total_weekly"]],
                               marker_color="#3498db"),
                        go.Bar(name="After", x=["Per Store/Wk", "Total Network/Wk"],
                               y=[di["new_weekly_demand_per_store"], di["new_total_weekly"]],
                               marker_color="#e74c3c" if d["price_direction"] == "increase" else "#2ecc71"),
                    ])
                    fig_dem.update_layout(barmode="group", title="Demand Before vs After",
                                           height=300, margin=dict(l=30, r=20, t=40, b=30))
                    st.plotly_chart(fig_dem, use_container_width=True)

                with col_r:
                    pos = d.get("po_adjustments", [])
                    if pos:
                        st.markdown("**Open PO Adjustments**")
                        po_df = pd.DataFrame(pos)
                        po_df["Recommendation"] = po_df["recommended_adjustment_units"].apply(
                            lambda x: f"{'Reduce' if x < 0 else 'Increase'} {abs(int(x))} units"
                        )
                        po_df["Adjustable"] = po_df["adjustable"].map({True: "Yes", False: "Too close"})
                        st.dataframe(po_df[["po_id", "current_units", "Recommendation",
                                            "eta_days", "Adjustable"]],
                                     use_container_width=True, hide_index=True)
                    else:
                        st.info("No open POs for this SKU.")

                st.subheader("Recommended Actions")
                for i, rec in enumerate(d.get("recommendations", []), 1):
                    st.markdown(f"**{i}.** {rec}")

                if d.get("asymmetric_elasticity_note") and "N/A" not in d.get("asymmetric_elasticity_note", ""):
                    st.warning(d["asymmetric_elasticity_note"])

                # ── Save to history ──
                import datetime as _dt
                st.session_state.hist_price.append({
                    "header": f"{pc_sku}  ${pc_old_price:.2f} → ${pc_new_price:.2f}",
                    "meta": f"Horizon: {pc_horizon}W  |  Demand Δ: {di['demand_change_pct']:+.1f}%",
                    "result": f"Revenue Δ: ${fi['net_revenue_change_usd']:+,.0f}  |  Margin Δ: ${fi['margin_change_usd']:+,.0f}",
                    "ts": _dt.datetime.now().strftime("%H:%M"),
                })


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SUPPLY ALERT
# ════════════════════════════════════════════════════════════════════════════
with tab_supply:
    st.subheader("Supply Disruption Analyzer")
    st.caption(
        "Model carrier strikes, port delays, and supplier bankruptcies. "
        "Regional carrier availability checked separately from national."
    )

    sa_main, sa_hist = st.columns([2, 1])

    with sa_hist:
        _render_history(st.session_state.hist_supply, "Run a disruption analysis to see history.")
        _render_formula_panel("supply")
        _render_data_sources("supply")

    with sa_main:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            sa_type = st.selectbox("Disruption Type",
                                   ["carrier_strike", "port_delay", "supplier_bankruptcy"], key="sa_type")
            sa_entity = st.selectbox("Affected Entity",
                                     ["TruckCo_B", "TruckCo_A", "TruckCo_C", "PORT-LA", "SUP-KIMBERLY"],
                                     key="sa_entity")
        with col2:
            defaults_dur = {"carrier_strike": 14, "port_delay": 45, "supplier_bankruptcy": 0}
            sa_duration = st.number_input("Duration (days) — 0 for bankruptcy (permanent)",
                                          value=defaults_dur[sa_type], min_value=0, max_value=180, key="sa_dur")
            sa_region = st.selectbox("Focus Region", ["all", "SE", "NW", "MW"], key="sa_region")
        with col3:
            sa_skus_all = st.multiselect("Override affected SKUs (blank = auto-detect)",
                                         list(mock_executor.PRODUCTS.keys()), key="sa_skus")
            st.markdown("")
            st.markdown("")
            run_sa = st.button("Analyze Disruption", type="primary", key="sa_run")

        # Duration model hint
        model_hints = {
            "carrier_strike": "Carrier strikes: typically resolve in **2-3 weeks**.",
            "port_delay": "Port delays (rerouting, congestion): typically **6-10 weeks**.",
            "supplier_bankruptcy": "Supplier bankruptcy: **permanent** until new supplier qualified (8-16 weeks).",
        }
        st.info(model_hints[sa_type])

        if run_sa:
            with st.spinner("Analyzing disruption impact..."):
                res = mock_executor.execute("get_supply_disruption_impact", {
                    "disruption_type": sa_type,
                    "affected_entity": sa_entity,
                    "duration_days": sa_duration,
                    "affected_skus": sa_skus_all or None,
                })

            if res.get("error"):
                st.error(res["error"])
            else:
                d = res["data"]
                crit = d.get("critical_count", 0)
                warn = d.get("warning_count", 0)

                if crit > 0:
                    st.error(f"CRITICAL: {crit} location(s) hit stockout before replenishment arrives. Immediate action required.")
                elif warn > 0:
                    st.warning(f"WARNING: {warn} location(s) in danger zone. Expedited replenishment recommended.")
                else:
                    st.success("All monitored locations within safety stock parameters.")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("SKUs Affected", len(d.get("affected_skus", [])))
                m2.metric("Critical Locations", crit)
                m3.metric("Warning Locations", warn)
                m4.metric("Revenue at Risk", f"${d.get('total_revenue_at_risk_usd', 0):,.0f}")

                # Stockout chart
                risks = d.get("stockout_risks", [])
                if risks:
                    risk_df = pd.DataFrame(risks)
                    fig_r = px.bar(
                        risk_df, x="store_id", y="days_to_store_stockout",
                        color="severity",
                        color_discrete_map={"critical": "#e74c3c", "warning": "#f39c12", "ok": "#2ecc71"},
                        title="Days to Store Stockout", labels={"days_to_store_stockout": "Days on Hand"},
                        height=320,
                    )
                    fig_r.add_hline(y=4, line_dash="dash", line_color="orange",
                                    annotation_text="Replenishment lag (4d)", annotation_position="top right")
                    fig_r.add_hline(y=0, line_color="red", line_width=1)
                    st.plotly_chart(fig_r, use_container_width=True)

                # Alternate carriers
                alts = d.get("alternates_by_region", {})
                if alts:
                    st.subheader("Alternate Carriers by Region")
                    alt_cols = st.columns(len(alts))
                    for col, (region, options) in zip(alt_cols, alts.items()):
                        with col:
                            st.markdown(f"**Region: {region}**")
                            if not options:
                                st.markdown('<span class="chip-red">No viable alternates</span>',
                                            unsafe_allow_html=True)
                            for opt in options:
                                if opt.get("available"):
                                    st.markdown(
                                        f'<span class="chip-green">✓ {opt["carrier_id"]}</span> '
                                        f'+{opt.get("additional_lead_time_days", 0)}d, '
                                        f'{opt.get("capacity_pct", 0)}% cap, '
                                        f'+{opt.get("cost_premium_pct", 0):.0f}% cost',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<span class="chip-red">✗ {opt["carrier_id"]}</span> '
                                        f'{opt.get("reason_unavailable", "")}',
                                        unsafe_allow_html=True
                                    )

                # Mitigation plan
                st.subheader("Mitigation Plan")
                for i, step in enumerate(d.get("mitigation_plan", []), 1):
                    if "IMMEDIATE" in step:
                        st.error(f"**{i}.** {step}")
                    elif "STRATEGIC" in step:
                        st.info(f"**{i}.** {step}")
                    else:
                        st.markdown(f"**{i}.** {step}")

                # ── Save to history ──
                import datetime as _dt
                severity_label = "CRITICAL" if crit > 0 else ("WARNING" if warn > 0 else "OK")
                st.session_state.hist_supply.append({
                    "header": f"{sa_entity} — {sa_type.replace('_', ' ').title()}",
                    "meta": f"{sa_duration}d duration  |  {len(d.get('affected_skus', []))} SKU(s) affected",
                    "result": f"{severity_label}: {crit} critical, {warn} warning  |  Rev@Risk: ${d.get('total_revenue_at_risk_usd', 0):,.0f}",
                    "ts": _dt.datetime.now().strftime("%H:%M"),
                })


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEMAND FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader("Demand Forecast — 15-Variable Model")
    st.caption("Demand is a class function, not a time series. Price, promo, tariffs, weather, and 11 more variables all move the signal.")

    fc_main, fc_hist = st.columns([2, 1])

    with fc_hist:
        _render_history(st.session_state.hist_forecast, "Run a forecast to see history.")
        _render_formula_panel("forecast")
        _render_data_sources("forecast")

    with fc_main:
        col1, col2 = st.columns([1, 1])
        with col1:
            fc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="fc_sku")
            fc_horizon = st.slider("Forecast Horizon (weeks)", 1, 32, 8, key="fc_horizon")
        with col2:
            st.markdown("**Override demand variables**")
            fc_c1, fc_c2 = st.columns(2)
            fc_price  = fc_c1.number_input("Price change %", value=0.0, step=1.0, key="fc_price")
            fc_promo  = fc_c2.number_input("Promo intensity %", value=0.0, step=5.0, key="fc_promo")
            fc_tariff = fc_c1.number_input("Tariff increase %", value=0.0, step=1.0, key="fc_tariff")
            fc_weather = fc_c2.number_input("Weather impact (0=none)", value=0.0, step=1.0, key="fc_weather")

        if st.button("Run Forecast", type="primary", key="fc_run"):
            overrides = {k: v for k, v in {
                "price": fc_price, "promo": fc_promo,
                "tariff": fc_tariff, "weather": fc_weather
            }.items() if v != 0}

            with st.spinner("Running 15-variable demand model..."):
                res = mock_executor.execute("get_demand_forecast", {
                    "sku": fc_sku, "horizon_weeks": fc_horizon,
                    "variable_overrides": overrides or None,
                })
                acc_res = mock_executor.execute("get_forecast_accuracy", {"sku": fc_sku, "horizon_weeks": fc_horizon})

            if res.get("error"):
                st.error(res["error"])
            else:
                d = res["data"]
                acc = d["forecast_accuracy_mape"]
                reliable = d["is_reliable"]

                # Accuracy gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=acc * 100,
                    title={"text": "Forecast Accuracy (MAPE)", "font": {"size": 16}},
                    delta={"reference": 85, "valueformat": ".1f", "suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ecc71" if acc >= 0.85 else "#f39c12" if acc >= 0.70 else "#e74c3c"},
                        "steps": [
                            {"range": [0, 60], "color": "#fadbd8"},
                            {"range": [60, 70], "color": "#fdebd0"},
                            {"range": [70, 85], "color": "#fef9e7"},
                            {"range": [85, 100], "color": "#eafaf1"},
                        ],
                        "threshold": {"line": {"color": "#2c3e50", "width": 3}, "value": 85},
                    },
                    number={"suffix": "%", "valueformat": ".1f"},
                ))
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))

                col_g, col_note = st.columns([1, 2])
                with col_g:
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with col_note:
                    st.markdown(f"**SKU:** `{fc_sku}` — {d['sku_name']}")
                    st.markdown(f"**Horizon:** {fc_horizon} weeks")
                    if not reliable:
                        st.error(
                            f"UNRELIABLE ({acc*100:.1f}%). DO NOT pass to PO system. "
                            "Accuracy is below 60% minimum threshold."
                        )
                    elif acc < 0.70:
                        st.warning(f"Below 70% threshold — use with caution. Widen safety stock.")
                    else:
                        st.success(f"Acceptable accuracy. Suitable as planning input.")

                    acc_d = acc_res.get("data", {})
                    st.markdown(
                        f"**Gap to benchmark (85%):** {round((0.85 - acc) * 100, 1)}pts  \n"
                        f"**Revenue impact:** ~{round((0.85 - acc) * 100, 1)}% revenue efficiency loss"
                    )
                    st.caption(d.get("ci_note", ""))

                # Fan chart
                points = d.get("weekly_forecast", [])
                if points:
                    wks  = [p["week"] for p in points]
                    pts  = [p["point_estimate"] for p in points]
                    l80  = [p["ci_80_lower"] for p in points]
                    u80  = [p["ci_80_upper"] for p in points]
                    l95  = [p["ci_95_lower"] for p in points]
                    u95  = [p["ci_95_upper"] for p in points]

                    fig_fc = go.Figure()
                    fig_fc.add_trace(go.Scatter(x=wks + wks[::-1], y=u95 + l95[::-1],
                                                 fill="toself", fillcolor="rgba(52,152,219,0.08)",
                                                 line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
                    fig_fc.add_trace(go.Scatter(x=wks + wks[::-1], y=u80 + l80[::-1],
                                                 fill="toself", fillcolor="rgba(52,152,219,0.22)",
                                                 line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
                    fig_fc.add_trace(go.Scatter(x=wks, y=pts, mode="lines+markers",
                                                 line=dict(color="#2980b9", width=2.5),
                                                 marker=dict(size=5), name="Point Estimate"))
                    if fc_horizon > 4:
                        fig_fc.add_vline(x=4, line_dash="dot", line_color="#adb5bd",
                                         annotation_text="CI widens past week 4", annotation_position="top right")
                    fig_fc.update_layout(
                        title=f"Demand Forecast — {d['sku_name']} ({fc_horizon}W)",
                        xaxis_title="Week", yaxis_title="Units / Store / Week",
                        height=360, margin=dict(l=40, r=30, t=55, b=40),
                    )
                    st.plotly_chart(fig_fc, use_container_width=True)

                # Variable contributions
                contribs = {k: v for k, v in d.get("variable_contributions_units_per_week", {}).items() if v != 0}
                if contribs:
                    cdf = pd.DataFrame(list(contribs.items()), columns=["Variable", "Impact"]).sort_values("Impact")
                    fig_c = px.bar(cdf, x="Impact", y="Variable", orientation="h",
                                   color="Impact", color_continuous_scale=["#e74c3c", "#ecf0f1", "#2ecc71"],
                                   color_continuous_midpoint=0,
                                   title="Demand Variable Contributions (units/store/week)",
                                   height=max(300, len(contribs) * 35))
                    fig_c.update_layout(margin=dict(l=10, r=30, t=45, b=30))
                    st.plotly_chart(fig_c, use_container_width=True)

                st.info(d.get("accuracy_note", ""))

                # ── Save to history ──
                import datetime as _dt
                rel = "Reliable" if d.get("is_reliable") else "UNRELIABLE"
                st.session_state.hist_forecast.append({
                    "header": f"{fc_sku}  |  {fc_horizon}W horizon",
                    "meta": f"Accuracy: {acc*100:.1f}%  |  {rel}",
                    "result": f"Point est: {d['weekly_forecast'][0]['point_estimate']:.0f} units/store/wk",
                    "ts": _dt.datetime.now().strftime("%H:%M"),
                })


# TAB 6 — SCENARIO PLANNER
# ════════════════════════════════════════════════════════════════════════════
with tab_scenario:
    st.subheader("Scenario Planner")
    st.caption("Compare up to 4 scenarios and detect conflicts. All require a time anchor — comparisons without dates are invalid.")

    sc_main, sc_hist = st.columns([2, 1])

    with sc_hist:
        _render_history(st.session_state.hist_scenario, "Run a scenario comparison to see history.")
        _render_formula_panel("scenario")
        _render_data_sources("scenario")

    with sc_main:
        sc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="sc_sku")
        sc_horizon = st.slider("Horizon (weeks)", 4, 32, 8, key="sc_horizon")
        base_price = mock_executor.PRODUCTS.get(sc_sku, {}).get("base_price", 12.99)

        st.markdown("**Define Scenarios**")
        cols = st.columns(4)
        labels = ["Baseline", "Scenario B", "Scenario C", "Scenario D"]
        defaults = [
            {"price": base_price, "promo": 0.0, "sup_red": 0.0, "tariff": 0.0,
             "ddir": "neutral", "sdir": "neutral"},
            {"price": base_price + 1.5, "promo": 0.0, "sup_red": 0.0, "tariff": 0.0,
             "ddir": "decrease", "sdir": "neutral"},
            {"price": base_price - 1.0, "promo": 15.0, "sup_red": 0.0, "tariff": 0.0,
             "ddir": "increase", "sdir": "neutral"},
            {"price": base_price, "promo": 0.0, "sup_red": 30.0, "tariff": 5.0,
             "ddir": "decrease", "sdir": "constrain"},
        ]

        scenarios, scenario_meta = [], []
        for i, (col, label, df) in enumerate(zip(cols, labels, defaults)):
            with col:
                st.markdown(f"**{label}**")
                price    = st.number_input("Price ($)", value=float(df["price"]), min_value=0.01, step=0.10, key=f"sc_p{i}")
                promo    = st.number_input("Promo %", value=float(df["promo"]), step=5.0, key=f"sc_pr{i}")
                sup_red  = st.number_input("Supply cut %", value=float(df["sup_red"]), min_value=0.0, max_value=100.0, step=5.0, key=f"sc_sr{i}")
                tariff   = st.number_input("Tariff cost %", value=float(df["tariff"]), step=1.0, key=f"sc_ta{i}")
                sdate    = st.date_input("Start date", key=f"sc_sd{i}")

                scenarios.append({"name": label, "price": price, "promo_uplift_pct": promo,
                                  "supply_reduction_pct": sup_red, "tariff_additional_cost_pct": tariff})
                scenario_meta.append({
                    "name": label,
                    "scenario_type": ("supply_disruption" if sup_red > 0 else "promotion" if promo > 0 else "price_change"),
                    "start_date": sdate.isoformat(),
                    "horizon_days": sc_horizon * 7,
                    "demand_direction": df["ddir"],
                    "supply_direction": df["sdir"],
                })

        if st.button("Compare + Detect Conflicts", type="primary", key="sc_run"):
            with st.spinner("Running scenario comparison and conflict detection..."):
                comp_res = mock_executor.execute("run_scenario_comparison",
                                                 {"sku": sc_sku, "scenarios": scenarios, "horizon_weeks": sc_horizon})
                conf_res = mock_executor.execute("detect_scenario_conflicts", {"scenarios": scenario_meta})

            # Conflicts first
            for conflict in conf_res.get("data", {}).get("conflicts", []):
                sev = conflict["severity"]
                icon = "CRITICAL" if sev == "critical" else "WARNING"
                msg = f"**{icon}:** '{conflict['scenario_a']}' + '{conflict['scenario_b']}' — {conflict['description']}\n\n**Rec:** {conflict['recommendation']}"
                (st.error if sev == "critical" else st.warning)(msg)

            if not conf_res.get("data", {}).get("conflicts"):
                st.success("No conflicts detected between the provided scenarios.")

            cd = comp_res.get("data", {})
            rows = cd.get("comparison", [])
            if rows:
                sc_df = pd.DataFrame(rows)
                col_l, col_r = st.columns(2)
                with col_l:
                    fig_r = px.bar(sc_df, x="scenario", y="total_revenue_usd", color="scenario",
                                   title="Total Revenue by Scenario", height=300)
                    st.plotly_chart(fig_r, use_container_width=True)
                with col_r:
                    fig_m = px.bar(sc_df, x="scenario", y="total_margin_usd", color="scenario",
                                   title="Total Margin by Scenario", height=300)
                    st.plotly_chart(fig_m, use_container_width=True)

                disp_cols = ["scenario", "price", "demand_per_store_week",
                             "weekly_revenue_usd", "total_revenue_usd", "total_margin_usd"]
                st.dataframe(sc_df[disp_cols], use_container_width=True, hide_index=True)
                st.success(cd.get("recommendation", ""))

                # ── Save to history ──
                import datetime as _dt
                best = max(rows, key=lambda r: r["total_revenue_usd"])
                n_conflicts = len(conf_res.get("data", {}).get("conflicts", []))
                st.session_state.hist_scenario.append({
                    "header": f"{sc_sku}  |  {sc_horizon}W  |  {len(rows)} scenarios",
                    "meta": f"Conflicts detected: {n_conflicts}",
                    "result": f"Best rev: {best['scenario']}  ${best['total_revenue_usd']:,.0f}",
                    "ts": _dt.datetime.now().strftime("%H:%M"),
                })


# TAB 7 — SHELF & STORE REPLENISHMENT
# ════════════════════════════════════════════════════════════════════════════
with tab_shelf:
    st.subheader("Shelf & Store Replenishment")
    st.caption("HQ → DC → Store with 3-4d lag, lead-time variability, perishable caps, and planogram constraints.")

    sh_main, sh_hist = st.columns([2, 1])

    with sh_hist:
        _render_history(st.session_state.hist_shelf, "Run a replenishment analysis to see history.")
        _render_formula_panel("shelf")
        _render_data_sources("shelf")

    with sh_main:
        col1, col2, col3 = st.columns(3)
        with col1:
            sh_sku   = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="sh_sku")
            sh_store = st.selectbox("Store", [f"STR-{i:03d}" for i in range(1, 31)], key="sh_store")
        with col2:
            sh_qty      = st.number_input("Proposed Replenishment Units", value=48, min_value=1, max_value=500, key="sh_qty")
            sh_priority = st.selectbox("Priority", ["standard", "expedited", "emergency"], key="sh_priority",
                                       help="standard=4d / expedited=2d (2x cost) / emergency=1d (3x cost)")
        with col3:
            sh_extra = st.number_input("Supply disruption extra delay (days)", value=0, min_value=0, key="sh_extra")
            st.markdown("")
            run_sh = st.button("Analyze & Recommend", type="primary", key="sh_run")

        if run_sh:
            store_meta = mock_executor.STORES.get(sh_store, mock_executor.STORES["STR-001"])
            dc_id = store_meta["dc"]
            prod_p = mock_executor.PRODUCTS.get(sh_sku, {})

            with st.spinner("Checking stockout risk, shelf capacity, perishable status..."):
                risk_res = mock_executor.execute("calculate_stockout_risk",
                                                 {"sku": sh_sku, "location_id": sh_store,
                                                  "supply_disruption_additional_days": sh_extra})
                cap_res  = mock_executor.execute("check_shelf_capacity",
                                                 {"sku": sh_sku, "store_id": sh_store,
                                                  "proposed_replenishment_units": sh_qty})
                per_res  = mock_executor.execute("check_perishable_status",
                                                 {"sku": sh_sku, "location_id": sh_store,
                                                  "proposed_replenishment_units": sh_qty}) if prod_p.get("perishable") else None
                rep_res  = mock_executor.execute("trigger_replenishment",
                                                 {"sku": sh_sku, "from_location": dc_id,
                                                  "to_location": sh_store, "quantity": sh_qty,
                                                  "priority": sh_priority})

            rd = risk_res.get("data", {})
            sev = rd.get("severity", "ok")
            (st.error if sev == "critical" else st.warning if sev == "warning" else st.success)(
                rd.get("recommendation", "")
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current Inventory", f"{rd.get('current_inventory_units', '?')} units")
            m2.metric("Days on Hand", f"{rd.get('days_on_hand', '?')}d")
            m3.metric("Effective Lag", f"{rd.get('effective_lag_days', '?')}d")
            m4.metric("Buffer Above Lag", f"{rd.get('days_above_lag', '?')}d",
                      delta_color="inverse" if (rd.get("days_above_lag", 1) or 1) < 0 else "normal")

            # Gantt
            chain = [
                {"Step": "HQ Signal Generated", "Start": 0, "Duration": 1},
                {"Step": "DC Pick & Ship", "Start": 1, "Duration": 2},
                {"Step": "Store Receive & Shelf", "Start": 3, "Duration": 1},
            ]
            if sh_extra > 0:
                chain.append({"Step": f"Disruption Delay (+{sh_extra}d)", "Start": 4, "Duration": sh_extra})
            cost_mult = {"standard": 1.0, "expedited": 2.0, "emergency": 3.0}[sh_priority]
            base_d = pd.Timestamp("2026-01-01")
            chain_df = pd.DataFrame([{
                "Task": c["Step"],
                "Start": base_d + pd.Timedelta(days=c["Start"]),
                "Finish": base_d + pd.Timedelta(days=c["Start"] + c["Duration"]),
            } for c in chain])
            fig_g = px.timeline(chain_df, x_start="Start", x_end="Finish", y="Task",
                                 color="Task", title=f"Replenishment Chain — {sh_store} ({sh_priority})",
                                 height=280)
            fig_g.update_yaxes(autorange="reversed")
            fig_g.update_layout(margin=dict(l=10, r=10, t=50, b=20))
            st.plotly_chart(fig_g, use_container_width=True)

            # Shelf cap + perishable
            col_c, col_p = st.columns(2)
            cd = cap_res.get("data", {})
            with col_c:
                st.markdown("**Planogram Check**")
                st.markdown(cd.get("recommendation", ""))
                if cd.get("overflow_to_back_storage_units", 0) > 0:
                    st.warning(f"{cd['overflow_to_back_storage_units']} units → back-of-store staging.")
            with col_p:
                if per_res:
                    pd_data = per_res.get("data", {})
                    st.markdown("**Perishable Cap Check**")
                    (st.error if pd_data.get("exceeds_perishable_cap") else st.success)(
                        pd_data.get("recommendation", "")
                    )
                    if pd_data.get("exceeds_perishable_cap"):
                        st.metric("Write-off Risk", f"${pd_data.get('write_off_risk_usd', 0):,.2f}")
                else:
                    st.markdown("**Perishable Cap Check**")
                    st.info("Not applicable — non-perishable SKU.")

            rep_d = rep_res.get("data", {})
            st.info(rep_d.get("confirmation", ""))
            st.metric("Freight Cost", f"${rep_d.get('freight_cost_usd', 0):,.2f}",
                      delta=f"{cost_mult}× standard rate")

            # ── Save to history ──
            import datetime as _dt
            st.session_state.hist_shelf.append({
                "header": f"{sh_sku}  →  {sh_store}  |  {sh_qty} units",
                "meta": f"Priority: {sh_priority}  |  Severity: {sev}",
                "result": f"DoH: {rd.get('days_on_hand','?')}d  |  Freight: ${rep_d.get('freight_cost_usd',0):,.0f}",
                "ts": _dt.datetime.now().strftime("%H:%M"),
            })


# TAB 8 — FINANCIAL IMPACT
# ════════════════════════════════════════════════════════════════════════════
with tab_finance:
    st.subheader("Financial Impact Calculator")
    st.caption("Revenue, margin, vendor trade dollars, VMI split, carrying cost. Tax flagged as approximate.")

    fi_main, fi_hist = st.columns([2, 1])

    with fi_hist:
        _render_history(st.session_state.hist_finance, "Run a P&L calculation to see history.")
        _render_formula_panel("finance")
        _render_data_sources("finance")

    with fi_main:
        col1, col2 = st.columns(2)
        with col1:
            fi_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="fi_sku")
            fi_prod = mock_executor.PRODUCTS.get(fi_sku, {})
            fi_old_p = st.number_input("Old Price ($)", value=float(fi_prod.get("base_price", 12.99)),
                                        min_value=0.01, step=0.10, key="fi_op")
            fi_new_p = st.number_input("New Price ($)", value=float(fi_prod.get("base_price", 12.99)) + 1.50,
                                        min_value=0.01, step=0.10, key="fi_np")
        with col2:
            fi_old_v = st.number_input("Old Volume (units, total horizon)", value=12000.0, step=100.0, key="fi_ov")
            fi_new_v = st.number_input("New Volume (units, total horizon)", value=10056.0, step=100.0, key="fi_nv")
            fi_jx    = st.selectbox("Tax Jurisdiction", ["US", "US-WA", "US-GA", "US-IL"], key="fi_jx")
            fi_trade = st.checkbox("Include vendor trade dollars", value=True, key="fi_trade")

        if st.button("Calculate P&L Impact", type="primary", key="fi_run"):
            with st.spinner("Calculating..."):
                rev_res  = mock_executor.execute("calculate_revenue_impact", {
                    "sku": fi_sku, "old_price": fi_old_p, "new_price": fi_new_p,
                    "old_volume_units": fi_old_v, "new_volume_units": fi_new_v,
                    "include_trade_dollars": fi_trade, "jurisdiction": fi_jx,
                })
                excess   = max(0.0, fi_old_v - fi_new_v)
                car_res  = mock_executor.execute("calculate_carrying_cost", {
                    "sku": fi_sku, "excess_units": excess, "carrying_weeks": 4.0,
                })

            rd = rev_res.get("data", {})
            cd = car_res.get("data", {})

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Gross Revenue Δ", f"${rd.get('gross_revenue_change_usd', 0):+,.0f}")
            k2.metric("Vendor Trade Offset", f"${rd.get('vendor_trade_offset_usd', 0):+,.0f}")
            k3.metric("Net Revenue Δ",    f"${rd.get('net_revenue_change_usd', 0):+,.0f}")
            k4.metric("Margin Δ",         f"${rd.get('margin_change_usd', 0):+,.0f}")

            fig_pl = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "total", "relative", "relative", "total"],
                x=["Gross Revenue", "Trade Dollars", "Net Revenue",
                   "Margin", "Tax (est.)", "Carrying Cost"],
                y=[
                    rd.get("gross_revenue_change_usd", 0),
                    rd.get("vendor_trade_offset_usd", 0),
                    rd.get("net_revenue_change_usd", 0),
                    rd.get("margin_change_usd", 0),
                    -rd.get("tax_on_margin_usd", 0),
                    -cd.get("total_carrying_cost_usd", 0),
                ],
                connector={"line": {"color": "#dee2e6"}},
                increasing={"marker": {"color": "#2ecc71"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                totals={"marker": {"color": "#3498db"}},
                texttemplate="%{y:+,.0f}",
                textposition="outside",
            ))
            fig_pl.update_layout(title="P&L Waterfall", height=380,
                                 margin=dict(l=40, r=40, t=55, b=40))
            st.plotly_chart(fig_pl, use_container_width=True)

            col_n1, col_n2 = st.columns(2)
            with col_n1:
                st.caption(f"Tax: {rd.get('tax_note', '')}")
                st.info(rd.get("trade_dollar_note", ""))
            with col_n2:
                st.caption(rd.get("vmi_carrying_note", ""))
                if excess > 0:
                    st.metric("Inventory Carrying Cost (4W)", f"${cd.get('total_carrying_cost_usd', 0):,.2f}",
                              help=f"On {int(cd.get('owned_units', 0))} owned units")

            # ── Save to history ──
            import datetime as _dt
            st.session_state.hist_finance.append({
                "header": f"{fi_sku}  ${fi_old_p:.2f} → ${fi_new_p:.2f}",
                "meta": f"Vol: {fi_old_v:.0f} → {fi_new_v:.0f} units",
                "result": f"Net Rev Δ: ${rd.get('net_revenue_change_usd',0):+,.0f}  |  Margin Δ: ${rd.get('margin_change_usd',0):+,.0f}",
                "ts": _dt.datetime.now().strftime("%H:%M"),
            })


# TAB 9 — DATA SOURCES
# ════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.subheader("Data Sources & Provenance")
    st.caption("Know which data is real-time vs batch before making operational decisions.")

    sources = [
        {"System": "OLTP (Transactional DB)", "Lag": "~5 min", "Use For": "Pricing, PO creation, financial posting",
         "Risk": "Low", "Tools Used": "simulate_price_change, calculate_revenue_impact"},
        {"System": "WMS (Warehouse Mgmt)", "Lag": "~15 min", "Use For": "Operational inventory, stockout, replenishment",
         "Risk": "Low-Medium", "Tools Used": "get_inventory_levels, calculate_stockout_risk"},
        {"System": "OLAP (Analytics DW)", "Lag": "24 hours", "Use For": "Trend analysis, financial reporting",
         "Risk": "HIGH for ops — use WMS instead", "Tools Used": "get_demand_forecast, get_forecast_accuracy"},
        {"System": "Carrier API", "Lag": "15–30 min", "Use For": "Carrier status, alternate availability",
         "Risk": "Medium — strikes may be delayed", "Tools Used": "get_carrier_status, find_alternate_carriers"},
        {"System": "Competitor Feed", "Lag": "4–6 hr", "Use For": "Competitive pricing context",
         "Risk": "Medium — treat as indicative", "Tools Used": "get_competitive_pricing"},
    ]
    st.dataframe(pd.DataFrame(sources), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Session Freshness Warnings")
    warnings = st.session_state.get("freshness_warnings", [])
    if warnings:
        for w in set(warnings):
            st.warning(w, icon="⏱")
    else:
        st.info("No freshness warnings yet. Run a query in the Chat tab.")

    st.divider()
    st.subheader("WMS vs OLAP Inventory Check")
    st.caption("Flags where the 24h batch and 15min WMS diverge — always use WMS for operational decisions.")

    disc_sku = st.selectbox("SKU", list(mock_executor.PRODUCTS.keys()), key="disc_sku")
    rows = []
    for dc_id, dc in mock_executor.DCS.items():
        inv = dc["inventory"].get(disc_sku, {"wms": 0, "olap": 0})
        diff = inv["wms"] - inv["olap"]
        diff_pct = round(diff / inv["olap"] * 100, 1) if inv["olap"] > 0 else 0
        rows.append({
            "DC": dc_id, "Region": dc["region"], "Location": dc["name"].split("—")[1].strip(),
            "WMS (15min)": inv["wms"], "OLAP (24h)": inv["olap"],
            "Diff (units)": diff, "Diff %": diff_pct,
            "Recommendation": "Use WMS" if abs(diff_pct) > 2 else "Either",
        })

    disc_df = pd.DataFrame(rows)
    st.dataframe(disc_df, use_container_width=True, hide_index=True)

    fig_d = px.bar(
        disc_df.melt(id_vars="DC", value_vars=["WMS (15min)", "OLAP (24h)"],
                     var_name="Source", value_name="Inventory"),
        x="DC", y="Inventory", color="Source", barmode="group",
        title=f"WMS vs OLAP — {disc_sku}",
        color_discrete_map={"WMS (15min)": "#2ecc71", "OLAP (24h)": "#3498db"},
        height=300,
    )
    st.plotly_chart(fig_d, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 10 — WORKFLOW (Integrated Step-by-Step Scenario Builder)
# ════════════════════════════════════════════════════════════════════════════
with tab_workflow:
    st.subheader("Integrated Scenario Workflow Builder")
    st.caption(
        "Walk through a structured decision flow — pick your trigger, context, and objective, "
        "then let the AI run the full cascade analysis and present ranked options with ripple-effect timelines."
    )

    wf_main, wf_hist = st.columns([2, 1])

    with wf_hist:
        _render_history(st.session_state.hist_workflow, "Run a workflow analysis to see history.")

    with wf_main:
        # ── Step 1: Business Trigger ──
        st.markdown("### Step 1 — Business Trigger")
        st.markdown("What event is driving this decision?")

        trigger_options = {
            "🔺 Price Change": "price_change",
            "🚛 Supply Disruption": "supply_disruption",
            "📉 Demand Shift": "demand_shift",
            "📦 Tariff / Cost Increase": "tariff",
            "🎄 Seasonal Event": "seasonal",
            "⚔️ Competitive Move": "competitive",
        }

        trigger_cols = st.columns(3)
        selected_trigger = st.session_state.get("wf_trigger", "🔺 Price Change")

        for i, label in enumerate(trigger_options):
            col = trigger_cols[i % 3]
            with col:
                is_selected = selected_trigger == label
                btn_style = (
                    "background:#0071ce;color:white;border:none;padding:10px;width:100%;border-radius:8px;cursor:pointer;font-weight:600;"
                    if is_selected else
                    "background:#f8f9fa;color:#212529;border:1px solid #dee2e6;padding:10px;width:100%;border-radius:8px;cursor:pointer;"
                )
                if st.button(label, key=f"wf_trigger_{label}", use_container_width=True):
                    st.session_state["wf_trigger"] = label
                    st.rerun()

        selected_trigger = st.session_state.get("wf_trigger", "🔺 Price Change")
        trigger_key = trigger_options[selected_trigger]

        st.divider()

        # ── Step 2: Context ──
        st.markdown("### Step 2 — Context")
        wf_col1, wf_col2, wf_col3 = st.columns(3)

        with wf_col1:
            sku_labels = {
                "HUG48-3": "Huggies Size 3 (Diapers)",
                "MLK-GAL": "Whole Milk Gallon",
                "CIG-PKT": "Marlboro Cigarettes",
                "OJ-64": "Tropicana OJ 64oz",
                "FORMULA-24": "Similac Formula 24pk",
            }
            wf_sku = st.selectbox(
                "SKU / Product",
                options=list(sku_labels.keys()),
                format_func=lambda x: sku_labels[x],
                key="wf_sku",
            )

        with wf_col2:
            wf_region = st.selectbox(
                "Region",
                ["All Regions", "SE — Southeast", "NW — Northwest", "NE — Northeast", "SW — Southwest"],
                key="wf_region",
            )

        with wf_col3:
            wf_horizon = st.selectbox(
                "Horizon",
                ["2 weeks (operational)", "4 weeks (tactical)", "8 weeks (strategic)"],
                key="wf_horizon",
            )

        # Trigger-specific inputs
        st.markdown("#### Trigger Parameters")

        if trigger_key == "price_change":
            p_col1, p_col2 = st.columns(2)
            base_price = mock_executor.PRODUCTS.get(wf_sku, {}).get("base_price", 10.0)
            with p_col1:
                wf_current_price = st.number_input(
                    "Current Price ($)", value=float(base_price), step=0.01, key="wf_cur_price"
                )
            with p_col2:
                wf_new_price = st.number_input(
                    "New Price ($)", value=float(base_price * 1.10), step=0.01, key="wf_new_price"
                )
            pct_change = round((wf_new_price - wf_current_price) / wf_current_price * 100, 1)
            direction = "increase" if pct_change > 0 else "decrease"
            st.caption(f"→ {abs(pct_change):.1f}% price {direction}")

        elif trigger_key == "supply_disruption":
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                wf_carrier = st.selectbox(
                    "Affected Carrier",
                    ["TruckCo A", "TruckCo B", "TruckCo C", "TruckCo D"],
                    index=1,
                    key="wf_carrier",
                )
            with d_col2:
                wf_disruption_days = st.slider("Expected Duration (days)", 1, 30, 14, key="wf_disrupt_days")

        elif trigger_key == "demand_shift":
            wf_demand_pct = st.slider(
                "Demand Change (%)", -30, 30, -10, key="wf_demand_pct",
                help="Negative = demand drop, Positive = demand surge"
            )
            wf_demand_driver = st.selectbox(
                "Primary Driver",
                ["Weather event", "Competitor promotion", "Demographic shift", "Media coverage", "Seasonal"],
                key="wf_demand_driver",
            )

        elif trigger_key == "tariff":
            wf_tariff_pct = st.slider("Cost Increase (%)", 1, 50, 25, key="wf_tariff_pct")
            st.caption("The AI will model how much of this cost can be passed to consumers vs. absorbed in margin.")

        elif trigger_key == "seasonal":
            wf_season = st.selectbox(
                "Seasonal Event",
                ["Back-to-School (Aug)", "Thanksgiving (Nov)", "Christmas (Dec)", "Super Bowl (Feb)", "Summer (Jun-Aug)"],
                key="wf_season",
            )
            wf_uplift_pct = st.slider("Expected Demand Uplift (%)", 5, 50, 20, key="wf_uplift_pct")

        elif trigger_key == "competitive":
            wf_competitor = st.selectbox(
                "Competitor",
                ["Costco", "Target", "Amazon", "Kroger", "Aldi"],
                key="wf_competitor",
            )
            wf_comp_pct = st.slider("Competitor Price Change (%)", -30, 0, -15, key="wf_comp_pct")

        st.divider()

        # ── Step 3: Decision Objective ──
        st.markdown("### Step 3 — Decision Objective")
        st.caption("Select one or more objectives to optimize for. The AI will rank options accordingly.")

        obj_col1, obj_col2, obj_col3, obj_col4 = st.columns(4)
        obj_revenue = obj_col1.checkbox("Maximize Revenue", value=True, key="wf_obj_rev")
        obj_margin = obj_col2.checkbox("Maximize Margin", value=True, key="wf_obj_margin")
        obj_service = obj_col3.checkbox("Service Level / Availability", value=False, key="wf_obj_service")
        obj_cost = obj_col4.checkbox("Minimize Cost", value=False, key="wf_obj_cost")

        objectives = []
        if obj_revenue: objectives.append("revenue maximization")
        if obj_margin: objectives.append("margin maximization")
        if obj_service: objectives.append("service level / stockout prevention")
        if obj_cost: objectives.append("cost minimization")
        if not objectives:
            objectives = ["revenue maximization"]

        st.divider()

        # ── Step 4: Run Analysis ──
        st.markdown("### Step 4 — Run AI Analysis")

        # Build the query from selections
        def _build_workflow_query():
            sku_name = sku_labels.get(wf_sku, wf_sku)
            region_str = wf_region.split(" — ")[0] if " — " in wf_region else wf_region
            horizon_str = wf_horizon.split(" ")[0] + " " + wf_horizon.split(" ")[1]
            obj_str = " and ".join(objectives)

            if trigger_key == "price_change":
                return (
                    f"Analyze a price change for {sku_name} ({wf_sku}) from ${wf_current_price:.2f} "
                    f"to ${wf_new_price:.2f} ({'+' if pct_change > 0 else ''}{pct_change:.1f}%) "
                    f"in the {region_str} region over a {horizon_str} horizon. "
                    f"Optimize for: {obj_str}. "
                    f"Cover: demand elasticity (asymmetric if applicable), inventory and replenishment impact, "
                    f"carrier capacity requirements, financial margin with vendor trade netting, "
                    f"and scenario conflict detection. "
                    f"Provide 3 ranked options (proceed / modify / defer) with trade-offs."
                )
            elif trigger_key == "supply_disruption":
                return (
                    f"{wf_carrier} is experiencing a supply disruption expected to last {wf_disruption_days} days, "
                    f"affecting {sku_name} ({wf_sku}) in the {region_str} region. "
                    f"Optimize for: {obj_str} over {horizon_str}. "
                    f"Analyze: stockout risk by DC, alternate carrier options with cost and coverage gaps, "
                    f"revenue at risk, and a mitigation plan with ranked options."
                )
            elif trigger_key == "demand_shift":
                driver = st.session_state.get("wf_demand_driver", "external event")
                change = st.session_state.get("wf_demand_pct", -10)
                direction_word = "increase" if change > 0 else "decrease"
                return (
                    f"Demand for {sku_name} ({wf_sku}) in {region_str} is expected to {direction_word} "
                    f"by {abs(change)}% due to {driver} over the next {horizon_str}. "
                    f"Optimize for: {obj_str}. "
                    f"Analyze: inventory adequacy, replenishment adjustment needs, "
                    f"financial impact, and 3 ranked response options."
                )
            elif trigger_key == "tariff":
                tariff = st.session_state.get("wf_tariff_pct", 25)
                return (
                    f"A {tariff}% cost increase (tariff/duty) is being applied to {sku_name} ({wf_sku}). "
                    f"Region: {region_str}. Horizon: {horizon_str}. Optimize for: {obj_str}. "
                    f"Model: cost pass-through scenarios (full / partial / absorb), "
                    f"demand elasticity impact, margin erosion, and vendor trade dollar adjustments. "
                    f"Provide 3 ranked response strategies."
                )
            elif trigger_key == "seasonal":
                season = st.session_state.get("wf_season", "Seasonal event")
                uplift = st.session_state.get("wf_uplift_pct", 20)
                return (
                    f"Plan for {season} seasonal demand uplift of {uplift}% for {sku_name} ({wf_sku}) "
                    f"in {region_str} over {horizon_str}. Optimize for: {obj_str}. "
                    f"Analyze: inventory build requirements, replenishment schedule, "
                    f"carrier capacity needs, and risk of over-ordering vs. stockout. "
                    f"Provide 3 ranked inventory strategies."
                )
            elif trigger_key == "competitive":
                competitor = st.session_state.get("wf_competitor", "Competitor")
                comp_pct = st.session_state.get("wf_comp_pct", -15)
                return (
                    f"{competitor} has reduced their price for a comparable product to {sku_name} ({wf_sku}) "
                    f"by {abs(comp_pct)}% in {region_str}. Horizon: {horizon_str}. Optimize for: {obj_str}. "
                    f"Analyze: expected demand cannibalization, price match vs. differentiate options, "
                    f"margin impact of matching, and 3 ranked competitive response strategies."
                )
            return f"Analyze the supply chain situation for {sku_name} in {region_str} over {horizon_str}."

        wf_query = _build_workflow_query()

        with st.expander("Preview AI Query", expanded=False):
            st.text(wf_query)

        run_col1, run_col2 = st.columns([2, 1])
        with run_col1:
            run_workflow = st.button(
                "Run Analysis (V2 — LangGraph)",
                type="primary",
                use_container_width=True,
                key="wf_run_btn",
            )
        with run_col2:
            wf_pipeline = st.selectbox(
                "Pipeline",
                ["V2 — LangGraph", "V1 — Agentic Loop"],
                key="wf_pipeline_sel",
                label_visibility="collapsed",
            )

        if run_workflow:
            use_v2_wf = "V2" in wf_pipeline
            wf_result = None

            with st.status("Running workflow analysis...", expanded=True) as wf_status:
                try:
                    if use_v2_wf:
                        st.write("→ Routing query through LangGraph nodes...")
                        from agents.langgraph_flow import run_langgraph
                        wf_result = run_langgraph(wf_query, [])
                        st.write(f"→ Nodes executed: {', '.join(wf_result.get('node_outputs', {}).keys())}")
                        wf_status.update(label="Analysis complete", state="complete")
                    else:
                        st.write("→ Running V1 agentic loop...")
                        orch = _get_orchestrator(_get_api_key())
                        wf_result = orch.run(wf_query, history=[], max_iterations=10)
                        wf_status.update(label="Analysis complete", state="complete")
                except Exception as exc:
                    wf_status.update(label=f"Error: {exc}", state="error")
                    st.error(str(exc))

            if wf_result:
                st.session_state["wf_last_result"] = wf_result
                st.session_state["wf_last_query"] = wf_query
                st.session_state["wf_last_trigger"] = trigger_key
                st.session_state["wf_last_sku"] = wf_sku

                # ── Save to history ──
                import datetime as _dt
                resp_preview = (wf_result.get("response_text") or wf_result.get("response") or "")[:100]
                n_nodes = len(wf_result.get("node_outputs", {}))
                n_tools_wf = len(wf_result.get("tool_calls_made") or wf_result.get("tool_calls") or [])
                st.session_state.hist_workflow.append({
                    "header": f"{selected_trigger}  |  {wf_sku}",
                    "meta": f"Region: {wf_region.split(' — ')[0]}  |  Horizon: {wf_horizon.split(' ')[0]+' '+wf_horizon.split(' ')[1]}  |  {wf_pipeline}",
                    "result": (f"{n_nodes} nodes, {n_tools_wf} tools" if n_nodes else f"{n_tools_wf} tools") + (f"  — {resp_preview[:80]}…" if resp_preview else ""),
                    "ts": _dt.datetime.now().strftime("%H:%M"),
                })

        # ── Step 5: Results ──
        wf_last = st.session_state.get("wf_last_result")
        if wf_last:
            st.divider()
            st.markdown("### Step 5 — Analysis Results")

            response_text = wf_last.get("response", "")
            if response_text:
                st.markdown(response_text)

            # Tool calls summary
            tool_calls = wf_last.get("tool_calls", [])
            if tool_calls:
                with st.expander(f"Tools Used ({len(tool_calls)})", expanded=False):
                    tc_rows = []
                    for tc in tool_calls:
                        tc_rows.append({
                            "Tool": tc.get("tool", ""),
                            "Input": str(tc.get("input", ""))[:80],
                            "Result Preview": str(tc.get("result", ""))[:80],
                        })
                    if tc_rows:
                        st.dataframe(pd.DataFrame(tc_rows), use_container_width=True, hide_index=True)

            # Node trace (V2 only)
            node_outputs = wf_last.get("node_outputs", {})
            if node_outputs:
                with st.expander("LangGraph Node Trace", expanded=False):
                    for node_name, node_out in node_outputs.items():
                        st.markdown(f"**{node_name}** → {str(node_out)[:200]}")

            st.divider()

            # ── Step 6: Ripple Effect Timeline ──
            st.markdown("### Step 6 — Decision Ripple Effect (Day 0 → 30)")
            st.caption("How the selected trigger propagates across the supply chain over 30 days.")

            last_trigger = st.session_state.get("wf_last_trigger", "price_change")
            last_sku = st.session_state.get("wf_last_sku", "HUG48-3")
            prod = mock_executor.PRODUCTS.get(last_sku, {})

            # Build timeline events based on trigger type
            if last_trigger == "price_change":
                elasticity = prod.get("elasticity", -1.2)
                is_asym = prod.get("asymmetric", False)
                new_p = st.session_state.get("wf_new_price", prod.get("base_price", 10) * 1.10)
                cur_p = st.session_state.get("wf_cur_price", prod.get("base_price", 10))
                pct = (new_p - cur_p) / cur_p
                demand_impact = pct * elasticity * 100
                if is_asym and pct < 0:
                    demand_impact *= prod.get("recovery_factor", 0.70)

                timeline_events = [
                    {"Day": 0, "Domain": "Pricing", "Event": f"Price changes: ${cur_p:.2f} → ${new_p:.2f}"},
                    {"Day": 1, "Domain": "Demand", "Event": f"Demand model updates: {demand_impact:+.1f}% volume shift"},
                    {"Day": 1, "Domain": "Finance", "Event": "Revenue forecast recalculated"},
                    {"Day": 2, "Domain": "Inventory", "Event": "Reorder point recalculated from new demand signal"},
                    {"Day": 3, "Domain": "Supply Chain", "Event": "Replenishment PO adjusted and transmitted to DC"},
                    {"Day": 4, "Domain": "Supply Chain", "Event": "HQ→DC leg of replenishment order in transit"},
                    {"Day": 7, "Domain": "Inventory", "Event": "DC receives adjusted stock; WMS updated"},
                    {"Day": 10, "Domain": "Inventory", "Event": "Store shelves reflect new inventory level"},
                    {"Day": 14, "Domain": "Finance", "Event": "14-day margin vs. forecast reconciliation"},
                    {"Day": 30, "Domain": "Finance", "Event": "Month-end P&L close: actual vs. projected impact"},
                ]
            elif last_trigger == "supply_disruption":
                dur = st.session_state.get("wf_disrupt_days", 14)
                timeline_events = [
                    {"Day": 0, "Domain": "Supply Chain", "Event": f"Disruption confirmed — carrier unavailable"},
                    {"Day": 1, "Domain": "Inventory", "Event": "DC-level days-of-supply clock starts ticking"},
                    {"Day": 1, "Domain": "Supply Chain", "Event": "Alternate carrier evaluation begins"},
                    {"Day": 2, "Domain": "Finance", "Event": "Revenue-at-risk quantified per region"},
                    {"Day": 2, "Domain": "Supply Chain", "Event": "Alternate carrier contracted (if available)"},
                    {"Day": 4, "Domain": "Inventory", "Event": "First DC hits safety stock threshold"},
                    {"Day": 5, "Domain": "Inventory", "Event": "Emergency replenishment via alternate carrier dispatched"},
                    {"Day": 7, "Domain": "Inventory", "Event": "Alternate carrier delivers first shipment (+45% cost)"},
                    {"Day": dur, "Domain": "Supply Chain", "Event": "Disruption resolved — primary carrier resumes"},
                    {"Day": dur + 3, "Domain": "Inventory", "Event": "Inventory normalized; DCs restocked"},
                    {"Day": 30, "Domain": "Finance", "Event": "Total disruption cost reconciled vs. revenue saved"},
                ]
            else:
                timeline_events = [
                    {"Day": 0, "Domain": "Strategy", "Event": "Decision triggered"},
                    {"Day": 1, "Domain": "Demand", "Event": "Demand model receives new signal"},
                    {"Day": 2, "Domain": "Inventory", "Event": "Inventory plan adjusted"},
                    {"Day": 3, "Domain": "Supply Chain", "Event": "Replenishment updated"},
                    {"Day": 7, "Domain": "Finance", "Event": "7-day financial impact measurement"},
                    {"Day": 30, "Domain": "Finance", "Event": "Month-end reconciliation"},
                ]

            timeline_df = pd.DataFrame(timeline_events)
            domain_colors = {
                "Pricing": "#0071ce",
                "Demand": "#f39c12",
                "Inventory": "#27ae60",
                "Supply Chain": "#e74c3c",
                "Finance": "#8e44ad",
                "Strategy": "#2980b9",
            }

            # Gantt-style scatter plot
            fig_timeline = go.Figure()

            for _, row in timeline_df.iterrows():
                color = domain_colors.get(row["Domain"], "#95a5a6")
                fig_timeline.add_trace(go.Scatter(
                    x=[row["Day"]],
                    y=[row["Domain"]],
                    mode="markers+text",
                    marker=dict(size=16, color=color, symbol="circle"),
                    text=[f"Day {row['Day']}"],
                    textposition="top center",
                    name=row["Domain"],
                    hovertemplate=f"<b>Day {row['Day']} — {row['Domain']}</b><br>{row['Event']}<extra></extra>",
                    showlegend=False,
                ))

            # Add invisible scatter for each domain to create connected lines
            for domain, grp in timeline_df.groupby("Domain"):
                color = domain_colors.get(domain, "#95a5a6")
                if len(grp) > 1:
                    fig_timeline.add_trace(go.Scatter(
                        x=grp["Day"].tolist(),
                        y=[domain] * len(grp),
                        mode="lines",
                        line=dict(color=color, width=2, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

            fig_timeline.update_layout(
                title="Decision Ripple Effect — Day 0 to 30",
                xaxis_title="Day",
                yaxis_title="Domain",
                height=380,
                xaxis=dict(range=[-1, 32]),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Event table
            with st.expander("Full Event Timeline", expanded=False):
                st.dataframe(timeline_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════════
# TAB 11 — FLOW MAP (LangGraph DAG Visualization)
# ════════════════════════════════════════════════════════════════════════════
with tab_flowmap:
    st.subheader("LangGraph Flow Map")
    st.caption(
        "Visual representation of the V2 LangGraph multi-agent graph. "
        "Green nodes = executed in your last Chat query. Gray = available but not invoked."
    )

    fm_main, fm_hist = st.columns([3, 1])

    with fm_hist:
        st.markdown('<div class="hist-panel"><h4>📋 Execution Log</h4>', unsafe_allow_html=True)
        chat_msgs_fm = [m for m in st.session_state.conversation_history if isinstance(m.get("content"), str)]
        user_msgs = [m["content"] for m in chat_msgs_fm if m["role"] == "user"]
        wf_runs = list(reversed(st.session_state.hist_workflow))
        all_entries = []
        for q in reversed(user_msgs[-8:]):
            q_short = q[:70] + ("…" if len(q) > 70 else "")
            all_entries.append(("Chat", q_short))
        for w in wf_runs[:5]:
            all_entries.append(("Workflow", w.get("header", "")))

        if not all_entries:
            st.markdown('<div class="hist-empty">Run queries in Chat or Workflow tabs — node execution will appear here.</div>', unsafe_allow_html=True)
        else:
            tc_map = {}
            for tc in st.session_state.get("last_tool_calls", []):
                tc_map[tc.get("tool", "")] = True
            node_count = len([n for n in [
                "price_cascade","supply_disruption","demand_forecast",
                "scenario_planning","shelf_replenishment","inventory_node",
                "carrier_node","accuracy_node","perishable_check","financial_impact"
            ] if any(t in tc_map for t in [n, n.replace("_node",""), n.replace("_"," ")])])

            for source, label in all_entries[:8]:
                badge = "Chat" if source == "Chat" else "WF"
                badge_color = "#0071ce" if source == "Chat" else "#28a745"
                st.markdown(
                    f'<div class="hist-item">'
                    f'<div class="hist-item-header" style="display:flex;gap:6px;align-items:center;">'
                    f'<span style="background:{badge_color};color:white;font-size:10px;padding:1px 6px;border-radius:4px;">{badge}</span>'
                    f'{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)

    with fm_main:
      # Node positions (x, y) — center of each node box
        NODE_POSITIONS = {
            "router":             (5.0, 9.5),
            "price_cascade":      (1.0, 7.5),
            "supply_disruption":  (3.0, 7.5),
            "demand_forecast":    (5.0, 7.5),
            "scenario_planning":  (7.0, 7.5),
            "shelf_replenishment":(9.0, 7.5),
            "inventory_node":     (1.0, 5.5),
            "carrier_node":       (3.0, 5.5),
            "accuracy_node":      (5.0, 5.5),
            "perishable_check":   (9.0, 5.5),
            "financial_impact":   (2.0, 3.5),
            "synthesizer":        (5.0, 1.5),
        }

        NODE_LABELS = {
            "router":             "ROUTER\n(intent + entity)",
            "price_cascade":      "Price\nCascade",
            "supply_disruption":  "Supply\nDisruption",
            "demand_forecast":    "Demand\nForecast",
            "scenario_planning":  "Scenario\nPlanning",
            "shelf_replenishment":"Shelf\nReplenishment",
            "inventory_node":     "Inventory\nNode",
            "carrier_node":       "Carrier\nNode",
            "accuracy_node":      "Accuracy\nGate",
            "perishable_check":   "Perishable\nCheck",
            "financial_impact":   "Financial\nImpact",
            "synthesizer":        "SYNTHESIZER\n(final response)",
        }

        NODE_TOOLS = {
            "router":             "—",
            "price_cascade":      "simulate_price_change\nget_competitive_pricing\nadjust_promotional_price",
            "supply_disruption":  "get_carrier_status\nget_dc_inventory",
            "demand_forecast":    "get_demand_forecast\nanalyze_demand_variables",
            "scenario_planning":  "detect_scenario_conflicts\ncompare_scenarios",
            "shelf_replenishment":"get_replenishment_schedule\ncalculate_stockout_risk",
            "inventory_node":     "get_inventory_levels\nget_reorder_recommendations",
            "carrier_node":       "find_alternate_carriers\ncalculate_revenue_impact",
            "accuracy_node":      "get_forecast_accuracy",
            "perishable_check":   "get_inventory_levels\ncalculate_stockout_risk",
            "financial_impact":   "calculate_revenue_impact\ncalculate_carrying_cost",
            "synthesizer":        "—",
        }

        # Directed edges (from_node, to_node)
        EDGES = [
            ("router", "price_cascade"),
            ("router", "supply_disruption"),
            ("router", "demand_forecast"),
            ("router", "scenario_planning"),
            ("router", "shelf_replenishment"),
            ("price_cascade", "inventory_node"),
            ("supply_disruption", "carrier_node"),
            ("demand_forecast", "accuracy_node"),
            ("shelf_replenishment", "perishable_check"),
            ("inventory_node", "financial_impact"),
            ("carrier_node", "synthesizer"),
            ("accuracy_node", "synthesizer"),
            ("perishable_check", "synthesizer"),
            ("financial_impact", "synthesizer"),
            ("scenario_planning", "synthesizer"),
            ("price_cascade", "financial_impact"),
        ]

        # Determine which nodes were executed in last Chat query
        executed_nodes = set()
        last_chat_result = None
        # Check session state for node_outputs from last Chat query
        for msg in reversed(st.session_state.get("conversation_history", [])):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                break
        # Use last_tool_calls to infer which nodes ran
        last_tool_calls_wf = st.session_state.get("last_tool_calls", [])
        if last_tool_calls_wf:
            executed_nodes.add("router")
            executed_nodes.add("synthesizer")
            # Infer nodes from tools called
            tool_to_node = {
                "simulate_price_change": "price_cascade",
                "get_competitive_pricing": "price_cascade",
                "adjust_promotional_price": "price_cascade",
                "get_carrier_status": "supply_disruption",
                "get_dc_inventory": "supply_disruption",
                "find_alternate_carriers": "carrier_node",
                "get_demand_forecast": "demand_forecast",
                "analyze_demand_variables": "demand_forecast",
                "get_forecast_accuracy": "accuracy_node",
                "detect_scenario_conflicts": "scenario_planning",
                "compare_scenarios": "scenario_planning",
                "get_replenishment_schedule": "shelf_replenishment",
                "get_inventory_levels": "inventory_node",
                "get_reorder_recommendations": "inventory_node",
                "calculate_stockout_risk": "inventory_node",
                "calculate_revenue_impact": "financial_impact",
                "calculate_carrying_cost": "financial_impact",
            }
            for tc in last_tool_calls_wf:
                tool_name = tc.get("tool", "")
                if tool_name in tool_to_node:
                    executed_nodes.add(tool_to_node[tool_name])
                    # Add downstream nodes based on execution chain
                    node = tool_to_node[tool_name]
                    if node == "price_cascade":
                        executed_nodes.add("inventory_node")
                        executed_nodes.add("financial_impact")
                    elif node == "supply_disruption":
                        executed_nodes.add("carrier_node")
                    elif node == "demand_forecast":
                        executed_nodes.add("accuracy_node")
                    elif node == "shelf_replenishment":
                        executed_nodes.add("perishable_check")

        # Color map
        def _node_color(node: str) -> str:
            if node in executed_nodes:
                return "#27ae60"   # green — executed
            return "#bdc3c7"       # gray — not executed

        def _node_text_color(node: str) -> str:
            if node in executed_nodes:
                return "white"
            return "#2c3e50"

        # Build Plotly figure
        fig_flow = go.Figure()

        # Draw edges first (behind nodes)
        for src, dst in EDGES:
            x0, y0 = NODE_POSITIONS[src]
            x1, y1 = NODE_POSITIONS[dst]
            edge_color = "#27ae60" if (src in executed_nodes and dst in executed_nodes) else "#dee2e6"
            edge_width = 2.5 if (src in executed_nodes and dst in executed_nodes) else 1.5

            fig_flow.add_annotation(
                x=x1, y=y1,
                ax=x0, ay=y0,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=edge_width,
                arrowcolor=edge_color,
            )

        # Draw nodes as scatter markers with text
        for node, (x, y) in NODE_POSITIONS.items():
            color = _node_color(node)
            text_color = _node_text_color(node)
            label = NODE_LABELS[node]
            tools_hint = NODE_TOOLS[node]

            # Large invisible marker for hover area
            fig_flow.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=60,
                    color=color,
                    line=dict(color="#2c3e50" if node in executed_nodes else "#adb5bd", width=2),
                    symbol="square",
                ),
                text=[label],
                textposition="middle center",
                textfont=dict(
                    size=10,
                    color=text_color,
                    family="monospace",
                ),
                name=node,
                hovertemplate=(
                    f"<b>{node}</b><br>"
                    f"Status: {'✅ Executed' if node in executed_nodes else '⬜ Not invoked'}<br>"
                    f"Tools:<br>{tools_hint.replace(chr(10), '<br>')}<extra></extra>"
                ),
                showlegend=False,
            ))

        # Legend annotation
        fig_flow.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="#27ae60", symbol="square"),
            name="Executed",
            showlegend=True,
        ))
        fig_flow.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color="#bdc3c7", symbol="square"),
            name="Not invoked",
            showlegend=True,
        ))

        fig_flow.update_layout(
            title=dict(
                text="LangGraph Multi-Agent Graph — V2 Pipeline",
                font=dict(size=16),
            ),
            xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[0.5, 11.0], showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor="white",
            paper_bgcolor="#f8f9fa",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig_flow, use_container_width=True)

        if executed_nodes:
            st.success(
                f"**Last query executed {len(executed_nodes)} nodes:** "
                + ", ".join(f"`{n}`" for n in sorted(executed_nodes))
            )
        else:
            st.info(
                "No query has been run yet. Run a question in the **Chat** tab, "
                "then return here to see which nodes were executed."
            )

        st.divider()

        # ── Architecture explanation ──
        st.markdown("### How the Graph Works")

        arch_col1, arch_col2 = st.columns(2)

        with arch_col1:
            st.markdown("""
    **Entry Point: Router**
    - Classifies query intent into one of 5 paths
    - Extracts SKU, region, and scenario entities
    - Sets `route` in state to direct flow

    **Domain Nodes (parallel candidates)**
    - `price_cascade` → handles pricing decisions
    - `supply_disruption` → handles carrier/supply events
    - `demand_forecast` → handles volume predictions
    - `scenario_planning` → handles multi-scenario comparison
    - `shelf_replenishment` → handles store-level replenishment

    **Supporting Nodes (depth layer)**
    - `inventory_node` → follows price_cascade
    - `carrier_node` → follows supply_disruption
    - `accuracy_node` → follows demand_forecast
    - `perishable_check` → follows shelf_replenishment
    - `financial_impact` → follows inventory_node
            """)

        with arch_col2:
            st.markdown("""
    **Exit Node: Synthesizer**
    - Receives outputs from all upstream nodes
    - Merges domain findings into a single coherent response
    - Applies provenance warnings (OLAP/WMS/OLTP freshness)
    - Returns final structured recommendation

    **Key Design Choices**
    - Each node has a specialized system prompt (focused scope)
    - Each node only receives the tools relevant to its domain
    - Conditional edges enforce domain-specific chains
    - State is accumulated via `add_messages` annotation
    - Synthesizer can access all prior node context

    **Execution Paths by Intent**
    | Intent | Path |
    |--------|------|
    | Price change | router → price_cascade → inventory → financial → synthesizer |
    | Supply disruption | router → supply_disruption → carrier → synthesizer |
    | Demand forecast | router → demand_forecast → accuracy → synthesizer |
    | Shelf replenishment | router → shelf_replenishment → perishable → synthesizer |
    | Scenario planning | router → scenario_planning → synthesizer |
            """)

        st.divider()
        st.markdown("### Node Tool Assignments")

        tool_rows = []
        for node, tools in NODE_TOOLS.items():
            status = "Executed" if node in executed_nodes else "Available"
            tool_rows.append({
                "Node": node,
                "Status": status,
                "Tools": tools.replace("\n", ", ") if tools != "—" else "—",
            })
        st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 12 — NETWORK GRAPH (Enhanced LangGraph DAG)
# ════════════════════════════════════════════════════════════════════════════
with tab_netgraph:
    import datetime as _dt

    st.subheader("🕸 Network Graph — LangGraph Agent Architecture")
    st.caption(
        "Logically layered multi-agent DAG. "
        "Run a query in Chat or Workflow, then use **Step-Through** to replay how data flowed."
    )

    ng_main, ng_hist = st.columns([3, 1])

    # ── Build execution context from session ──
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
                if nd == "price_cascade":   _exec_nodes.update(["inventory_node", "financial_impact"])
                if nd == "supply_disruption": _exec_nodes.add("carrier_node")
                if nd == "demand_forecast":  _exec_nodes.add("accuracy_node")
                if nd == "shelf_replenishment": _exec_nodes.add("perishable_check")

    # ── Layered positions ──
    # Layer 0 (top): User Input  Layer 1: Router  Layer 2: Domain nodes
    # Layer 3: Support nodes  Layer 4: Financial  Layer 5: Synthesizer  Layer 6: Output
    _LAYERS = {
        "user_input":         (5.0, 10.0),
        "router":             (5.0,  8.5),
        "price_cascade":      (1.5,  7.0),
        "supply_disruption":  (3.2,  7.0),
        "demand_forecast":    (5.0,  7.0),
        "scenario_planning":  (6.8,  7.0),
        "shelf_replenishment":(8.5,  7.0),
        "inventory_node":     (1.5,  5.5),
        "carrier_node":       (3.2,  5.5),
        "accuracy_node":      (5.0,  5.5),
        "perishable_check":   (8.5,  5.5),
        "financial_impact":   (2.3,  4.0),
        "synthesizer":        (5.0,  2.5),
        "output":             (5.0,  1.0),
    }
    _LABELS = {
        "user_input":         "👤 User Query",
        "router":             "🔀 Router\n(intent + entity)",
        "price_cascade":      "💰 Price\nCascade",
        "supply_disruption":  "🚛 Supply\nDisruption",
        "demand_forecast":    "📈 Demand\nForecast",
        "scenario_planning":  "⚖️ Scenario\nPlanning",
        "shelf_replenishment":"🏪 Shelf\nReplenishment",
        "inventory_node":     "📦 Inventory\nNode",
        "carrier_node":       "🛤 Carrier\nNode",
        "accuracy_node":      "🎯 Accuracy\nGate",
        "perishable_check":   "🥛 Perishable\nCheck",
        "financial_impact":   "💵 Financial\nImpact",
        "synthesizer":        "🔮 Synthesizer",
        "output":             "📋 Response",
    }
    _DOMAIN_COLOR = {
        "user_input":         "#6c757d",
        "router":             "#0071ce",
        "price_cascade":      "#e67e22",
        "supply_disruption":  "#e74c3c",
        "demand_forecast":    "#27ae60",
        "scenario_planning":  "#8e44ad",
        "shelf_replenishment":"#16a085",
        "inventory_node":     "#d35400",
        "carrier_node":       "#c0392b",
        "accuracy_node":      "#1e8449",
        "perishable_check":   "#117a65",
        "financial_impact":   "#1a5276",
        "synthesizer":        "#0071ce",
        "output":             "#6c757d",
    }
    _TOOLS_HINT = {
        "user_input": "Natural language query",
        "router": "Classifies intent → routes to domain",
        "price_cascade": "simulate_price_change\nadjust_promotional_price\nget_competitive_pricing",
        "supply_disruption": "get_carrier_status\nget_supply_disruption_impact",
        "demand_forecast": "get_demand_forecast\nanalyze_demand_variables",
        "scenario_planning": "run_scenario_comparison\ndetect_scenario_conflicts",
        "shelf_replenishment": "check_shelf_capacity\ntrigger_replenishment",
        "inventory_node": "get_inventory_levels\ncalculate_stockout_risk",
        "carrier_node": "find_alternate_carriers",
        "accuracy_node": "get_forecast_accuracy",
        "perishable_check": "check_perishable_status",
        "financial_impact": "calculate_revenue_impact\ncalculate_carrying_cost",
        "synthesizer": "Merges all node outputs → final response",
        "output": "Structured recommendation delivered",
    }
    _EDGES_NG = [
        ("user_input", "router"),
        ("router", "price_cascade"), ("router", "supply_disruption"),
        ("router", "demand_forecast"), ("router", "scenario_planning"),
        ("router", "shelf_replenishment"),
        ("price_cascade", "inventory_node"), ("price_cascade", "financial_impact"),
        ("supply_disruption", "carrier_node"),
        ("demand_forecast", "accuracy_node"),
        ("shelf_replenishment", "perishable_check"),
        ("inventory_node", "financial_impact"),
        ("carrier_node", "synthesizer"), ("accuracy_node", "synthesizer"),
        ("perishable_check", "synthesizer"), ("financial_impact", "synthesizer"),
        ("scenario_planning", "synthesizer"),
        ("synthesizer", "output"),
    ]

    # ── Step-through controls ──
    all_steps = ["all"] + sorted(_exec_nodes) if _exec_nodes else ["all"]
    if "ng_step" not in st.session_state:
        st.session_state["ng_step"] = "all"

    with ng_hist:
        st.markdown('<div class="hist-panel"><h4>📋 Execution Log</h4>', unsafe_allow_html=True)
        if not _exec_nodes:
            st.markdown('<div class="hist-empty">Run a Chat or Workflow query first — active nodes will appear here.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hist-item"><div class="hist-item-header">Active nodes: {len(_exec_nodes)}</div>', unsafe_allow_html=True)
            for nd in sorted(_exec_nodes):
                col_hex = _DOMAIN_COLOR.get(nd, "#6c757d")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
                    f'<span style="width:10px;height:10px;border-radius:50%;background:{col_hex};display:inline-block;"></span>'
                    f'<span style="font-size:12px;">{_LABELS.get(nd, nd)}</span></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)
        wf_hist_items = list(reversed(st.session_state.hist_workflow))[:4]
        chat_qs = [m["content"][:60] for m in st.session_state.conversation_history if m.get("role") == "user"][-4:]
        for q in reversed(chat_qs):
            st.markdown(f'<div class="hist-item"><div class="hist-item-meta">💬 {q}…</div></div>', unsafe_allow_html=True)
        for w in wf_hist_items:
            st.markdown(f'<div class="hist-item"><div class="hist-item-meta">🔄 {w.get("header","")}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ng_main:
        # Step-through mode controls
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2, 1, 1, 1])
        with ctrl_c1:
            view_mode = st.radio(
                "View mode",
                ["Full graph", "Execution path only", "Step-through"],
                horizontal=True, key="ng_view_mode",
            )
        with ctrl_c2:
            show_labels = st.toggle("Tool hints", value=True, key="ng_show_tools")
        with ctrl_c3:
            show_layers = st.toggle("Layer bands", value=True, key="ng_layers")
        with ctrl_c4:
            if st.button("🔄 Reset step", key="ng_reset"):
                st.session_state["ng_step"] = "all"

        # Step slider (only in step-through mode)
        step_node = None
        if view_mode == "Step-through" and _exec_nodes:
            exec_order = ["user_input", "router"] + [
                n for n in ["price_cascade","supply_disruption","demand_forecast",
                             "scenario_planning","shelf_replenishment",
                             "inventory_node","carrier_node","accuracy_node","perishable_check",
                             "financial_impact","synthesizer","output"]
                if n in _exec_nodes or n in ("synthesizer", "output")
            ]
            step_idx = st.slider(
                "Execution step",
                min_value=0, max_value=len(exec_order) - 1,
                value=st.session_state.get("ng_step_idx", 0),
                format="Step %d", key="ng_step_slider",
            )
            st.session_state["ng_step_idx"] = step_idx
            step_node = exec_order[step_idx] if step_idx < len(exec_order) else None
            if step_node:
                hint = _TOOLS_HINT.get(step_node, "")
                st.info(f"**Step {step_idx}: `{step_node}`** — {hint.replace(chr(10), ', ')}")

        # ── Build Plotly figure ──
        fig_ng = go.Figure()

        # Layer background bands
        if show_layers:
            layer_bands = [
                (9.3, 10.7, "rgba(108,117,125,0.05)", "Input"),
                (7.7,  9.2, "rgba(0,113,206,0.06)",   "Router"),
                (6.2,  7.6, "rgba(230,126,34,0.06)",   "Domain Nodes"),
                (4.7,  6.1, "rgba(39,174,96,0.06)",    "Support Nodes"),
                (3.2,  4.6, "rgba(26,82,118,0.06)",    "Financial"),
                (1.7,  3.1, "rgba(0,113,206,0.06)",    "Synthesizer"),
                (0.2,  1.6, "rgba(108,117,125,0.05)",  "Output"),
            ]
            for y0, y1, color, label in layer_bands:
                fig_ng.add_hrect(
                    y0=y0, y1=y1, fillcolor=color, line_width=0,
                    annotation_text=label,
                    annotation_position="right",
                    annotation_font=dict(size=10, color="#adb5bd"),
                )

        # Determine which nodes to show
        if view_mode == "Execution path only" and _exec_nodes:
            visible_nodes = _exec_nodes | {"user_input", "output"}
            visible_edges = [(s, d) for s, d in _EDGES_NG if s in visible_nodes and d in visible_nodes]
        else:
            visible_nodes = set(_LAYERS.keys())
            visible_edges = _EDGES_NG

        # Draw edges
        for src, dst in visible_edges:
            x0, y0 = _LAYERS[src]
            x1, y1 = _LAYERS[dst]
            both_exec = src in _exec_nodes and dst in _exec_nodes
            step_active = (step_node and (src == step_node or dst == step_node))

            if view_mode == "Step-through":
                if step_active:
                    edge_color, edge_width, dash = "#f39c12", 3.5, "solid"
                elif both_exec:
                    edge_color, edge_width, dash = "#27ae60", 2.5, "solid"
                else:
                    edge_color, edge_width, dash = "#dee2e6", 1.0, "dot"
            else:
                edge_color = "#27ae60" if both_exec else "#dee2e6"
                edge_width = 2.5 if both_exec else 1.2
                dash = "solid"

            fig_ng.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.3,
                arrowwidth=edge_width, arrowcolor=edge_color,
            )

        # Draw nodes
        for node in visible_nodes:
            x, y = _LAYERS[node]
            base_color = _DOMAIN_COLOR.get(node, "#6c757d")
            is_exec = node in _exec_nodes
            is_step = (step_node == node)

            if view_mode == "Step-through":
                if is_step:
                    fill_color, border_color, border_width, opacity = "#f39c12", "#e67e22", 4, 1.0
                elif is_exec:
                    fill_color, border_color, border_width, opacity = base_color, "white", 2, 1.0
                else:
                    fill_color, border_color, border_width, opacity = "#ecf0f1", "#bdc3c7", 1, 0.5
            else:
                if is_exec:
                    fill_color, border_color, border_width, opacity = base_color, "white", 2.5, 1.0
                else:
                    fill_color, border_color, border_width, opacity = "#ecf0f1", "#bdc3c7", 1, 0.85

            text_color = "white" if (is_exec or is_step) else "#495057"
            node_size = 72 if is_step else (60 if is_exec else 48)

            hint = _TOOLS_HINT.get(node, "").replace("\n", "<br>") if show_labels else ""
            status_txt = "⚡ Active step" if is_step else ("✅ Executed" if is_exec else "⬜ Available")

            fig_ng.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(
                    size=node_size, color=fill_color, opacity=opacity,
                    line=dict(color=border_color, width=border_width),
                    symbol="square",
                ),
                text=[_LABELS.get(node, node)],
                textposition="middle center",
                textfont=dict(size=9 if node_size < 60 else 10, color=text_color, family="Arial"),
                name=node,
                hovertemplate=(
                    f"<b>{_LABELS.get(node, node)}</b><br>"
                    f"Status: {status_txt}<br>"
                    + (f"Tools:<br>{hint}<br>" if hint else "")
                    + "<extra></extra>"
                ),
                showlegend=False,
            ))

        # Legend
        for label, color in [("Executed", "#27ae60"), ("Active Step", "#f39c12"), ("Available", "#bdc3c7")]:
            fig_ng.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=label, showlegend=True,
            ))

        fig_ng.update_layout(
            title=dict(text="LangGraph Multi-Agent Network — Layered Architecture", font=dict(size=15)),
            xaxis=dict(range=[0, 10.5], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[0.2, 11.2], showgrid=False, zeroline=False, showticklabels=False),
            height=640,
            plot_bgcolor="white",
            paper_bgcolor="#f8f9fa",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=10, r=120, t=60, b=20),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        st.plotly_chart(fig_ng, use_container_width=True)

        # ── Live execution summary ──
        if _exec_nodes:
            ex_c1, ex_c2, ex_c3, ex_c4 = st.columns(4)
            ex_c1.metric("Nodes executed", len(_exec_nodes))
            ex_c2.metric("Tool calls", len(_last_tcs))
            domain_nodes = [n for n in _exec_nodes if n in (
                "price_cascade","supply_disruption","demand_forecast","scenario_planning","shelf_replenishment")]
            ex_c3.metric("Domain nodes", len(domain_nodes))
            support_nodes = [n for n in _exec_nodes if n in (
                "inventory_node","carrier_node","accuracy_node","perishable_check","financial_impact")]
            ex_c4.metric("Support nodes", len(support_nodes))

            with st.expander("📋 Node-level tool breakdown", expanded=False):
                rows_ng = []
                for nd in sorted(_exec_nodes):
                    tcs_for_node = [tc for tc in _last_tcs if _TOOL_TO_NODE.get(tc.get("tool","")) == nd]
                    rows_ng.append({
                        "Node": nd,
                        "Domain": _LABELS.get(nd, nd).replace("\n", " "),
                        "Tools called": len(tcs_for_node),
                        "Tool names": ", ".join(tc["tool"] for tc in tcs_for_node) or "—",
                        "Provenance": ", ".join(set(tc.get("provenance","?") for tc in tcs_for_node)) or "—",
                    })
                if rows_ng:
                    st.dataframe(pd.DataFrame(rows_ng), use_container_width=True, hide_index=True)
        else:
            st.info("No query has been run yet. Run a question in the **Chat** tab, then return here to see the execution trace.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 13 — SCENARIO BUILDER (Drag & Drop Interactive Canvas)
# ════════════════════════════════════════════════════════════════════════════
with tab_builder:
    st.subheader("🧩 Scenario Builder — Interactive Decision Canvas")
    st.caption("Click **+ Add** on any scenario block to place it on the canvas. Adjust impact values with sliders and click **▶ Run Analysis** for full financial breakdown.")

    # ── Session state ──
    if "builder_blocks" not in st.session_state:
        st.session_state.builder_blocks = []

    _BLK_DEF = {
        "PRICING": [
            ("price-up","Price Increase",8.0,5.0,-14.0,"Med","#388bfd"),
            ("price-dn","Price Cut",-3.0,-8.0,12.0,"Low","#388bfd"),
            ("edlp","EDLP Rollback",1.0,-3.0,8.0,"Low","#388bfd"),
            ("markdown","Markdown Clearance",-5.0,-10.0,22.0,"Med","#388bfd"),
            ("price-match","Competitor Match",-2.0,-5.0,6.0,"Med","#388bfd"),
        ],
        "PROMOTIONS": [
            ("tpr","Temp Price Reduction",4.0,-6.0,20.0,"Low","#3fb950"),
            ("display","Display / End Cap",6.0,-2.0,15.0,"Low","#3fb950"),
            ("bogo","BOGO Event",3.0,-12.0,30.0,"Low","#3fb950"),
            ("coupon","Digital Coupon Drop",2.0,-8.0,18.0,"Low","#3fb950"),
            ("promo","Shopper Marketing",5.0,-3.0,12.0,"Low","#3fb950"),
        ],
        "SUPPLY CHAIN": [
            ("strike","Carrier Strike",-12.0,-9.0,0.0,"Crit","#f85149"),
            ("port-delay","Port Delay",-7.0,-6.0,0.0,"High","#f85149"),
            ("dc-closure","DC Closure",-10.0,-8.0,0.0,"Crit","#f85149"),
            ("supplier-bk","Supplier Bankruptcy",-15.0,-12.0,0.0,"Crit","#f85149"),
            ("prestock","Pre-Build Inventory",3.0,-2.0,5.0,"Low","#f85149"),
            ("alt-carrier","Alt Carrier Switch",-2.0,-3.0,0.0,"Med","#f85149"),
        ],
        "DEMAND SIGNALS": [
            ("surge","Demand Surge",15.0,10.0,20.0,"Low","#2ea043"),
            ("demand-drop","Demand Drop",-10.0,-7.0,-15.0,"High","#2ea043"),
            ("seasonal-pk","Seasonal Peak",12.0,8.0,18.0,"Med","#2ea043"),
            ("weather-ev","Weather Event",-6.0,-5.0,-10.0,"High","#2ea043"),
            ("new-item","New Item Launch",20.0,5.0,25.0,"Med","#2ea043"),
            ("comp-oos","Competitor OOS",8.0,6.0,10.0,"Low","#2ea043"),
        ],
        "TRADE & EXTERNAL": [
            ("tariff","Tariff Increase",-4.0,-6.0,-8.0,"High","#d29922"),
            ("fx-impact","FX / Currency Move",-3.0,-4.0,-5.0,"Med","#d29922"),
            ("regulatory","Regulatory Change",-5.0,-4.0,0.0,"High","#d29922"),
            ("comp-move","Competitor Promo",-3.0,-3.0,-8.0,"Med","#d29922"),
            ("inflation","Input Cost Inflation",-2.0,-8.0,-4.0,"High","#d29922"),
        ],
        "OPERATIONS": [
            ("safety-stk","Safety Stock Raise",-1.0,-2.0,3.0,"Med","#58a6ff"),
            ("vmi-switch","VMI Transition",2.0,3.0,1.0,"Med","#58a6ff"),
            ("dc-optim","DC Optimization",3.0,4.0,2.0,"Low","#58a6ff"),
            ("planogram","Planogram Reset",4.0,2.0,5.0,"Med","#58a6ff"),
            ("trade-mgt","Trade Spend Realloc",3.0,-1.0,8.0,"Med","#58a6ff"),
        ],
    }
    _CONF_PAIRS = [
        ("tpr","strike"),("bogo","port-delay"),("bogo","strike"),
        ("display","strike"),("price-up","price-dn"),("edlp","price-up"),
        ("promo","supplier-bk"),("surge","dc-closure"),
        ("seasonal-pk","strike"),("new-item","dc-closure"),
    ]
    _RISK_COLOR = {"Crit":"#f85149","High":"#d29922","Med":"#388bfd","Low":"#3fb950"}

    pal_col, canvas_col, kpi_col = st.columns([2, 4, 3])

    # ── PALETTE ──
    with pal_col:
        st.markdown("#### 📦 Scenario Palette")
        base_rev_b = st.number_input("Base Revenue ($M)", 1.0, 9999.0, 100.0, 10.0, key="bldr_base")
        st.caption("Click **+ Add** to place on canvas")
        for cat, blocks in _BLK_DEF.items():
            cat_color = blocks[0][6]
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;color:{cat_color};'
                f'letter-spacing:.08em;text-transform:uppercase;margin-top:10px;'
                f'padding:3px 0;border-bottom:1px solid {cat_color}44;">{cat}</div>',
                unsafe_allow_html=True,
            )
            for bid, bname, brev, bmgn, bdem, brisk, bclr in blocks:
                already = any(b["id"] == bid for b in st.session_state.builder_blocks)
                rc = _RISK_COLOR.get(brisk, "#8b949e")
                rs = f"+{brev:.0f}" if brev >= 0 else f"{brev:.0f}"
                ms = f"+{bmgn:.0f}" if bmgn >= 0 else f"{bmgn:.0f}"
                ds = f"+{bdem:.0f}" if bdem >= 0 else f"{bdem:.0f}"
                ca, cb = st.columns([3,1])
                with ca:
                    st.markdown(
                        f'<div style="font-size:11px;font-weight:600;color:#cdd9e5;">{bname} '                        f'<span style="font-size:9px;color:{rc};font-weight:700;">({brisk})</span></div>'                        f'<div style="font-size:9px;color:#8b949e;">Rev {rs}% | Mgn {ms}% | Dem {ds}%</div>',
                        unsafe_allow_html=True,
                    )
                with cb:
                    if already:
                        if st.button("✕", key=f"brm_{bid}", help="Remove from canvas"):
                            st.session_state.builder_blocks = [b for b in st.session_state.builder_blocks if b["id"] != bid]
                            st.rerun()
                    else:
                        if st.button("＋", key=f"bad_{bid}", help=f"Add {bname}"):
                            st.session_state.builder_blocks.append({
                                "id":bid,"label":bname,
                                "rev":brev,"mgn":bmgn,"dem":bdem,"risk":brisk
                            })
                            st.rerun()

    # ── CANVAS ──
    with canvas_col:
        st.markdown("#### 🎯 Decision Canvas")
        blocks = st.session_state.builder_blocks
        if not blocks:
            st.info("👈 Click **＋** on any scenario block in the palette to add it here.")
        else:
            for i, blk in enumerate(blocks):
                clr = _RISK_COLOR.get(blk["risk"], "#8b949e")
                with st.container():
                    st.markdown(
                        f'<div style="border-left:3px solid {clr};padding-left:8px;margin-bottom:2px;">'                        f'<span style="font-size:13px;font-weight:700;color:#cdd9e5;">{blk["label"]}</span> '                        f'<span style="font-size:10px;color:{clr};">({blk["risk"]})</span></div>',
                        unsafe_allow_html=True,
                    )
                    c1,c2,c3,c4 = st.columns([3,3,3,1])
                    with c1:
                        nrev = st.slider("Rev %",min_value=-30.0,max_value=30.0,
                                         value=float(blk["rev"]),step=0.5,key=f"br_{blk['id']}_{i}")
                        st.session_state.builder_blocks[i]["rev"] = nrev
                    with c2:
                        nmgn = st.slider("Mgn %",min_value=-20.0,max_value=20.0,
                                         value=float(blk["mgn"]),step=0.5,key=f"bm_{blk['id']}_{i}")
                        st.session_state.builder_blocks[i]["mgn"] = nmgn
                    with c3:
                        ndem = st.slider("Dem %",min_value=-40.0,max_value=40.0,
                                         value=float(blk["dem"]),step=0.5,key=f"bd_{blk['id']}_{i}")
                        st.session_state.builder_blocks[i]["dem"] = ndem
                    with c4:
                        st.write("")
                        if st.button("✕", key=f"bx_{blk['id']}_{i}"):
                            st.session_state.builder_blocks.pop(i)
                            st.rerun()
                    st.divider()

            if st.button("🗑 Clear Canvas", key="bldr_clear"):
                st.session_state.builder_blocks = []
                st.rerun()

    # ── KPI + ANALYSIS ──
    with kpi_col:
        st.markdown("#### 📊 Live Impact")
        blocks = st.session_state.builder_blocks
        base = base_rev_b * 1_000_000
        n = len(blocks)

        def _fp(v): return f"+{v:.1f}%" if v>=0 else f"{v:.1f}%"
        def _fd(v):
            s="+" if v>=0 else "-"; a=abs(v)
            if a>=1e9: return f"{s}${a/1e9:.1f}B"
            if a>=1e6: return f"{s}${a/1e6:.1f}M"
            if a>=1e3: return f"{s}${a/1e3:.0f}K"
            return f"{s}${a:.0f}"

        if n == 0:
            st.info("Add blocks to see live KPIs")
        else:
            tot_r = sum(b["rev"] for b in blocks)
            tot_m = sum(b["mgn"] for b in blocks)
            tot_d = sum(b["dem"] for b in blocks)
            pen = max(0, (n-2)*2.0)
            adj_r = tot_r - pen
            adj_m = tot_m - pen
            ids = [b["id"] for b in blocks]
            conflicts = [(a,b2) for a,b2 in _CONF_PAIRS if a in ids and b2 in ids]

            k1,k2 = st.columns(2)
            k1.metric("Revenue", _fp(adj_r), _fd(base*adj_r/100))
            k2.metric("Margin",  _fp(adj_m), _fd(base*adj_m/100))
            k3,k4 = st.columns(2)
            k3.metric("Demand",  _fp(tot_d))
            k4.metric("Blocks",  str(n))

            if conflicts:
                for a,b2 in conflicts:
                    an = next((x["label"] for x in blocks if x["id"]==a), a)
                    bn = next((x["label"] for x in blocks if x["id"]==b2), b2)
                    st.error(f"⚡ Conflict: **{an}** ↔ **{bn}**")
            else:
                st.success("✓ No conflicts")

            if pen > 0:
                st.warning(f"⚠ Compound penalty: −{pen:.0f}% ({n-2} extra blocks)")

            st.markdown("---")
            if st.button("▶ Run Analysis", type="primary", use_container_width=True, key="bldr_run"):
                st.markdown("**Block Breakdown**")
                rows = []
                for b in blocks:
                    rows.append({
                        "Scenario": b["label"],
                        "Rev %": _fp(b["rev"]), "Rev $": _fd(base*b["rev"]/100),
                        "Mgn %": _fp(b["mgn"]), "Mgn $": _fd(base*b["mgn"]/100),
                        "Demand %": _fp(b["dem"]), "Risk": b["risk"],
                    })
                if pen > 0:
                    rows.append({
                        "Scenario": f"⚠ Penalty ({n-2} extra)",
                        "Rev %": _fp(-pen), "Rev $": _fd(-base*pen/100),
                        "Mgn %": _fp(-pen), "Mgn $": _fd(-base*pen/100),
                        "Demand %": "—", "Risk": "—",
                    })
                rows.append({
                    "Scenario": "🔵 NET IMPACT",
                    "Rev %": _fp(adj_r), "Rev $": _fd(base*adj_r/100),
                    "Mgn %": _fp(adj_m), "Mgn $": _fd(base*adj_m/100),
                    "Demand %": _fp(tot_d), "Risk": "—",
                })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("**Recommendations**")
                recs = []
                if conflicts:
                    recs.append("🔴 Resolve conflicts before activating both blocks.")
                if "price-up" in ids and "prestock" in ids:
                    recs.append("✅ Pre-Build Inventory before Price Increase to protect shelf stock.")
                if any(x in ids for x in ["tpr","display","bogo"]):
                    recs.append("📦 Align promotions with DC replenishment to avoid stockout.")
                if any(x in ids for x in ["strike","port-delay","dc-closure"]):
                    recs.append("🚚 Activate alternate carriers within 48h of disruption.")
                if adj_r < 0 and adj_m < 0:
                    recs.append("⚠️ Net negative — consider removing high-risk blocks.")
                if adj_r > 10:
                    recs.append(f"📈 Strong upside ({_fp(adj_r)}) — confirm DC stock covers demand uplift.")
                if not recs:
                    recs.append("✔ Chain balanced. Review timing with merchant and supply teams.")
                for r in recs:
                    st.markdown(f"- {r}")


with tab_canvas:
    st.subheader("🎯 Strategy Canvas")
    st.caption("Click **+ Add** on any scenario block to add it to your canvas. Adjust the impact sliders live and run the analysis.")

    # ── Scenario block definitions ──
    _SC_BLOCKS = {
        "PRICING": [
            {"id": "price-up",    "label": "Price Increase",      "rev":  8.0, "mgn":  5.0, "dem": -14.0, "risk": "Med"},
            {"id": "price-dn",    "label": "Price Cut",            "rev": -3.0, "mgn": -8.0, "dem":  12.0, "risk": "Low"},
            {"id": "edlp",        "label": "EDLP Rollback",        "rev":  1.0, "mgn": -3.0, "dem":   8.0, "risk": "Low"},
            {"id": "markdown",    "label": "Markdown Clearance",   "rev": -5.0, "mgn":-10.0, "dem":  22.0, "risk": "Med"},
            {"id": "price-match", "label": "Competitor Match",     "rev": -2.0, "mgn": -5.0, "dem":   6.0, "risk": "Med"},
        ],
        "PROMOTIONS": [
            {"id": "tpr",     "label": "Temp Price Reduction", "rev":  4.0, "mgn": -6.0, "dem":  20.0, "risk": "Low"},
            {"id": "display", "label": "Display / End Cap",    "rev":  6.0, "mgn": -2.0, "dem":  15.0, "risk": "Low"},
            {"id": "bogo",    "label": "BOGO Event",           "rev":  3.0, "mgn":-12.0, "dem":  30.0, "risk": "Low"},
            {"id": "coupon",  "label": "Digital Coupon Drop",  "rev":  2.0, "mgn": -8.0, "dem":  18.0, "risk": "Low"},
            {"id": "promo",   "label": "Shopper Marketing",    "rev":  5.0, "mgn": -3.0, "dem":  12.0, "risk": "Low"},
        ],
        "SUPPLY CHAIN": [
            {"id": "strike",      "label": "Carrier Strike",       "rev":-12.0, "mgn": -9.0, "dem":   0.0, "risk": "Crit"},
            {"id": "port-delay",  "label": "Port Delay",           "rev": -7.0, "mgn": -6.0, "dem":   0.0, "risk": "High"},
            {"id": "dc-closure",  "label": "DC Closure",           "rev":-10.0, "mgn": -8.0, "dem":   0.0, "risk": "Crit"},
            {"id": "supplier-bk", "label": "Supplier Bankruptcy",  "rev":-15.0, "mgn":-12.0, "dem":   0.0, "risk": "Crit"},
            {"id": "prestock",    "label": "Pre-Build Inventory",  "rev":  3.0, "mgn": -2.0, "dem":   5.0, "risk": "Low"},
            {"id": "alt-carrier", "label": "Alt Carrier Switch",   "rev": -2.0, "mgn": -3.0, "dem":   0.0, "risk": "Med"},
        ],
        "DEMAND SIGNALS": [
            {"id": "surge",      "label": "Demand Surge",      "rev": 15.0, "mgn": 10.0, "dem":  20.0, "risk": "Low"},
            {"id": "drop",       "label": "Demand Drop",       "rev":-10.0, "mgn": -7.0, "dem": -15.0, "risk": "High"},
            {"id": "seasonal-pk","label": "Seasonal Peak",     "rev": 12.0, "mgn":  8.0, "dem":  18.0, "risk": "Med"},
            {"id": "weather-ev", "label": "Weather Event",     "rev": -6.0, "mgn": -5.0, "dem": -10.0, "risk": "High"},
            {"id": "new-item",   "label": "New Item Launch",   "rev": 20.0, "mgn":  5.0, "dem":  25.0, "risk": "Med"},
            {"id": "comp-oos",   "label": "Competitor OOS",   "rev":  8.0, "mgn":  6.0, "dem":  10.0, "risk": "Low"},
        ],
        "TRADE & EXTERNAL": [
            {"id": "tariff",     "label": "Tariff Increase",     "rev": -4.0, "mgn": -6.0, "dem":  -8.0, "risk": "High"},
            {"id": "fx-impact",  "label": "FX / Currency Move",  "rev": -3.0, "mgn": -4.0, "dem":  -5.0, "risk": "Med"},
            {"id": "regulatory", "label": "Regulatory Change",   "rev": -5.0, "mgn": -4.0, "dem":   0.0, "risk": "High"},
            {"id": "comp-move",  "label": "Competitor Promo",    "rev": -3.0, "mgn": -3.0, "dem":  -8.0, "risk": "Med"},
            {"id": "inflation",  "label": "Input Cost Inflation", "rev":-2.0, "mgn": -8.0, "dem":  -4.0, "risk": "High"},
        ],
        "OPERATIONS": [
            {"id": "safety-stk", "label": "Safety Stock Raise",  "rev": -1.0, "mgn": -2.0, "dem":   3.0, "risk": "Med"},
            {"id": "vmi-switch", "label": "VMI Transition",       "rev":  2.0, "mgn":  3.0, "dem":   1.0, "risk": "Med"},
            {"id": "dc-optim",   "label": "DC Optimization",      "rev":  3.0, "mgn":  4.0, "dem":   2.0, "risk": "Low"},
            {"id": "planogram",  "label": "Planogram Reset",      "rev":  4.0, "mgn":  2.0, "dem":   5.0, "risk": "Med"},
            {"id": "trade-mgt",  "label": "Trade Spend Realloc",  "rev":  3.0, "mgn": -1.0, "dem":   8.0, "risk": "Med"},
        ],
    }

    _SC_CONFLICTS = [
        ("tpr", "strike"), ("bogo", "port-delay"), ("bogo", "strike"),
        ("display", "strike"), ("price-up", "price-dn"), ("edlp", "price-up"),
        ("promo", "supplier-bk"), ("surge", "dc-closure"), ("seasonal-pk", "strike"),
        ("new-item", "dc-closure"),
    ]

    _CAT_COLORS = {
        "PRICING": "#388bfd", "PROMOTIONS": "#3fb950",
        "SUPPLY CHAIN": "#f85149", "DEMAND SIGNALS": "#3fb950",
        "TRADE & EXTERNAL": "#d29922", "OPERATIONS": "#58a6ff",
    }

    # Session state
    if "canvas_blocks" not in st.session_state:
        st.session_state.canvas_blocks = []   # list of block dicts with editable rev/mgn/dem

    left_col, right_col = st.columns([2, 3])

    # ── LEFT: Palette ──
    with left_col:
        st.markdown("### Scenario Palette")
        base_rev = st.number_input("Base Annual Revenue ($M)", min_value=1.0, max_value=9999.0,
                                   value=100.0, step=10.0, key="sc_base_rev")

        for cat, blocks in _SC_BLOCKS.items():
            cat_color = _CAT_COLORS.get(cat, "#8b949e")
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;color:{cat_color};'
                f'text-transform:uppercase;letter-spacing:.08em;margin-top:12px;'
                f'border-top:1px solid {cat_color}44;padding-top:6px;">{cat}</div>',
                unsafe_allow_html=True,
            )
            for blk in blocks:
                blk_id = blk["id"]
                already = any(b["id"] == blk_id for b in st.session_state.canvas_blocks)
                risk_color = {"Crit": "#f85149", "High": "#d29922", "Med": "#388bfd", "Low": "#3fb950"}.get(blk["risk"], "#8b949e")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    rev_s = f"+{blk['rev']:.0f}" if blk["rev"] >= 0 else f"{blk['rev']:.0f}"
                    mgn_s = f"+{blk['mgn']:.0f}" if blk["mgn"] >= 0 else f"{blk['mgn']:.0f}"
                    dem_s = f"+{blk['dem']:.0f}" if blk["dem"] >= 0 else f"{blk['dem']:.0f}"
                    st.markdown(
                        f'<div style="font-size:12px;font-weight:600;color:#cdd9e5;">'
                        f'{blk["label"]}'
                        f' <span style="font-size:10px;color:{risk_color}">({blk["risk"]})</span></div>'
                        f'<div style="font-size:10px;color:#8b949e;">Rev {rev_s}% | Mgn {mgn_s}% | Dem {dem_s}%</div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    if already:
                        if st.button("✕ Remove", key=f"rm_{blk_id}"):
                            st.session_state.canvas_blocks = [
                                b for b in st.session_state.canvas_blocks if b["id"] != blk_id
                            ]
                            st.rerun()
                    else:
                        if st.button("+ Add", key=f"add_{blk_id}"):
                            st.session_state.canvas_blocks.append({
                                "id": blk["id"], "label": blk["label"],
                                "rev": blk["rev"], "mgn": blk["mgn"], "dem": blk["dem"],
                                "risk": blk["risk"],
                            })
                            st.rerun()

    # ── RIGHT: Canvas + Analysis ──
    with right_col:
        st.markdown("### Active Canvas")

        blocks = st.session_state.canvas_blocks
        if not blocks:
            st.info("No blocks added yet. Click **+ Add** on any scenario in the palette.")
        else:
            # Editable block cards
            for i, blk in enumerate(blocks):
                cat_color = _CAT_COLORS.get(
                    next((c for c, bs in _SC_BLOCKS.items() if any(b["id"] == blk["id"] for b in bs)), ""), "#8b949e"
                )
                with st.expander(f"**{blk['label']}**", expanded=True):
                    c1, c2, c3, c4 = st.columns([3, 3, 3, 1])
                    with c1:
                        new_rev = st.slider("Rev %", -30.0, 30.0, float(blk["rev"]), 0.5,
                                            key=f"cv_rev_{blk['id']}_{i}")
                        st.session_state.canvas_blocks[i]["rev"] = new_rev
                    with c2:
                        new_mgn = st.slider("Mgn %", -20.0, 20.0, float(blk["mgn"]), 0.5,
                                            key=f"cv_mgn_{blk['id']}_{i}")
                        st.session_state.canvas_blocks[i]["mgn"] = new_mgn
                    with c3:
                        new_dem = st.slider("Dem %", -40.0, 40.0, float(blk["dem"]), 0.5,
                                            key=f"cv_dem_{blk['id']}_{i}")
                        st.session_state.canvas_blocks[i]["dem"] = new_dem
                    with c4:
                        if st.button("✕", key=f"del_{blk['id']}_{i}"):
                            st.session_state.canvas_blocks.pop(i)
                            st.rerun()

            st.markdown("---")

            # ── Live KPI calculation ──
            if st.button("▶ Run Analysis", type="primary", use_container_width=True, key="cv_run"):
                base = base_rev * 1_000_000
                ids = [b["id"] for b in blocks]
                n = len(blocks)

                tot_rev = sum(b["rev"] for b in blocks)
                tot_mgn = sum(b["mgn"] for b in blocks)
                tot_dem = sum(b["dem"] for b in blocks)

                # Compound execution penalty
                penalty = max(0, (n - 2) * 2.0)
                adj_rev = tot_rev - penalty
                adj_mgn = tot_mgn - penalty

                rev_dollars = base * adj_rev / 100
                mgn_dollars = base * adj_mgn / 100

                def fmt_d(v):
                    s = "+" if v >= 0 else "-"
                    a = abs(v)
                    if a >= 1e9:   return f"{s}${a/1e9:.1f}B"
                    if a >= 1e6:   return f"{s}${a/1e6:.1f}M"
                    if a >= 1e3:   return f"{s}${a/1e3:.0f}K"
                    return f"{s}${a:.0f}"

                def fmt_p(v):
                    return f"+{v:.1f}%" if v >= 0 else f"{v:.1f}%"

                # Conflict detection
                conflicts = []
                for c1_id, c2_id in _SC_CONFLICTS:
                    if c1_id in ids and c2_id in ids:
                        c1_lbl = next((b["label"] for b in blocks if b["id"] == c1_id), c1_id)
                        c2_lbl = next((b["label"] for b in blocks if b["id"] == c2_id), c2_id)
                        conflicts.append(f"{c1_lbl} ↔ {c2_lbl}")

                # KPI tiles
                k1, k2, k3, k4 = st.columns(4)
                rc = "normal" if adj_rev >= 0 else "inverse"
                mc = "normal" if adj_mgn >= 0 else "inverse"
                dc = "normal" if tot_dem >= 0 else "inverse"
                k1.metric("Revenue Impact", fmt_p(adj_rev), fmt_d(rev_dollars))
                k2.metric("Margin Impact",  fmt_p(adj_mgn), fmt_d(mgn_dollars))
                k3.metric("Demand Shift",   fmt_p(tot_dem))
                k4.metric("Blocks Active",  str(n))

                # Per-block table
                st.markdown("**Block-by-Block Breakdown**")
                rows = []
                for b in blocks:
                    r_d = base * b["rev"] / 100
                    m_d = base * b["mgn"] / 100
                    rows.append({
                        "Scenario": b["label"],
                        "Rev %": fmt_p(b["rev"]),
                        "Rev $": fmt_d(r_d),
                        "Mgn %": fmt_p(b["mgn"]),
                        "Mgn $": fmt_d(m_d),
                        "Demand %": fmt_p(b["dem"]),
                        "Risk": b["risk"],
                    })
                if penalty > 0:
                    rows.append({
                        "Scenario": f"⚠ Compound Penalty ({n-2} extra blocks)",
                        "Rev %": fmt_p(-penalty), "Rev $": fmt_d(-base * penalty / 100),
                        "Mgn %": fmt_p(-penalty), "Mgn $": fmt_d(-base * penalty / 100),
                        "Demand %": "—", "Risk": "—",
                    })
                rows.append({
                    "Scenario": "NET IMPACT",
                    "Rev %": fmt_p(adj_rev), "Rev $": fmt_d(rev_dollars),
                    "Mgn %": fmt_p(adj_mgn), "Mgn $": fmt_d(mgn_dollars),
                    "Demand %": fmt_p(tot_dem), "Risk": "—",
                })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Conflicts
                if conflicts:
                    for cf in conflicts:
                        st.error(f"⚡ Conflict detected: **{cf}** — these blocks produce opposing effects. Resolve before activating.")
                else:
                    st.success("✓ No conflicts detected. Chain is executable as planned.")

                # Recommendations
                st.markdown("**Recommendations**")
                recs = []
                if conflicts:
                    recs.append("🔴 **Resolve conflicts first** before activating conflicting blocks simultaneously.")
                if "price-up" in ids and "prestock" in ids:
                    recs.append("✅ **Recommended sequence**: Execute Pre-Build Inventory before Price Increase to protect shelf availability.")
                if any(x in ids for x in ["tpr", "display", "bogo"]):
                    recs.append("📦 **Align promotions** with DC replenishment cycle to prevent stockout during demand uplift.")
                if any(x in ids for x in ["strike", "port-delay", "dc-closure"]):
                    recs.append("🚚 **Activate alternate carriers** in affected regions within 48h of disruption signal.")
                if adj_rev < 0 and adj_mgn < 0:
                    recs.append("⚠️ **Net negative impact** — consider removing or mitigating high-risk blocks.")
                if adj_rev > 10:
                    recs.append(f"📈 **Strong revenue upside** ({fmt_p(adj_rev)}) — ensure DC stock covers demand uplift.")
                if not recs:
                    recs.append("✔ Scenario chain looks balanced. Review timing with merchant and supply planning teams.")
                for r in recs:
                    st.markdown(f"- {r}")

        if blocks:
            if st.button("🗑 Clear Canvas", key="cv_clear"):
                st.session_state.canvas_blocks = []
                st.rerun()
