"""
Retail Supply Chain Optimization — Streamlit UI (Full)

Tabs:
  1. Dashboard     — live network status + KPI tiles
  2. Chat          — streaming multi-agent conversational interface
  3. Price Cascade — price change → full downstream simulation
  4. Supply Alert  — carrier strike / port delay / supplier bankruptcy
  5. Demand Forecast — 15-variable model with CI fan chart
  6. Scenario Planner — multi-scenario comparison + conflict detection
  7. Shelf & Store — HQ→DC→Store replenishment chain
  8. Financial Impact — P&L waterfall, trade dollars, carrying cost
  9. Data Sources  — provenance and freshness tracker
"""

import sys
import json
import time
import logging
from pathlib import Path

import streamlit as st
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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─── Orchestrator (lazy, cached) ─────────────────────────────────────────────

@st.cache_resource
def _get_orchestrator():
    from agents.orchestrator import Orchestrator
    return Orchestrator()

def get_orchestrator():
    try:
        return _get_orchestrator()
    except Exception as exc:
        return None, str(exc)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏪 Retail Supply Chain AI")
    st.markdown("*Powered by Claude claude-sonnet-4-6*")
    st.divider()

    # ── API Key Status ──
    st.markdown("### 🔑 API Status")
    key_ok = bool(settings.ANTHROPIC_API_KEY)
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

(tab_dash, tab_chat, tab_price, tab_supply, tab_forecast,
 tab_scenario, tab_shelf, tab_finance, tab_data) = st.tabs([
    "Dashboard",
    "Chat",
    "Price Cascade",
    "Supply Alert",
    "Demand Forecast",
    "Scenario Planner",
    "Shelf & Store",
    "Financial Impact",
    "Data Sources",
])

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
            "API key not detected. Add `ANTHROPIC_API_KEY` to your `.env` file. "
            "All other tabs work without an API key."
        )

    # ── Render existing history ──
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
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not key_ok:
                st.error("Set ANTHROPIC_API_KEY in .env to enable the AI agent.")
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

                            client = _anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
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


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SUPPLY ALERT
# ════════════════════════════════════════════════════════════════════════════
with tab_supply:
    st.subheader("Supply Disruption Analyzer")
    st.caption(
        "Model carrier strikes, port delays, and supplier bankruptcies. "
        "Regional carrier availability checked separately from national."
    )

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEMAND FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.subheader("Demand Forecast — 15-Variable Model")
    st.caption("Demand is a class function, not a time series. Price, promo, tariffs, weather, and 11 more variables all move the signal.")

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — SCENARIO PLANNER
# ════════════════════════════════════════════════════════════════════════════
with tab_scenario:
    st.subheader("Scenario Planner")
    st.caption("Compare up to 4 scenarios and detect conflicts. All require a time anchor — comparisons without dates are invalid.")

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — SHELF & STORE REPLENISHMENT
# ════════════════════════════════════════════════════════════════════════════
with tab_shelf:
    st.subheader("Shelf & Store Replenishment")
    st.caption("HQ → DC → Store with 3-4d lag, lead-time variability, perishable caps, and planogram constraints.")

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


# ════════════════════════════════════════════════════════════════════════════
# TAB 8 — FINANCIAL IMPACT
# ════════════════════════════════════════════════════════════════════════════
with tab_finance:
    st.subheader("Financial Impact Calculator")
    st.caption("Revenue, margin, vendor trade dollars, VMI split, carrying cost. Tax flagged as approximate.")

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


# ════════════════════════════════════════════════════════════════════════════
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
