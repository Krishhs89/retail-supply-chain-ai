"""
LangGraph Multi-Agent Workflow — Retail Supply Chain Optimization
=================================================================

Graph topology
--------------

                         ┌──────────────┐
         user query ───► │    ROUTER    │  classifies intent + extracts SKU
                         └──────┬───────┘
                                │
            ┌───────────────────┼───────────────────────┐
            │                   │                       │
            ▼                   ▼                       ▼
     [price_cascade]   [supply_disruption]   [demand_forecast]
            │                   │                       │
            ▼                   ▼                       ▼
     [inventory_node]  [carrier_node]         [accuracy_node]
            │                   │                       │
            └───────────────────┼───────────────────────┘
                                │
                         ┌──────▼───────┐
                         │ SYNTHESIZER  │  combines all node outputs
                         └──────┬───────┘
                                │
                              END

Additional paths:
  [scenario_planner]  ──────────────────────────► SYNTHESIZER
  [shelf_replenishment] + [perishable_check] ───► SYNTHESIZER
  [financial_impact] ──────────────────────────► SYNTHESIZER

Each node:
  • Has a specialized system prompt focused on its domain
  • Calls a curated subset of tools from mock_executor
  • Annotates results with provenance (OLAP/WMS/OLTP)
  • Returns partial state that the synthesizer merges

Run standalone:
  python agents/langgraph_flow.py

Run from orchestrator:
  from agents.langgraph_flow import run_langgraph
  result = run_langgraph(query, history)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config.settings import settings
from tools import mock_executor

logger = logging.getLogger(__name__)

# ─── LLM ─────────────────────────────────────────────────────────────────────

def _llm(max_tokens: int = 2048) -> ChatAnthropic:
    import os as _os
    api_key = _os.getenv("ANTHROPIC_API_KEY", "") or settings.ANTHROPIC_API_KEY
    return ChatAnthropic(
        model=settings.MODEL_ID,
        api_key=api_key,
        max_tokens=max_tokens,
    )

# ─── Graph State ──────────────────────────────────────────────────────────────

class RetailState(TypedDict):
    """Shared state passed between all nodes in the graph."""
    # Core I/O
    query: str
    messages: Annotated[List[BaseMessage], add_messages]
    final_response: str

    # Routing
    intent: str          # price_cascade | supply_disruption | demand_forecast |
                         # scenario_planning | shelf_replenishment | financial_impact | general
    sku: str             # primary SKU extracted by router
    entities: Dict[str, Any]  # carrier, DC, store, price, etc.

    # Node outputs (accumulated)
    tool_calls_made: List[Dict[str, Any]]
    freshness_warnings: List[str]
    node_outputs: Dict[str, Any]  # node_name → structured output dict

    # Meta
    iterations: int
    route: str           # mirrors intent — used for conditional edges


# ─── System Prompts ───────────────────────────────────────────────────────────

_ROUTER_PROMPT = """You are a retail supply chain query router.
Classify the user query into exactly one intent and extract key entities.

Intents:
  price_cascade        — price change and downstream effects
  supply_disruption    — carrier strike, port delay, supplier bankruptcy
  demand_forecast      — demand forecasting, elasticity, forecast accuracy
  scenario_planning    — multi-scenario comparison, conflict detection
  shelf_replenishment  — store replenishment, stockout, planogram, perishables
  financial_impact     — revenue, margin, carrying cost, P&L
  general              — anything else or multi-topic

Respond ONLY with valid JSON:
{
  "intent": "<one of the intents above>",
  "sku": "<SKU if mentioned, else 'HUG48-3'>",
  "entities": {
    "carrier": "<carrier ID if mentioned>",
    "store": "<store ID if mentioned>",
    "dc": "<DC ID if mentioned>",
    "old_price": <number or null>,
    "new_price": <number or null>,
    "disruption_type": "<carrier_strike|port_delay|supplier_bankruptcy or null>",
    "duration_days": <number or null>,
    "horizon_weeks": <number or null>
  }
}"""

_PRICING_PROMPT = """You are the Pricing Agent for a Walmart-scale retail optimization system.
Your job: analyze the full downstream cascade of a price change — demand, inventory, POs, financials.

Key rules:
• Asymmetric elasticity applies to diapers, tobacco, alcohol, formula.
• Always net vendor trade dollars before reporting margin impact.
• PO adjustments are only possible for POs with ETA > 14 days.
• Report: price_change_pct, demand_change_pct, net_revenue_change, margin_change, PO adjustments.
"""

_SUPPLY_PROMPT = """You are the Supply Chain Agent for a Walmart-scale retail optimization system.
Your job: analyze supply disruptions (carrier strikes, port delays, supplier bankruptcies).

Key rules:
• Check regional alternate carrier availability — not just national.
• Flag explicitly when NO viable alternates exist in the affected region.
• Disruption types have different duration models:
    carrier_strike = 2-3 weeks, port_delay = 6-10 weeks, supplier_bankruptcy = permanent.
• Always compute revenue at risk and provide a time-ordered mitigation plan.
"""

_DEMAND_PROMPT = """You are the Demand Forecasting Agent for a Walmart-scale retail optimization system.
Your job: produce demand forecasts and accuracy analysis using a 15-variable model.

Key rules:
• Forecast accuracy < 60% → mark as UNRELIABLE, do NOT pass to PO system.
• Confidence intervals MUST widen ~15% per 4-week horizon period.
• Report all 15 variable contributions (price, promo, markdown, trade_spend,
  competitor_price, weather, seasonality, tariff, supply_availability, store_count,
  customer_income_index, trend, social_media_buzz, regulatory_change, external_event).
• Each 1% accuracy improvement ≈ 1% revenue improvement at scale.
"""

_SCENARIO_PROMPT = """You are the Scenario Planning Agent for a Walmart-scale retail optimization system.
Your job: compare scenarios and detect conflicts.

Key rules:
• All scenarios MUST have a start_date — comparisons without time anchors are invalid.
• CRITICAL conflict: promotion (demand up) + supply disruption (supply constrained) simultaneously.
• Report best-revenue and best-margin scenarios separately.
• Factor in asymmetric elasticity when computing demand for price scenarios.
"""

_INVENTORY_PROMPT = """You are the Inventory & Shelf Agent for a Walmart-scale retail optimization system.
Your job: manage store-level replenishment across the HQ → DC → Store chain.

Key rules:
• Full replenishment lag = 3-4 days + 30% chance of 2-4 extra days.
• If days_on_hand < effective lag → CRITICAL: trigger emergency transfer.
• Perishables (dairy, produce): max days-of-supply cap = 3 days. Excess → write-off risk.
• Planogram constraint: replenishment exceeding shelf capacity goes to back-of-store, NOT shelf.
"""

_FINANCIAL_PROMPT = """You are the Financial Impact Agent for a Walmart-scale retail optimization system.
Your job: calculate revenue, margin, carrying cost, and vendor trade impact.

Key rules:
• Always net vendor trade dollars against promotional cost before reporting margin.
• Carrying cost applies ONLY to owned inventory (not VMI).
• Tax calculations are ALWAYS flagged as approximate.
• Report gross revenue Δ, trade offset, net revenue Δ, margin Δ, carrying cost separately.
"""

_SYNTHESIZER_PROMPT = """You are the Synthesis Agent for a retail supply chain optimization system.
You receive structured outputs from specialized domain agents and produce a coherent,
executive-quality response.

Format your response as:
1. Summary (2-3 sentences)
2. Key Findings (bullet points by domain)
3. Recommended Actions (numbered, prioritized)
4. Dollar Impact (range — gross and net)
5. Data Caveats (any OLAP staleness or low-accuracy forecasts)

Be precise with numbers. Flag asymmetric elasticity when applicable.
"""


# ─── Helper: invoke LLM and parse JSON ────────────────────────────────────────

def _chat(prompt: str, system: str, max_tokens: int = 1024) -> str:
    llm = _llm(max_tokens)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    return response.content


def _safe_json(text: str) -> Dict:
    """Extract JSON from an LLM response that may have markdown fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _append_tool(state: RetailState, tool_name: str, tool_input: Dict,
                 result: Dict) -> None:
    """Mutate state lists in-place (nodes work on copies)."""
    prov = result.get("provenance", "OLTP")
    fresh = result.get("freshness_minutes", 5)
    state["tool_calls_made"].append({
        "tool": tool_name,
        "input": tool_input,
        "result_summary": result.get("error") or str(result.get("data", {}))[:200],
        "provenance": prov,
        "freshness_minutes": fresh,
        "had_error": bool(result.get("error")),
    })
    if result.get("is_stale") or prov == "OLAP":
        state["freshness_warnings"].append(
            f"'{tool_name}' uses {prov} data ({fresh}min lag)."
        )
    data = result.get("data", {})
    if isinstance(data, dict):
        for fw in data.get("freshness_warnings", []):
            state["freshness_warnings"].append(fw)


# ─── Node: Router ─────────────────────────────────────────────────────────────

def router_node(state: RetailState) -> RetailState:
    """Classify intent and extract entities from the query."""
    logger.info("[Router] Classifying query: %s", state["query"][:80])

    raw = _chat(
        prompt=f"Query: {state['query']}",
        system=_ROUTER_PROMPT,
        max_tokens=400,
    )
    parsed = _safe_json(raw)

    intent  = parsed.get("intent", "general")
    sku     = parsed.get("sku", "HUG48-3")
    entities = parsed.get("entities", {})

    logger.info("[Router] intent=%s sku=%s", intent, sku)

    return {
        **state,
        "intent": intent,
        "route": intent,
        "sku": sku,
        "entities": entities,
        "tool_calls_made": state.get("tool_calls_made", []),
        "freshness_warnings": state.get("freshness_warnings", []),
        "node_outputs": state.get("node_outputs", {}),
        "iterations": state.get("iterations", 0) + 1,
    }


# ─── Node: Price Cascade ──────────────────────────────────────────────────────

def price_cascade_node(state: RetailState) -> RetailState:
    """Run price change simulation + elasticity analysis."""
    logger.info("[PriceCascade] sku=%s", state["sku"])
    ent = state.get("entities", {})
    sku = state["sku"]
    prod = mock_executor.PRODUCTS.get(sku, mock_executor.PRODUCTS["HUG48-3"])

    old_price = ent.get("old_price") or prod["base_price"]
    new_price = ent.get("new_price") or (prod["base_price"] + 1.50)
    horizon   = ent.get("horizon_weeks") or 8

    # Tool 1: simulate price change
    t1_in = {"sku": sku, "old_price": old_price, "new_price": new_price, "horizon_weeks": horizon}
    t1 = mock_executor.execute("simulate_price_change", t1_in)
    _append_tool(state, "simulate_price_change", t1_in, t1)

    # Tool 2: competitive pricing context
    t2_in = {"sku": sku}
    t2 = mock_executor.execute("get_competitive_pricing", t2_in)
    _append_tool(state, "get_competitive_pricing", t2_in, t2)

    # Tool 3: inventory levels (to quantify overstock)
    t3_in = {"sku": sku, "location_type": "dc"}
    t3 = mock_executor.execute("get_inventory_levels", t3_in)
    _append_tool(state, "get_inventory_levels", t3_in, t3)

    node_out = {
        "price_cascade": t1.get("data", {}),
        "competitive_pricing": t2.get("data", {}),
        "dc_inventory": t3.get("data", {}),
    }

    # LLM synthesis for this node
    summary_prompt = (
        f"Query: {state['query']}\n\n"
        f"Price cascade result: {json.dumps(t1.get('data', {}), indent=2)[:1500]}\n"
        f"Competitive pricing: {json.dumps(t2.get('data', {}))}\n"
        f"DC inventory: {json.dumps(t3.get('data', {}))[:600]}"
    )
    node_summary = _chat(summary_prompt, _PRICING_PROMPT, max_tokens=600)

    node_out["pricing_summary"] = node_summary
    state["node_outputs"]["price_cascade"] = node_out
    state["iterations"] += 1
    return state


# ─── Node: Inventory (called after price cascade) ────────────────────────────

def inventory_node(state: RetailState) -> RetailState:
    """Check stockout risk and shelf capacity after a price change or disruption."""
    logger.info("[Inventory] sku=%s", state["sku"])
    sku = state["sku"]
    ent = state.get("entities", {})
    store = ent.get("store") or "STR-011"  # default SE store (most at risk)

    # Tool: stockout risk at store level
    t1_in = {"sku": sku, "location_id": store}
    t1 = mock_executor.execute("calculate_stockout_risk", t1_in)
    _append_tool(state, "calculate_stockout_risk", t1_in, t1)

    # Tool: shelf capacity
    t2_in = {"sku": sku, "store_id": store, "proposed_replenishment_units": 48}
    t2 = mock_executor.execute("check_shelf_capacity", t2_in)
    _append_tool(state, "check_shelf_capacity", t2_in, t2)

    state["node_outputs"]["inventory"] = {
        "stockout_risk": t1.get("data", {}),
        "shelf_capacity": t2.get("data", {}),
    }
    state["iterations"] += 1
    return state


# ─── Node: Supply Disruption ──────────────────────────────────────────────────

def supply_disruption_node(state: RetailState) -> RetailState:
    """Analyze a supply disruption — carrier strike, port delay, or bankruptcy."""
    logger.info("[SupplyDisruption] sku=%s", state["sku"])
    ent = state.get("entities", {})
    sku = state["sku"]

    d_type    = ent.get("disruption_type") or "carrier_strike"
    entity    = ent.get("carrier") or "TruckCo_B"
    duration  = ent.get("duration_days") or 14

    # Tool 1: full disruption impact
    t1_in = {"disruption_type": d_type, "affected_entity": entity, "duration_days": duration}
    t1 = mock_executor.execute("get_supply_disruption_impact", t1_in)
    _append_tool(state, "get_supply_disruption_impact", t1_in, t1)

    # Tool 2: carrier status overview
    t2_in = {"carrier_id": "all"}
    t2 = mock_executor.execute("get_carrier_status", t2_in)
    _append_tool(state, "get_carrier_status", t2_in, t2)

    node_summary_prompt = (
        f"Query: {state['query']}\n\n"
        f"Disruption impact: {json.dumps(t1.get('data', {}), indent=2)[:2000]}\n"
        f"All carrier status: {json.dumps(t2.get('data', {}))[:500]}"
    )
    node_summary = _chat(node_summary_prompt, _SUPPLY_PROMPT, max_tokens=700)

    state["node_outputs"]["supply_disruption"] = {
        "disruption_impact": t1.get("data", {}),
        "carrier_status": t2.get("data", {}),
        "supply_summary": node_summary,
    }
    state["iterations"] += 1
    return state


# ─── Node: Alternate Carrier ──────────────────────────────────────────────────

def carrier_node(state: RetailState) -> RetailState:
    """Find alternate carriers for each region affected by the disruption."""
    logger.info("[CarrierNode] sku=%s", state["sku"])
    sku = state["sku"]
    ent = state.get("entities", {})
    carrier = ent.get("carrier") or "TruckCo_B"
    duration = ent.get("duration_days") or 14

    alts_by_region = {}
    for region in ["NW", "SE", "MW"]:
        t_in = {"sku": sku, "disrupted_carrier_id": carrier,
                "region": region, "duration_days": duration}
        t = mock_executor.execute("find_alternate_carriers", t_in)
        _append_tool(state, "find_alternate_carriers", t_in, t)
        alts_by_region[region] = t.get("data", {})

    state["node_outputs"]["alternate_carriers"] = alts_by_region
    state["iterations"] += 1
    return state


# ─── Node: Demand Forecast ────────────────────────────────────────────────────

def demand_forecast_node(state: RetailState) -> RetailState:
    """Run demand forecast with confidence intervals."""
    logger.info("[DemandForecast] sku=%s", state["sku"])
    sku = state["sku"]
    ent = state.get("entities", {})
    horizon = ent.get("horizon_weeks") or 8

    t1_in = {"sku": sku, "horizon_weeks": horizon}
    t1 = mock_executor.execute("get_demand_forecast", t1_in)
    _append_tool(state, "get_demand_forecast", t1_in, t1)

    node_prompt = (
        f"Query: {state['query']}\n\n"
        f"Forecast result: {json.dumps(t1.get('data', {}), indent=2)[:2000]}"
    )
    node_summary = _chat(node_prompt, _DEMAND_PROMPT, max_tokens=600)

    state["node_outputs"]["demand_forecast"] = {
        "forecast": t1.get("data", {}),
        "demand_summary": node_summary,
    }
    state["iterations"] += 1
    return state


# ─── Node: Forecast Accuracy ─────────────────────────────────────────────────

def accuracy_node(state: RetailState) -> RetailState:
    """Add forecast accuracy and revenue-impact analysis."""
    logger.info("[AccuracyNode] sku=%s", state["sku"])
    sku = state["sku"]
    ent = state.get("entities", {})
    horizon = ent.get("horizon_weeks") or 8

    t_in = {"sku": sku, "horizon_weeks": horizon}
    t = mock_executor.execute("get_forecast_accuracy", t_in)
    _append_tool(state, "get_forecast_accuracy", t_in, t)

    state["node_outputs"]["forecast_accuracy"] = t.get("data", {})
    state["iterations"] += 1
    return state


# ─── Node: Scenario Planning ──────────────────────────────────────────────────

def scenario_planning_node(state: RetailState) -> RetailState:
    """Compare scenarios and detect conflicts."""
    logger.info("[ScenarioPlanning] sku=%s", state["sku"])
    sku = state["sku"]
    prod = mock_executor.PRODUCTS.get(sku, mock_executor.PRODUCTS["HUG48-3"])
    base_price = prod["base_price"]

    # Default 3-scenario comparison
    scenarios = [
        {"name": "Baseline", "price": base_price},
        {"name": "Price Up", "price": base_price + 1.50},
        {"name": "Promo",    "price": base_price - 1.00, "promo_uplift_pct": 15},
    ]
    t1_in = {"sku": sku, "scenarios": scenarios, "horizon_weeks": 8}
    t1 = mock_executor.execute("run_scenario_comparison", t1_in)
    _append_tool(state, "run_scenario_comparison", t1_in, t1)

    # Conflict detection
    scenario_meta = [
        {"name": "Promo", "scenario_type": "promotion",
         "start_date": "2026-05-01", "horizon_days": 30,
         "demand_direction": "increase", "supply_direction": "neutral"},
        {"name": "Strike", "scenario_type": "supply_disruption",
         "start_date": "2026-05-05", "horizon_days": 14,
         "demand_direction": "neutral", "supply_direction": "constrain"},
    ]
    t2_in = {"scenarios": scenario_meta}
    t2 = mock_executor.execute("detect_scenario_conflicts", t2_in)
    _append_tool(state, "detect_scenario_conflicts", t2_in, t2)

    node_prompt = (
        f"Query: {state['query']}\n\n"
        f"Comparison: {json.dumps(t1.get('data', {}), indent=2)[:1200]}\n"
        f"Conflicts: {json.dumps(t2.get('data', {}))[:600]}"
    )
    node_summary = _chat(node_prompt, _SCENARIO_PROMPT, max_tokens=600)

    state["node_outputs"]["scenario_planning"] = {
        "comparison": t1.get("data", {}),
        "conflicts": t2.get("data", {}),
        "scenario_summary": node_summary,
    }
    state["iterations"] += 1
    return state


# ─── Node: Shelf Replenishment ────────────────────────────────────────────────

def shelf_replenishment_node(state: RetailState) -> RetailState:
    """Check stockout, planogram, and perishable status — recommend replenishment."""
    logger.info("[ShelfReplenishment] sku=%s", state["sku"])
    sku = state["sku"]
    ent = state.get("entities", {})
    store = ent.get("store") or "STR-005"
    store_meta = mock_executor.STORES.get(store, mock_executor.STORES["STR-001"])
    dc_id = store_meta["dc"]
    prod = mock_executor.PRODUCTS.get(sku, {})

    t1_in = {"sku": sku, "location_id": store}
    t1 = mock_executor.execute("calculate_stockout_risk", t1_in)
    _append_tool(state, "calculate_stockout_risk", t1_in, t1)

    t2_in = {"sku": sku, "store_id": store, "proposed_replenishment_units": 48}
    t2 = mock_executor.execute("check_shelf_capacity", t2_in)
    _append_tool(state, "check_shelf_capacity", t2_in, t2)

    t3 = None
    if prod.get("perishable"):
        t3_in = {"sku": sku, "location_id": store, "proposed_replenishment_units": 48}
        t3 = mock_executor.execute("check_perishable_status", t3_in)
        _append_tool(state, "check_perishable_status", t3_in, t3)

    t4_in = {"sku": sku, "from_location": dc_id, "to_location": store,
             "quantity": 48, "priority": "standard"}
    t4 = mock_executor.execute("trigger_replenishment", t4_in)
    _append_tool(state, "trigger_replenishment", t4_in, t4)

    node_prompt = (
        f"Query: {state['query']}\n\n"
        f"Stockout risk: {json.dumps(t1.get('data', {}))}\n"
        f"Shelf capacity: {json.dumps(t2.get('data', {}))}\n"
        f"Replenishment: {json.dumps(t4.get('data', {}))}"
        + (f"\nPerishable: {json.dumps(t3.get('data', {}))}" if t3 else "")
    )
    node_summary = _chat(node_prompt, _INVENTORY_PROMPT, max_tokens=600)

    state["node_outputs"]["shelf_replenishment"] = {
        "stockout_risk": t1.get("data", {}),
        "shelf_capacity": t2.get("data", {}),
        "perishable": t3.get("data", {}) if t3 else None,
        "replenishment": t4.get("data", {}),
        "shelf_summary": node_summary,
    }
    state["iterations"] += 1
    return state


# ─── Node: Perishable Check (sub-node) ───────────────────────────────────────

def perishable_check_node(state: RetailState) -> RetailState:
    """Extra perishable validation for dairy/produce SKUs."""
    sku = state["sku"]
    prod = mock_executor.PRODUCTS.get(sku, {})
    if not prod.get("perishable"):
        state["node_outputs"]["perishable_check"] = {"applicable": False}
        return state

    store = state.get("entities", {}).get("store") or "STR-001"
    t_in = {"sku": sku, "location_id": store, "proposed_replenishment_units": 80}
    t = mock_executor.execute("check_perishable_status", t_in)
    _append_tool(state, "check_perishable_status", t_in, t)
    state["node_outputs"]["perishable_check"] = t.get("data", {})
    state["iterations"] += 1
    return state


# ─── Node: Financial Impact ───────────────────────────────────────────────────

def financial_impact_node(state: RetailState) -> RetailState:
    """Calculate revenue, margin, and carrying cost impact."""
    logger.info("[FinancialImpact] sku=%s", state["sku"])
    sku = state["sku"]
    ent = state.get("entities", {})
    prod = mock_executor.PRODUCTS.get(sku, mock_executor.PRODUCTS["HUG48-3"])

    old_p = ent.get("old_price") or prod["base_price"]
    new_p = ent.get("new_price") or (prod["base_price"] + 1.50)
    base_vol = prod["base_demand_per_store_week"] * 30 * 8  # 8-week network
    elasticity = prod["elasticity"]
    pct_change = (new_p - old_p) / old_p
    new_vol = base_vol * (1 + elasticity * pct_change)

    t1_in = {"sku": sku, "old_price": old_p, "new_price": new_p,
              "old_volume_units": base_vol, "new_volume_units": max(0, new_vol),
              "include_trade_dollars": True}
    t1 = mock_executor.execute("calculate_revenue_impact", t1_in)
    _append_tool(state, "calculate_revenue_impact", t1_in, t1)

    excess = max(0.0, base_vol - new_vol) if new_p > old_p else 0
    t2_in = {"sku": sku, "excess_units": excess, "carrying_weeks": 4.0}
    t2 = mock_executor.execute("calculate_carrying_cost", t2_in)
    _append_tool(state, "calculate_carrying_cost", t2_in, t2)

    node_prompt = (
        f"Query: {state['query']}\n\n"
        f"Revenue impact: {json.dumps(t1.get('data', {}), indent=2)[:1200]}\n"
        f"Carrying cost: {json.dumps(t2.get('data', {}))}"
    )
    node_summary = _chat(node_prompt, _FINANCIAL_PROMPT, max_tokens=500)

    state["node_outputs"]["financial_impact"] = {
        "revenue_impact": t1.get("data", {}),
        "carrying_cost": t2.get("data", {}),
        "financial_summary": node_summary,
    }
    state["iterations"] += 1
    return state


# ─── Node: Synthesizer ────────────────────────────────────────────────────────

def synthesizer_node(state: RetailState) -> RetailState:
    """Merge all node outputs into a single coherent executive response."""
    logger.info("[Synthesizer] merging %d node outputs", len(state["node_outputs"]))

    node_summaries = []
    for node_name, output in state["node_outputs"].items():
        for key, val in output.items():
            if key.endswith("_summary") and isinstance(val, str):
                node_summaries.append(f"[{node_name.upper()}]\n{val}")

    if not node_summaries:
        # Fallback: use raw data
        node_summaries = [json.dumps(state["node_outputs"], indent=2)[:3000]]

    synth_prompt = (
        f"Original query: {state['query']}\n\n"
        f"Node agent outputs:\n\n"
        + "\n\n---\n\n".join(node_summaries)
        + (
            f"\n\nFreshness warnings: {'; '.join(set(state['freshness_warnings']))}"
            if state.get("freshness_warnings") else ""
        )
    )

    final = _chat(synth_prompt, _SYNTHESIZER_PROMPT, max_tokens=1500)
    state["final_response"] = final
    state["iterations"] += 1
    return state


# ─── Routing Logic ────────────────────────────────────────────────────────────

def _route_after_router(state: RetailState) -> str:
    """Conditional edge: router → domain node."""
    return state.get("route", "general")


def _route_after_supply(state: RetailState) -> str:
    """After supply disruption node, always go to carrier node."""
    return "carrier_node"


def _route_after_demand(state: RetailState) -> str:
    """After demand forecast, go to accuracy node."""
    return "accuracy_node"


def _route_after_shelf(state: RetailState) -> str:
    """After shelf replenishment, check perishable if applicable."""
    sku = state.get("sku", "")
    prod = mock_executor.PRODUCTS.get(sku, {})
    return "perishable_check_node" if prod.get("perishable") else "synthesizer"


def _route_after_price_cascade(state: RetailState) -> str:
    """After price cascade, run inventory impact."""
    return "inventory_node"


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph() -> Any:
    workflow = StateGraph(RetailState)

    # Register nodes
    workflow.add_node("router",                router_node)
    workflow.add_node("price_cascade",         price_cascade_node)
    workflow.add_node("inventory_node",        inventory_node)
    workflow.add_node("supply_disruption",     supply_disruption_node)
    workflow.add_node("carrier_node",          carrier_node)
    workflow.add_node("demand_forecast",       demand_forecast_node)
    workflow.add_node("accuracy_node",         accuracy_node)
    workflow.add_node("scenario_planning",     scenario_planning_node)
    workflow.add_node("shelf_replenishment",   shelf_replenishment_node)
    workflow.add_node("perishable_check_node", perishable_check_node)
    workflow.add_node("financial_impact",      financial_impact_node)
    workflow.add_node("synthesizer",           synthesizer_node)

    # Entry point
    workflow.set_entry_point("router")

    # Router → domain nodes (conditional)
    workflow.add_conditional_edges("router", _route_after_router, {
        "price_cascade":     "price_cascade",
        "supply_disruption": "supply_disruption",
        "demand_forecast":   "demand_forecast",
        "scenario_planning": "scenario_planning",
        "shelf_replenishment": "shelf_replenishment",
        "financial_impact":  "financial_impact",
        "general":           "price_cascade",   # general → full price cascade pipeline
    })

    # Domain node chains
    workflow.add_conditional_edges("price_cascade",   _route_after_price_cascade,
                                   {"inventory_node": "inventory_node"})
    workflow.add_edge("inventory_node", "financial_impact")
    workflow.add_edge("financial_impact", "synthesizer")

    workflow.add_conditional_edges("supply_disruption", _route_after_supply,
                                   {"carrier_node": "carrier_node"})
    workflow.add_edge("carrier_node", "synthesizer")

    workflow.add_conditional_edges("demand_forecast", _route_after_demand,
                                   {"accuracy_node": "accuracy_node"})
    workflow.add_edge("accuracy_node", "synthesizer")

    workflow.add_edge("scenario_planning", "synthesizer")

    workflow.add_conditional_edges("shelf_replenishment", _route_after_shelf, {
        "perishable_check_node": "perishable_check_node",
        "synthesizer": "synthesizer",
    })
    workflow.add_edge("perishable_check_node", "synthesizer")

    # Synthesizer → END
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

_GRAPH = None

def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_langgraph(query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Run the LangGraph workflow for a query.

    Returns same shape as orchestrator.run():
      response_text, tool_calls_made, iterations_used,
      data_freshness_warnings, node_outputs, error
    """
    graph = _get_graph()

    initial_state: RetailState = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "final_response": "",
        "intent": "",
        "route": "",
        "sku": "HUG48-3",
        "entities": {},
        "tool_calls_made": [],
        "freshness_warnings": [],
        "node_outputs": {},
        "iterations": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
        return {
            "response_text": final_state.get("final_response", "No response generated."),
            "tool_calls_made": final_state.get("tool_calls_made", []),
            "iterations_used": final_state.get("iterations", 0),
            "data_freshness_warnings": list(set(final_state.get("freshness_warnings", []))),
            "node_outputs": final_state.get("node_outputs", {}),
            "intent": final_state.get("intent", "unknown"),
            "sku": final_state.get("sku", ""),
            "updated_history": [],
            "error": None,
        }
    except Exception as exc:
        logger.exception("LangGraph execution error")
        return {
            "response_text": f"LangGraph error: {exc}",
            "tool_calls_made": [],
            "iterations_used": 0,
            "data_freshness_warnings": [],
            "node_outputs": {},
            "intent": "error",
            "sku": "",
            "updated_history": [],
            "error": str(exc),
        }


# ─── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "TruckCo B is on strike. What is the stockout risk for diapers?"
    print(f"\nQuery: {q}\n{'='*60}")
    result = run_langgraph(q)
    print(f"\nIntent:  {result['intent']}")
    print(f"SKU:     {result['sku']}")
    print(f"Iters:   {result['iterations_used']}")
    print(f"Tools:   {[t['tool'] for t in result['tool_calls_made']]}")
    print(f"\n--- Final Response ---\n{result['response_text']}")
