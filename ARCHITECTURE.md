# Retail Supply Chain Optimization AI — Architecture & Workflow Documentation

> **Version:** 2.0 — LangGraph multi-agent edition  
> **Model:** Claude claude-sonnet-4-6 (Anthropic)  
> **Stack:** Python 3.13 · Streamlit · LangGraph · LangChain Anthropic · Plotly

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Map](#2-module-map)
3. [Data Layer](#3-data-layer)
4. [Tool Layer](#4-tool-layer)
5. [Agent Layer — Direct Orchestrator (V1)](#5-agent-layer--direct-orchestrator-v1)
6. [Agent Layer — LangGraph Workflow (V2)](#6-agent-layer--langgraph-workflow-v2)
7. [Scenario Workflows — Step by Step](#7-scenario-workflows--step-by-step)
8. [How All Scenarios Are Integrated](#8-how-all-scenarios-are-integrated)
9. [UI Layer](#9-ui-layer)
10. [Edge Cases & Guardrails](#10-edge-cases--guardrails)
11. [Extending the System](#11-extending-the-system)

---

## 1. System Overview

This system is the **connective tissue** missing from retail supply chain operations at Walmart scale.

**The problem it solves:** A price change happens. Nobody tells the demand planners. The demand planners cut a stale forecast. The buyers cut a wrong PO. The warehouse over-stocks. The shelf gets marked down at a loss. The CFO sees a revenue miss three months later.

**The solution:** A multi-agent AI system that receives one query ("raise diaper price by $1.50") and instantly surfaces every downstream consequence — demand shifts, PO adjustments, DC rebalancing, financial impact, and recommended actions — across HQ, DCs, and 30 stores.

### Three Core Use Cases (from the business transcript)

| Use Case | Trigger | Chain of effects modelled |
|---|---|---|
| **Price Cascade** | Merchant raises/cuts a price | Demand → Inventory → PO → DC allocation → Finance |
| **Supply Disruption** | Carrier strike / Port delay / Supplier bankruptcy | Stockout timeline → Regional alternate carriers → Revenue at risk → Mitigation plan |
| **Demand Forecasting** | Accuracy analysis or forecast request | 15-variable demand model → CI fan chart → Accuracy gate → Revenue impact |

Plus three supporting workflows: Scenario Planning, Shelf Replenishment, and Financial Impact.

---

## 2. Module Map

```
Optimization/
│
├── config/
│   └── settings.py          # All tuneable constants (API key, model, iteration limits,
│                            #   freshness thresholds, replenishment lag, CI widening rate)
│
├── data/
│   └── schemas.py           # Pydantic models for every data structure:
│                            #   ToolResult, DemandForecast, StockoutRisk,
│                            #   PriceCascadeResult, SupplyDisruptionResult,
│                            #   ScenarioConflict, RevenueImpact, AgentResponse
│
├── tools/
│   ├── tool_definitions.py  # 17 Claude tool schemas (JSON Schema) used by both
│   │                        #   the direct orchestrator and LangGraph nodes
│   └── mock_executor.py     # Mock retail data + all tool implementations
│                            #   execute(tool_name, tool_input) → ToolResult dict
│                            #   Never raises — all failures return error dict
│
├── agents/
│   ├── orchestrator.py      # V1: Direct Anthropic SDK agentic loop
│   │                        #   • Configurable max iterations (6–20 by complexity)
│   │                        #   • History summarization at 12+ messages
│   │                        #   • Provenance tracking (OLAP/WMS/OLTP)
│   └── langgraph_flow.py    # V2: LangGraph multi-agent graph
│                            #   • 12 specialized nodes
│                            #   • Conditional routing by intent
│                            #   • Domain-specific system prompts per node
│                            #   • Synthesizer node merges all outputs
│
└── ui/
    └── streamlit_app.py     # 9-tab Streamlit frontend
                             #   • Dashboard, Chat (V1+V2), Price Cascade,
                             #     Supply Alert, Demand Forecast, Scenario Planner,
                             #     Shelf & Store, Financial Impact, Data Sources
```

---

## 3. Data Layer

### 3.1 Mock Retail Data (mock_executor.py)

The system ships with realistic Walmart-scale mock data. No external DB is needed.

#### Products (6 SKUs)

| SKU | Name | Price | Elasticity | Asymmetric | Perishable | VMI% |
|---|---|---|---|---|---|---|
| HUG48-3 | Huggies Size 3 48-ct | $12.99 | −1.40 | Yes | No | 0% |
| PAM72-5 | Pampers Size 5 72-ct | $18.99 | −1.35 | Yes | No | 0% |
| MLK-GAL | Whole Milk 1 Gallon | $4.29 | −0.60 | No | **Yes** (14d) | 30% |
| TAB-DIN | Dinnerware Set 16-pc | $29.99 | −0.90 | No | No | 10% |
| BLK-THR | Fleece Throw Blanket | $14.99 | −1.10 | No | No | 0% |
| CIG-PKT | Cigarettes Premium | $8.99 | −0.50 | **Very** (0.4x) | No | 0% |

**Asymmetric elasticity** means: for diapers/tobacco/alcohol/formula, a price *increase* suppresses demand at the full elasticity rate, but a price *decrease* only recovers demand at a fraction (recovery_factor) of that rate. Demand is sticky downward.

#### Carriers (4 carriers, 1 on strike)

| Carrier | Status | Cargo | Regions | Lead Time |
|---|---|---|---|---|
| TruckCo_A | Active | Tableware, Linen | NW, MW | 3 days |
| **TruckCo_B** | **ON STRIKE** | **Diapers, Formula** | **SE, MW** | 2 days |
| TruckCo_C | Active | Dairy, Produce | NW, SE | 2 days |
| TruckCo_D | Active | General, Mixed | All | 4 days (+45% cost premium for diapers) |

#### Distribution Centers (3 DCs, 10 stores each)

| DC | Location | Region | Diaper Inventory (WMS) |
|---|---|---|---|
| DC-NW | Seattle, WA | NW | 8,500 units |
| DC-SE | Atlanta, GA | SE | 3,200 units ← most at risk |
| DC-MW | Chicago, IL | MW | 6,100 units |

### 3.2 Data Source Provenance

Every tool result returns provenance metadata:

```python
{
  "data": {...},
  "error": None,           # structured error if tool failed — never empty zeros
  "provenance": "WMS",     # OLAP | OLTP | WMS
  "freshness_minutes": 15, # minutes since last upstream refresh
  "is_stale": False        # True if below accuracy threshold or beyond freshness window
}
```

| Source | Lag | Use For | Risk |
|---|---|---|---|
| OLTP | ~5 min | Pricing, PO creation, financial posting | Low |
| WMS | ~15 min | Operational inventory, stockout, replenishment | Low-Medium |
| OLAP | 24 hours | Trend analysis, reporting, forecast inputs | **HIGH for ops** — use WMS |
| Carrier API | 15–30 min | Carrier status, alternate availability | Medium |
| Competitor Feed | 4–6 hr | Competitive pricing context | Medium |

---

## 4. Tool Layer

### 4.1 The 17 Tools

```
PRICING TOOLS
  simulate_price_change       Full cascade: demand → inventory → PO → financial
  get_competitive_pricing     Competitor price context for decision

DEMAND TOOLS
  get_demand_forecast         15-variable model, CI bands, accuracy gate
  calculate_price_elasticity  Asymmetric elasticity by product class
  get_forecast_accuracy       MAPE at any horizon, gap-to-benchmark analysis

INVENTORY TOOLS
  get_inventory_levels        WMS + OLAP comparison, discrepancy flagging
  check_shelf_capacity        Planogram constraint — overflow to back-storage
  calculate_stockout_risk     Lag + variability modelling, emergency threshold
  check_perishable_status     Dairy/produce max-DOS cap, write-off risk
  trigger_replenishment       Standard / expedited / emergency with cost model

SUPPLY CHAIN TOOLS
  get_carrier_status          All carriers: active/strike/delayed + cargo map
  find_alternate_carriers     Regional (not national) availability check
  get_supply_disruption_impact Full impact: stockout timeline + mitigation plan

FINANCIAL TOOLS
  calculate_revenue_impact    Gross → trade offset → net → margin → tax (approx.)
  calculate_carrying_cost     Owned inventory only (VMI excluded)

SCENARIO TOOLS
  detect_scenario_conflicts   Promo + strike = CRITICAL; time-anchored required
  run_scenario_comparison     Up to 4 scenarios, demand + revenue + margin
```

### 4.2 Tool Execution Pattern

```python
# Every tool call goes through this wrapper — never raises, never returns empty zeros
result = mock_executor.execute(tool_name, tool_input)

# Result always has this shape:
{
  "data": {...},          # structured output
  "error": None,          # or "Tool 'X' failed: ... Retry recommended."
  "provenance": "OLTP",   # data source
  "freshness_minutes": 5, # lag
  "is_stale": False       # staleness flag
}
```

---

## 5. Agent Layer — Direct Orchestrator (V1)

**File:** `agents/orchestrator.py`  
**Pattern:** Single-agent agentic loop (standard Anthropic tool-use pattern)

### Flow Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  ORCHESTRATOR                                        │
│                                                      │
│  1. Detect complexity → set max_iterations (6-20)    │
│  2. Summarize history if ≥ 12 messages               │
│  3. Send to Claude with ALL 17 tools + system prompt │
│                                                      │
│  ┌─────────────────────────────────────────────┐    │
│  │  AGENTIC LOOP (max N iterations)             │    │
│  │                                              │    │
│  │  Claude decides which tools to call          │    │
│  │       ↓                                      │    │
│  │  mock_executor.execute(tool, input)          │    │
│  │       ↓                                      │    │
│  │  Result + provenance fed back to Claude      │    │
│  │       ↓                                      │    │
│  │  Claude reasons, decides next tool or DONE  │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  4. Track OLAP/WMS provenance → freshness warnings   │
│  5. Return: response_text + tool_calls + warnings    │
└─────────────────────────────────────────────────────┘
    │
    ▼
Structured AgentResponse → Streamlit UI
```

### Complexity-Based Iteration Caps

| Query type | Max iterations |
|---|---|
| Simple lookup ("what is current inventory") | 6 |
| Standard (price cascade, single SKU) | 10 |
| Complex (tariff shock, multi-SKU, multi-DC) | 20 |

### History Summarization

At ≥ 12 messages, older turns are condensed by Claude into a running summary. Only the last 4 messages + summary are passed forward, keeping the context window manageable across long sessions.

---

## 6. Agent Layer — LangGraph Workflow (V2)

**File:** `agents/langgraph_flow.py`  
**Pattern:** Directed acyclic graph with specialized nodes per domain

### Full Graph Topology

```
                         ┌──────────────┐
         user query ───► │    ROUTER    │  intent classification + entity extraction
                         └──────┬───────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          │                     │                      │
          ▼                     ▼                      ▼
  [price_cascade]    [supply_disruption]   [demand_forecast]   [scenario_planning]
  (pricing agent)    (supply chain agent)  (demand agent)      (scenario agent)
          │                     │                      │              │
          ▼                     ▼                      ▼              │
  [inventory_node]   [carrier_node]        [accuracy_node]            │
  (inventory agent)  (carrier lookup)      (MAPE analysis)            │
          │                     │                      │              │
          ▼                     │                      │              │
  [financial_impact] ◄──────────┘                      │              │
  (financial agent)                                    │              │
          │                                            │              │
          └────────────────────────────────────────────┘──────────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │   SYNTHESIZER   │ merges all outputs
                                              └────────┬────────┘
                                                       │
                                                      END

Additional paths:
  [shelf_replenishment] ──► [perishable_check_node] (if dairy/produce) ──► SYNTHESIZER
  [shelf_replenishment] ────────────────────────────────────────────────► SYNTHESIZER
```

### Node Responsibilities

| Node | Domain | Tools Called | System Prompt Focus |
|---|---|---|---|
| **router** | Classification | LLM only | Extract intent + SKU + entities from query |
| **price_cascade** | Pricing | simulate_price_change, get_competitive_pricing, get_inventory_levels | Asymmetric elasticity, PO adjustments, cascade |
| **inventory_node** | Inventory | calculate_stockout_risk, check_shelf_capacity | Lag + variability, emergency threshold |
| **supply_disruption** | Supply Chain | get_supply_disruption_impact, get_carrier_status | Duration model, stockout timelines |
| **carrier_node** | Logistics | find_alternate_carriers (×3 regions) | Regional gaps, cost premium, capacity % |
| **demand_forecast** | Demand | get_demand_forecast | 15 variables, CI widening, accuracy gate |
| **accuracy_node** | Demand | get_forecast_accuracy | MAPE gap, revenue impact of improvement |
| **scenario_planning** | Strategy | run_scenario_comparison, detect_scenario_conflicts | Conflict detection, time-anchored |
| **shelf_replenishment** | Operations | calculate_stockout_risk, check_shelf_capacity, trigger_replenishment | Planogram cap, lag, perishable cap |
| **perishable_check_node** | Operations | check_perishable_status | Dairy/produce max-DOS, write-off risk |
| **financial_impact** | Finance | calculate_revenue_impact, calculate_carrying_cost | Trade dollars, VMI split, tax approx. |
| **synthesizer** | All | LLM only | Executive summary: findings + actions + dollar impact + caveats |

### Routing Logic (Conditional Edges)

```python
# After router: route by intent
router → price_cascade       (intent == "price_cascade")
router → supply_disruption   (intent == "supply_disruption")
router → demand_forecast     (intent == "demand_forecast")
router → scenario_planning   (intent == "scenario_planning")
router → shelf_replenishment (intent == "shelf_replenishment")
router → financial_impact    (intent == "financial_impact")
router → price_cascade       (intent == "general"  # default full pipeline)

# Chained after domain nodes:
price_cascade      → inventory_node → financial_impact → synthesizer
supply_disruption  → carrier_node → synthesizer
demand_forecast    → accuracy_node → synthesizer
scenario_planning  → synthesizer
shelf_replenishment → perishable_check_node (if perishable SKU) → synthesizer
                   → synthesizer (non-perishable shortcut)
financial_impact   → synthesizer
```

### V1 vs V2 Comparison

| Feature | V1 — Direct Orchestrator | V2 — LangGraph |
|---|---|---|
| Architecture | Single agent, all 17 tools | 12 specialized nodes, curated tool subsets |
| Routing | Claude decides implicitly | Explicit conditional edges by intent |
| System prompt | One global prompt | Per-node domain-focused prompt |
| Tool visibility | All 17 tools always available | Only relevant tools per node |
| Auditability | Tool call list | Node-by-node trace with node_outputs dict |
| Parallelism | Sequential | Supports parallel node execution |
| Context management | History summarization | Stateful graph state (TypedDict) |
| Best for | Complex multi-turn conversation | Single-query deep analysis |

---

## 7. Scenario Workflows — Step by Step

### 7.1 Price Cascade Workflow

**Trigger:** "Raise Huggies diapers from $12.99 to $14.49"

```
Step 1 — Router classifies intent as "price_cascade", extracts SKU=HUG48-3

Step 2 — price_cascade_node calls:
  ├── simulate_price_change(HUG48-3, 12.99, 14.49, horizon=8)
  │     → price_change_pct = +11.55%
  │     → elasticity = -1.4 (asymmetric — price increase uses full elasticity)
  │     → demand_change_pct = -16.17%
  │     → old_demand = 50 units/store/week → new_demand = 41.9
  │     → excess_units = 1,944 over 8 weeks (30 stores)
  │     → PO-2025-0341: reduce by 810 units (ETA 45d, adjustable)
  │     → PO-2025-0389: reduce by 486 units (ETA 75d, adjustable)
  ├── get_competitive_pricing(HUG48-3)
  │     → Target: $13.49, Costco: $11.89 → we'd be above Costco at $14.49
  └── get_inventory_levels(HUG48-3, dc)
        → DC-SE only has 3,200 units (14d supply) — most vulnerable

Step 3 — inventory_node calls:
  ├── calculate_stockout_risk(HUG48-3, STR-011)
  │     → days_on_hand = 3.1d, effective_lag = 4.9d → CRITICAL
  └── check_shelf_capacity(HUG48-3, STR-011, 48)
        → fits within planogram (48 max facings)

Step 4 — financial_impact_node calls:
  ├── calculate_revenue_impact(HUG48-3, 12.99, 14.49, 12000, 10056)
  │     → gross_revenue_change = -$25,890 (volume loss > price gain)
  │     → vendor_trade_offset = $0 (no promo — trade doesn't apply)
  │     → net_revenue_change = -$25,890
  │     → margin_change = -$5,696
  └── calculate_carrying_cost(HUG48-3, excess=1944, weeks=4)
        → carrying_cost = $2,840 (100% owned, 25% annual rate)

Step 5 — synthesizer merges:
  → Executive summary with: price change %, demand drop, PO adjustments needed,
    financial impact, competitive position warning, recommended actions
```

**Key insight:** A $1.50 price increase on diapers generates a $25K revenue *loss* per 8-week period at network scale because the volume decline outweighs the per-unit gain. Asymmetric elasticity means if you reverse the price later, you won't recover the lost volume at the same rate.

---

### 7.2 Supply Disruption Workflow (Carrier Strike)

**Trigger:** "TruckCo B is on strike — what do we do?"

```
Step 1 — Router: intent=supply_disruption, carrier=TruckCo_B, duration=14 days

Step 2 — supply_disruption_node calls:
  ├── get_supply_disruption_impact(carrier_strike, TruckCo_B, 14)
  │     → affected_skus = [HUG48-3, PAM72-5]  (TruckCo B cargo)
  │     → DC-SE: 3,200 units / 714 daily = 4.5 days DC supply
  │     → STR-011: 22 units / 7.1 daily = 3.1 days store supply
  │     → Stockout in stores: Day 3
  │     → Stockout in DC: Day 14 (matches strike duration)
  │     → Revenue at risk: $2.84M
  └── get_carrier_status(all)
        → TruckCo_B: STRIKE ⚠  │  TruckCo_A: active  │  TruckCo_C: active
          TruckCo_D: active

Step 3 — carrier_node calls find_alternate_carriers for each region:
  ├── Region NW: TruckCo_A ✓ (can handle diapers at 60% capacity, +25% cost, +1d)
  ├── Region SE: TruckCo_C ✗ (refrigerated — cannot carry diapers)
  │             TruckCo_D ✓ (available, 100% capacity, +45% cost, +2d)
  │             ⚠ WARNING: Only 1 viable alternate in SE at 45% premium
  └── Region MW: TruckCo_A ✓ (40% capacity, +35% cost, +2d)
                TruckCo_D ✓ (100% capacity, +45% cost, +1d)

Step 4 — synthesizer generates:
  → IMMEDIATE: Trigger emergency order via TruckCo D (SE), TruckCo A (NW)
  → IMMEDIATE: Price hold to suppress demand, extend days-on-hand
  → 24h: Inter-DC transfer DC-NW surplus → DC-SE (8,500 → 3,200 rebalance)
  → 24h: Prioritize SE allocation to high-volume (tier-1) stores first
  → Ongoing: Monitor TruckCo B expected resolution 2026-04-24
```

**Key insight:** Regional availability matters more than national. TruckCo C is active nationally but serves dairy — it cannot carry diapers. The SE region has only one viable alternate (TruckCo D) at a 45% cost premium. This is a single point of failure that must be flagged.

---

### 7.3 Demand Forecast Workflow

**Trigger:** "What is our forecast accuracy for diapers? What's the revenue impact of the gap?"

```
Step 1 — Router: intent=demand_forecast, sku=HUG48-3, horizon=8

Step 2 — demand_forecast_node calls:
  └── get_demand_forecast(HUG48-3, horizon=8)
        → accuracy = 78% MAPE at 8W (above 60% minimum, below 85% benchmark)
        → base demand = 50 units/store/week
        → Week 1: point=50.0, 80% CI [46.5, 53.5], 95% CI [43.2, 56.8]
        → Week 8: point=52.8, 80% CI [40.1, 65.5], 95% CI [33.6, 72.0]
          (CI widens 15% per 4-week period)
        → Key drivers: seasonality (+2.3), competitor_price (+1.1), tariff (-0.5)
        → is_reliable = True (≥ 60%)

Step 3 — accuracy_node calls:
  └── get_forecast_accuracy(HUG48-3, 8)
        → accuracy = 0.78
        → benchmark = 0.85
        → gap = 0.07 (7 percentage points)
        → revenue_impact: "A 7-point accuracy gap costs ~7% in revenue efficiency.
          At $100B retail scale, each 1% = ~$1B revenue impact."

Step 4 — synthesizer:
  → 78% accuracy at 8W — acceptable but 7pts below benchmark
  → Improving by 7pts could represent $7B+ in revenue efficiency at Walmart scale
  → CI widens dramatically past week 4 — beyond this horizon, treat as directional only
  → Key levers: incorporate competitor_price signal, improve seasonality model
```

**Key insight:** Demand is a class function, not a time series. The 15-variable model captures price, promo, markdown, tariff, weather, seasonality, and 9 more. Each variable has a learned elasticity coefficient. A 1% improvement in accuracy ≈ 1% improvement in top-line revenue.

---

### 7.4 Scenario Planning + Conflict Detection Workflow

**Trigger:** "We're planning a 10% promo on diapers. But TruckCo B is on strike."

```
Step 1 — Router: intent=scenario_planning, sku=HUG48-3

Step 2 — scenario_planning_node calls:
  ├── run_scenario_comparison(HUG48-3, [Baseline, PriceUp, Promo], 8W)
  │     → Baseline:  $12.99, 50 units/store, $584K revenue
  │     → Price Up:  $14.49, 41.9 units/store, $558K revenue (volume loss dominates)
  │     → Promo:     $11.99 + 15% uplift, 60.5 units/store, $626K revenue
  │                  (but: supply constrained by strike!)
  └── detect_scenario_conflicts([Promo, Strike])
        → time overlap: both active May 1–24
        → Promo: demand_direction = "increase"
        → Strike: supply_direction = "constrain"
        → CRITICAL CONFLICT: demand spike during supply constraint
          → "Promotion during TruckCo B strike creates peak demand at zero supply."
          → Recommendation: defer promotion until strike resolves (est. Apr 24)

Step 3 — synthesizer:
  → Revenue: Promo scenario wins at $626K BUT it has a CRITICAL conflict with the strike
  → Safe choice: Hold price at baseline until TruckCo B resolves
  → After resolution: run Promo scenario for maximum revenue
  → Never run Promo + Strike simultaneously — stockout at peak demand = customer churn
```

---

### 7.5 Shelf & Store Replenishment Workflow

**Trigger:** "STR-005 has low milk inventory. What should we do?"

```
Step 1 — Router: intent=shelf_replenishment, sku=MLK-GAL, store=STR-005

Step 2 — shelf_replenishment_node calls:
  ├── calculate_stockout_risk(MLK-GAL, STR-005)
  │     → current_inventory = 18 units
  │     → daily_demand = 200/7 = 28.6 units/day
  │     → days_on_hand = 0.63d ← CRITICAL
  │     → effective_lag = 4d (normal) + 0.9d (variability buffer) = 4.9d
  │     → days_above_lag = 0.63 - 4.9 = -4.3d (SHELF HIT ZERO BEFORE REPLENISHMENT)
  │     → Severity: CRITICAL — trigger emergency transfer NOW
  ├── check_shelf_capacity(MLK-GAL, STR-005, 48)
  │     → shelf_capacity = 60 units
  │     → current_on_shelf = 18
  │     → space_available = 42 units → 48 units won't fit (overflow = 6 to back-storage)
  └── trigger_replenishment(MLK-GAL, DC-NW, STR-005, 48, priority=emergency)
        → ETA: 1 day (emergency), cost = $14.40 (3× standard rate)

Step 3 — perishable_check_node (milk is perishable):
  └── check_perishable_status(MLK-GAL, STR-005, 48)
        → shelf_life = 14 days, max_dos_cap = 3 days
        → projected_total = 18 + 48 = 66 units
        → projected_dos = 66 / 28.6 = 2.3 days ← within 3-day cap ✓
        → No write-off risk — OK to proceed

Step 4 — synthesizer:
  → CRITICAL: Shelf empty in 0.63 days. Emergency replenishment ordered (ETA 1d).
  → 42 units to shelf, 6 units to back-storage (planogram constraint).
  → Perishable cap check passed — 2.3 days projected supply within 3-day limit.
  → Freight cost: $14.40 (emergency premium). Revenue protected: $4.29 × 28.6 × 4+ days = $491.
```

---

## 8. How All Scenarios Are Integrated

The system is not a collection of isolated calculators. All scenarios share the same:

### 8.1 Shared State (LangGraph)

```python
class RetailState(TypedDict):
    query: str
    messages: List[BaseMessage]
    intent: str          # routes the graph
    sku: str             # propagated to all nodes
    entities: Dict       # price, carrier, store — shared across nodes
    tool_calls_made: List[Dict]   # accumulated by every node
    freshness_warnings: List[str] # accumulated across all tool calls
    node_outputs: Dict           # each node deposits its result here
    final_response: str          # synthesizer writes final answer
```

Every node reads from and writes to this shared state. The synthesizer sees all node outputs.

### 8.2 Cross-Scenario Data Flow

```
Price change
    │
    ├──► affects demand
    │         │
    │         ├──► affects PO quantities
    │         │         │
    │         │         └──► affects inventory levels
    │         │                   │
    │         │                   └──► affects stockout risk
    │         │                             │
    │         └──► affects revenue           └──► triggers replenishment
    │                   │
    │                   └──► affects margin (via trade dollars, VMI, carrying cost)
    │
    └──► promo + supply disruption → CONFLICT DETECTION
              │
              └──► if both active → CRITICAL warning → defer promo
```

### 8.3 The Connecting Thread: Demand as the PO Driver

From the business transcript: *"The demand becomes the actual buy quantity over a period of time."*

```
Price change ──► Demand change (elasticity)
Promo        ──► Demand change (promo coefficient)
Tariff       ──► Demand change (tariff coefficient)
Strike       ──► Supply constraint (demand signal unchanged but supply = 0)
                      │
                      ▼
               DEMAND SIGNAL
                      │
                      ▼
               PO SYSTEM cuts purchase orders against this signal
                      │
                      ├──► wrong signal → wrong PO → wrong inventory
                      └──► right signal → right PO → right shelf fill rate
```

Improving forecast accuracy by 7 points converts wrong demand signal → right PO at scale. At Walmart ($500B revenue), that's $3.5B+ in revenue efficiency.

---

## 9. UI Layer

**File:** `ui/streamlit_app.py`  
9 tabs + sidebar. Works without an API key for all analytical tabs (direct mock data). Chat tab requires API key.

### Tab Map

| Tab | Works without API key | Key components |
|---|---|---|
| **Dashboard** | Yes | Carrier status cards, DC inventory snapshot, strike timeline chart |
| **Chat** | No | st.status (live tool tracking), st.write_stream (streaming response), session stats |
| **Price Cascade** | Yes | Waterfall chart, demand before/after bars, PO adjustment table |
| **Supply Alert** | Yes | Stockout bar chart, alternate carrier chips, mitigation plan |
| **Demand Forecast** | Yes | Accuracy gauge, fan chart with CI bands, variable contributions |
| **Scenario Planner** | Yes | 4-scenario comparison, conflict alerts, revenue + margin bars |
| **Shelf & Store** | Yes | Gantt replenishment chain, planogram check, perishable cap |
| **Financial Impact** | Yes | P&L waterfall with vendor trade, carrying cost, VMI note |
| **Data Sources** | Yes | Provenance table, WMS vs OLAP discrepancy chart |

### Sidebar: Pipeline Version Toggle

```
Sidebar ──► Pipeline Version: [Agentic Loop (V1)]  [LangGraph (V2)]
                                   │                        │
                              orchestrator.py        langgraph_flow.py
                            (full tool loop)         (specialized nodes)
```

---

## 10. Edge Cases & Guardrails

All edge cases from the original business transcript are handled:

| Edge Case | Where Handled | Behavior |
|---|---|---|
| Asymmetric elasticity (diapers, tobacco) | mock_executor._simulate_price_change | recovery_factor applied on price decrease |
| Forecast accuracy < 60% | mock_executor._get_demand_forecast | is_reliable=False, marked UNRELIABLE, not passed downstream |
| CI widening at long horizons | mock_executor._ci_bounds | 15% wider per 4-week period |
| Regional carrier gap (no alternates) | mock_executor._find_alternate_carriers | Explicit critical warning, not just "unavailable" |
| Replenishment lag variability | mock_executor._calculate_stockout_risk | 30% × 3 extra days added to effective lag |
| Perishable cap (dairy 3-day max) | mock_executor._check_perishable_status | Write-off risk in USD, markdown recommendation |
| Planogram overflow | mock_executor._check_shelf_capacity | Excess → back-of-store, not shelf overflow |
| Scenario conflicts (promo + strike) | mock_executor._detect_scenario_conflicts | CRITICAL flag if demand spike + supply constraint overlap |
| Missing time anchor on scenarios | detect_scenario_conflicts tool note | Warns comparison is invalid without start_date |
| OLAP vs WMS discrepancy | get_inventory_levels | Both values shown; discrepancy flagged > 100 units |
| VMI vs owned carrying cost | calculate_carrying_cost | Cost only on owned fraction (vmi_pct excluded) |
| Vendor trade dollars | calculate_revenue_impact | Net against promotional cost before margin reporting |
| Tax jurisdiction variance | calculate_revenue_impact | Per-jurisdiction rate; always flagged as approximate |
| Tool failure / timeout | mock_executor.execute() | try/except → error dict with retry recommendation; never returns empty zeros |
| Context window overflow | orchestrator._summarize_history | Condense at ≥ 12 messages using Claude summarization |
| Max iterations exceeded | orchestrator.run() | User-facing warning + partial analysis note |
| Supplier bankruptcy (permanent) | get_supply_disruption_impact | duration=120d modelled; strategic supplier qualification triggered |

---

## 11. Extending the System

### Add a new SKU

In `tools/mock_executor.py`, add an entry to `PRODUCTS` with all required fields:
```python
"NEW-SKU": {
    "name": "...", "category": "...", "product_class": "...",
    "base_price": X.XX, "base_demand_per_store_week": N,
    "elasticity": -X.X, "asymmetric": True/False, "recovery_factor": 0.X,
    "margin_pct": 0.X, "vendor_trade_pct": 0.X, "vmi_pct": 0.X,
    "freight_term": "FOB", "perishable": True/False,
    "shelf_facings_max": N, "safety_stock_days": N, "open_pos": [],
}
```

### Add a new tool

1. Implement the function in `tools/mock_executor.py` following the `_result()` / `_error_result()` pattern
2. Add the tool schema to `tools/tool_definitions.py` (ALL_TOOLS list)
3. Register in `TOOL_MAP` at the bottom of `mock_executor.py`
4. Add to a relevant LangGraph node in `agents/langgraph_flow.py`

### Add a new agent node (LangGraph)

1. Write a `new_node(state: RetailState) -> RetailState` function
2. Add `workflow.add_node("new_node", new_node)`
3. Add edges to/from the new node
4. Update the router's conditional edge map if it's a new intent

### Connect a real data source

Replace `mock_executor.execute()` calls with real API calls — the return shape is identical. The provenance, freshness, and is_stale fields are the integration contract.

```python
def _get_inventory_levels(sku, location_type, location_id):
    try:
        # Replace mock with:
        result = snowflake_client.query("SELECT ... FROM inventory WHERE sku=?", sku)
        return _result(result, provenance="WMS", freshness=15)
    except Exception as exc:
        return _error_result("get_inventory_levels", exc)
```

---

*Last updated: 2026-04-11*
