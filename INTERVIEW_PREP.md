# Interview Preparation Guide — Retail Supply Chain Optimization AI

> **Audience:** Technical + non-technical interview rounds for AI/ML Engineering, Data Science, Product, and Solutions Architecture roles.  
> **Format:** Each section answers *Why*, *What*, and *How* so you can speak to any depth the interviewer probes.

---

## Table of Contents

1. [Domain Knowledge — Retail Supply Chain](#1-domain-knowledge--retail-supply-chain)
2. [Business Case — The Problem, Value, and ROI](#2-business-case--the-problem-value-and-roi)
3. [Cost Structure — LLM, Infrastructure, and Optimization](#3-cost-structure--llm-infrastructure-and-optimization)
4. [Technical Architecture — Component Deep-Dives](#4-technical-architecture--component-deep-dives)
5. [Design Decisions and Trade-offs](#5-design-decisions-and-trade-offs)
6. [Interview Q&A — Expected Questions with Model Answers](#6-interview-qa--expected-questions-with-model-answers)

---

## 1. Domain Knowledge — Retail Supply Chain

### 1.1 The Demand Signal: The Driver of Everything

Every supply chain decision flows from demand. When demand is wrong, everything downstream is wrong.

```
Price change ──► Demand change (elasticity model)
Promo        ──► Demand uplift (coefficient × base)
Tariff       ──► Demand suppression (cost pass-through)
Weather      ──► Demand spike or drop (seasonal signals)
                      │
                      ▼
               DEMAND SIGNAL (15-variable model)
                      │
                      ▼
        PO System ── Buyer commits purchase orders
                      │
                      ▼
        DC Inventory ── Warehouse receives and allocates
                      │
                      ▼
        Store Shelf ── Customer buys
```

**Interview point:** "Demand is the actual buy quantity over a period. If the demand model is wrong by 7%, the PO is wrong by 7%, the DC stocks wrong by 7%, and the store shelf is wrong by 7%. At $500B Walmart scale, 1% demand accuracy improvement = ~$5B in revenue efficiency."

---

### 1.2 Price Elasticity — Including Asymmetric Behavior

**What it is:** Elasticity measures how sensitive demand is to price. Elasticity = −1.4 means a 10% price increase reduces demand by 14%.

**Formula:**
```
Demand_delta% = Elasticity × Price_delta%
```

**Asymmetric elasticity (critical for interview):**
For diapers, tobacco, alcohol, infant formula — *customers hurt more by price increases than they benefit from price cuts.*

```
Price ↑: Demand drops at full elasticity rate (×1.0)
Price ↓: Demand recovers at only 60–70% of that rate

Example (Huggies HUG48-3, elasticity = -1.4, recovery_factor = 0.70):
  +10% price → -14% demand          (full elasticity)
  -10% price →  +9.8% demand        (70% of 14%)
  
Net: You can lose 14% demand but only get 9.8% back. Promotions are
expensive; reversals are lossy. Never assume symmetric recovery.
```

**Why it matters:** A pricing analyst who ignores asymmetry will over-estimate promotion lift and under-estimate the damage of a price reversal. This is a $10M+ mistake at scale.

---

### 1.3 Demand Forecasting — 15-Variable Multiplicative Model

**Model structure:**
```
D(t) = Base_Demand × Π(1 + fᵢ × wᵢ)

Where fᵢ = factor signal (observed), wᵢ = learned weight

15 variables:
  price, promo, tariff, weather, seasonality, trend,
  competitor_price, social_sentiment, days_supply,
  channel_mix, planogram_compliance, regional_income,
  household_penetration, repeat_rate, new_item_velocity
```

**Confidence intervals (CI):**
- Week 1 CI is narrow (±7%). Week 8 CI is wide (±30%).
- Formula: `CI_width = Base_MAPE × (1 + 0.05 × Horizon_weeks)`
- **Interview rule:** Never cite a single forecast point as truth. Always give the range.

**Accuracy gate:**
- If forecast MAPE < 60%: mark `is_reliable=False`, block from PO system.
- Reason: An unreliable forecast fed to automated PO creation is worse than no forecast.

**MAPE vs. Benchmark (interview question: "how do you measure forecast quality"):**
```
MAPE (Mean Absolute Percentage Error) = mean(|actual - forecast| / actual)
Our model: 78% at 8 weeks
Industry benchmark: 85%
Gap: 7 percentage points

Revenue impact of gap: at $500B scale, 7% accuracy gap ≈ $35B in suboptimal purchasing
```

---

### 1.4 Inventory Management — Safety Stock, ROP, VMI

**Reorder Point (ROP):**
```
ROP = Lead_Time_days × Avg_Daily_Demand + Safety_Stock
Safety_Stock = Z × σ_demand × √Lead_Time

Where Z = 1.65 for 95% service level
```

**Replenishment lag and variability (interview critical):**
```
HQ → DC → Store = 3–4 days standard
But: 30% probability of +3 extra days (weather, customs, DC congestion)

Effective_Lag = base_lag + (0.30 × 3_extra_days) = 3.9–4.9 days

If days_on_hand < effective_lag: STOCKOUT GUARANTEED before delivery
→ Trigger emergency replenishment NOW, not when shelf is empty
```

**VMI (Vendor-Managed Inventory):**
- Some inventory (e.g., 30% of milk) is owned and managed by the vendor, not by you.
- **Carrying cost applies only to OWNED inventory.**
- `Carrying_Cost = Owned_Units × Annual_Rate × (Days / 365)`
- Interview trap: Many candidates calculate carrying cost on total inventory, missing VMI exclusion.

**Perishable cap (dairy example):**
```
Max_Order = (Shelf_Life_days - Lead_Time) × Avg_Daily_Demand × Planogram_Cap
Dairy max DOS = 3 days (hard cap)
Never order more than 3 days of dairy regardless of demand signal.
Why: expired inventory write-off cost > stockout cost
```

---

### 1.5 Supply Chain Logistics — Carriers, DCs, Regional Coverage

**Network topology (this system):**
```
3 DCs: DC-NW (Seattle), DC-SE (Atlanta), DC-MW (Chicago)
30 stores: 10 per DC, spread across NW / SE / MW regions
4 carriers: TruckCo_A (tableware), TruckCo_B (diapers — STRIKE),
            TruckCo_C (dairy), TruckCo_D (general, all regions)
```

**Key insight — regional availability:**
> "A carrier available nationally but absent from the affected region provides zero benefit."

TruckCo_C is active nationally but handles only dairy. It cannot carry diapers. The SE region has only TruckCo_D for diaper delivery — at a 45% cost premium. This single-point-of-failure is invisible to anyone looking at national carrier capacity.

**Days-to-stockout calculation:**
```
DTS = Current_Inventory / Avg_Daily_Demand
Revenue_at_Risk = max(0, Replenishment_Lag - DTS) × Daily_Revenue

If DTS < Lag: shelf goes empty before the truck arrives
```

---

### 1.6 Data Freshness — OLAP vs. WMS vs. OLTP

| Source | Lag | Use For | Risk |
|--------|-----|---------|------|
| OLTP | ~5 min | Pricing, PO creation, GL entries | Low |
| WMS | ~15 min | Operational inventory, stockout decisions | Low-Medium |
| OLAP | 24 hours | Trend analysis, forecasting inputs | **HIGH for ops** |

**Interview scenario:** "WMS says DC-SE has 3,200 diaper units. OLAP says 4,100. Which do you use?"
→ **WMS for operational decisions** (15-min lag). OLAP is from yesterday's batch. The discrepancy (900 units) may reflect shipments that went out overnight and are now en route to stores.

---

## 2. Business Case — The Problem, Value, and ROI

### 2.1 The Problem Statement

**In retail today:**
- Pricing team changes a price → no one tells the demand planner.
- Demand planner cuts a stale forecast → buyer places wrong PO.
- Buyer over-orders based on wrong forecast → DC overstocks → markdown at a loss.
- CFO sees revenue miss 3 months later with no root cause trace.

**Root cause:** Supply chain domains (pricing, demand, inventory, supply, finance) are siloed. Decisions don't propagate. No system connects them in real-time.

**This system's solution:** One query → simultaneous cascade across all 5 domains → ranked recommendation with quantified dollar impact in seconds.

---

### 2.2 Value Proposition by Stakeholder

| Stakeholder | Pain Point Solved | Quantified Value |
|-------------|------------------|-----------------|
| Category Manager | "I don't know if this price change will hurt or help" | Price cascade simulation in <10s vs. 2-4 hrs manual |
| Supply Chain Lead | "I found out about the carrier strike from the news" | Proactive stockout timeline + alternate routing |
| Buyer | "I don't know how much to order next quarter" | 15-variable forecast with CI bounds; accuracy gate blocks bad forecasts |
| CFO | "I can't trace why margin missed" | Full P&L waterfall from trigger to net margin in one query |
| VP Operations | "Each team makes decisions in a vacuum" | Conflict detection: promo + strike = CRITICAL, surfaced before execution |

---

### 2.3 The ROI Case

**Demand accuracy improvement:**
- Current MAPE: 78% (7 pts below 85% benchmark)
- At $500B revenue, each 1% improvement ≈ $5B in revenue efficiency
- 7-point improvement potential: **$35B revenue efficiency**

**Stockout prevention:**
- A single SKU (Huggies) stockout across 10 SE stores over 14 days:
  `= 10 stores × 7.1 units/day × 14 days × $14.49 = ~$145K revenue loss`
- At network scale (30 stores, 6 SKUs): $2.84M per 14-day strike window

**Wrong PO cost:**
- Over-ordering: carrying cost = 25% annual rate on excess inventory
- Under-ordering: stockout + customer lifetime value erosion
- A 5% PO accuracy improvement on $50B in annual purchases = $250M in savings

**Promo ROI:**
- A promotion that conflicts with a supply disruption can cost 3–5× more than it generates.
- Detecting and deferring one such conflict per quarter: $500K–$2M saved.

---

### 2.4 Why Multi-Agent AI vs. Traditional Analytics

| Approach | Time | Connected? | Actionable? | Scales? |
|----------|------|-----------|------------|---------|
| Human analyst | 2–4 hours | Partially | Sometimes | No |
| Traditional BI dashboard | Instant (historical) | No | No | Yes |
| Rules-based system | Instant | Partly | Yes | Partly |
| **Multi-agent AI (this system)** | **<10 seconds** | **Yes — all domains** | **Yes — ranked actions** | **Yes** |

**Key differentiator:** The system reasons across pricing, demand, inventory, supply, and finance *simultaneously* — not in sequence, not in isolation. That's what a 20-year expert supply chain VP does intuitively. This externalizes that cognition.

---

## 3. Cost Structure — LLM, Infrastructure, and Optimization

### 3.1 LLM API Cost Model

**Pricing basis (Claude claude-sonnet-4-6):**
```
Input tokens:  ~$3 per million tokens
Output tokens: ~$15 per million tokens
```

**Per-query cost estimate:**
```
Simple query (1 tool call, 6 iterations max):
  Input:  ~2,000 tokens (system prompt + history + query + tool schemas)
  Output: ~500 tokens (response)
  Cost:   ~$0.014 per query

Complex query (5 tool calls, 20 iterations):
  Input:  ~8,000 tokens (system + history + 5× tool results)
  Output: ~1,500 tokens (detailed response)
  Cost:   ~$0.047 per query

History summarization call (at 12+ messages):
  Input:  ~3,000 tokens
  Output: ~600 tokens (summary)
  Cost:   ~$0.018 per summarization
```

**Monthly cost at 1,000 queries/day:**
```
Avg cost/query: $0.025
Daily cost:     $25
Monthly cost:   ~$750
Annual cost:    ~$9,000

For an enterprise deployment (100K queries/day): ~$75,000/month
```

---

### 3.2 Cost Optimization Strategies (What We Implemented)

**1. Token differentiation by complexity:**
```python
_MAX_TOKENS_SUMMARY  = 600    # history compression — cheap, short
_MAX_TOKENS_MAIN     = 4096   # standard queries
_MAX_TOKENS_COMPLEX  = 8096   # tariff shocks, multi-SKU, full cascades

# Before: ALL queries allocated 8096 tokens
# After:  Simple queries use 4096 (50% reduction for standard calls)
# Impact: ~30% reduction in output token spend for typical query mix
```

**2. _slim_tool_result() — strip provenance before LLM feed-back:**
```python
# Tool result with provenance (what the tool returns):
{
  "data": {...},
  "provenance": "WMS",       # → stripped
  "freshness_minutes": 15,   # → stripped
  "is_stale": False,         # → stripped
  "error": None
}

# Slimmed result (what the LLM sees):
{"data": {...}, "error": None}

# Impact: ~30% reduction in tool-result token input per iteration
# Provenance still tracked in tool_calls_made for the UI (not wasted)
```

**3. TTL cache — identical tool calls return instantly:**
```
OLTP tools (pricing, PO): 5-minute TTL
WMS tools (inventory):    15-minute TTL
OLAP tools (analytics):   24-hour TTL

Cache hit = zero API calls = zero cost
Example: Two users query the same SKU inventory within 15 minutes
  → 2nd query: free. No API call. No LLM token cost.

At 1,000 queries/day with ~20% cache hit rate: saves ~$50/day
```

**4. Per-session rate limiting:**
```
30 queries per browser session maximum
Purpose: Prevent a single malicious or runaway session from
         exhausting the API budget
At Streamlit Cloud: protects the shared deployment budget
```

**5. History summarization at 12+ messages:**
```
Without summarization: message list grows unbounded
  → At 20 messages × ~200 tokens each = 4,000 extra input tokens per call
  → At 40 messages: 8,000 extra tokens = $0.024 per call just for history

With summarization: compress to ~600-token summary + last 4 messages
  → Fixed overhead regardless of conversation length
  → At 40 messages: saves ~7,200 tokens = $0.022 per call
```

---

### 3.3 Infrastructure Cost Breakdown

| Component | Cost Model | Estimate |
|-----------|-----------|----------|
| Streamlit Community Cloud | Free (public apps) | $0 |
| Anthropic Claude API | Per-token pricing | $0.02–$0.05/query |
| Vector DB (if added: Pinecone) | $70/month (starter) | $70/month |
| Production Streamlit (Teams) | $500/month | $500/month |
| AWS/GCP hosting (enterprise) | $500–$2,000/month | Variable |
| **Total (current demo)** | | **~$0–$10/month** |
| **Total (production MVP)** | | **~$1,500/month** |

---

### 3.4 Build vs. Buy Analysis

| Component | Build | Buy |
|-----------|-------|-----|
| LLM reasoning | ❌ Years of research | ✅ Anthropic API ($0.025/query) |
| Agentic loop | ✅ 200 lines of Python | ✅ LangChain agents (more opaque) |
| Multi-agent graph | ✅ LangGraph (open source) | ❌ Custom orchestrators (expensive) |
| UI | ✅ Streamlit (open source) | ❌ Custom React app (weeks of work) |
| Data layer | ✅ Mock data (proof of concept) | ✅ SAP/Snowflake APIs (production) |
| Vector memory | ❌ Too complex to build | ✅ Pinecone / pgvector ($70/month) |

**Decision principle:** Buy the undifferentiated infrastructure (LLM, vector DB, cloud). Build the domain logic (tool definitions, prompt engineering, edge case handling, business rules). The business rules ARE the moat.

---

## 4. Technical Architecture — Component Deep-Dives

### 4.1 Anthropic Claude claude-sonnet-4-6 — The Reasoning Engine

**Why Claude claude-sonnet-4-6:**
- Superior instruction-following for complex system prompts (17 non-negotiables)
- Native tool-use support (structured JSON in/out) without prompt hacks
- 200K context window — can hold full 12-message conversation without truncation
- Extended thinking available for step-by-step reasoning trace
- Better than GPT-4o on structured business reasoning benchmarks (internal testing)

**What it does in this system:**
- Reads the system prompt (47 supply chain business rules)
- Receives user query + conversation history
- Decides which tools to call and in what order (agentic reasoning)
- Interprets tool results and decides whether to continue or synthesize
- Generates final response: actions + dollar impact + data caveats

**How tool use works:**
```python
# 1. Send query + tools to Claude
response = client.messages.create(
    model="claude-sonnet-4-6",
    tools=ALL_TOOLS,          # 17 tool schemas (JSON Schema)
    messages=messages,
    system=SYSTEM_PROMPT,
)

# 2. Claude returns stop_reason="tool_use" with tool call(s)
for block in response.content:
    if block.type == "tool_use":
        result = mock_executor.execute(block.name, block.input)

# 3. Feed results back as "tool_result" messages
messages.append({"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": block.id,
     "content": json.dumps(result)}
]})

# 4. Claude continues until stop_reason="end_turn" (analysis complete)
```

---

### 4.2 V1 — Single-Agent Agentic Loop

**Why use it:**
- Simple, transparent, easy to debug
- Flexible — Claude chooses its own tool chain per query
- Great for conversational, exploratory, multi-turn interactions

**What it does:**
- Runs a while loop: send → get tools → execute → feed back → repeat
- Adaptive iteration caps by query complexity (6 / 10 / 20)
- Tracks provenance warnings at every tool call
- Summarizes history at 12+ messages to prevent context blowup

**How complexity detection works:**
```python
_COMPLEX_KEYWORDS = ["tariff", "multi-sku", "all regions", "supplier bankruptcy", ...]
_SIMPLE_KEYWORDS  = ["what is the price", "current inventory", "how many", ...]

def _detect_max_iterations(query: str) -> int:
    q = query.lower()
    if any(k in q for k in _COMPLEX_KEYWORDS): return 20
    if any(k in q for k in _SIMPLE_KEYWORDS):  return 6
    return 10
```

**When to use V1 vs. V2:** V1 shines for open-ended conversation where the query complexity and tool needs are unpredictable. V2 shines for structured, repeatable workflows where you want explicit control and auditability.

---

### 4.3 V2 — LangGraph Multi-Agent Graph

**Why LangGraph:**
- Explicit routing: you control which agent handles which query type
- Shared state TypedDict flows through all nodes — no repetition
- Each node sees only its relevant tools (not all 17) — cleaner reasoning
- Built-in support for parallel node execution (not yet utilized here)
- Node-by-node audit trail for compliance and debugging

**What it does:**
```
user query
    │
  ROUTER node
  (LLM classifies intent + extracts SKU, carrier, store)
    │
    ├──► price_cascade node  ──► inventory_node  ──► financial_impact
    ├──► supply_disruption   ──► carrier_node    ──►     │
    ├──► demand_forecast     ──► accuracy_node   ──►     │
    ├──► scenario_planning                       ──►     │
    └──► shelf_replenishment ──► perishable_check ──►    │
                                                         ▼
                                                   SYNTHESIZER
                                                   (merges all outputs)
```

**How LangGraph state works:**
```python
class RetailState(TypedDict):
    query: str
    messages: List[BaseMessage]
    intent: str               # set by router, used for routing
    sku: str                  # propagated to all nodes
    entities: Dict            # price, carrier, store — shared
    tool_calls_made: List[Dict]   # accumulated across ALL nodes
    freshness_warnings: List[str] # accumulated across ALL nodes
    node_outputs: Dict            # each node deposits its result here
    final_response: str           # synthesizer writes this last

# Each node function signature:
def price_cascade_node(state: RetailState) -> RetailState:
    # reads state.sku, state.entities
    # calls tools, stores results in state.node_outputs["price"]
    # accumulates tool_calls_made
    return state  # modified state flows to next node
```

**Routing (conditional edges):**
```python
workflow.add_conditional_edges(
    "router",
    lambda s: s["intent"],
    {
        "price_cascade": "price_cascade",
        "supply_disruption": "supply_disruption",
        "demand_forecast": "demand_forecast",
        "general": "price_cascade",   # default: full pipeline
    }
)
```

---

### 4.4 Tool Use Pattern — The 17 Tools

**Why tools instead of RAG (Retrieval-Augmented Generation):**
- RAG retrieves static documents — tools compute live values.
- Inventory, stockout risk, and revenue impact are calculations, not lookups.
- Tools give the LLM the ability to act, not just recall.

**Tool schema structure (JSON Schema):**
```python
{
    "name": "simulate_price_change",
    "description": "Simulate full downstream impact of a retail price change...",
    "input_schema": {
        "type": "object",
        "properties": {
            "sku_id":    {"type": "string",  "description": "Product SKU"},
            "old_price": {"type": "number",  "description": "Current price"},
            "new_price": {"type": "number",  "description": "Proposed new price"},
            "horizon_weeks": {"type": "integer", "description": "Forecast horizon"},
        },
        "required": ["sku_id", "old_price", "new_price"]
    }
}
```

**Tool execution pattern (never raises, always returns):**
```python
def execute(tool_name: str, tool_input: dict) -> dict:
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return {"data": {}, "error": f"Unknown tool: {tool_name}", ...}

    validation_error = _validate_input(tool_name, tool_input)
    if validation_error:
        return {"data": {}, "error": validation_error, ...}

    try:
        result = fn(**tool_input)
        _cache_set(cache_key, result)
        return result
    except Exception as e:
        return {"data": {}, "error": str(e), ...}
```

**Why tools never raise exceptions:** If a tool raises, the LLM crashes. If a tool returns an error dict, the LLM can reason about the error and decide what to do next (retry, use an alternate tool, or warn the user).

---

### 4.5 Input Validation + Security

**Why validate at the tool layer:**
- Prevents LLM hallucinations from reaching business logic (e.g., LLM invents SKU "HUG99-X")
- Prevents prompt injection via malicious tool inputs
- Prevents accidental API cost blowup from infinite loops on bad inputs

**What is validated:**
```python
VALID_SKUS = {"HUG48-3", "PAM72-5", "MLK-GAL", "TAB-DIN", "BLK-THR", "CIG-PKT"}
VALID_CARRIERS = {"TruckCo_A", "TruckCo_B", "TruckCo_C", "TruckCo_D"}
VALID_LOCATION_IDS = {f"STR-{i:03d}" for i in range(1, 31)} | {"DC-NW", "DC-SE", "DC-MW"}

# Price bounds: $0.01 – $9,999.99
# Horizon: 1–104 weeks
# Duration: 0–730 days (2 years max)
```

---

### 4.6 TTL Cache — Idempotent Tool Results

**Why cache tool results:**
- Identical tool calls within a session are common (e.g., three different nodes all check the same DC inventory)
- Without cache: 3× API tool calls = 3× latency + 3× computation cost
- With cache: 2nd and 3rd calls return instantly, zero re-computation

**How the cache works:**
```python
_CACHE_TTL_SECONDS = {"OLTP": 300, "WMS": 900, "OLAP": 86400}
_cache: Dict[str, Tuple[float, dict]] = {}   # key → (timestamp, result)

cache_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"

# Check: is there a cached result within TTL?
if (time.time() - _cache[key][0]) < TTL:
    return _cache[key][1]

# Execute and store
result = fn(**tool_input)
_cache[cache_key] = (time.time(), result)
return result
```

**Write-through tools bypass cache:**
- `trigger_replenishment`, `adjust_promotional_price` — these are state-changing operations.
- Caching a mutation is dangerous. These always execute live.

---

### 4.7 History Summarization — Context Window Management

**The problem:** A 20-message conversation accumulates ~5,000–8,000 tokens of context. At 40 messages: ~12,000 tokens of input on every call. This is expensive and approaches the context window limit for long sessions.

**The solution:**
```python
if len(messages) >= 12:
    recent = messages[-4:]          # keep last 4 verbatim
    older  = messages[:-4]          # compress everything before

    summary = claude_summarize(older)   # dedicated call, max 600 tokens output

    messages = [
        {"role": "user",      "content": f"[SUMMARY]: {summary}"},
        {"role": "assistant", "content": "Understood."},
    ] + recent
```

**Cost tradeoff:**
- Summarization call: ~$0.018 (one-time)
- Savings per call after: ~$0.022 in context tokens
- Break-even: 1 summarization call pays for itself after 1 subsequent query.

---

### 4.8 Pydantic Schemas — Data Contract Layer

**Why Pydantic:**
- Validates data at module boundaries (between tools and agents)
- Self-documenting — schemas are the specification
- Prevents silent data corruption (wrong type silently ignored)

**Key schemas:**
```python
class ToolResult(BaseModel):
    data: Dict[str, Any]
    error: Optional[str]
    provenance: Literal["OLTP", "WMS", "OLAP"]
    freshness_minutes: int
    is_stale: bool

class DemandForecast(BaseModel):
    sku_id: str
    accuracy: float
    is_reliable: bool      # False if accuracy < 0.60
    weekly_forecasts: List[WeeklyForecast]
    confidence_intervals: Dict[str, Tuple[float, float]]
    key_drivers: List[str]

class ScenarioConflict(BaseModel):
    conflict_type: str
    severity: Literal["CRITICAL", "WARNING", "INFO"]
    scenarios_involved: List[str]
    recommendation: str
```

---

### 4.9 Streamlit — The UI Layer

**Why Streamlit:**
- Python-native: no JavaScript, no React, no separate frontend codebase
- `st.status()` for live tool call tracking during agentic loops
- `st.write_stream()` for token-by-token response streaming
- `st.session_state` for per-user conversation isolation (in-process)
- Deployable to Streamlit Community Cloud for free (public demos)

**Limitations (important for interviews):**
- Session state dies on page refresh (no persistence)
- Single-process: all users share the same Python process (not truly multi-user)
- Not suitable for production enterprise deployment at scale
- Production path: FastAPI backend + React/Next.js frontend with proper session management

**Version selector pattern (v1/v2/v3 UI):**
```python
app_version = st.sidebar.radio(
    "Interface Version",
    ["v1 — Core (2 tabs)", "v2 — Full (15 tabs)", "v3 — Simplified (7 tabs)"]
)
# Conditional rendering based on version — one codebase, three UIs
if "v1" in app_version:
    _render_v1()
elif "v2" in app_version:
    _render_v2()
else:
    _render_v3()   # default
```

---

## 5. Design Decisions and Trade-offs

### 5.1 Why Two Pipelines (V1 + V2)?

| Dimension | V1 Decision | V2 Decision |
|-----------|------------|------------|
| Architecture | Single agent | Multi-agent graph |
| **Why** | Faster to build, simpler to debug, sufficient for most queries | More control, per-domain prompting, explicit audit trail |
| **Trade-off** | Less predictable routing, all tools always visible | More latency (12 node hops), harder to extend |
| **Best for** | Exploratory conversation | Structured, repeatable workflows |

**Interview answer:** "We built both to demonstrate the trade-off. V1 gives you flexibility — the LLM decides what to do. V2 gives you control — you decide what the LLM can do in each step. In production, you'd choose based on your governance requirements. Financial services needs V2 (auditability). Customer support needs V1 (flexibility)."

---

### 5.2 Why Mock Data Instead of Real Integrations?

**Decision:** Build a rich, realistic mock data layer instead of integrating with real APIs.

**Reasons:**
1. Faster demonstration — real API integrations take weeks of vendor negotiation
2. No credentials, no data governance overhead
3. Mock data can model edge cases (e.g., TruckCo_B always on strike) that real data rarely surfaces in a demo
4. Identical interface — switching to real data = replacing mock functions, no architectural change

**Production path:** Replace `mock_executor.execute()` with real API calls. Return shape is identical. The contract is the `ToolResult` schema.

---

### 5.3 Why Not Use RAG (Retrieval-Augmented Generation)?

RAG retrieves relevant documents to augment LLM responses. We use **tools** instead.

| RAG | Tool Use |
|-----|---------|
| Retrieves static text | Executes live calculations |
| Good for: knowledge bases, documentation | Good for: real-time state, computation |
| Output: text passage | Output: structured JSON |
| Can't "compute" stockout risk | Can calculate stockout risk precisely |

**Both have a place:** A future enhancement would use RAG for retrieving historical similar scenarios (e.g., "find the last 3 times we had a SE carrier disruption and what we did"). Tools handle the live computation; RAG handles the institutional memory.

---

### 5.4 Why 17 Tools and Not 170?

**Design principle:** Tools should be at the right level of granularity — not too coarse (does too much, hard to compose), not too fine (LLM must call 30 tools to answer one question).

**Rules applied:**
- One tool per business question the LLM needs to answer
- Each tool has one clear responsibility (single-responsibility principle)
- Tools return structured data, not pre-formatted text (LLM formats)
- No tool depends on another tool's state (pure functions where possible)

---

## 6. Interview Q&A — Expected Questions with Model Answers

### Q1: "Walk me through what happens when a user sends a query."

**Answer:**
"The query lands in the Orchestrator. It first classifies complexity using keyword detection — simple queries get 6 iterations, complex ones get 20. If the conversation history is 12+ messages, it summarizes older turns to stay within context limits.

Then it sends the query to Claude with all 17 tool schemas and the supply chain system prompt. Claude reads the query, reasons about it, and returns a tool call — say, `simulate_price_change`.

The executor validates the input (SKU whitelist, price bounds), checks the cache, and runs the tool function. The result comes back with data + provenance tags. We strip the provenance before feeding back to Claude (saves ~30% tokens) but track it separately for the UI.

Claude may call more tools — inventory levels, financial impact, competitive pricing. After the final tool call, it synthesizes a response: actions, dollar impact, data caveats. That text streams back to the user."

---

### Q2: "How does the system handle a carrier strike scenario?"

**Answer:**
"The query is classified as `supply_disruption`. In V2 / LangGraph, the router sends it to the `supply_disruption` node.

That node calls `get_supply_disruption_impact` with the carrier ID and estimated duration. It gets back: affected SKUs (diapers, formula), days-to-stockout per DC, and revenue at risk.

Then the `carrier_node` calls `find_alternate_carriers` three times — once per region. This is critical: it checks regional availability, not national. TruckCo_C is active but handles dairy. The SE region only has TruckCo_D at +45% cost premium.

That regional gap is surfaced as a CRITICAL warning. The synthesizer generates: immediate emergency order via TruckCo_D for SE, inter-DC transfer from NW (surplus) to SE, price hold to suppress demand and extend days-on-hand, and a monitoring schedule for strike resolution."

---

### Q3: "How do you handle prompt injection or malicious inputs?"

**Answer:**
"Three layers of defense.

First, input validation in `mock_executor.py` — all SKU IDs, carrier IDs, and location IDs are checked against whitelists before reaching any business logic. An injected SKU like `'; DROP TABLE products; --` fails the whitelist check and returns a structured error, not an exception.

Second, the system prompt is structured as non-negotiables — the LLM is given very specific instructions about what to do and what not to do. Prompt injection through user queries would have to override these.

Third, rate limiting — 30 queries per session maximum. Even if someone tries to use the system as a proxy for something else, they're capped at 30 calls.

For production, I'd add server-side authentication, input sanitization on the HTTP layer, and API key scoping so each user has their own budget."

---

### Q4: "Why did you choose LangGraph over CrewAI or AutoGen?"

**Answer:**
"LangGraph gives explicit control over routing via conditional edges. You define exactly which node handles which intent, and you can see the full state at every step. CrewAI and AutoGen are more autonomous — agents negotiate their own task delegation, which is less predictable.

For supply chain decisions with dollar consequences, predictability and auditability matter more than autonomy. A CFO asking about margin impact needs to trust that the financial agent specifically ran the P&L calculation, not that 'some agent did something.'

LangGraph also has first-class Python support, integrates natively with LangChain tools and Anthropic's SDK, and the state TypedDict pattern maps cleanly to the shared data model we already had."

---

### Q5: "What would you do differently if you were productionizing this?"

**Answer:**
"Several things:

1. **Replace Streamlit with FastAPI + React** — Streamlit is great for demos but shares a process across all users and has no real persistence. Production needs a proper API layer with authentication, rate limiting per user, and a React frontend with proper state management.

2. **Connect real data sources** — the mock executor would be replaced by actual Snowflake queries (OLAP), Blue Yonder WMS API calls (inventory), and SAP APIs (pricing, POs). The ToolResult contract stays the same.

3. **Add vector memory** — Pinecone or pgvector to store historical decisions and outcomes. The LLM can then recall 'last time we had a SE carrier disruption, we did X and it worked.' That's institutional memory.

4. **Human-in-the-loop for execution** — recommendations today are text. Productionizing means a confirmation step where the system proposes a PO adjustment and the buyer clicks 'Approve' to push it to SAP.

5. **Decision logging and feedback loop** — log every recommendation + actual outcome 30/60/90 days later. Feed the delta back into elasticity parameters. The model improves over time instead of being static."

---

### Q6: "What is asymmetric elasticity and why does it matter for pricing decisions?"

**Answer:**
"Asymmetric elasticity means that a price increase suppresses demand more aggressively than a price decrease recovers it. It's empirically observed in tobacco, alcohol, diapers, and infant formula — categories with habit-driven purchasing.

For Huggies diapers (elasticity = -1.4, recovery factor = 0.70):
- Raise price 10% → demand drops 14% (full elasticity applies)
- Cut price 10% → demand rises only 9.8% (70% of expected recovery)

Why this matters: A pricing team that models both directions symmetrically will over-estimate promotion lift by ~30% and under-estimate the risk of a price increase. If you raise price by $1.50, lose 14% volume, and then reverse the increase — you recover only 9.8%, not 14%. You can't price your way back to where you were. Net permanent demand loss.

At Walmart scale on diapers alone (~$2B annual), a 4% permanent demand loss from a mismodeled price decision = $80M in revenue."

---

### Q7: "How does the forecast accuracy gate work and why is 60% the threshold?"

**Answer:**
"The forecast accuracy gate checks MAPE (Mean Absolute Percentage Error) before passing a forecast downstream to the PO system. If accuracy falls below 60%, the system sets `is_reliable=False` and explicitly blocks the forecast from being used.

The 60% threshold is based on the principle that a forecast worse than random guessing does active harm. At 60% MAPE, you're wrong by 40% on average — that's still actionable directionally. Below 60%, the noise exceeds the signal.

A better answer for 'why 60%' in production would be: run a simulation with historical PO data and actual demand outcomes. Find the MAPE threshold below which order accuracy gets worse with the forecast than without it. That's your gate.

The key design decision is: it's better to surface 'I don't know' than to surface a false-confidence number that a buyer then over-commits to."

---

### Q8: "How do you handle the cold start problem — the system has no user history on first use?"

**Answer:**
"The mock data layer provides a warm start — the system knows the current inventory, carrier status, SKU prices, and elasticity parameters without any user history. The first query gets a full, context-rich response even with zero conversation history.

For real production cold start:
1. Pre-load recent transaction history as context for the first call
2. Use organizational defaults for parameters (e.g., company-wide safety stock policy) rather than asking the user each time
3. The history summarization system means that once a session has 12+ messages, older context is compressed and preserved — so 'cold start' really only applies to the literal first message

Longer-term, a vector memory layer (Pinecone/pgvector) would store decisions across sessions so 'cold start' becomes 'slightly warmer start' — the system recalls the last time this user asked about Huggies pricing."

---

### Q9: "What's the token spend breakdown in a typical complex query?"

**Answer:**
"For a complex query like 'TruckCo B is on strike — run full supply disruption analysis for all DCs':

```
System prompt:           ~650 tokens   (fixed per call)
Tool schemas (17 tools): ~3,500 tokens  (fixed per call)
Conversation history:    ~1,000 tokens  (grows with conversation)
User query:              ~30 tokens
                         ─────────────
Input (first call):      ~5,180 tokens

Tool call 1 result (supply disruption):  ~400 tokens back
Tool call 2 result (carrier status):     ~300 tokens back
Tool call 3 result (alternate carriers): ~500 tokens back (×3 regions)
                         ─────────────
Total input across 4 iterations: ~8,500 tokens
Output (final synthesis): ~1,200 tokens

Total cost: (8,500 × $3 + 1,200 × $15) / 1,000,000 = $0.044

With _slim_tool_result() stripping provenance: saves ~300 tokens per tool result
= saves ~900 tokens across 3 tool calls = ~$0.003 per complex query
```

Over 10,000 complex queries: that's $30 saved by one optimization."

---

### Q10: "What would you add to make this system truly production-grade?"

**Answer:**
"Five additions, in order of impact:

**1. Real-time data connectors** (highest impact): The system is only as good as its data. Connect Blue Yonder WMS (live inventory, 15-min refresh), SAP S/4HANA (pricing, POs), and Snowflake (OLAP analytics). The ToolResult contract already handles this — just replace the mock functions.

**2. ERP execution hooks**: Right now recommendations are text. Close the loop: 'AI proposes PO reduction of 810 units, buyer clicks Approve → SAP API call places the adjusted PO.' Human-in-the-loop with one-click execution.

**3. Decision memory (vector DB)**: Store every recommendation + outcome. Future queries can recall 'last time we had SE carrier disruption: we used TruckCo_D, it cost 45% premium, resolve in 8 days.' Chroma or Pinecone, embedded via Claude Embeddings API.

**4. Proactive alert queue**: Instead of reactive queries, run a daily agent sweep across all SKUs and DCs. Surface the top 5 decisions that need attention today, ranked by revenue-at-risk. Push to Slack or Teams.

**5. Elasticity recalibration pipeline**: Log predicted vs. actual demand outcomes quarterly. Feed delta back into the elasticity model. The system gets smarter over time rather than relying on static parameters."

---

*Document last updated: 2026-04-14*  
*System version: v3.2 — Streamlit deployed at https://retail-supply-chain-ai-hp2hz8kf9cjqfkr82wogkt.streamlit.app/*
