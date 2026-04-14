---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background: #ffffff;
    color: #1a1a2e;
    padding: 40px 60px;
  }
  section.title {
    background: linear-gradient(135deg, #0071ce 0%, #004a8f 100%);
    color: white;
    text-align: center;
    padding: 60px;
  }
  section.title h1 { font-size: 2.4em; font-weight: 800; margin-bottom: 0.2em; }
  section.title h2 { font-size: 1.1em; font-weight: 300; opacity: 0.9; }
  section.section-header {
    background: #0071ce; color: white;
    display: flex; align-items: center; justify-content: center; text-align: center;
  }
  section.section-header h1 { font-size: 2.2em; font-weight: 700; }
  h1 { color: #0071ce; font-size: 1.5em; border-bottom: 3px solid #0071ce; padding-bottom: 6px; margin-bottom: 16px; }
  h2 { color: #1a1a2e; font-size: 1.15em; margin-bottom: 8px; }
  h3 { color: #0071ce; font-size: 1em; margin-bottom: 4px; }
  table { width: 100%; font-size: 0.72em; border-collapse: collapse; margin-top: 8px; }
  th { background: #0071ce; color: white; padding: 7px 10px; text-align: left; }
  td { padding: 5px 10px; border-bottom: 1px solid #e9ecef; }
  tr:nth-child(even) td { background: #f8f9fa; }
  code { background: #f0f4ff; color: #0071ce; padding: 2px 6px; border-radius: 4px; font-size: 0.82em; }
  pre { background: #1a1a2e; color: #e6edf3; padding: 14px; border-radius: 8px; font-size: 0.7em; line-height: 1.5; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
  .columns3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2em; }
  .highlight { background: #fff3cd; border-left: 4px solid #f39c12; padding: 10px 14px; border-radius: 4px; margin: 8px 0; font-size: 0.85em; }
  .critical  { background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px 14px; border-radius: 4px; font-size: 0.85em; }
  .success   { background: #d4edda; border-left: 4px solid #28a745; padding: 10px 14px; border-radius: 4px; font-size: 0.85em; }
  .info      { background: #cce5ff; border-left: 4px solid #004085; padding: 10px 14px; border-radius: 4px; font-size: 0.85em; }
  ul { font-size: 0.85em; line-height: 1.7; }
  li { margin-bottom: 2px; }
  footer { font-size: 0.65em; color: #6c757d; }
---

<!-- _class: title -->

# 🏪 Retail Supply Chain Optimization AI

## Multi-Agent Agentic AI System for Enterprise Retail Decision-Making

---
**Claude claude-sonnet-4-6 · LangGraph · Streamlit · Python 3.13**
Interview Kickstart — Agentic AI Capstone | 2026

---

<!-- _class: section-header -->

# 🏭 Part 1: Domain — The Retail Supply Chain Problem

---

# The Core Problem: Decisions Don't Propagate

**Every supply chain decision creates a cascade across 5 domains simultaneously**

```
A 10% diaper price increase triggers — in sequence:

  Price Change  →  Demand drops 14%  →  Replenishment PO cut by 14%
      ↓                   ↓                        ↓
  Revenue risk        DC over-stock              Carrier load drops
      ↓                   ↓                        ↓
  Margin impact    Carrying cost rises        Regional gap exposed
```

<div class="highlight">
⏱ A human supply chain expert reasons through this chain in <strong>2–4 hours</strong>.
This system does it in <strong>&lt; 10 seconds</strong> — across all domains simultaneously.
</div>

> **$1B+ daily decisions are made in retail without connected, real-time AI reasoning across pricing, demand, inventory, supply, and finance.**

---

# Domain 1: Price Elasticity — Including Asymmetric Behavior

**What it is:** Elasticity = how sensitive demand is to price changes.

| SKU | Elasticity | Asymmetric? | Recovery Factor | What It Means |
|-----|-----------|------------|----------------|---------------|
| Huggies HUG48-3 | **−1.4** | ✅ Yes | 0.70 | +10% price → −14% volume |
| Pampers PAM72-5 | −1.35 | ✅ Yes | 0.72 | Demand sticky on price reversal |
| Cigarettes CIG-PKT | −0.50 | ✅ Very | **0.40** | Promos barely move volume |
| Milk MLK-GAL | −0.60 | ❌ No | 1.00 | Symmetric (commodity) |

**Critical insight — Asymmetric Elasticity:**
```
Price ↑ 10%: demand DROPS 14%   (full elasticity applies)
Price ↓ 10%: demand recovers 9.8%  (only 70% of expected recovery)

Net effect: You can lose 14% demand but only win back 9.8%. Promotions are
lossy. Price reversals don't restore volume. This is a $10M+ modeling error at scale.
```

---

# Domain 2: 15-Variable Demand Forecast Model

**Multiplicative demand model captures all demand signals simultaneously:**

```
D(t) = Base_Demand × Π(1 + factor_signal_i × learned_weight_i)

15 variables:  price, promo, tariff, weather, seasonality, trend,
               competitor_price, social_sentiment, days_supply, channel_mix,
               planogram_compliance, regional_income, household_penetration,
               repeat_rate, new_item_velocity
```

<div class="columns">

**Confidence Intervals (CI) widen with horizon:**

| Week | Point | 80% CI | 95% CI |
|------|-------|--------|--------|
| 1 | 50 | [46.5, 53.5] | [43.2, 56.8] |
| 4 | 51.5 | [44.0, 59.0] | [37.8, 65.2] |
| 8 | 52.8 | [40.1, 65.5] | [33.6, 72.0] |

**Accuracy Gate:**
- MAPE ≥ 60% → `is_reliable = True` → enters PO system
- MAPE < 60% → `is_reliable = False` → **BLOCKED**
- Rule: never cite a point forecast as truth

</div>

---

# Domain 3: Inventory, Replenishment, and the Lag Problem

**Replenishment chain — the hidden risk:**

```
Today:  HQ approves PO
Day 2:  Vendor picks + ships
Day 3:  DC receives → picks → truck dispatched
Day 4:  Store receives (best case)
+ 30% chance: +3 extra days (weather, DC congestion, customs)

Effective Lag = 3–4 days + (0.30 × 3 extra days) = 3.9–4.9 days
```

<div class="critical">
❌ If days_on_hand &lt; effective_lag → shelf goes empty BEFORE the truck arrives.<br>
→ Trigger emergency replenishment NOW, not when the shelf is empty.
</div>

**VMI (Vendor-Managed Inventory) impact:**
- Milk: 30% VMI → only 70% is your inventory → carrying cost on 70% only
- Interview trap: Many calculate carrying cost on 100% of inventory, missing the VMI exclusion

**Perishable hard cap:**
- Dairy: max 3 days-of-supply regardless of demand signal
- `Max_Order = (Shelf_Life - Lead_Time) × Avg_Daily_Demand × Planogram_Cap`

---

# Domain 4: Supply Chain Network — Regional Reality vs. National Illusion

**Network topology:**

| DC | Location | Region | Diaper Stock (WMS) | Risk Level |
|----|----------|--------|--------------------|-----------|
| DC-NW | Seattle | Northwest | 8,500 units | Low |
| DC-SE | Atlanta | Southeast | **3,200 units** | **CRITICAL** |
| DC-MW | Chicago | Midwest | 6,100 units | Moderate |

**Carrier coverage — regional gaps:**

| Carrier | Status | Cargo | Regions | SE Diaper Alt? |
|---------|--------|-------|---------|----------------|
| TruckCo_A | ✅ Active | Tableware | NW, MW | ❌ Wrong cargo |
| **TruckCo_B** | 🔴 **STRIKE** | **Diapers, Formula** | **SE, MW** | N/A |
| TruckCo_C | ✅ Active | Dairy only | NW, SE | ❌ Wrong cargo |
| TruckCo_D | ✅ Active | General | All | ✅ **+45% cost** |

> **Key insight:** SE region has exactly ONE viable diaper carrier when TruckCo_B strikes — at a 45% cost premium. National carrier availability is meaningless. Regional availability is everything.

---

# Domain 5: Data Freshness — When Stale Data Costs Millions

| Source | Lag | Use For | Risk if Used for Ops |
|--------|-----|---------|---------------------|
| **OLTP** | ~5 min | Pricing, POs, GL entries | Low |
| **WMS** | ~15 min | Inventory, stockout decisions | Low-Medium |
| **OLAP** | **24 hours** | Analytics, reporting, trend | **HIGH — day-old ops data** |

**Interview scenario:** *WMS says DC-SE: 3,200 units. OLAP says 4,100 units. Which do you use?*

→ **WMS for operations.** OLAP is from yesterday's batch. The 900-unit difference represents overnight shipments already at stores. Acting on OLAP inventory for an emergency order means over-ordering by 900 units.

**How this system handles it:**
```python
# Every tool result carries provenance metadata
{"provenance": "OLAP", "freshness_minutes": 1440, "is_stale": True}

# Orchestrator surfaces this as a warning to the user — not silently ignored
freshness_warnings.append("⚠ Tool used OLAP data (24-hour lag). Use WMS for operational decisions.")
```

---

<!-- _class: section-header -->

# 💼 Part 2: Business Case — Value, ROI, and Stakeholders

---

# The Business Case — Why This System Exists

<div class="columns">

**What breaks today without it:**

- Pricing team raises a price
- Nobody tells demand planners
- Stale forecast → wrong PO
- Wrong PO → DC over-stock or under-stock
- Markdown at loss OR stockout + customer churn
- CFO sees revenue miss 3 months later, no root cause

**What this system provides:**

- One query → cascade across 5 domains
- Demand impact + PO adjustment + carrier routing + P&L — simultaneously
- Conflict detection before decisions are executed
- Ranked recommendations with quantified dollar impact
- Data provenance flags on every number

</div>

---

# ROI: Quantifying the Value at Retail Scale

| Opportunity | Calculation | Impact |
|------------|------------|--------|
| **Demand accuracy +7 pts** | At $500B revenue, 1% improvement ≈ $5B | **$35B efficiency potential** |
| **Stockout prevention (14-day strike)** | 10 stores × 7.1 units/day × 14d × $14.49 | **$145K per SKU per event** |
| **Network-scale strike impact** | 30 stores × 6 SKUs × $2.84M | **$2.84M revenue at risk** |
| **Wrong PO cost** | 5% PO accuracy improvement × $50B annual purchasing | **$250M savings** |
| **Promo + disruption conflict avoided** | One averted conflict per quarter | **$500K–$2M saved** |
| **Wrong asymmetric elasticity model** | 4% permanent demand loss × $2B diaper revenue | **$80M annual error** |

<div class="success">
✅ The system doesn't need to capture all of these to justify its cost. <strong>One averted conflict per quarter pays for a year of API usage.</strong>
</div>

---

# Stakeholder Value Map

| Role | Daily Pain Point | What This System Delivers |
|------|-----------------|--------------------------|
| **Category Manager** | "Will this price change help or hurt?" | Price cascade in <10s: demand + revenue + margin + competitive position |
| **Supply Chain Lead** | "I found out about the strike from the news" | Proactive stockout timeline + regional carrier gap analysis |
| **Buyer** | "How much should I order next quarter?" | 15-variable forecast + CI bands + accuracy gate blocks bad forecasts |
| **VP Operations** | "Each team makes decisions in isolation" | Conflict detection: promo + strike = CRITICAL, before execution |
| **CFO** | "I can't trace why margin missed" | Full P&L waterfall from trigger to net margin in one query |
| **Data Scientist** | "My model doesn't account for asymmetric behavior" | Built-in: asymmetric elasticity, CI widening, MAPE gate, VMI exclusion |

---

# Live Scenario: Carrier Strike + Promotion Conflict

<div class="highlight">
📋 <strong>Situation:</strong> TruckCo B (diapers, SE+MW) goes on strike. Category manager simultaneously wants to launch a 10% Huggies promotional price cut.
</div>

**What the AI reasons through in one query:**

1. **Conflict Detection** → CRITICAL: promotion creates +14% demand surge while supply is at ZERO
2. **Regional Gap** → SE region: only TruckCo_D available at **+45% cost premium**
3. **Days-of-Supply Clock** → DC-SE hits critical threshold in **4.5 days** at current demand
4. **Financial Impact** → Revenue-at-risk: **$2.84M** over 14-day strike window
5. **Recommendation** → Defer promotion until strike resolves; pre-build DC-SE inventory via inter-DC transfer from DC-NW (8,500 surplus units)

<div class="critical">
❌ Running promotion + strike simultaneously = stockout at peak demand = customer churn + 45% freight premium + permanent demand loss (asymmetric recovery)
</div>

---

# Live Scenario: Full Price Cascade Analysis

**Query:** *"Raise HUG48-3 from $12.99 to $14.49 — simulate full cascade"*

| Step | What Happens | Numbers |
|------|-------------|---------|
| Price change | +$1.50 / unit | $12.99 → $14.49 (+11.5%) |
| Demand impact (asymmetric) | Elasticity = −1.4, full rate applied (price increase) | **−16.1% volume** |
| Volume: 30 stores | 50 → 41.9 units/store/week | −1,944 units over 8 weeks |
| PO recalculation | 2 open POs adjustable | PO-0341: −810 units · PO-0389: −486 units |
| Competitive check | Target $13.49, Costco $11.89 | **Above Costco at new price** |
| Revenue impact | Price gain vs. volume loss | Net: **+4.2% revenue** (price wins) |
| Carrying cost | Fewer owned units in DC | **−$2,840 annual carrying cost** |
| Future reversal risk | Recovery factor = 0.70 | A future price cut recovers only 70% of lost demand |

> **Bottom line:** Price increase is accretive. Proceed — but flag Costco competitive risk and the irreversibility of asymmetric demand loss.

---

<!-- _class: section-header -->

# 💰 Part 3: Cost Structure — LLM, Infrastructure, and Optimization

---

# LLM API Cost Model

**Claude claude-sonnet-4-6 pricing:**
```
Input tokens:  ~$3.00 per million tokens
Output tokens: ~$15.00 per million tokens
```

| Query Type | Input Tokens | Output Tokens | Cost per Query |
|-----------|-------------|--------------|---------------|
| Simple (1 tool, 6 iterations) | ~2,000 | ~500 | **~$0.014** |
| Standard (3 tools, 10 iterations) | ~5,000 | ~1,000 | **~$0.030** |
| Complex (5+ tools, 20 iterations) | ~8,500 | ~1,500 | **~$0.047** |
| History summarization call | ~3,000 | ~600 | **~$0.018** |

**At scale:**
| Volume | Daily Cost | Monthly Cost |
|--------|-----------|-------------|
| Demo (100 queries/day) | $3 | $90 |
| SMB (1,000/day) | $30 | $900 |
| Enterprise (100K/day) | $3,000 | $90,000 |

---

# 5 Cost Optimizations We Implemented

<div class="columns">

**1. Token differentiation**
```python
_MAX_TOKENS_SUMMARY  = 600
_MAX_TOKENS_MAIN     = 4096
_MAX_TOKENS_COMPLEX  = 8096
# Before: all queries got 8096
# After: ~30% savings on standard queries
```

**2. _slim_tool_result()**
```python
# Strip provenance before LLM feed-back
# Tool returns: data + provenance + freshness
# LLM sees: data + error only
# Savings: ~30% per tool-result token
```

**3. TTL Cache**
```python
OLTP: 5-min TTL
WMS:  15-min TTL
OLAP: 24-hr  TTL
# Cache hit = $0 API cost
# ~20% hit rate → saves ~$50/day at 1K/day
```

**4. History summarization**
```python
# At 12+ messages:
# Compress older turns → 600-token summary
# Fixed context overhead regardless of length
# Saves ~7,200 tokens on 40-message sessions
```

</div>

**5. Rate limiting:** 30 queries/session max — prevents API budget exhaustion from a single runaway session.

---

# Build vs. Buy Decision Framework

| Component | Decision | Rationale | Cost |
|-----------|----------|-----------|------|
| LLM reasoning | **Buy** (Anthropic API) | Years of research, $0.025/query | $0.025/query |
| Multi-agent graph | **Buy** (LangGraph OSS) | Open source, production-grade | Free |
| UI layer | **Buy** (Streamlit OSS) | Python-native, free cloud hosting | Free |
| Tool schemas | **Build** | Domain-specific — this IS the moat | Engineering time |
| Business rules | **Build** | Asymmetric elasticity, conflict detection | Engineering time |
| Mock data layer | **Build** | Proof of concept without vendor dependency | Engineering time |
| Vector memory | **Buy** (Pinecone) | Complex to build, cheap to buy | $70/month |
| ERP integration | **Buy** (SAP/Oracle APIs) | Not reinventing ERP | Vendor contract |

> **Principle:** Buy the undifferentiated infrastructure. Build the domain logic. The business rules, prompt engineering, and edge case handling ARE the competitive moat.

---

# Infrastructure Cost — Demo vs. Production

<div class="columns">

**Current (Demo) — ~$0–$10/month**
- Streamlit Community Cloud: **Free**
- Anthropic API: **Pay-per-query** (~$0–$10/mo demo usage)
- No database, no auth layer
- Suitable for: demos, prototypes, hackathons

**Production MVP — ~$1,500/month**
- Streamlit Teams or FastAPI on AWS: $500/month
- Anthropic API (10K queries/day): $300/month
- Pinecone (vector memory): $70/month
- PostgreSQL/RDS (decision logging): $50/month
- CloudWatch/observability: $30/month
- Support buffer: $550/month

</div>

**Enterprise (100K queries/day): ~$90,000–$120,000/month**
Amortized against: a single stockout averted = $145K+, single conflict avoided = $500K+

---

<!-- _class: section-header -->

# ⚙️ Part 4: Technical Architecture — Why, What, and How

---

# System Architecture — Two Pipelines, One Data Layer

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                   PIPELINE SELECTOR (Sidebar)            │
│  V1 — Agentic Loop          │   V2 — LangGraph Graph     │
│  orchestrator.py            │   langgraph_flow.py        │
│  Single agent, all 17 tools │   12 specialist nodes      │
│  6/10/20 iteration caps     │   Router → Domain → Synth  │
└──────────────┬──────────────┴──────────────┬────────────┘
               │                              │
               └──────────────┬───────────────┘
                              ▼
                  ┌─────────────────────┐
                  │   TOOL LAYER        │
                  │  17 tools via       │
                  │  mock_executor.py   │
                  │  TTL cache          │
                  │  Input validation   │
                  └────────┬────────────┘
                           ▼
                  Mock Data (Pydantic schemas)
                  PRODUCTS · CARRIERS · DCS · STORES
```

---

# Technical Component 1: Claude claude-sonnet-4-6 — The Reasoning Engine

**Why this model:**
- Native tool-use support — structured JSON in/out without prompt engineering hacks
- 200K context window — holds 12-message conversation without truncation
- Superior instruction-following for 47 non-negotiable supply chain rules
- Best-in-class structured business reasoning (vs. GPT-4o benchmarks)

**How tool use works (agentic loop):**

```python
# 1. Send query + 17 tool schemas to Claude
response = client.messages.create(model="claude-sonnet-4-6",
    tools=ALL_TOOLS, system=SYSTEM_PROMPT, messages=messages)

# 2. Claude returns stop_reason="tool_use" — it decided what to call
while response.stop_reason == "tool_use" and iteration < max_iter:
    for block in response.content:
        if block.type == "tool_use":
            result = mock_executor.execute(block.name, block.input)
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": json.dumps(_slim_tool_result(result))  # strip provenance
            })
    # 3. Feed results back — Claude continues reasoning
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": tool_results})
    response = client.messages.create(...)  # next iteration
```

---

# Technical Component 2: V1 Agentic Loop vs V2 LangGraph

| Dimension | V1 — Agentic Loop | V2 — LangGraph |
|-----------|------------------|----------------|
| **Architecture** | Single agent, 17 tools always visible | 12 specialist nodes, curated tool subsets |
| **Routing** | Claude decides implicitly | Explicit conditional edges by intent |
| **Auditability** | Tool call list | Node-by-node trace with `node_outputs` dict |
| **Context** | History summarization at 12 msgs | Stateful `RetailState` TypedDict |
| **Parallelism** | Sequential only | Supports parallel branches |
| **Prompt** | One global system prompt | Per-node domain-focused prompt |
| **Best for** | Exploratory conversation | Structured, repeatable workflows |
| **Latency** | Faster for simple | Better for complex multi-domain |
| **Governance** | Tool names + inputs | Full node output map — audit-ready |

> **Interview answer:** "V1 gives the LLM freedom; V2 gives you control. Financial services chooses V2 (auditability). Customer support chooses V1 (flexibility)."

---

# Technical Component 3: LangGraph — Multi-Agent Graph

**Why LangGraph over CrewAI / AutoGen:**
- **Explicit routing** — you define conditional edges, not agent negotiation
- **Shared state TypedDict** — every node reads and writes to one state object
- **Per-node prompting** — `price_cascade` node only knows about pricing; no cross-domain confusion
- **Audit trail** — `node_outputs` dict captures every node's reasoning

```python
class RetailState(TypedDict):
    query: str;  intent: str;  sku: str
    entities: Dict          # price, carrier, store — shared across all nodes
    tool_calls_made: List   # accumulated by EVERY node
    freshness_warnings: List # accumulated across ALL tool calls
    node_outputs: Dict      # each node deposits: {"price": ..., "supply": ...}
    final_response: str     # synthesizer writes last

# LangGraph graph construction
workflow = StateGraph(RetailState)
workflow.add_node("router", router_node)
workflow.add_node("price_cascade", price_cascade_node)
workflow.add_conditional_edges("router", lambda s: s["intent"],
    {"price_cascade": "price_cascade", "supply_disruption": "supply_disruption", ...})
```

---

# Technical Component 4: Tool Design — 17 Tools, One Contract

**Tool schema (JSON Schema — what Claude sees):**
```python
{
    "name": "simulate_price_change",
    "description": "Simulate full downstream impact of a retail price change on demand, inventory, POs, and financials.",
    "input_schema": {
        "type": "object",
        "properties": {
            "sku_id":        {"type": "string",  "description": "Product SKU (e.g. HUG48-3)"},
            "old_price":     {"type": "number",  "description": "Current retail price"},
            "new_price":     {"type": "number",  "description": "Proposed new price"},
            "horizon_weeks": {"type": "integer", "description": "Forecast horizon in weeks"}
        },
        "required": ["sku_id", "old_price", "new_price"]
    }
}
```

**Why tools instead of RAG:**
- RAG retrieves static text — tools **compute live values**
- Stockout risk, elasticity-adjusted demand, and carrying cost are calculations, not lookups
- Tools give the LLM the ability to act and reason, not just recall

**Tool execution contract (never raises, always returns a structured dict):**

| Field | Type | Purpose |
|-------|------|---------|
| `data` | dict | Business payload — what the LLM reasons on |
| `error` | str/None | Structured error — LLM can decide how to handle |
| `provenance` | OLTP/WMS/OLAP | Tracked for UI, stripped before LLM feed-back |
| `is_stale` | bool | Triggers freshness warning in UI |

---

# Technical Component 5: Input Validation + TTL Cache + Rate Limiting

<div class="columns">

**Input Validation (security layer)**
```python
VALID_SKUS = {"HUG48-3", "PAM72-5", ...}
VALID_CARRIERS = {"TruckCo_A", ...}

# Checked before ANY tool logic runs:
# Prevents LLM hallucinations reaching business logic
# Prevents prompt injection via tool inputs
# Prevents invalid prices ($0, negative, $99999)
```

**TTL Cache (cost + latency)**
```python
_CACHE_TTL = {"OLTP": 300, "WMS": 900, "OLAP": 86400}

# Cache key = tool_name + sorted JSON input
# Cache hit = instant response, zero API cost
# Write-through tools bypass cache:
#   trigger_replenishment — state-changing
#   adjust_promotional_price — state-changing
```

</div>

**Rate Limiting (budget protection)**
```python
_SESSION_QUERY_LIMIT = 30   # max queries per browser session

def _check_rate_limit() -> bool:
    return st.session_state.get("session_queries", 0) < _SESSION_QUERY_LIMIT
# Shown as progress bar in sidebar — user sees their remaining budget
```

---

# Technical Component 6: History Summarization

**The problem:** Long conversations blow up input token counts.

```
At 20 messages (avg 200 tokens each): +4,000 extra input tokens per call = +$0.012
At 40 messages:                       +8,000 extra input tokens per call = +$0.024
```

**The solution — compress at 12+ messages:**

```python
if len(messages) >= 12:
    recent = messages[-4:]    # keep last 4 verbatim (active context)
    older  = messages[:-4]    # everything older → summarize

    summary = client.messages.create(
        model=MODEL, max_tokens=600,    # short summary only
        system="Concise summarizer. No preamble.",
        messages=[{"role":"user", "content": "Summarize: " + older_text}]
    )
    # Replace N older messages with 2 messages (summary + ack)
    messages = [
        {"role": "user",      "content": "[SUMMARY]: " + summary},
        {"role": "assistant", "content": "Understood."},
    ] + recent
```

**Cost tradeoff:** Summarization call costs $0.018 (one-time). Saves $0.022+ per subsequent call. Break-even: 1 call after summarization.

---

<!-- _class: section-header -->

# 🎯 Part 5: Design Decisions, Trade-offs & Production Path

---

# Key Design Decisions and Rationale

| Decision | What We Chose | Why | Trade-off |
|----------|--------------|-----|-----------|
| **Mock data** | Rich mock layer over real APIs | Faster demo; no vendor credentials; edge cases always available | Must replace before production |
| **Two pipelines** | Both V1 + V2 | Demonstrates the spectrum: autonomy vs. control | Double maintenance surface |
| **Streamlit** | Streamlit over React | Python-native, free cloud deploy, `st.status()` for agentic loops | Not production-scale multi-user |
| **17 tools** | Domain-specific tools over monolithic | Right granularity; single responsibility; composable | More schema maintenance |
| **Pydantic schemas** | Strict typing at all boundaries | Catches silent data corruption; self-documenting | Small overhead |
| **Tools over RAG** | Compute over retrieval | Live values, not static text | RAG still needed for institutional memory |
| **prompts/system_prompt.txt** | Prompt in file over inline | Edit prompt without touching Python; versionable separately | Runtime file read on startup |

---

# What Would Change in Production

<div class="columns">

**Architecture changes:**
- FastAPI backend + React/Next.js frontend (not Streamlit)
- Proper session management with PostgreSQL
- Authentication: OAuth2 + API key per user
- Rate limiting per user, not per session
- Kubernetes deployment with horizontal scaling

**Data changes:**
- Replace `mock_executor.py` with real API calls
- Blue Yonder WMS → live inventory (15-min refresh)
- SAP S/4HANA → live pricing and POs
- Snowflake → OLAP analytics (24-hr batch)
- ToolResult contract stays identical — swap the data source only

</div>

**New capabilities to add (in order of ROI):**
1. **Real-time data connectors** — the #1 value unlock
2. **ERP execution hooks** — close the loop: AI recommends → buyer approves → SAP places PO
3. **Vector memory (Pinecone)** — institutional recall across sessions
4. **Proactive alert queue** — daily agent sweep → top 5 revenue-at-risk decisions
5. **Elasticity recalibration pipeline** — predicted vs. actual demand → model learns over time

---

# V1 vs V2 Pipeline Comparison — Full Picture

| Dimension | V1 — Agentic Loop | V2 — LangGraph |
|-----------|------------------|----------------|
| **When to use** | Exploratory, conversational, unpredictable queries | Structured, repeatable, governance-required workflows |
| **Routing** | Claude decides autonomously | Explicit conditional edges |
| **Debugging** | "Why did it call that tool?" is unclear | Every node's output is captured |
| **Latency** | Faster (no routing overhead) | Slightly slower (12 hops) |
| **Parallelism** | Sequential only | Parallel branches possible |
| **Compliance** | Tool list + inputs | Full node-by-node audit trail |
| **Extension** | Add a tool to TOOL_MAP | Add a node + edges to the graph |
| **System prompts** | One global prompt | Per-node domain-focused prompts |

> **Both pipelines share the same 17 tools, mock data layer, and ToolResult contract. Results are comparable. Reasoning paths differ.**

---

<!-- _class: section-header -->

# ❓ Part 6: Interview Q&A — Key Questions and Answers

---

# Q1: Walk me through what happens when a user sends a query

**The full agentic loop:**

```
1. Classify complexity  →  6 / 10 / 20 max iterations
2. Summarize history    →  if ≥ 12 messages: compress older turns (max 600 tokens)
3. Send to Claude       →  system_prompt + 17 tool schemas + history + query
4. Claude responds      →  stop_reason = "tool_use" → tool call(s) specified
5. Validate input       →  SKU whitelist, price bounds, location whitelist
6. Check cache          →  cache hit → return instantly, no API call
7. Execute tool         →  mock_executor.execute(name, input) → ToolResult
8. Track provenance     →  OLAP? → freshness warning stored separately
9. Slim the result      →  strip provenance/freshness before LLM feed-back (~30% tokens)
10. Feed back to Claude →  append tool_result messages → next iteration
11. Repeat 4–10         →  until stop_reason = "end_turn" or max_iter reached
12. Extract final text  →  + data caveats + action list → stream to UI
```

---

# Q2: Why did you choose LangGraph over CrewAI or AutoGen?

**Short answer:** Control and auditability over autonomy.

| Criterion | LangGraph | CrewAI | AutoGen |
|-----------|-----------|--------|---------|
| **Routing** | Explicit conditional edges | Agent negotiation | Agent negotiation |
| **State** | TypedDict — structured | Less structured | Message-based |
| **Auditability** | Node-by-node trace | Agent log | Message log |
| **Determinism** | High (you define edges) | Lower (agents decide) | Lower |
| **Domain prompting** | Per-node system prompts | Role-based | Role-based |
| **Production fit** | High for governed workflows | Good for autonomous tasks | Good for research agents |

**For supply chain with dollar consequences:** predictability and auditability matter more than autonomy. A CFO needs to trust the financial node specifically ran the P&L — not that 'some agent did something.'

---

# Q3: How do you handle prompt injection or malicious inputs?

**Three layers of defense:**

**Layer 1 — Input validation (tool layer):**
```python
VALID_SKUS = {"HUG48-3", "PAM72-5", ...}
# Injected SKU "'; DROP TABLE products;--" → whitelist check fails → structured error
# No SQL, no shell, no external calls — mock layer is pure Python
```

**Layer 2 — System prompt guardrails:**
- 8 non-negotiables in the system prompt constrain LLM behavior
- Injection through user queries must override 47 rules — practically impossible

**Layer 3 — Rate limiting:**
- 30 queries/session maximum
- Even a persistent attacker is capped at 30 API calls

**For production, I would add:**
- Server-side authentication (OAuth2)
- Input sanitization on the HTTP layer
- API key scoping: each user has their own budget ceiling
- Anomaly detection: flag sessions that hit 30 queries in <5 minutes

---

# Q4: What's the token spend on a complex query?

```
System prompt (loaded from file):    650 tokens   (fixed)
17 tool schemas:                   3,500 tokens   (fixed)
Conversation history (4 msgs):     1,000 tokens   (grows)
User query:                           30 tokens
─────────────────────────────────────────────────
Input (first call):                5,180 tokens

Tool result 1 (supply disruption): + 400 tokens   (slimmed)
Tool result 2 (carrier status):    + 300 tokens   (slimmed)
Tool result 3-5 (alternates ×3):   + 500 tokens   (slimmed)
─────────────────────────────────────────────────
Total input across 4 iterations:   6,380 tokens
Output (synthesis):                1,200 tokens

Cost: (6,380 × $3 + 1,200 × $15) / 1,000,000 = $0.037

Without _slim_tool_result(): +300 tokens per tool result = +$0.005 per query
At 10,000 complex queries:  $50 saved by one 10-line optimization
```

---

# Q5: What would you change to productionize this?

<div class="columns">

**Must-have for production:**
1. **FastAPI + React** — Streamlit is single-process, not truly multi-user
2. **Real data connectors** — mock → Blue Yonder WMS, SAP, Snowflake APIs
3. **ERP execution hooks** — AI recommends → buyer approves → SAP places PO
4. **Authentication + per-user rate limiting** — not just per-session
5. **PostgreSQL decision log** — every recommendation + outcome stored

**High-value enhancements:**
6. **Vector memory (Pinecone)** — institutional recall across sessions
7. **Proactive alert queue** — daily agent sweep → top 5 risks surfaced
8. **Elasticity recalibration** — predicted vs. actual demand → model learns
9. **Multi-horizon view** — 2W ops / 8W tactical / 26W strategic simultaneously
10. **Stakeholder framing** — same analysis, CFO view vs. Buyer view vs. Category Mgr view

</div>

---

# Key Numbers Every Interviewer May Test You On

| Parameter | Value | Why It Matters |
|-----------|-------|----------------|
| Diaper elasticity | **−1.4** | 10% price hike → 14% volume drop |
| Diaper recovery factor | **0.70** | Price cuts only recover 70% of lost demand |
| Tobacco recovery factor | **0.40** | Promos barely move cigarette volume |
| Dairy max days-of-supply | **3 days** | Hard cap — write-off risk above this |
| VMI share (milk) | **30%** | Carrying cost on 70% owned only |
| Replenishment lag | **3–4 days** | Order today → shelves in 4 days minimum |
| Late delivery probability | **30%** | 30% chance of +3 extra days on top |
| Forecast accuracy gate | **60% MAPE** | Below this: forecast blocked from PO system |
| CI widening rate | **+15% per 4 weeks** | At 8W: ±30% uncertainty band |
| TruckCo_D premium | **+45%** | Only SE diaper alternate, at high cost |
| Annual carrying rate | **25%** | Standard retail inventory holding cost |
| History summarize threshold | **12 messages** | Context window management trigger |
| Token budget: summary | **600** | History compression — cheap, short |
| Token budget: standard | **4,096** | Normal queries |
| Token budget: complex | **8,096** | Multi-SKU, tariff shock, multi-DC |

---

<!-- _class: section-header -->

# 🚀 Live Demo + Project Summary

---

# What Was Built — Complete Checklist

<div class="columns">

**Technical Achievements**
✅ V1: Full agentic loop (orchestrator.py)
✅ V2: LangGraph 12-node directed graph
✅ 17 domain tools with realistic mock data
✅ TTL cache (OLTP/WMS/OLAP TTL)
✅ Input validation + rate limiting
✅ Token differentiation (600/4096/8096)
✅ _slim_tool_result() — 30% token savings
✅ History summarization at 12+ messages
✅ Pydantic schemas across all layers
✅ Streaming UI with live tool tracking

**Business Logic Achievements**
✅ Asymmetric price elasticity modeling
✅ 15-variable demand forecast + CI
✅ Forecast accuracy gate (60% floor)
✅ Replenishment lag + variability model
✅ Perishable cap (3-day dairy rule)
✅ VMI exclusion from carrying cost
✅ Vendor trade dollar netting
✅ Regional carrier gap detection
✅ Scenario conflict detection (CRITICAL)
✅ P&L waterfall with full traceability

</div>

---

# Architecture Decision Log

| Decision | Chose | Rejected | Reason |
|----------|-------|----------|--------|
| LLM provider | **Anthropic Claude** | OpenAI GPT-4o | Better instruction-following, native tool use, larger context |
| Multi-agent framework | **LangGraph** | CrewAI, AutoGen | Explicit routing, auditability, TypedDict state |
| UI framework | **Streamlit** | Flask+React, Dash | Python-native, free cloud, st.status() for agentic loops |
| Memory strategy | **In-session summarization** | Full conversation, external DB | Balance cost vs. context quality |
| Tool execution | **Pure Python mock** | SQLite, REST APIs | Zero dependency, edge cases always available |
| Prompt storage | **prompts/system_prompt.txt** | Inline Python string | Editable without code deploy; versionable |
| Token optimization | **Differentiated budgets** | Single global max_tokens | 30% cost reduction for standard queries |

---

<!-- _class: section-header -->

# 🎯 Live Demo

## [retail-supply-chain-ai.streamlit.app](https://retail-supply-chain-ai-hp2hz8kf9cjqfkr82wogkt.streamlit.app/)

**GitHub:** [github.com/Krishhs89/retail-supply-chain-ai](https://github.com/Krishhs89/retail-supply-chain-ai)

**Try these queries in the Chat tab:**
1. *"TruckCo B is on strike — what's the impact on diapers?"*
2. *"Raise HUG48-3 from $12.99 to $14.49 — simulate full cascade"*
3. *"We want to run a 10% promo on diapers while TruckCo B is on strike — is that safe?"*

---

<!-- _class: title -->

# Thank You

## Retail Supply Chain Optimization AI
### Claude claude-sonnet-4-6 · LangGraph · Streamlit · Python 3.13

---
*Interview Kickstart — Agentic AI Capstone · 2026*
*[INTERVIEW_PREP.md](INTERVIEW_PREP.md) — full domain + business + cost + technical Q&A guide*
