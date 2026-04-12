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
  }
  section.title {
    background: linear-gradient(135deg, #0071ce 0%, #004a8f 100%);
    color: white;
    text-align: center;
  }
  section.title h1 {
    font-size: 2.4em;
    font-weight: 800;
    margin-bottom: 0.2em;
  }
  section.title h2 {
    font-size: 1.1em;
    font-weight: 300;
    opacity: 0.9;
  }
  section.section-header {
    background: #0071ce;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
  }
  section.section-header h1 {
    font-size: 2.2em;
    font-weight: 700;
  }
  h1 { color: #0071ce; font-size: 1.6em; border-bottom: 3px solid #0071ce; padding-bottom: 8px; }
  h2 { color: #1a1a2e; font-size: 1.2em; }
  h3 { color: #0071ce; font-size: 1em; }
  table { width: 100%; font-size: 0.75em; border-collapse: collapse; }
  th { background: #0071ce; color: white; padding: 8px 12px; }
  td { padding: 6px 12px; border-bottom: 1px solid #e9ecef; }
  tr:nth-child(even) td { background: #f8f9fa; }
  code { background: #f0f4ff; color: #0071ce; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
  .highlight { background: #fff3cd; border-left: 4px solid #f39c12; padding: 12px 16px; border-radius: 4px; margin: 8px 0; }
  .critical { background: #f8d7da; border-left: 4px solid #dc3545; padding: 12px 16px; border-radius: 4px; }
  .success { background: #d4edda; border-left: 4px solid #28a745; padding: 12px 16px; border-radius: 4px; }
  footer { font-size: 0.7em; color: #6c757d; }
---

<!-- _class: title -->

# 🏪 Retail Supply Chain<br>Optimization AI

## Multi-Agent Agentic AI System for Enterprise Retail Decision-Making

---
**Powered by Claude claude-sonnet-4-6 · LangGraph · Streamlit**
Built for Interview Kickstart — Agentic AI Capstone Project

---

# The Problem: Decisions Don't Live in Isolation

Every supply chain decision creates a **cascade across 5 domains simultaneously**

<div class="columns">

**A 10% price increase on diapers triggers:**
- 📉 14% demand volume drop (elasticity = -1.4)
- 📦 Replenishment PO recalculation across 4 DCs
- 🚛 Carrier load reduction in SE and MW regions
- 💰 Margin change after vendor trade dollar netting
- ⚠️ Potential scenario conflict with active promotions

**A human expert takes 2–4 hours per decision.**
**This system does it in seconds.**

</div>

> **$1B+ revenue decisions are made daily in retail without connected, real-time AI reasoning across pricing, supply, demand, and finance.**

---

# What We Built

<div class="columns">

**Two AI Reasoning Pipelines**

🔵 **V1 — Agentic Loop**
Single Claude claude-sonnet-4-6 agent
17 tools, adaptive iterations (6 / 10 / 20)
Auto history summarization at 12 messages

🟢 **V2 — LangGraph Multi-Agent**
12 specialist nodes in a directed graph
Router → Domain Nodes → Synthesizer
Node-by-node audit trail

**17 Operational Tools covering:**
- Pricing & demand simulation
- Inventory & replenishment
- Carrier & supply chain
- Financial P&L
- Scenario conflict detection

</div>

---

# System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│           PIPELINE TOGGLE           │
│    V1: Agentic Loop  │  V2: LangGraph│
└─────────────┬───────────────┬───────┘
              │               │
         [Claude]       [ROUTER NODE]
         17 tools            │
              │    ┌──────────┼──────────┐
              │    ▼          ▼          ▼
              │  Price    Supply     Demand
              │  Cascade  Disruption Forecast
              │    │          │          │
              │  Inventory Carrier   Accuracy
              │  Node      Node      Gate
              │    └──────────┼──────────┘
              │               ▼
              └──────► [SYNTHESIZER]
                             │
                           END
```

---
<!-- _class: section-header -->

# 🔑 Key Capabilities & Edge Cases

---

# 17 Production-Grade Edge Cases

| Edge Case | Behavior |
|-----------|----------|
| **Asymmetric price elasticity** | Tobacco/diapers recover only 40–70% of demand on price cuts |
| **Perishable cap** | Dairy orders capped at 3 days of supply — no exceptions |
| **Regional carrier gap** | SE has only TruckCo D for diapers at +45% cost premium |
| **Forecast accuracy gate** | < 60% accuracy blocks forecast from entering PO system |
| **CI widening** | Confidence intervals widen 15% per 4-week horizon period |
| **VMI exclusion** | Vendor-managed inventory excluded from carrying cost |
| **Trade dollar netting** | Vendor subsidies netted before margin reporting |
| **Replenishment lag** | 3–4 days + 30% chance of +3 extra days |
| **Scenario conflict** | Promo + supply disruption = CRITICAL flag |
| **Data provenance** | OLAP (24h) vs WMS (15min) — warns on stale ops decisions |

---

# Live Scenario: Carrier Strike + Price Promotion

<div class="highlight">
⚠️ <strong>Scenario:</strong> TruckCo B (diapers, SE + MW) goes on strike. Simultaneously, category manager wants to launch a 10% promotional price cut on Huggies HUG48-3.
</div>

**What the AI reasons through:**

1. **Conflict Detection** → CRITICAL: Promotion creates demand surge (+14%) while supply is constrained
2. **Regional Gap Analysis** → SE has no diaper carrier except TruckCo D (+45% cost)
3. **Days-of-Supply Clock** → DC-SE hits safety stock in **4 days** without alternate carrier
4. **Financial Impact** → Revenue-at-risk: ~$890K over 14-day strike window
5. **Recommendation** → Defer promotion until carrier restored OR pre-build DC-SE inventory

<div class="critical">
❌ Proceed with promotion = stockout risk in SE within 4 days + 45% carrier cost premium
</div>

---

# Live Scenario: Price Cascade Analysis

**Query:** *"Raise HUG48-3 diaper price from $12.99 to $14.49 — simulate full cascade"*

| Step | What Happens | Numbers |
|------|-------------|---------|
| Price change | +$1.50 / unit (+11.5%) | $12.99 → $14.49 |
| Demand impact | Elasticity = -1.4 | **-16.1% volume** |
| Asymmetric check | Recovery factor = 0.70 | Applied to any future reversal |
| PO recalculation | Lower demand → smaller orders | -16% replenishment |
| Margin impact | Higher price × lower volume | Net: **+4.2% revenue** |
| Vendor trade | 8% subsidy on promos only | Not triggered here |
| Carrying cost | Fewer owned units | -$12K annual carrying cost |

> **Bottom line:** Price increase is accretive to revenue and margin. Carrier capacity headroom exists. Recommend proceed.

---

# The 9-Tab Streamlit Interface

| Tab | What It Shows |
|-----|--------------|
| **Dashboard** | Live network status — carrier chips, DC KPIs, strike timeline |
| **Chat** | Streaming multi-agent conversation with live tool tracking |
| **Price Cascade** | Price change → demand → PO → financial waterfall |
| **Supply Alert** | Carrier strike → stockout bars → alternate carrier chips |
| **Demand Forecast** | 8-week fan chart with CI widening + variable contributions |
| **Scenario Planner** | 3-scenario side-by-side comparison + conflict detection |
| **Shelf & Store** | Replenishment Gantt + perishable cap + planogram check |
| **Financial Impact** | P&L waterfall: revenue → margin → trade → VMI → net |
| **Data Sources** | Provenance tracker — WMS vs OLAP freshness comparison |
| **Workflow** | Step-by-step integrated scenario builder (new) |
| **Flow Map** | LangGraph DAG with live execution trace (new) |

---
<!-- _class: section-header -->

# 📊 Results & What Was Achieved

---

# What Was Achieved

<div class="columns">

**Technical Achievements**
✅ Full multi-agent agentic loop (V1)
✅ LangGraph 12-node directed graph (V2)
✅ 17 domain tools with mock data
✅ Asymmetric elasticity modeling
✅ Scenario conflict detection engine
✅ Data provenance tracking system
✅ Context window management (12-msg threshold)
✅ Streaming UI with live tool tracking
✅ Deployed on Streamlit Community Cloud

**Business Logic Achievements**
✅ Price → demand → PO → financial cascade
✅ Regional carrier gap detection
✅ Perishable inventory capping
✅ Vendor trade dollar netting
✅ VMI exclusion from carrying costs
✅ Forecast accuracy gating (60% floor)
✅ Multi-scenario comparison with ranking
✅ Replenishment chain lag modeling

</div>

---

# V1 vs V2 Pipeline Comparison

| Dimension | V1 — Agentic Loop | V2 — LangGraph |
|-----------|------------------|----------------|
| **Architecture** | Single agent, all 17 tools | 12 specialist nodes |
| **Reasoning** | Autonomous tool selection | Structured routing |
| **Auditability** | Tool call list | Node-by-node trace |
| **Latency** | Faster for simple queries | Better for complex chains |
| **Configurability** | Max iterations slider | Automatic per intent |
| **Best for** | Exploratory / conversational | Repeatable / structured |
| **Context** | History summarization | State graph accumulation |
| **Transparency** | Tool names + inputs | Full node output map |

> **Both pipelines share the same 17 tools and mock data layer — results are comparable, reasoning paths differ.**

---
<!-- _class: section-header -->

# 🚀 10 Improvement Suggestions

---

# Improvements: From Decision Support → Decision Execution

| # | Capability | Business Impact |
|---|-----------|----------------|
| 1 | **Decision Memory & Learning** | Recalibrate elasticity from actual outcomes |
| 2 | **Alert Triage Queue** | Proactively surface top 5 revenue-at-risk decisions daily |
| 3 | **Cross-Category Elasticity** | Huggies price up → store-brand diapers up 9% |
| 4 | **Probabilistic Outcome Trees** | 40% revenue up / 35% flat / 25% down |
| 5 | **NL ERP Execution** | One-click: AI recommendation → SAP PO creation |
| 6 | **Multi-Horizon Planning** | 2W operational / 8W tactical / 26W strategic simultaneous |
| 7 | **Stakeholder Framing** | Same analysis, Buyer vs. CFO vs. Category Manager view |
| 8 | **What-If Branching** | Named scenario versions with rollback |
| 9 | **Competitor Move Simulation** | Costco drops price 15% → recommend counter-strategy |
| 10 | **Executive Summary Auto-Gen** | 1-page brief for steering committee |

---

# The Macro → Micro Decision Mental Model

```
TRIGGER
(price / supply / demand / tariff / seasonal / competitive)
        │
        ▼
DEMAND IMPACT          ← elasticity, promotional uplift, forecast accuracy
        │
        ▼
INVENTORY IMPACT       ← reorder points, safety stock, perishable cap
        │
        ▼
SUPPLY CHAIN IMPACT    ← carrier capacity, regional gaps, lead time risk
        │
        ▼
FINANCIAL IMPACT       ← revenue × margin × trade netting × VMI × carrying cost
        │
        ▼
CONFLICT DETECTION     ← scenario overlap, CRITICAL flags, defer/proceed/modify
        │
        ▼
RANKED RECOMMENDATION  ← 3 options with trade-off table
```

---

# Key Numbers Every Specialist Should Know

| Parameter | Value | Why It Matters |
|-----------|-------|----------------|
| Diaper elasticity | **-1.4** | 10% price hike → 14% volume drop |
| Diaper recovery factor | **0.70** | Price cuts only recover 70% of lost demand |
| Tobacco recovery factor | **0.40** | Very sticky — promotions don't move volume |
| Dairy max days-of-supply | **3 days** | Hard cap — no exceptions |
| VMI share (milk) | **30%** | 30% of milk inventory owned by vendor |
| Replenishment lag | **3–4 days** | Order today = shelves in 4 days minimum |
| Late delivery probability | **30%** | 30% chance of +3 extra days on top |
| Forecast accuracy gate | **60%** | Below this: forecast blocked from PO system |
| TruckCo D premium | **+45%** | SE region alternate carrier cost |
| Annual carrying rate | **25%** | Cost of holding inventory for a year |

---
<!-- _class: section-header -->

# 🎯 Live Demo

### [retail-supply-chain-ai.streamlit.app](https://retail-supply-chain-ai-hp2hz8kf9cjqfkr82wogkt.streamlit.app/)

---
**GitHub:** [github.com/Krishhs89/retail-supply-chain-ai](https://github.com/Krishhs89/retail-supply-chain-ai)

---

<!-- _class: title -->

# Thank You

## Retail Supply Chain Optimization AI
### Built with Claude claude-sonnet-4-6 · LangGraph · Streamlit

---
*Interview Kickstart — Agentic AI Capstone · 2026*
