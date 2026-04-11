# Retail Supply Chain Optimization AI — Specialist Guide

> **Audience:** Category managers, supply chain VPs, buyers, and operations leads who use this system to make daily and strategic decisions. This guide explains what the system does, what's been built, how to use it, and where it can grow.

---

## 1. What This System Does

### The Problem It Solves

In retail, a single decision — changing a price, absorbing a carrier strike, promoting a product — does not live in isolation. It propagates:

- A **10% price increase** on diapers reduces demand by 14%, which reduces the replenishment PO, which reduces carrier load, which reduces DC throughput, which ultimately changes carrying cost and net margin — all within 96 hours.
- A **carrier strike** in the Southeast doesn't just delay shipments. It triggers a domino: stockout risk rises at 3 DCs within 4 days, alternate carrier capacity becomes critical, and a parallel promotion becomes a liability because you can't fill demand you just created.

A human expert reasoning through these chains takes 2–4 hours per decision. This system does it in seconds — across all interconnected domains simultaneously.

### The Elevator Pitch

**A multi-agent AI assistant that simulates the full supply chain cascade from a single decision input — pricing, supply disruption, demand shift, or combined scenario — and delivers a ranked, explainable recommendation with quantified financial and operational impact.**

### What It Covers

| Domain | What the AI Reasons About |
|--------|--------------------------|
| **Price Cascade** | Demand elasticity (asymmetric), competitive response, PO recalculation, DC inventory adjustment, financial margin |
| **Supply Disruption** | Carrier status, alternate routing, regional coverage gaps, days-of-supply risk by DC |
| **Demand Forecasting** | 15-variable demand model, 8-week horizon with confidence intervals, forecast accuracy gate |
| **Inventory Management** | Reorder points, safety stock, replenishment lag (3–7 days), planogram capacity |
| **Perishable Logic** | 3-day max days-of-supply cap for dairy, shelf-life aware order sizing |
| **Financial Impact** | Gross margin, vendor trade dollar netting, VMI exclusions, carrying cost, revenue-at-risk |
| **Scenario Conflict** | Simultaneous detection (e.g., promo + strike = CRITICAL conflict) |

---

## 2. What Has Been Achieved

### Capabilities Built

**Two reasoning pipelines:**

- **V1 — Agentic Loop:** A single Claude claude-sonnet-4-6 agent that autonomously selects and chains up to 17 tools, with complexity-adaptive iteration limits (simple: 6, default: 10, complex: 20) and automatic history summarization at 12+ messages.

- **V2 — LangGraph:** A structured multi-agent graph with 12 specialist nodes (router → domain nodes → synthesizer), each with curated tool access and domain-specific prompting. Produces traceable node-by-node audit trails.

**17 operational tools** covering the full decision chain:

```
Pricing:         simulate_price_change, get_competitive_pricing, adjust_promotional_price
Demand:          get_demand_forecast, get_forecast_accuracy, analyze_demand_variables
Inventory:       get_inventory_levels, calculate_stockout_risk, get_reorder_recommendations
Supply Chain:    get_carrier_status, find_alternate_carriers, get_dc_inventory
Finance:         calculate_revenue_impact, calculate_carrying_cost
Scenario:        detect_scenario_conflicts, compare_scenarios, get_replenishment_schedule
```

**17 production-grade edge cases handled:**

| Edge Case | Behavior |
|-----------|----------|
| Asymmetric price elasticity | Tobacco, alcohol, formula, diapers recover only 40–70% of lost demand on price cuts |
| Perishable cap | Dairy orders never exceed 3 days of supply regardless of signal |
| Regional carrier gap | SE region has no diaper-capable carrier except TruckCo D at +45% cost |
| Forecast accuracy gate | <60% accuracy blocks forecast from entering PO system |
| CI widening | Confidence intervals widen 15% per 4-week horizon period |
| VMI exclusion | Vendor-managed inventory excluded from carrying cost calculations |
| Vendor trade dollar netting | Promotional subsidies netted before margin reporting |
| Replenishment lag | HQ→DC→Store = 3–4 days + 30% probability of +3 extra days |
| Planogram overflow | Orders capped at available shelf space; excess routes to back-of-store |
| Scenario conflict detection | Simultaneous promotion + supply disruption flags CRITICAL |
| Data provenance | OLAP (24h lag) vs WMS (15min) vs OLTP (5min) — warns when stale data drives ops decisions |
| Context window management | History auto-summarized at ≥12 messages |
| Multi-DC reasoning | 4 DCs (NW, NE, SE, SW) with different inventory, carrier access, and regional risk profiles |
| Complexity detection | Query keyword analysis selects appropriate iteration depth |
| Seasonal demand uplift | Holiday and back-to-school uplift baked into demand variables |
| Stockout cost estimation | Lost margin + customer lifetime value erosion calculated |
| Multi-scenario comparison | 3 simultaneous scenario branches ranked by revenue and margin |

---

## 3. How a Specialist Uses This System

### Role: Category Manager (Pricing & Promotion)

**Daily workflow:**
1. Open the **Price Cascade tab** → select SKU → set new price → see the full demand, revenue, and margin impact in under 10 seconds.
2. Before launching a promotion, use the **Chat tab**: *"We're planning a 10% diaper price cut during TruckCo B strike — is this safe?"* The system detects the conflict and recommends deferring.
3. Use the **Scenario Planner** to compare Hold / Raise / Promote side-by-side with asymmetric elasticity applied — pick the option that maximizes margin, not just revenue.

**Key things to watch:**
- Asymmetric elasticity: dropping diaper price doesn't recover proportionally — only 70% of demand returns after a price cut versus a price hold. Promotions are more expensive than they appear.
- Vendor trade dollar netting: that 8% vendor subsidy on Huggies changes your effective promo cost significantly. The system accounts for it.

### Role: Supply Chain / Operations Lead

**Daily workflow:**
1. Monitor the **Dashboard tab** — carrier status chips and DC inventory KPIs give the network snapshot.
2. When a disruption hits, go to **Supply Alert tab** → enter carrier and region → get alternate routing options with cost and coverage gaps.
3. For replenishment decisions, use **Shelf & Store tab** → the system handles perishable caps, planogram constraints, and replenishment lag automatically.

**Key things to watch:**
- SE region is uniquely vulnerable: only TruckCo D handles diapers there, at a 45% cost premium. Any diaper promo or volume increase in SE should be pressure-tested against TruckCo D capacity.
- Replenishment lag: even an approved PO takes 3–4 days (plus 30% chance of 7 days). Plan replenishment triggers 4–7 days before projected stockout, not on the day of.

### Role: Buyer

**Daily workflow:**
1. Use **Demand Forecast tab** — review 8-week horizon with confidence intervals. At 8 weeks, CI widens to ±30%. Only commit POs you can adjust within that uncertainty band.
2. Check forecast accuracy gate: if accuracy drops below 60%, the system automatically flags the forecast as unreliable — do not place large orders based on it.
3. VMI products (e.g., Milk at 30% VMI) reduce your carrying cost exposure. The system excludes VMI inventory from your carrying cost calculations automatically.

### Role: Supply Chain VP / Finance

**Strategic workflow:**
1. Use the **Financial Impact tab** for P&L waterfall: revenue → gross margin → trade dollars → carrying cost → net margin.
2. Use the **Chat tab** for multi-scenario strategic questions: *"Compare 8-week revenue impact of holding diaper price vs. 10% increase given current carrier strike and forecast accuracy at 67%."*
3. The **Scenario Planner** is your go-to for capital allocation questions: which scenario maximizes revenue? Which maximizes margin? Which has acceptable service level risk?

---

## 4. Macro → Micro: How Decisions Ripple

Every decision in this system creates a cascade. Here is the mental model:

```
TRIGGER (price change / supply event / demand shift)
    │
    ▼
DEMAND IMPACT
  └── Elasticity-adjusted volume change
  └── Promotional uplift or drag
  └── Forecast accuracy applied to volume confidence
    │
    ▼
INVENTORY IMPACT
  └── Replenishment PO recalculation
  └── Safety stock adequacy check
  └── Days-of-supply check (perishable cap applied)
  └── Planogram capacity check
    │
    ▼
SUPPLY CHAIN IMPACT
  └── Carrier capacity requirement change
  └── Regional coverage gap exposure
  └── Lead time risk (3–4d + 30% late probability)
    │
    ▼
FINANCIAL IMPACT
  └── Revenue: price × volume (elasticity-adjusted)
  └── Margin: gross margin % × revenue
  └── Vendor trade netting
  └── VMI exclusion from carrying cost
  └── Carrying cost: owned units × 25% annual rate × period
    │
    ▼
CONFLICT DETECTION
  └── Simultaneous scenario analysis
  └── CRITICAL if promo + supply risk in same window
```

When you make a change in the Chat tab, the agent is reasoning through ALL of these layers — not just the one you asked about.

---

## 5. Improvement Suggestions for Sophisticated Specialist Experience

These are the next 10 capabilities that would meaningfully elevate this from a decision support tool to a decision execution system:

### 5.1 Decision Memory & Learning
**What:** Log every recommendation the system made, what the specialist decided, and what actually happened 30/60/90 days later.
**Why it matters:** Over time, you learn whether your elasticity assumptions are right. If diaper demand fell 18% on a 10% price increase but the model predicted 14%, the model recalibrates.
**Implementation path:** Persist recommendations + outcomes to a database. Feed delta back to elasticity parameters quarterly.

### 5.2 Alert Triage Queue
**What:** Instead of reactive chat queries, proactively surface the top 5 decisions that need attention today, ranked by revenue-at-risk.
**Why it matters:** A supply chain VP doesn't have time to ask questions — they need the system to tell them what to ask.
**Implementation path:** A daily scheduled agent run across all SKUs and DCs; surface anomalies with quantified impact as a triage list.

### 5.3 Cross-Category Elasticity
**What:** When diaper prices rise, some customers substitute to a competing brand or a different size. Model cross-elasticity between SKUs and between categories.
**Why it matters:** A 10% diaper price increase may drop Huggies volume 14% but increase store-brand diaper volume 9% — net category impact is different from SKU impact.
**Implementation path:** Add a cross-elasticity matrix to the product catalog; update demand forecast to factor in substitute products.

### 5.4 Probabilistic Outcome Trees
**What:** Instead of a single point estimate, show the probability distribution of outcomes: "40% chance revenue increases, 35% chance flat, 25% chance decreases."
**Why it matters:** Executives make better decisions with uncertainty quantified, not hidden.
**Implementation path:** Monte Carlo sampling over elasticity ranges, demand variable uncertainty, and carrier recovery probability. Render as fan chart (already partially built in Demand Forecast tab).

### 5.5 Natural Language ERP Execution
**What:** After the AI recommends an action (e.g., "place PO for 4,200 units of HUG48-3 to DC-SE with priority flag"), allow one-click execution to the ERP/WMS via API.
**Why it matters:** Closing the loop from recommendation to action is where the most time is currently lost. A specialist still has to manually enter the PO.
**Implementation path:** Add ERP integration layer (SAP/Oracle Retail API). Gate with human-in-the-loop confirmation step.

### 5.6 Multi-Horizon Planning View
**What:** Show the same SKU/scenario across 3 horizons simultaneously — 2-week operational, 8-week tactical, 26-week strategic.
**Why it matters:** A price change decision looks different at 2 weeks (revenue) vs. 26 weeks (brand equity, competitive response, cost structure).
**Implementation path:** Run three separate forecast chains in parallel; render as a synchronized timeline view.

### 5.7 Stakeholder-Specific Recommendation Framing
**What:** The same underlying analysis, but reframed for: Buyer (inventory and service level focus), Category Manager (price and margin focus), CFO (net margin and cash flow focus).
**Why it matters:** The same data needs different emphasis for different decision-makers. A buyer doesn't need the P&L waterfall; a CFO doesn't need the planogram constraint.
**Implementation path:** Add a role selector to the sidebar; apply role-specific system prompts to the synthesizer node.

### 5.8 What-If Branching with Version Control
**What:** Create named "branches" of a scenario — "Base Case", "Aggressive Promo", "Conservative Hold" — and compare them side by side with the ability to roll back.
**Why it matters:** Planning sessions involve multiple stakeholders proposing variants. Today, each variant overwrites the previous conversation.
**Implementation path:** Store scenario state as named snapshots in session storage. Add a scenario selector that loads a past state for comparison.

### 5.9 Competitor Move Simulation
**What:** "If Costco drops their diaper price 15%, what should we do?" The system simulates the competitive response and recommends a counter-strategy.
**Why it matters:** Competitive response is one of the top 3 daily concerns for a category manager, but today the system only provides static competitive pricing data.
**Implementation path:** Add a competitive response agent that models cross-retailer demand shift and recommends price/promotion response options.

### 5.10 Executive Summary Auto-Generation
**What:** After any complex query, auto-generate a 1-page executive summary: situation, recommendation, financial impact, risks, next actions.
**Why it matters:** Most AI-generated content is too detailed for a 5-minute briefing. A summary layer makes the system useful for steering committee presentations.
**Implementation path:** Add a post-synthesizer summarization node that condenses the full response into a structured brief with configurable detail level.

---

## 6. System Architecture at a Glance

```
User Query
    │
    ▼
[Pipeline Router — sidebar toggle]
    │
    ├── V1: Single Agent (Claude claude-sonnet-4-6)
    │       → 17 tools available
    │       → Adaptive iteration (6/10/20)
    │       → History summarization at 12 msgs
    │
    └── V2: LangGraph Multi-Agent Graph
            → Router node classifies intent + extracts entities
            → Domain nodes (price / supply / demand / scenario / shelf)
            → Supporting nodes (inventory / carrier / accuracy / perishable / financial)
            → Synthesizer merges all outputs into final response

[Mock Data Layer]
    → PRODUCTS: 5 SKUs with elasticity, margins, perishable flags
    → CARRIERS: 4 carriers with status, region coverage, cargo restrictions
    → DCS: 4 distribution centers with WMS + OLAP inventory
    → STORES: 10 stores across 4 regions

[Data Provenance]
    → OLTP: 5-minute lag (pricing, PO)
    → WMS: 15-minute lag (inventory operations)
    → OLAP: 24-hour lag (analytics — flagged stale for ops decisions)
```

---

## 7. Quick Reference: Key Numbers to Know

| Parameter | Value | Why It Matters |
|-----------|-------|----------------|
| Diaper elasticity | -1.4 | 10% price increase → ~14% volume drop |
| Diaper recovery factor | 0.70 | Price cuts only recover 70% of lost demand |
| Tobacco recovery factor | 0.40 | Very sticky — promotions don't drive volume |
| Dairy max days-of-supply | 3 days | Never order more than 3 days of dairy |
| VMI share (milk) | 30% | 30% of milk inventory owned by vendor |
| Replenishment lag | 3–4 days | Order today = shelves in 4 days minimum |
| Late delivery probability | 30% | 30% chance of +3 extra days on top |
| Forecast accuracy gate | 60% | Below this: forecast blocked from PO system |
| CI widening | +15% per 4 weeks | At 8 weeks: ±30% uncertainty band |
| Annual carrying rate | 25% | Cost of holding inventory for a year |
| TruckCo D diaper premium | +45% | SE region alternate carrier cost |
| Vendor trade pct (diapers) | 8% | Vendor subsidizes 8% of diaper promo cost |

---

*This system is a foundation. The architecture is designed to be extended — real inventory feeds, live carrier APIs, ERP execution hooks, and a decision memory layer are the natural next steps. The goal is to make every supply chain decision faster, more connected, and better quantified.*
