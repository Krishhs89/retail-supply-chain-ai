"""
UI metadata — data source panels and formula definitions for each tool tab.

Extracted from streamlit_app.py to keep the UI file lean.
Imported by ui/streamlit_app.py as:
    from data.ui_metadata import _DATA_SOURCES, _FORMULAS
"""

from __future__ import annotations

# ─── Data Source Panels ───────────────────────────────────────────────────────
# Keys map to tab identifiers used by _render_data_sources(tab_key).
# Each value has: title (str), sources (list of 3-tuples), cloud_note (str).

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

# ─── Formula Panels ───────────────────────────────────────────────────────────
# Keys map to the same tab identifiers used by _render_formula_panel(tab_key).
# Each value has: title (str), formula (raw markdown/LaTeX str),
#   params (list of 6-tuples: label, key, default, min, max, step).

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
