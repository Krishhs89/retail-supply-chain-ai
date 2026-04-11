"""
Claude tool schemas for the retail supply-chain optimization system.

Each tool maps 1-to-1 with a function in mock_executor.py.
All tools return a structured dict with: data, error, provenance, freshness_minutes, is_stale.
"""

ALL_TOOLS = [

    # ──────────────────────────────────────────────────────────────────────────
    # PRICING TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "simulate_price_change",
        "description": (
            "Simulate the full cascade effect of a SKU price change across HQ, DC, and stores. "
            "Returns demand impact (with asymmetric elasticity for diapers/tobacco/alcohol/formula), "
            "inventory adjustment, open PO recommendations, and financial impact (gross and net of vendor trade dollars)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string", "description": "Product SKU (e.g. HUG48-3, PAM72-5, MLK-GAL)"},
                "old_price": {"type": "number", "description": "Current retail price in USD"},
                "new_price": {"type": "number", "description": "Proposed new retail price in USD"},
                "horizon_weeks": {
                    "type": "integer",
                    "description": "Planning horizon in weeks (default 8). Longer horizons have wider CI.",
                    "default": 8
                },
                "affected_dc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "DCs to include (default: all). e.g. ['DC-NW', 'DC-SE']"
                },
            },
            "required": ["sku", "old_price", "new_price"]
        }
    },
    {
        "name": "get_competitive_pricing",
        "description": "Get current competitor pricing for a SKU to contextualize a price change decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "competitors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Competitor names (default: all). e.g. ['Target', 'Costco', 'Amazon']"
                }
            },
            "required": ["sku"]
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # DEMAND TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "get_demand_forecast",
        "description": (
            "Get demand forecast with 80% and 95% confidence intervals. "
            "Confidence intervals widen ~15% per 4-week horizon. "
            "IMPORTANT: if forecast accuracy < 60%, this returns is_reliable=false and "
            "you must NOT pass those numbers downstream as planning inputs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "horizon_weeks": {
                    "type": "integer",
                    "description": "Forecast horizon in weeks (1–32). CI widens significantly beyond week 4."
                },
                "variable_overrides": {
                    "type": "object",
                    "description": (
                        "Override specific demand variable values. Keys: price, promo, markdown, trade_spend, "
                        "competitor_price, weather, seasonality, tariff, supply_availability, store_count, "
                        "customer_income_index, trend, social_media_buzz, regulatory_change, external_event"
                    )
                }
            },
            "required": ["sku", "horizon_weeks"]
        }
    },
    {
        "name": "calculate_price_elasticity",
        "description": (
            "Calculate demand response to a price change using product-class-specific elasticity. "
            "Handles ASYMMETRIC elasticity: for diapers, formula, tobacco, alcohol — "
            "a price increase suppresses demand more than a price decrease recovers it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "price_change_pct": {
                    "type": "number",
                    "description": "Percentage price change. Positive = increase, negative = decrease."
                },
                "base_demand_units_per_week": {
                    "type": "number",
                    "description": "Current weekly demand per store (leave blank to use system baseline)."
                }
            },
            "required": ["sku", "price_change_pct"]
        }
    },
    {
        "name": "get_forecast_accuracy",
        "description": "Get current MAPE forecast accuracy for a SKU or category at a given horizon.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "description": "SKU or 'all' for category-level rollup"
                },
                "horizon_weeks": {
                    "type": "integer",
                    "description": "Which horizon to evaluate. Default: 8.",
                    "default": 8
                }
            },
            "required": ["sku"]
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # INVENTORY TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "get_inventory_levels",
        "description": (
            "Get current inventory at HQ/DC/store for a SKU. "
            "Returns both WMS (15-min lag) and OLAP (24-hr lag) values. "
            "Flags discrepancies between them — use WMS for operational decisions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "location_type": {
                    "type": "string",
                    "enum": ["all", "dc", "store", "hq"],
                    "description": "Filter by location type. Default: all."
                },
                "location_id": {
                    "type": "string",
                    "description": "Specific location (e.g. DC-NW, STR-005). Omit for all."
                }
            },
            "required": ["sku"]
        }
    },
    {
        "name": "check_shelf_capacity",
        "description": (
            "Check planogram shelf capacity for a SKU at a store. "
            "Returns max facings and whether the proposed replenishment quantity would exceed shelf capacity. "
            "Excess must go to back-of-store staging, not the shelf."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "store_id": {"type": "string"},
                "proposed_replenishment_units": {
                    "type": "integer",
                    "description": "How many units you intend to put on the shelf."
                }
            },
            "required": ["sku", "store_id", "proposed_replenishment_units"]
        }
    },
    {
        "name": "calculate_stockout_risk",
        "description": (
            "Calculate stockout risk for a SKU at a location. "
            "Accounts for full replenishment lag (HQ→DC→Store = 3-4 days) and "
            "lead time variability (30% of POs arrive 2-4 days late). "
            "CRITICAL: if days_on_hand < replenishment_lag, recommend emergency transfer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "location_id": {"type": "string", "description": "DC or store ID"},
                "demand_adjustment_pct": {
                    "type": "number",
                    "description": "Percentage demand change to apply (e.g. -16 for demand drop). Default 0.",
                    "default": 0
                },
                "supply_disruption_additional_days": {
                    "type": "integer",
                    "description": "Extra delay days from an active supply disruption. Default 0.",
                    "default": 0
                }
            },
            "required": ["sku", "location_id"]
        }
    },
    {
        "name": "check_perishable_status",
        "description": (
            "For perishable SKUs (dairy, produce, deli): check expiration risk and whether "
            "proposed replenishment would exceed the perishable max days-of-supply cap. "
            "Returns markdown recommendation if at risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "location_id": {"type": "string"},
                "proposed_replenishment_units": {"type": "integer"}
            },
            "required": ["sku", "location_id", "proposed_replenishment_units"]
        }
    },
    {
        "name": "trigger_replenishment",
        "description": (
            "Recommend/trigger a replenishment order between locations. "
            "Respects planogram constraints and perishable caps. "
            "Sets priority level which affects lead time and cost."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "from_location": {"type": "string", "description": "Source DC (e.g. DC-NW)"},
                "to_location": {"type": "string", "description": "Destination store (e.g. STR-005)"},
                "quantity": {"type": "integer"},
                "priority": {
                    "type": "string",
                    "enum": ["standard", "expedited", "emergency"],
                    "description": "standard=4 days, expedited=2 days, emergency=1 day (3x cost)",
                    "default": "standard"
                }
            },
            "required": ["sku", "from_location", "to_location", "quantity"]
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SUPPLY CHAIN TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "get_carrier_status",
        "description": "Get status of all carriers (active/strike/delayed), their cargo assignments, and regional coverage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "carrier_id": {
                    "type": "string",
                    "description": "Specific carrier ID or 'all'. e.g. 'TruckCo_B', 'all'"
                },
                "region": {
                    "type": "string",
                    "description": "Filter by region (NW | SE | MW | all). Default: all.",
                    "default": "all"
                }
            },
            "required": []
        }
    },
    {
        "name": "find_alternate_carriers",
        "description": (
            "Find alternate carriers for a disrupted lane. "
            "Checks REGIONAL availability — not just national. "
            "Flags explicitly when NO viable alternates exist in the affected geography. "
            "Returns lead-time and cost premium for each viable alternate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "disrupted_carrier_id": {"type": "string"},
                "region": {
                    "type": "string",
                    "description": "Region where disruption applies (NW | SE | MW)"
                },
                "duration_days": {
                    "type": "integer",
                    "description": "Expected disruption duration in days"
                }
            },
            "required": ["sku", "disrupted_carrier_id", "region"]
        }
    },
    {
        "name": "get_supply_disruption_impact",
        "description": (
            "Calculate full impact of a supply disruption. "
            "Maps disruption type to appropriate duration model: "
            "carrier_strike=2-3 weeks, port_delay=6-10 weeks, supplier_bankruptcy=permanent. "
            "Returns stockout timelines, revenue at risk, and mitigation plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "disruption_type": {
                    "type": "string",
                    "enum": ["carrier_strike", "port_delay", "supplier_bankruptcy"]
                },
                "affected_entity": {
                    "type": "string",
                    "description": "Carrier/port/supplier ID (e.g. TruckCo_B, PORT-LA, SUP-KIMBERLY)"
                },
                "duration_days": {
                    "type": "integer",
                    "description": "Expected duration. Use 0 for supplier_bankruptcy (treated as permanent)."
                },
                "affected_skus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific SKUs affected. Omit to auto-detect from carrier cargo."
                }
            },
            "required": ["disruption_type", "affected_entity", "duration_days"]
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # FINANCIAL TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "calculate_revenue_impact",
        "description": (
            "Calculate revenue and margin impact of a change. "
            "IMPORTANT: nets vendor trade dollars against retail revenue before reporting margin. "
            "A promotion funded by vendor trade spend may be revenue-neutral or positive at the net level. "
            "Tax calculations are flagged as approximate — jurisdiction matters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "old_price": {"type": "number"},
                "new_price": {"type": "number"},
                "old_volume_units": {"type": "number"},
                "new_volume_units": {"type": "number"},
                "include_trade_dollars": {
                    "type": "boolean",
                    "description": "Net vendor trade funding against promotional cost. Default true.",
                    "default": True
                },
                "jurisdiction": {
                    "type": "string",
                    "description": "Tax jurisdiction (US-WA, US-GA, US-IL). Default: US.",
                    "default": "US"
                }
            },
            "required": ["sku", "old_price", "new_price", "old_volume_units", "new_volume_units"]
        }
    },
    {
        "name": "calculate_carrying_cost",
        "description": (
            "Calculate inventory carrying cost for excess units. "
            "Applies ONLY to owned inventory — not VMI (vendor-managed inventory). "
            "Returns VMI vs owned split so you know which portion incurs carrying cost."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "excess_units": {"type": "number"},
                "carrying_weeks": {"type": "number"},
                "annual_carrying_rate": {
                    "type": "number",
                    "description": "Annual carrying rate as decimal (default 0.25 = 25%).",
                    "default": 0.25
                }
            },
            "required": ["sku", "excess_units", "carrying_weeks"]
        }
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SCENARIO TOOLS
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "detect_scenario_conflicts",
        "description": (
            "Detect conflicts between two or more scenarios running simultaneously. "
            "Key conflict: promotion (demand spike) + supply disruption (supply constraint) = dangerous. "
            "Requires a time anchor (start_date) for each scenario — comparison is invalid without it. "
            "Returns conflict severity and recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "scenarios": {
                    "type": "array",
                    "description": "List of scenarios to check for conflicts.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "scenario_type": {
                                "type": "string",
                                "description": "price_change | promotion | supply_disruption | tariff | markdown"
                            },
                            "start_date": {"type": "string", "description": "ISO date (YYYY-MM-DD)"},
                            "horizon_days": {"type": "integer"},
                            "demand_direction": {
                                "type": "string",
                                "enum": ["increase", "decrease", "neutral"],
                                "description": "Direction of demand impact for this scenario"
                            },
                            "supply_direction": {
                                "type": "string",
                                "enum": ["constrain", "expand", "neutral"],
                                "description": "Direction of supply impact for this scenario"
                            }
                        },
                        "required": ["name", "scenario_type", "start_date", "horizon_days"]
                    }
                }
            },
            "required": ["scenarios"]
        }
    },
    {
        "name": "run_scenario_comparison",
        "description": "Compare multiple scenarios side-by-side on demand, revenue, and inventory metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "horizon_weeks": {"type": "integer", "default": 8},
                "scenarios": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "price": {"type": "number"},
                            "promo_uplift_pct": {"type": "number", "default": 0},
                            "supply_reduction_pct": {"type": "number", "default": 0},
                            "tariff_additional_cost_pct": {"type": "number", "default": 0}
                        },
                        "required": ["name"]
                    }
                }
            },
            "required": ["sku", "scenarios"]
        }
    },
]
