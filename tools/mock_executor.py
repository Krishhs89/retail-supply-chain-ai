"""
Mock tool executor — simulates all retail system responses.

Design principles:
• Every execute() call is wrapped in try/except; failures return a structured error dict
  (never empty data that silently becomes zeros downstream).
• All returns include provenance (OLAP/OLTP/WMS) and freshness_minutes.
• Asymmetric elasticity is applied for diaper, formula, tobacco, alcohol product classes.
• Regional carrier availability is checked separately from national availability.
• Lead time variability is modelled as a distribution, not a point estimate.
• OLAP vs WMS inventory discrepancies are flagged explicitly.
"""

from __future__ import annotations

import json
import math
import random
import time
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── TTL Cache ────────────────────────────────────────────────────────────────
# Avoids redundant re-computation for identical tool calls within the same
# agent loop. TTL mirrors the real data-source refresh cadences.

_CACHE_TTL_SECONDS: Dict[str, int] = {
    "OLTP": 300,    # 5 min  — POS / transactional
    "WMS":  900,    # 15 min — warehouse management
    "OLAP": 86400,  # 24 hr  — analytics data warehouse
}
_cache: Dict[str, tuple] = {}   # key → (result_dict, timestamp_float)


def _cache_get(key: str) -> Optional[Dict]:
    """Return cached result if still within TTL, else None."""
    entry = _cache.get(key)
    if not entry:
        return None
    result, ts = entry
    ttl = _CACHE_TTL_SECONDS.get(result.get("provenance", "OLTP"), 300)
    if time.time() - ts < ttl:
        return result
    del _cache[key]
    return None


def _cache_set(key: str, result: Dict) -> None:
    _cache[key] = (result, time.time())

# ─── Static Mock Data ─────────────────────────────────────────────────────────

PRODUCTS: Dict[str, Dict] = {
    "HUG48-3": {
        "name": "Huggies Size 3 Diapers 48-count",
        "category": "diaper",
        "product_class": "diaper",
        "base_price": 12.99,
        "base_demand_per_store_week": 50,
        "elasticity": -1.4,
        "asymmetric": True,       # price ↑ drops demand more than price ↓ recovers it
        "recovery_factor": 0.70,  # price decrease recovers only 70% of lost demand
        "margin_pct": 0.22,
        "vendor_trade_pct": 0.08, # vendor funds 8% of retail as trade spend
        "vmi_pct": 0.00,          # 100% owned inventory
        "freight_term": "FOB",
        "perishable": False,
        "shelf_facings_max": 48,
        "safety_stock_days": 7,
        "open_pos": [
            {"po_id": "PO-2025-0341", "units": 5000, "eta_days": 45, "status": "open"},
            {"po_id": "PO-2025-0389", "units": 3000, "eta_days": 75, "status": "open"},
        ]
    },
    "PAM72-5": {
        "name": "Pampers Size 5 Diapers 72-count",
        "category": "diaper",
        "product_class": "diaper",
        "base_price": 18.99,
        "base_demand_per_store_week": 35,
        "elasticity": -1.35,
        "asymmetric": True,
        "recovery_factor": 0.70,
        "margin_pct": 0.20,
        "vendor_trade_pct": 0.09,
        "vmi_pct": 0.00,
        "freight_term": "FOB",
        "perishable": False,
        "shelf_facings_max": 36,
        "safety_stock_days": 7,
        "open_pos": [
            {"po_id": "PO-2025-0401", "units": 2500, "eta_days": 50, "status": "open"},
        ]
    },
    "MLK-GAL": {
        "name": "Whole Milk 1 Gallon",
        "category": "dairy",
        "product_class": "dairy",
        "base_price": 4.29,
        "base_demand_per_store_week": 200,
        "elasticity": -0.6,
        "asymmetric": False,
        "recovery_factor": 1.0,
        "margin_pct": 0.12,
        "vendor_trade_pct": 0.02,
        "vmi_pct": 0.30,          # 30% vendor-managed
        "freight_term": "DDP",
        "perishable": True,
        "shelf_life_days": 14,
        "max_dos": 3,
        "shelf_facings_max": 60,
        "safety_stock_days": 2,
        "open_pos": []
    },
    "TAB-DIN": {
        "name": "Dinnerware Set 16-piece",
        "category": "tableware",
        "product_class": "tableware",
        "base_price": 29.99,
        "base_demand_per_store_week": 8,
        "elasticity": -0.9,
        "asymmetric": False,
        "recovery_factor": 1.0,
        "margin_pct": 0.38,
        "vendor_trade_pct": 0.05,
        "vmi_pct": 0.10,
        "freight_term": "FOB",
        "perishable": False,
        "shelf_facings_max": 12,
        "safety_stock_days": 14,
        "open_pos": [
            {"po_id": "PO-2025-0412", "units": 800, "eta_days": 30, "status": "open"},
        ]
    },
    "BLK-THR": {
        "name": "Fleece Throw Blanket 50x60",
        "category": "linen",
        "product_class": "linen",
        "base_price": 14.99,
        "base_demand_per_store_week": 12,
        "elasticity": -1.1,
        "asymmetric": False,
        "recovery_factor": 1.0,
        "margin_pct": 0.42,
        "vendor_trade_pct": 0.04,
        "vmi_pct": 0.00,
        "freight_term": "FOB",
        "perishable": False,
        "shelf_facings_max": 20,
        "safety_stock_days": 14,
        "open_pos": []
    },
    "CIG-PKT": {
        "name": "Cigarettes Premium Pack",
        "category": "tobacco",
        "product_class": "tobacco",
        "base_price": 8.99,
        "base_demand_per_store_week": 45,
        "elasticity": -0.5,
        "asymmetric": True,
        "recovery_factor": 0.40,  # very sticky — demand barely recovers on price drop
        "margin_pct": 0.08,
        "vendor_trade_pct": 0.00,
        "vmi_pct": 0.00,
        "freight_term": "FOB",
        "perishable": False,
        "shelf_facings_max": 30,
        "safety_stock_days": 5,
        "open_pos": []
    },
}

CARRIERS: Dict[str, Dict] = {
    "TruckCo_A": {
        "name": "TruckCo Alpha Logistics",
        "status": "active",
        "cargo_types": ["tableware", "linen", "general"],
        "regions_served": ["NW", "MW"],
        "normal_lead_time_days": 3,
        "capacity_units_per_week": 15000,
        "rate_per_mile": 2.10,
        "can_handle_diapers": True,   # not primary, but can handle
        "diaper_capacity_pct": 60,    # only 60% of normal capacity for diapers
        "diaper_cost_premium_pct": 25,
    },
    "TruckCo_B": {
        "name": "TruckCo Beta Distribution",
        "status": "strike",           # ← ON STRIKE
        "strike_start_date": "2026-04-10",
        "expected_resolution_date": "2026-04-24",
        "cargo_types": ["diaper", "formula"],
        "regions_served": ["SE", "MW"],
        "normal_lead_time_days": 2,
        "capacity_units_per_week": 20000,
        "rate_per_mile": 1.85,
        "can_handle_diapers": True,
        "diaper_capacity_pct": 100,
    },
    "TruckCo_C": {
        "name": "TruckCo Chill Express",
        "status": "active",
        "cargo_types": ["dairy", "produce", "linen"],
        "regions_served": ["NW", "SE"],
        "normal_lead_time_days": 2,
        "capacity_units_per_week": 12000,
        "rate_per_mile": 2.40,
        "can_handle_diapers": False,  # refrigerated trucks, not configured for diapers
        "diaper_capacity_pct": 0,
    },
    "TruckCo_D": {
        "name": "TruckCo Delta Freight",
        "status": "active",
        "cargo_types": ["general", "mixed"],
        "regions_served": ["NW", "SE", "MW"],  # national
        "normal_lead_time_days": 4,
        "capacity_units_per_week": 25000,
        "rate_per_mile": 2.65,
        "can_handle_diapers": True,
        "diaper_capacity_pct": 100,
        "diaper_cost_premium_pct": 45,  # higher cost as a spot carrier
    },
}

DCS: Dict[str, Dict] = {
    "DC-NW": {
        "name": "Northwest DC — Seattle, WA",
        "region": "NW",
        "stores_served": ["STR-001", "STR-002", "STR-003", "STR-004", "STR-005",
                          "STR-006", "STR-007", "STR-008", "STR-009", "STR-010"],
        "inventory": {
            "HUG48-3": {"wms": 8500, "olap": 8620},   # OLAP is 24h old
            "PAM72-5": {"wms": 5200, "olap": 5180},
            "MLK-GAL": {"wms": 1800, "olap": 1900},
            "TAB-DIN": {"wms": 620,  "olap": 640},
            "BLK-THR": {"wms": 890,  "olap": 910},
            "CIG-PKT": {"wms": 3400, "olap": 3390},
        }
    },
    "DC-SE": {
        "name": "Southeast DC — Atlanta, GA",
        "region": "SE",
        "stores_served": ["STR-011", "STR-012", "STR-013", "STR-014", "STR-015",
                          "STR-016", "STR-017", "STR-018", "STR-019", "STR-020"],
        "inventory": {
            "HUG48-3": {"wms": 3200, "olap": 3380},   # WMS lower — sold through since batch
            "PAM72-5": {"wms": 2100, "olap": 2200},
            "MLK-GAL": {"wms": 950,  "olap": 980},
            "TAB-DIN": {"wms": 310,  "olap": 320},
            "BLK-THR": {"wms": 430,  "olap": 440},
            "CIG-PKT": {"wms": 2800, "olap": 2820},
        }
    },
    "DC-MW": {
        "name": "Midwest DC — Chicago, IL",
        "region": "MW",
        "stores_served": ["STR-021", "STR-022", "STR-023", "STR-024", "STR-025",
                          "STR-026", "STR-027", "STR-028", "STR-029", "STR-030"],
        "inventory": {
            "HUG48-3": {"wms": 6100, "olap": 6050},
            "PAM72-5": {"wms": 3800, "olap": 3770},
            "MLK-GAL": {"wms": 1400, "olap": 1420},
            "TAB-DIN": {"wms": 490,  "olap": 500},
            "BLK-THR": {"wms": 670,  "olap": 680},
            "CIG-PKT": {"wms": 3100, "olap": 3090},
        }
    },
}

STORES: Dict[str, Dict] = {
    f"STR-{i:03d}": {
        "dc": "DC-NW" if i <= 10 else ("DC-SE" if i <= 20 else "DC-MW"),
        "region": "NW" if i <= 10 else ("SE" if i <= 20 else "MW"),
        "volume_tier": "high" if i in [1,2,11,12,21,22] else ("medium" if i in [3,4,5,13,14,15,23,24,25] else "low"),
        "inventory": {
            "HUG48-3": max(5, int(50 * (0.8 if i <= 10 else 0.6 if i <= 20 else 0.7) + random.randint(-5, 5))),
            "PAM72-5": max(3, int(35 * 0.7 + random.randint(-3, 3))),
            "MLK-GAL": max(10, int(200 * 0.1 + random.randint(-5, 5))),
            "TAB-DIN": max(1, int(8 * 1.5 + random.randint(-2, 2))),
            "BLK-THR": max(2, int(12 * 2 + random.randint(-3, 3))),
            "CIG-PKT": max(5, int(45 * 0.5 + random.randint(-5, 5))),
        },
        "shelf_capacity": {
            "HUG48-3": 48, "PAM72-5": 36, "MLK-GAL": 60,
            "TAB-DIN": 12, "BLK-THR": 20, "CIG-PKT": 30,
        }
    }
    for i in range(1, 31)
}

DEMAND_VARIABLE_COEFFICIENTS: Dict[str, Dict[str, float]] = {
    "HUG48-3": {
        "price": -1.40, "promo": 0.80, "markdown": 0.60, "trade_spend": 0.30,
        "competitor_price": 0.50, "weather": 0.00, "seasonality": 0.10,
        "tariff": -0.30, "supply_availability": 0.90, "store_count": 0.40,
        "customer_income_index": 0.20, "trend": 0.10, "social_media_buzz": 0.05,
        "regulatory_change": -0.10, "external_event": 0.15,
    },
    "PAM72-5": {
        "price": -1.35, "promo": 0.75, "markdown": 0.55, "trade_spend": 0.28,
        "competitor_price": 0.45, "weather": 0.00, "seasonality": 0.10,
        "tariff": -0.28, "supply_availability": 0.85, "store_count": 0.35,
        "customer_income_index": 0.25, "trend": 0.08, "social_media_buzz": 0.04,
        "regulatory_change": -0.08, "external_event": 0.12,
    },
    "MLK-GAL": {
        "price": -0.60, "promo": 0.55, "markdown": 0.45, "trade_spend": 0.15,
        "competitor_price": 0.30, "weather": 0.10, "seasonality": 0.25,
        "tariff": -0.05, "supply_availability": 0.70, "store_count": 0.30,
        "customer_income_index": 0.10, "trend": 0.05, "social_media_buzz": 0.02,
        "regulatory_change": -0.15, "external_event": 0.08,
    },
    "TAB-DIN": {
        "price": -0.90, "promo": 1.20, "markdown": 0.90, "trade_spend": 0.40,
        "competitor_price": 0.55, "weather": 0.05, "seasonality": 0.60,
        "tariff": -0.50, "supply_availability": 0.60, "store_count": 0.50,
        "customer_income_index": 0.45, "trend": 0.20, "social_media_buzz": 0.10,
        "regulatory_change": -0.05, "external_event": 0.25,
    },
}

FORECAST_ACCURACY: Dict[str, Dict] = {
    "HUG48-3": {4: 0.87, 8: 0.78, 12: 0.68, 16: 0.60, 20: 0.54, 32: 0.48},
    "PAM72-5": {4: 0.85, 8: 0.76, 12: 0.65, 16: 0.57, 20: 0.50, 32: 0.44},
    "MLK-GAL": {4: 0.91, 8: 0.84, 12: 0.74, 16: 0.65, 20: 0.58, 32: 0.50},
    "TAB-DIN": {4: 0.72, 8: 0.62, 12: 0.50, 16: 0.44, 20: 0.38, 32: 0.32},
    "BLK-THR": {4: 0.75, 8: 0.65, 12: 0.54, 16: 0.46, 20: 0.40, 32: 0.34},
    "CIG-PKT": {4: 0.93, 8: 0.88, 12: 0.80, 16: 0.72, 20: 0.65, 32: 0.58},
}

COMPETITIVE_PRICING: Dict[str, Dict[str, float]] = {
    "HUG48-3": {"Target": 13.49, "Costco": 11.89, "Amazon": 12.49, "Kroger": 13.19},
    "PAM72-5": {"Target": 19.49, "Costco": 17.49, "Amazon": 18.49, "Kroger": 19.99},
    "MLK-GAL": {"Target": 4.39,  "Costco": 3.99,  "Amazon": 4.59,  "Kroger": 4.19},
    "TAB-DIN": {"Target": 31.99, "Costco": 27.99, "Amazon": 29.49, "Kroger": None},
    "BLK-THR": {"Target": 15.99, "Costco": 13.99, "Amazon": 14.49, "Kroger": None},
    "CIG-PKT": {"Target": 9.29,  "Costco": None,  "Amazon": None,  "Kroger": 8.79},
}

REPLENISHMENT_COSTS = {
    "standard":  {"lead_time_days": 4, "cost_multiplier": 1.0},
    "expedited": {"lead_time_days": 2, "cost_multiplier": 2.0},
    "emergency": {"lead_time_days": 1, "cost_multiplier": 3.0},
}

TAX_RATES = {
    "US":    {"effective_rate": 0.25, "note": "Blended federal + state. Approximate."},
    "US-WA": {"effective_rate": 0.28, "note": "WA state rate applied. Approximate. No income tax — B&O tax modelled."},
    "US-GA": {"effective_rate": 0.24, "note": "GA state rate applied. Approximate."},
    "US-IL": {"effective_rate": 0.26, "note": "IL state rate applied. Approximate."},
}


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _get_product(sku: str) -> Dict:
    return PRODUCTS.get(sku, PRODUCTS["HUG48-3"])  # default to diaper if unknown


def _get_accuracy_for_horizon(sku: str, horizon_weeks: int) -> float:
    acc_map = FORECAST_ACCURACY.get(sku, {4: 0.70, 8: 0.62, 12: 0.55})
    keys = sorted(acc_map.keys())
    # Return the accuracy for the nearest horizon key that is >= horizon_weeks
    for k in keys:
        if horizon_weeks <= k:
            return acc_map[k]
    return acc_map[keys[-1]]  # worst case


def _ci_bounds(point: float, accuracy: float, horizon_period: int) -> Dict:
    """Compute 80% and 95% CI. Widens 15% per 4-week period and inversely with accuracy."""
    base_error = (1.0 - accuracy) * point
    widening = 1.0 + (0.15 * horizon_period)
    ci_80 = base_error * widening * 1.28   # z=1.28 for 80% CI
    ci_95 = base_error * widening * 1.96   # z=1.96 for 95% CI
    return {
        "ci_80_lower": round(max(0, point - ci_80), 1),
        "ci_80_upper": round(point + ci_80, 1),
        "ci_95_lower": round(max(0, point - ci_95), 1),
        "ci_95_upper": round(point + ci_95, 1),
    }


def _carrying_cost_per_unit_per_week(sku: str, annual_rate: float = 0.25) -> float:
    """Weekly carrying cost per unit = (price × annual_rate) / 52."""
    price = _get_product(sku)["base_price"]
    return (price * annual_rate) / 52


def _result(data: Any, provenance: str = "OLTP", freshness: int = 5,
            stale: bool = False) -> Dict:
    return {
        "data": data,
        "error": None,
        "provenance": provenance,
        "freshness_minutes": freshness,
        "is_stale": stale,
    }


def _error_result(tool_name: str, exc: Exception) -> Dict:
    return {
        "data": {},
        "error": f"Tool '{tool_name}' failed: {type(exc).__name__}: {exc}. "
                 "Retry recommended. Do NOT pass empty data downstream as zeros.",
        "provenance": "OLTP",
        "freshness_minutes": 0,
        "is_stale": True,
    }


# ─── Tool Implementations ─────────────────────────────────────────────────────

def _simulate_price_change(sku: str, old_price: float, new_price: float,
                            horizon_weeks: int = 8,
                            affected_dc_ids: Optional[List[str]] = None) -> Dict:
    prod = _get_product(sku)
    base_demand = prod["base_demand_per_store_week"]
    elasticity = prod["elasticity"]
    asymmetric = prod["asymmetric"]
    recovery_factor = prod.get("recovery_factor", 1.0)
    margin_pct = prod["margin_pct"]
    vendor_trade_pct = prod["vendor_trade_pct"]
    vmi_pct = prod.get("vmi_pct", 0.0)
    total_stores = 30

    price_change_pct = (new_price - old_price) / old_price * 100
    is_increase = price_change_pct > 0

    # Asymmetric elasticity: increases suppress demand more than decreases recover it
    if asymmetric and not is_increase:
        effective_elasticity = elasticity * recovery_factor
    else:
        effective_elasticity = elasticity

    demand_change_pct = effective_elasticity * (price_change_pct / 100) * 100
    new_demand = base_demand * (1 + demand_change_pct / 100)
    weekly_demand_delta = new_demand - base_demand

    # Inventory impact over horizon
    total_excess_units = abs(weekly_demand_delta) * horizon_weeks * total_stores if is_increase else 0
    carrying_cost = (_carrying_cost_per_unit_per_week(sku) * total_excess_units *
                     (horizon_weeks / 2))  # average half-horizon exposure

    # PO adjustments
    po_adjustments = []
    for po in prod.get("open_pos", []):
        demand_fraction = po["eta_days"] / (horizon_weeks * 7)
        adjustment = int(weekly_demand_delta * total_stores * demand_fraction)
        po_adjustments.append({
            "po_id": po["po_id"],
            "current_units": po["units"],
            "recommended_adjustment_units": adjustment,
            "eta_days": po["eta_days"],
            "adjustable": po["eta_days"] > 14,  # can only adjust POs > 14 days out
        })

    # Financial impact
    old_weekly_revenue = old_price * base_demand * total_stores
    new_weekly_revenue = new_price * new_demand * total_stores
    weekly_revenue_change = new_weekly_revenue - old_weekly_revenue
    total_revenue_change = weekly_revenue_change * horizon_weeks

    # Vendor trade dollars (apply only for price decrease with promo)
    trade_offset = 0
    if not is_increase and vendor_trade_pct > 0:
        promo_cost = abs(price_change_pct / 100) * old_price * new_demand * total_stores * horizon_weeks
        trade_offset = min(promo_cost, vendor_trade_pct * old_price * new_demand * total_stores * horizon_weeks)

    net_revenue_change = total_revenue_change + trade_offset
    margin_change = net_revenue_change * margin_pct

    dcs = list(DCS.keys()) if not affected_dc_ids else affected_dc_ids
    affected_nodes = ["HQ"] + dcs + [f"STR-001 to STR-030 ({total_stores} stores)"]

    recs = []
    for adj in po_adjustments:
        if abs(adj["recommended_adjustment_units"]) > 50 and adj["adjustable"]:
            direction = "Reduce" if adj["recommended_adjustment_units"] < 0 else "Increase"
            recs.append(f"{direction} PO {adj['po_id']} by "
                        f"{abs(adj['recommended_adjustment_units'])} units (ETA {adj['eta_days']}d)")
    if is_increase and total_excess_units > 500:
        recs.append(f"Rebalance DC-SE allocation downward — {int(total_excess_units/3)} excess units projected")
    if not is_increase and asymmetric:
        recs.append("Note asymmetric elasticity: demand recovery on price reversal will be slower "
                    f"({int(recovery_factor*100)}% of suppression rate)")
    recs.append("Notify Finance to update Q3 revenue model with revised price and volume assumptions")
    if abs(weekly_demand_delta) * total_stores > 200:
        recs.append("Alert Demand Planning team — weekly demand signal shifts by "
                    f"{abs(int(weekly_demand_delta * total_stores))} units across network")

    return _result({
        "sku": sku,
        "sku_name": prod["name"],
        "old_price": old_price,
        "new_price": new_price,
        "price_change_pct": round(price_change_pct, 2),
        "price_direction": "increase" if is_increase else "decrease",
        "elasticity_applied": round(effective_elasticity, 2),
        "asymmetric_elasticity_note": (
            f"Asymmetric: recovery factor {recovery_factor} applied (demand only recovers "
            f"{int(recovery_factor*100)}% as fast on price decrease)"
        ) if asymmetric and not is_increase else "N/A",
        "demand_impact": {
            "demand_change_pct": round(demand_change_pct, 2),
            "old_weekly_demand_per_store": base_demand,
            "new_weekly_demand_per_store": round(new_demand, 1),
            "total_stores": total_stores,
            "old_total_weekly": base_demand * total_stores,
            "new_total_weekly": round(new_demand * total_stores, 0),
            "weekly_unit_delta": round(weekly_demand_delta * total_stores, 0),
        },
        "inventory_impact": {
            "excess_units_over_horizon": round(total_excess_units, 0) if is_increase else 0,
            "shortage_units_over_horizon": round(abs(total_excess_units), 0) if not is_increase else 0,
            "carrying_cost_increase_usd": round(carrying_cost, 2),
            "vmi_pct": vmi_pct,
            "owned_inventory_pct": 1 - vmi_pct,
            "carrying_cost_note": f"Applied only to {int((1-vmi_pct)*100)}% owned inventory",
        },
        "po_adjustments": po_adjustments,
        "financial_impact": {
            "gross_revenue_change_usd": round(total_revenue_change, 2),
            "vendor_trade_offset_usd": round(trade_offset, 2),
            "net_revenue_change_usd": round(net_revenue_change, 2),
            "margin_change_usd": round(margin_change, 2),
            "carrying_cost_increase_usd": round(carrying_cost, 2),
            "combined_bottom_line_impact_usd": round(margin_change - carrying_cost, 2),
        },
        "affected_nodes": affected_nodes,
        "recommendations": recs,
    })


def _get_competitive_pricing(sku: str, competitors: Optional[List[str]] = None) -> Dict:
    comp_data = COMPETITIVE_PRICING.get(sku, {})
    our_price = _get_product(sku)["base_price"]
    if competitors:
        comp_data = {k: v for k, v in comp_data.items() if k in competitors}
    available = {k: v for k, v in comp_data.items() if v is not None}
    cheaper = {k: v for k, v in available.items() if v < our_price}
    return _result({
        "sku": sku,
        "our_price": our_price,
        "competitor_prices": comp_data,
        "cheaper_competitors": cheaper,
        "price_position": (
            "below_market" if not cheaper
            else f"above {len(cheaper)} competitor(s): " + ", ".join(f"{k}=${v}" for k, v in cheaper.items())
        ),
    })


def _get_demand_forecast(sku: str, horizon_weeks: int,
                          variable_overrides: Optional[Dict] = None) -> Dict:
    prod = _get_product(sku)
    base = prod["base_demand_per_store_week"]
    accuracy = _get_accuracy_for_horizon(sku, horizon_weeks)
    is_reliable = accuracy >= 0.60
    coeffs = DEMAND_VARIABLE_COEFFICIENTS.get(sku, {})

    # Apply variable overrides
    variable_contributions: Dict[str, float] = {}
    adjusted_base = base
    overrides = variable_overrides or {}
    for var, coeff in coeffs.items():
        override_val = overrides.get(var, 0)
        if override_val != 0:
            contribution = coeff * override_val * base / 100
            variable_contributions[var] = round(contribution, 2)
            adjusted_base += contribution
        else:
            variable_contributions[var] = 0.0

    # Add some realistic seasonality
    import datetime
    month = datetime.date.today().month
    seasonal_boost = base * coeffs.get("seasonality", 0) * (0.5 if month in [11, 12] else
                                                              0.3 if month in [1, 2] else 0)
    adjusted_base += seasonal_boost
    if seasonal_boost:
        variable_contributions["seasonality"] = round(seasonal_boost, 2)

    weekly_points = []
    for w in range(1, horizon_weeks + 1):
        period = (w - 1) // 4  # 4-week period index
        point = round(adjusted_base * (1 + (w - 1) * 0.005), 1)  # slight upward trend
        bounds = _ci_bounds(point, accuracy, period)
        weekly_points.append({
            "week": w,
            "point_estimate": point,
            **bounds,
        })

    key_drivers = sorted(
        [(k, abs(v)) for k, v in variable_contributions.items() if v != 0],
        key=lambda x: x[1], reverse=True
    )[:3]

    acc_note = (
        f"Accuracy {round(accuracy * 100, 1)}% MAPE at {horizon_weeks}-week horizon. "
        + ("RELIABLE — suitable as planning input." if accuracy >= 0.70
           else "WARNING: below 70% threshold — use with caution." if accuracy >= 0.60
           else "UNRELIABLE — below 60% minimum. DO NOT pass to PO system as planning input.")
    )

    return _result(
        {
            "sku": sku,
            "sku_name": prod["name"],
            "horizon_weeks": horizon_weeks,
            "forecast_accuracy_mape": round(accuracy, 3),
            "is_reliable": is_reliable,
            "weekly_forecast": weekly_points,
            "variable_contributions_units_per_week": variable_contributions,
            "key_drivers": [f"{k[0]} ({'+' if variable_contributions[k[0]] >= 0 else ''}{variable_contributions[k[0]]})" for k in key_drivers if k[0] in variable_contributions],
            "accuracy_note": acc_note,
            "ci_note": "Confidence intervals widen ~15% per 4-week horizon period.",
        },
        provenance="OLAP",
        freshness=1440,
        stale=(accuracy < 0.70),
    )


def _calculate_price_elasticity(sku: str, price_change_pct: float,
                                  base_demand_units_per_week: Optional[float] = None) -> Dict:
    prod = _get_product(sku)
    base = base_demand_units_per_week or prod["base_demand_per_store_week"]
    elasticity = prod["elasticity"]
    asymmetric = prod["asymmetric"]
    recovery_factor = prod.get("recovery_factor", 1.0)
    is_increase = price_change_pct > 0

    if asymmetric and not is_increase:
        effective_elasticity = elasticity * recovery_factor
        note = (f"Asymmetric elasticity applied: {prod['product_class']} category. "
                f"Price decrease recovers demand at {int(recovery_factor*100)}% of suppression rate.")
    else:
        effective_elasticity = elasticity
        note = f"Symmetric elasticity applied."

    demand_change_pct = effective_elasticity * (price_change_pct / 100) * 100
    new_demand = base * (1 + demand_change_pct / 100)

    return _result({
        "sku": sku,
        "product_class": prod["product_class"],
        "base_elasticity_coefficient": elasticity,
        "effective_elasticity_coefficient": round(effective_elasticity, 2),
        "asymmetric": asymmetric,
        "price_change_pct": price_change_pct,
        "demand_change_pct": round(demand_change_pct, 2),
        "old_demand_units_per_store_week": base,
        "new_demand_units_per_store_week": round(new_demand, 1),
        "note": note,
    })


def _get_forecast_accuracy(sku: str, horizon_weeks: int = 8) -> Dict:
    if sku == "all":
        summary = {s: _get_accuracy_for_horizon(s, horizon_weeks) for s in FORECAST_ACCURACY}
        avg = sum(summary.values()) / len(summary)
        return _result({
            "category": "all",
            "horizon_weeks": horizon_weeks,
            "accuracy_by_sku": {k: round(v, 3) for k, v in summary.items()},
            "network_average": round(avg, 3),
            "benchmark": 0.85,
            "gap_to_benchmark": round(0.85 - avg, 3),
            "revenue_impact_estimate": (
                f"A {round((0.85 - avg) * 100, 1)}-point accuracy gap costs approximately "
                f"{round((0.85 - avg) * 100, 1)}% in revenue efficiency. "
                "At $100B retail scale, each 1% = ~$1B revenue impact."
            )
        }, provenance="OLAP", freshness=1440)

    accuracy = _get_accuracy_for_horizon(sku, horizon_weeks)
    return _result({
        "sku": sku,
        "horizon_weeks": horizon_weeks,
        "current_accuracy_mape": round(accuracy, 3),
        "benchmark": 0.85,
        "gap": round(0.85 - accuracy, 3),
        "status": ("above_benchmark" if accuracy >= 0.85
                   else "acceptable" if accuracy >= 0.70
                   else "warning" if accuracy >= 0.60
                   else "unreliable"),
    }, provenance="OLAP", freshness=1440)


def _get_inventory_levels(sku: str, location_type: str = "all",
                           location_id: Optional[str] = None) -> Dict:
    prod = _get_product(sku)
    daily_demand = prod["base_demand_per_store_week"] / 7
    results = []
    freshness_warnings = []

    # DCs
    if location_type in ("all", "dc"):
        for dc_id, dc in DCS.items():
            if location_id and dc_id != location_id:
                continue
            inv = dc["inventory"].get(sku, {"wms": 0, "olap": 0})
            wms_qty = inv["wms"]
            olap_qty = inv["olap"]
            dc_daily = daily_demand * len(dc["stores_served"])
            days_on_hand = wms_qty / dc_daily if dc_daily > 0 else 999
            discrepancy = abs(wms_qty - olap_qty)
            if discrepancy > 100:
                freshness_warnings.append(
                    f"WMS vs OLAP discrepancy at {dc_id} for {sku}: "
                    f"WMS={wms_qty}, OLAP={olap_qty} (diff={discrepancy}). "
                    "Use WMS for operational decisions — OLAP is 24h behind."
                )
            results.append({
                "location_type": "dc",
                "location_id": dc_id,
                "location_name": dc["name"],
                "region": dc["region"],
                "sku": sku,
                "quantity_on_hand_wms": wms_qty,
                "quantity_on_hand_olap": olap_qty,
                "daily_velocity_units": round(dc_daily, 1),
                "days_on_hand": round(days_on_hand, 1),
                "safety_stock_days": prod["safety_stock_days"],
                "at_or_below_safety_stock": days_on_hand <= prod["safety_stock_days"],
                "wms_source": "WMS (15-min lag)",
                "olap_source": "OLAP (24-hr lag)",
            })

    # Stores
    if location_type in ("all", "store"):
        sample_stores = list(STORES.keys())[:5]  # return 5 representative stores
        if location_id:
            sample_stores = [location_id] if location_id in STORES else []
        for store_id in sample_stores:
            store = STORES[store_id]
            qty = store["inventory"].get(sku, 0)
            store_dos = qty / daily_demand if daily_demand > 0 else 999
            results.append({
                "location_type": "store",
                "location_id": store_id,
                "region": store["region"],
                "dc": store["dc"],
                "volume_tier": store["volume_tier"],
                "sku": sku,
                "quantity_on_hand": qty,
                "days_on_hand": round(store_dos, 1),
                "safety_stock_days": prod["safety_stock_days"],
                "at_or_below_safety_stock": store_dos <= prod["safety_stock_days"],
                "source": "WMS (15-min lag)",
            })

    return _result({
        "sku": sku,
        "locations": results,
        "freshness_warnings": freshness_warnings,
        "note": "WMS data used for operational decisions. OLAP shown for reference only.",
    }, provenance="WMS", freshness=15)


def _check_shelf_capacity(sku: str, store_id: str, proposed_replenishment_units: int) -> Dict:
    prod = _get_product(sku)
    store = STORES.get(store_id, STORES["STR-001"])
    max_facings = store["shelf_capacity"].get(sku, prod["shelf_facings_max"])
    current_on_shelf = store["inventory"].get(sku, 0)
    space_available = max(0, max_facings - current_on_shelf)
    overflow = max(0, proposed_replenishment_units - space_available)
    fits = overflow == 0

    return _result({
        "sku": sku,
        "store_id": store_id,
        "shelf_capacity_units": max_facings,
        "current_on_shelf_units": current_on_shelf,
        "space_available_units": space_available,
        "proposed_replenishment_units": proposed_replenishment_units,
        "fits_on_shelf": fits,
        "overflow_to_back_storage_units": overflow,
        "recommendation": (
            f"OK — {proposed_replenishment_units} units fit within {max_facings}-unit planogram capacity."
            if fits
            else f"Planogram exceeded: {space_available} units can go to shelf, "
                 f"{overflow} units must go to back-of-store staging."
        ),
    })


def _calculate_stockout_risk(sku: str, location_id: str,
                              demand_adjustment_pct: float = 0,
                              supply_disruption_additional_days: int = 0) -> Dict:
    prod = _get_product(sku)
    base_demand_per_week = prod["base_demand_per_store_week"]
    adjusted_daily = base_demand_per_week / 7 * (1 + demand_adjustment_pct / 100)

    # Get inventory
    if location_id.startswith("DC"):
        dc = DCS.get(location_id, DCS["DC-SE"])
        qty = dc["inventory"].get(sku, {}).get("wms", 0)
        num_stores = len(dc["stores_served"])
        total_daily = adjusted_daily * num_stores
    else:
        store = STORES.get(location_id, STORES["STR-001"])
        qty = store["inventory"].get(sku, 0)
        total_daily = adjusted_daily

    dos = qty / total_daily if total_daily > 0 else 999

    # Replenishment lag + variability
    normal_lag = 4  # HQ→DC→Store
    # Add lead time variability: 30% chance of +3 days
    p30_extra = 3 * 0.30
    effective_lag = normal_lag + supply_disruption_additional_days + p30_extra

    days_to_stockout = dos - effective_lag

    if days_to_stockout < 0:
        severity = "critical"
        rec = ("IMMEDIATE ACTION: Shelf hits zero before next replenishment arrives. "
               "Trigger emergency inter-DC transfer or store-to-store transfer NOW.")
        emergency_transfer = True
    elif days_to_stockout < 2:
        severity = "critical"
        rec = "Trigger emergency replenishment immediately. < 2 days buffer above lag."
        emergency_transfer = True
    elif days_to_stockout < 5:
        severity = "warning"
        rec = "Trigger expedited replenishment within 24 hours."
        emergency_transfer = False
    else:
        severity = "ok"
        rec = "Within safety stock parameters. Standard replenishment sufficient."
        emergency_transfer = False

    return _result({
        "sku": sku,
        "location_id": location_id,
        "current_inventory_units": qty,
        "adjusted_daily_demand": round(total_daily, 1),
        "days_on_hand": round(dos, 1),
        "replenishment_lag_days": normal_lag,
        "lead_time_variability_buffer_days": round(p30_extra, 1),
        "supply_disruption_extra_days": supply_disruption_additional_days,
        "effective_lag_days": round(effective_lag, 1),
        "days_above_lag": round(days_to_stockout, 1),
        "severity": severity,
        "emergency_transfer_recommended": emergency_transfer,
        "recommendation": rec,
        "demand_adjustment_applied_pct": demand_adjustment_pct,
    })


def _check_perishable_status(sku: str, location_id: str,
                              proposed_replenishment_units: int) -> Dict:
    prod = _get_product(sku)
    if not prod.get("perishable"):
        return _result({
            "sku": sku,
            "perishable": False,
            "note": f"{sku} ({prod['name']}) is not a perishable category. No expiration risk."
        })

    shelf_life = prod.get("shelf_life_days", 14)
    max_dos = prod.get("max_dos", 3)
    daily_demand = prod["base_demand_per_store_week"] / 7

    current_inventory = STORES.get(location_id, STORES["STR-001"])["inventory"].get(sku, 0)
    total_after_replenishment = current_inventory + proposed_replenishment_units
    projected_dos = total_after_replenishment / daily_demand if daily_demand > 0 else 999

    exceeds_cap = projected_dos > max_dos
    at_risk_units = max(0, int((projected_dos - max_dos) * daily_demand)) if exceeds_cap else 0
    write_off_risk_usd = at_risk_units * prod["base_price"]

    return _result({
        "sku": sku,
        "product_class": prod["product_class"],
        "perishable": True,
        "shelf_life_days": shelf_life,
        "max_dos_cap": max_dos,
        "current_inventory_units": current_inventory,
        "proposed_replenishment_units": proposed_replenishment_units,
        "projected_total_units": total_after_replenishment,
        "projected_days_of_supply": round(projected_dos, 1),
        "exceeds_perishable_cap": exceeds_cap,
        "at_risk_units": at_risk_units,
        "write_off_risk_usd": round(write_off_risk_usd, 2),
        "recommendation": (
            f"REDUCE replenishment by {at_risk_units} units to stay within {max_dos}-day perishable cap. "
            f"Sending full order risks ${round(write_off_risk_usd, 2)} write-off."
            if exceeds_cap
            else f"OK — {round(projected_dos, 1)} days-of-supply within {max_dos}-day perishable cap."
        ),
    })


def _trigger_replenishment(sku: str, from_location: str, to_location: str,
                            quantity: int, priority: str = "standard") -> Dict:
    prod = _get_product(sku)
    config = REPLENISHMENT_COSTS.get(priority, REPLENISHMENT_COSTS["standard"])
    base_freight_per_unit = 0.15
    freight_cost = base_freight_per_unit * quantity * config["cost_multiplier"]
    eta_days = config["lead_time_days"]

    # Check shelf capacity
    if to_location.startswith("STR"):
        cap = _check_shelf_capacity(sku, to_location, quantity)["data"]
        shelf_note = cap.get("recommendation", "")
        actual_to_shelf = min(quantity, cap.get("space_available_units", quantity))
        to_back_storage = quantity - actual_to_shelf
    else:
        shelf_note = "DC transfer — no shelf capacity constraint."
        actual_to_shelf = quantity
        to_back_storage = 0

    return _result({
        "sku": sku,
        "from_location": from_location,
        "to_location": to_location,
        "ordered_quantity": quantity,
        "to_shelf_units": actual_to_shelf,
        "to_back_storage_units": to_back_storage,
        "priority": priority,
        "eta_days": eta_days,
        "freight_cost_usd": round(freight_cost, 2),
        "cost_per_unit_usd": round(freight_cost / quantity, 3) if quantity > 0 else 0,
        "shelf_note": shelf_note,
        "confirmation": f"Replenishment order created: {quantity} units {from_location}→{to_location}, "
                        f"ETA {eta_days} day(s), cost ${round(freight_cost, 2)}.",
    })


def _get_carrier_status(carrier_id: str = "all", region: str = "all") -> Dict:
    carriers = CARRIERS if carrier_id == "all" else {
        k: v for k, v in CARRIERS.items() if k == carrier_id
    }
    if region != "all":
        carriers = {k: v for k, v in carriers.items() if region in v.get("regions_served", [])}

    result = []
    for cid, c in carriers.items():
        result.append({
            "carrier_id": cid,
            "name": c["name"],
            "status": c["status"],
            "cargo_types": c["cargo_types"],
            "regions_served": c["regions_served"],
            "lead_time_days": c["normal_lead_time_days"],
            "on_strike": c["status"] == "strike",
            "strike_detail": {
                "start": c.get("strike_start_date"),
                "expected_resolution": c.get("expected_resolution_date"),
            } if c["status"] == "strike" else None,
        })

    on_strike = [r for r in result if r["on_strike"]]
    return _result({
        "carriers": result,
        "total": len(result),
        "on_strike": [r["carrier_id"] for r in on_strike],
        "active": [r["carrier_id"] for r in result if not r["on_strike"]],
        "alert": (
            f"WARNING: {len(on_strike)} carrier(s) on strike: "
            + ", ".join(r["carrier_id"] for r in on_strike)
            if on_strike else "All carriers operational."
        ),
    })


def _find_alternate_carriers(sku: str, disrupted_carrier_id: str,
                              region: str, duration_days: int = 14) -> Dict:
    prod = _get_product(sku)
    product_class = prod["product_class"]
    disrupted = CARRIERS.get(disrupted_carrier_id, {})
    alternates = []
    no_viable = True

    for cid, c in CARRIERS.items():
        if cid == disrupted_carrier_id:
            continue
        if c["status"] == "strike":
            continue

        in_region = region in c.get("regions_served", []) or "all" in c.get("regions_served", [])
        can_carry = c.get("can_handle_diapers", False) if product_class in ("diaper", "formula") else (
            product_class in c.get("cargo_types", []) or "general" in c.get("cargo_types", [])
            or "mixed" in c.get("cargo_types", [])
        )

        available = in_region and can_carry
        reason = None
        if not in_region:
            reason = f"Does not serve {region} region"
        elif not can_carry:
            reason = f"Cannot handle {product_class} cargo"

        additional_lead_time = max(0, c["normal_lead_time_days"] - disrupted.get("normal_lead_time_days", 2))
        capacity_pct = c.get("diaper_capacity_pct", 100) if product_class in ("diaper", "formula") else 100
        cost_premium = c.get("diaper_cost_premium_pct", 0) if product_class in ("diaper", "formula") else 0

        if available:
            no_viable = False

        alternates.append({
            "carrier_id": cid,
            "name": c["name"],
            "available": available,
            "reason_unavailable": reason,
            "additional_lead_time_days": additional_lead_time if available else None,
            "capacity_pct": capacity_pct if available else 0,
            "cost_premium_pct": cost_premium if available else None,
            "in_region": in_region,
            "can_carry_product_class": can_carry,
        })

    viable = [a for a in alternates if a["available"]]
    regional_gap_warning = None
    if no_viable:
        regional_gap_warning = (
            f"CRITICAL: No viable alternate carriers for {product_class} in {region} region. "
            f"All capable carriers are either on strike or outside {region}. "
            "Escalate to supply chain leadership immediately."
        )
    elif len(viable) == 1:
        regional_gap_warning = (
            f"WARNING: Only 1 viable alternate ({viable[0]['carrier_id']}) in {region} region "
            f"at {viable[0]['cost_premium_pct']}% cost premium and +{viable[0]['additional_lead_time_days']}d lead time. "
            "Single point of failure — explore inter-DC transfers as backup."
        )

    return _result({
        "sku": sku,
        "product_class": product_class,
        "disrupted_carrier": disrupted_carrier_id,
        "region": region,
        "duration_days": duration_days,
        "alternates": alternates,
        "viable_alternates": viable,
        "no_viable_alternates": no_viable,
        "regional_gap_warning": regional_gap_warning,
        "recommendation": (
            f"Use {viable[0]['carrier_id']} at {viable[0]['cost_premium_pct']}% premium, "
            f"+{viable[0]['additional_lead_time_days']}d lead time."
            if viable else "No alternate available in region. Escalate immediately."
        ),
    })


def _get_supply_disruption_impact(disruption_type: str, affected_entity: str,
                                   duration_days: int, affected_skus: Optional[List[str]] = None) -> Dict:
    # Duration model by disruption type
    duration_notes = {
        "carrier_strike":     "Carrier strikes typically resolve in 2-3 weeks (union negotiations).",
        "port_delay":         "Port delays (rerouting, congestion) typically last 6-10 weeks.",
        "supplier_bankruptcy":"Permanent until new supplier onboarded (typically 8-16 weeks for qualification).",
    }

    # Auto-detect affected SKUs from carrier cargo
    if not affected_skus:
        carrier = CARRIERS.get(affected_entity, {})
        cargo_types = carrier.get("cargo_types", [])
        affected_skus = [
            sku for sku, p in PRODUCTS.items()
            if p["product_class"] in cargo_types or p["category"] in cargo_types
        ]

    stockout_risks = []
    total_revenue_at_risk = 0

    for sku in affected_skus:
        prod = _get_product(sku)
        daily_demand = prod["base_demand_per_store_week"] / 7

        for dc_id, dc in DCS.items():
            region = dc["region"]
            carrier = CARRIERS.get(affected_entity, {})
            if region not in carrier.get("regions_served", []):
                continue

            dc_inv = dc["inventory"].get(sku, {}).get("wms", 0)
            dc_stores = len(dc["stores_served"])
            dc_daily = daily_demand * dc_stores
            dc_dos = dc_inv / dc_daily if dc_daily > 0 else 999

            for store_id in list(dc["stores_served"])[:3]:  # sample 3 stores per DC
                store = STORES.get(store_id, STORES["STR-001"])
                store_inv = store["inventory"].get(sku, 0)
                store_dos = store_inv / daily_demand if daily_demand > 0 else 999

                effective_disruption = duration_days if disruption_type != "supplier_bankruptcy" else 120
                lag_with_disruption = 4 + effective_disruption
                days_to_stockout = store_dos - lag_with_disruption

                if days_to_stockout < 0:
                    severity = "critical"
                elif days_to_stockout < 5:
                    severity = "warning"
                else:
                    severity = "ok"

                if severity in ("critical", "warning"):
                    weeks_empty = min(effective_disruption / 7,
                                      (abs(days_to_stockout) if days_to_stockout < 0 else 0) / 7)
                    rev_at_risk = (prod["base_price"] * daily_demand * weeks_empty * 7
                                   * (3 if store["volume_tier"] == "high" else
                                      1.5 if store["volume_tier"] == "medium" else 1.0))
                    total_revenue_at_risk += rev_at_risk

                    stockout_risks.append({
                        "sku": sku,
                        "dc_id": dc_id,
                        "store_id": store_id,
                        "store_tier": store["volume_tier"],
                        "store_inventory_days": round(store_dos, 1),
                        "dc_inventory_days": round(dc_dos, 1),
                        "disruption_duration_days": effective_disruption,
                        "days_to_store_stockout": round(store_dos, 1),
                        "estimated_days_empty": round(max(0, effective_disruption - store_dos), 0),
                        "revenue_at_risk_usd": round(rev_at_risk, 2),
                        "severity": severity,
                    })

    # Alternates summary per region
    alternates_by_region = {}
    for region in ["NW", "SE", "MW"]:
        carrier_obj = CARRIERS.get(affected_entity, {})
        if region in carrier_obj.get("regions_served", []):
            alts = []
            for sku in affected_skus[:1]:  # check first affected SKU
                result = _find_alternate_carriers(sku, affected_entity, region, duration_days)
                alts = result["data"].get("viable_alternates", [])
                break
            alternates_by_region[region] = alts

    mitigation = []
    if any(r["severity"] == "critical" for r in stockout_risks):
        mitigation.append("IMMEDIATE: Trigger emergency replenishment via available alternate carriers")
        mitigation.append("IMMEDIATE: Implement price hold or slight increase to suppress demand and extend days-on-hand")
    mitigation.append("24h: Rebalance DC-NW surplus inventory toward impacted DCs via inter-DC transfer")
    mitigation.append("24h: Prioritize allocation to highest-volume (tier-1) stores")
    if disruption_type == "carrier_strike":
        mitigation.append(f"ONGOING: Monitor {affected_entity} strike resolution — expected {CARRIERS.get(affected_entity, {}).get('expected_resolution_date', 'TBD')}")
    elif disruption_type == "supplier_bankruptcy":
        mitigation.append("STRATEGIC: Begin emergency supplier qualification — 8-16 week timeline")
        mitigation.append("STRATEGIC: Contact top 3 backup suppliers immediately")
    mitigation.append("48h: Escalate to supply chain leadership if no alternate capacity confirmed")

    return _result({
        "disruption_type": disruption_type,
        "affected_entity": affected_entity,
        "affected_skus": affected_skus,
        "duration_days": duration_days,
        "duration_note": duration_notes.get(disruption_type, ""),
        "stockout_risks": stockout_risks[:10],  # cap at 10 for readability
        "critical_count": sum(1 for r in stockout_risks if r["severity"] == "critical"),
        "warning_count": sum(1 for r in stockout_risks if r["severity"] == "warning"),
        "alternates_by_region": alternates_by_region,
        "total_revenue_at_risk_usd": round(total_revenue_at_risk, 2),
        "mitigation_plan": mitigation,
    }, provenance="WMS", freshness=15)


def _calculate_revenue_impact(sku: str, old_price: float, new_price: float,
                               old_volume_units: float, new_volume_units: float,
                               include_trade_dollars: bool = True,
                               jurisdiction: str = "US") -> Dict:
    prod = _get_product(sku)
    margin_pct = prod["margin_pct"]
    vendor_trade_pct = prod["vendor_trade_pct"]
    vmi_pct = prod.get("vmi_pct", 0.0)

    gross_old = old_price * old_volume_units
    gross_new = new_price * new_volume_units
    gross_change = gross_new - gross_old

    # Vendor trade offsets promotional cost (only applies to price reductions)
    trade_offset = 0
    if include_trade_dollars and new_price < old_price and vendor_trade_pct > 0:
        promo_price_diff = (old_price - new_price) * new_volume_units
        trade_offset = min(promo_price_diff, vendor_trade_pct * gross_new)

    net_change = gross_change + trade_offset
    margin_change = net_change * margin_pct

    tax_info = TAX_RATES.get(jurisdiction, TAX_RATES["US"])
    tax_on_margin = margin_change * tax_info["effective_rate"]

    # Carrying cost for owned inventory
    owned_pct = 1 - vmi_pct
    carrying_note = (
        f"VMI split: {int(vmi_pct*100)}% vendor-managed, {int(owned_pct*100)}% owned. "
        "Carrying cost applies to owned portion only."
    ) if vmi_pct > 0 else "100% owned inventory — full carrying cost applies."

    return _result({
        "sku": sku,
        "old_price": old_price,
        "new_price": new_price,
        "old_volume_units": old_volume_units,
        "new_volume_units": new_volume_units,
        "gross_revenue_change_usd": round(gross_change, 2),
        "vendor_trade_offset_usd": round(trade_offset, 2),
        "net_revenue_change_usd": round(net_change, 2),
        "margin_change_usd": round(margin_change, 2),
        "tax_on_margin_usd": round(tax_on_margin, 2),
        "tax_note": f"Tax calculation approximate — {tax_info['note']}",
        "vmi_carrying_note": carrying_note,
        "trade_dollar_note": (
            f"Vendor trade funding ({int(vendor_trade_pct*100)}% of cost) "
            f"offsets ${round(trade_offset, 2)} of promotional price reduction."
            if trade_offset > 0 else "No vendor trade funding applicable."
        ),
    })


def _calculate_carrying_cost(sku: str, excess_units: float, carrying_weeks: float,
                              annual_carrying_rate: float = 0.25) -> Dict:
    prod = _get_product(sku)
    price = prod["base_price"]
    vmi_pct = prod.get("vmi_pct", 0.0)
    owned_units = excess_units * (1 - vmi_pct)
    weekly_rate = (price * annual_carrying_rate) / 52
    total_cost = owned_units * weekly_rate * carrying_weeks

    return _result({
        "sku": sku,
        "excess_units": excess_units,
        "owned_units": owned_units,
        "vmi_units": excess_units * vmi_pct,
        "carrying_weeks": carrying_weeks,
        "annual_carrying_rate_pct": int(annual_carrying_rate * 100),
        "weekly_cost_per_unit_usd": round(weekly_rate, 4),
        "total_carrying_cost_usd": round(total_cost, 2),
        "note": (
            f"Carrying cost applied only to {int((1-vmi_pct)*100)}% owned inventory "
            f"({int(owned_units)} units). {int(vmi_pct*100)}% VMI excluded."
            if vmi_pct > 0
            else "100% owned — full carrying cost applied."
        ),
    })


def _detect_scenario_conflicts(scenarios: List[Dict]) -> Dict:
    conflicts = []
    from itertools import combinations

    for s1, s2 in combinations(scenarios, 2):
        # Check time overlap
        try:
            s1_start = date.fromisoformat(s1["start_date"])
            s2_start = date.fromisoformat(s2["start_date"])
            s1_end = s1_start + timedelta(days=s1.get("horizon_days", 30))
            s2_end = s2_start + timedelta(days=s2.get("horizon_days", 30))
            overlap = max(date.today(), min(s1_start, s2_start)) <= min(s1_end, s2_end)
        except (KeyError, ValueError):
            overlap = True  # assume overlap if dates missing

        if not overlap:
            continue

        d1 = s1.get("demand_direction", "neutral")
        d2 = s2.get("demand_direction", "neutral")
        sup1 = s1.get("supply_direction", "neutral")
        sup2 = s2.get("supply_direction", "neutral")

        # Critical: demand spike + supply constrain
        if (d1 == "increase" and sup2 == "constrain") or (d2 == "increase" and sup1 == "constrain"):
            conflicts.append({
                "scenario_a": s1["name"],
                "scenario_b": s2["name"],
                "conflict_type": "demand_spike_supply_constrained",
                "severity": "critical",
                "description": (
                    f"'{s1['name']}' drives demand up while '{s2['name']}' constrains supply. "
                    "A promotional event during a supply disruption creates exactly the wrong conditions: "
                    "stockouts at peak demand. This combination has historically caused write-offs "
                    "and customer churn."
                ),
                "recommendation": (
                    "DO NOT run both scenarios simultaneously. "
                    "Either defer the promotion until supply is restored, "
                    "or cancel the promotion and focus on supply recovery. "
                    "If both must proceed, cap promotional volume to available supply."
                ),
            })
        # Warning: both increase demand
        elif d1 == "increase" and d2 == "increase":
            conflicts.append({
                "scenario_a": s1["name"],
                "scenario_b": s2["name"],
                "conflict_type": "double_demand_driver",
                "severity": "warning",
                "description": (
                    f"Both '{s1['name']}' and '{s2['name']}' drive demand up simultaneously. "
                    "Combined demand may exceed 3-week supply buffer. Inventory position must be verified."
                ),
                "recommendation": "Verify DC inventory can support combined demand spike. Pre-position stock.",
            })
        # Warning: both constrain supply
        elif sup1 == "constrain" and sup2 == "constrain":
            conflicts.append({
                "scenario_a": s1["name"],
                "scenario_b": s2["name"],
                "conflict_type": "cumulative_supply_risk",
                "severity": "warning",
                "description": (
                    f"'{s1['name']}' and '{s2['name']}' both constrain supply simultaneously. "
                    "Cumulative supply reduction may be more severe than either scenario alone."
                ),
                "recommendation": "Model combined supply reduction. Extend safety stock temporarily.",
            })

    if not conflicts:
        conflicts_msg = "No conflicts detected between the provided scenarios."
    else:
        conflicts_msg = f"{len(conflicts)} conflict(s) detected."

    return _result({
        "scenarios_checked": [s["name"] for s in scenarios],
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "critical_count": sum(1 for c in conflicts if c["severity"] == "critical"),
        "summary": conflicts_msg,
        "important_note": "All scenarios must include a start_date for valid conflict detection. "
                          "Scenarios without time anchors cannot be meaningfully compared.",
    })


def _run_scenario_comparison(sku: str, scenarios: List[Dict], horizon_weeks: int = 8) -> Dict:
    prod = _get_product(sku)
    base_price = prod["base_price"]
    base_demand = prod["base_demand_per_store_week"]
    total_stores = 30
    margin_pct = prod["margin_pct"]

    rows = []
    for sc in scenarios:
        price = sc.get("price", base_price)
        price_change_pct = (price - base_price) / base_price * 100

        # Demand
        elasticity = prod["elasticity"]
        asymmetric = prod["asymmetric"]
        recovery_factor = prod.get("recovery_factor", 1.0)
        is_increase = price_change_pct > 0
        eff_elasticity = elasticity * recovery_factor if (asymmetric and not is_increase) else elasticity
        demand_from_price = eff_elasticity * (price_change_pct / 100)
        promo_uplift = sc.get("promo_uplift_pct", 0) / 100
        supply_reduction = sc.get("supply_reduction_pct", 0) / 100
        tariff_cost = sc.get("tariff_additional_cost_pct", 0) / 100

        net_demand_change = demand_from_price + promo_uplift - supply_reduction * 0.5
        scenario_demand = base_demand * (1 + net_demand_change)
        weekly_revenue = price * scenario_demand * total_stores
        total_revenue = weekly_revenue * horizon_weeks
        total_margin = total_revenue * margin_pct * (1 - tariff_cost)

        rows.append({
            "scenario": sc["name"],
            "price": price,
            "price_change_pct": round(price_change_pct, 2),
            "demand_per_store_week": round(scenario_demand, 1),
            "total_weekly_demand": round(scenario_demand * total_stores, 0),
            "weekly_revenue_usd": round(weekly_revenue, 2),
            "total_revenue_usd": round(total_revenue, 2),
            "total_margin_usd": round(total_margin, 2),
            "notes": (
                f"Promo uplift +{sc.get('promo_uplift_pct', 0)}%. " if sc.get('promo_uplift_pct') else ""
            ) + (
                f"Supply -${sc.get('supply_reduction_pct', 0)}% demand suppression. " if sc.get('supply_reduction_pct') else ""
            ) + (
                f"Tariff adds {sc.get('tariff_additional_cost_pct', 0)}% cost. " if sc.get('tariff_additional_cost_pct') else ""
            )
        })

    baseline = next((r for r in rows if r["scenario"].lower() in ("baseline", "base")), rows[0])

    return _result({
        "sku": sku,
        "sku_name": prod["name"],
        "horizon_weeks": horizon_weeks,
        "comparison": rows,
        "baseline_scenario": baseline["scenario"],
        "best_revenue_scenario": max(rows, key=lambda r: r["total_revenue_usd"])["scenario"],
        "best_margin_scenario": max(rows, key=lambda r: r["total_margin_usd"])["scenario"],
        "recommendation": (
            f"Highest revenue: {max(rows, key=lambda r: r['total_revenue_usd'])['scenario']}. "
            f"Highest margin: {max(rows, key=lambda r: r['total_margin_usd'])['scenario']}."
        ),
    })


# ─── Main Dispatcher ─────────────────────────────────────────────────────────

TOOL_MAP = {
    "simulate_price_change":     _simulate_price_change,
    "get_competitive_pricing":   _get_competitive_pricing,
    "get_demand_forecast":       _get_demand_forecast,
    "calculate_price_elasticity": _calculate_price_elasticity,
    "get_forecast_accuracy":     _get_forecast_accuracy,
    "get_inventory_levels":      _get_inventory_levels,
    "check_shelf_capacity":      _check_shelf_capacity,
    "calculate_stockout_risk":   _calculate_stockout_risk,
    "check_perishable_status":   _check_perishable_status,
    "trigger_replenishment":     _trigger_replenishment,
    "get_carrier_status":        _get_carrier_status,
    "find_alternate_carriers":   _find_alternate_carriers,
    "get_supply_disruption_impact": _get_supply_disruption_impact,
    "calculate_revenue_impact":  _calculate_revenue_impact,
    "calculate_carrying_cost":   _calculate_carrying_cost,
    "detect_scenario_conflicts": _detect_scenario_conflicts,
    "run_scenario_comparison":   _run_scenario_comparison,
}


def _validate_input(tool_name: str, tool_input: Dict[str, Any]) -> Optional[str]:
    """
    Validate tool inputs before execution.
    Returns an error string if invalid, None if valid.
    """
    # SKU validation — any tool that accepts a 'sku' parameter
    if "sku" in tool_input:
        sku = tool_input["sku"]
        if not isinstance(sku, str) or sku not in PRODUCTS:
            return (
                f"Unknown SKU '{sku}'. Valid SKUs: {', '.join(PRODUCTS.keys())}. "
                "Ask the user to clarify the product."
            )

    # Price bounds — must be a positive number within sane retail range
    for price_field in ("old_price", "new_price"):
        if price_field in tool_input:
            try:
                price = float(tool_input[price_field])
            except (TypeError, ValueError):
                return f"'{price_field}' must be a number, got: {tool_input[price_field]}"
            if not (0.01 <= price <= 9_999.99):
                return f"'{price_field}' = {price} is outside valid range $0.01–$9,999.99."

    # Horizon weeks — positive integer
    if "horizon_weeks" in tool_input:
        try:
            h = int(tool_input["horizon_weeks"])
        except (TypeError, ValueError):
            return f"'horizon_weeks' must be an integer, got: {tool_input['horizon_weeks']}"
        if not (1 <= h <= 104):
            return f"'horizon_weeks' = {h} must be between 1 and 104."

    # Duration days — non-negative integer
    if "duration_days" in tool_input:
        try:
            d = int(tool_input["duration_days"])
        except (TypeError, ValueError):
            return f"'duration_days' must be an integer, got: {tool_input['duration_days']}"
        if d < 0 or d > 730:
            return f"'duration_days' = {d} must be between 0 and 730."

    # Location / store ID
    if "location_id" in tool_input:
        loc = tool_input["location_id"]
        valid_locations = set(STORES.keys()) | set(DCS.keys())
        if loc not in valid_locations:
            return (
                f"Unknown location '{loc}'. "
                f"Valid stores: STR-001…STR-030. Valid DCs: {', '.join(DCS.keys())}."
            )

    return None  # all good


def execute(tool_name: str, tool_input: Dict[str, Any]) -> Dict:
    """
    Execute a tool by name with the given input.
    - Validates inputs before execution (returns structured error on failure)
    - Caches results by TTL matching the data source refresh cadence
    - Always returns a structured dict — never raises
    """
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return _error_result(tool_name, ValueError(f"Unknown tool: '{tool_name}'. Available tools: {', '.join(TOOL_MAP.keys())}"))

    # Input validation
    validation_error = _validate_input(tool_name, tool_input)
    if validation_error:
        return {
            "data": {},
            "error": f"Input validation failed for '{tool_name}': {validation_error}",
            "provenance": "OLTP",
            "freshness_minutes": 0,
            "is_stale": False,
        }

    # Cache lookup — skip cache for write-like tools that have side effects
    _NO_CACHE_TOOLS = {"trigger_replenishment", "adjust_promotional_price"}
    cache_key = f"{tool_name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
    if tool_name not in _NO_CACHE_TOOLS:
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", tool_name)
            return cached

    # Execute
    try:
        result = fn(**tool_input)
    except Exception as exc:
        return _error_result(tool_name, exc)

    # Store in cache
    if tool_name not in _NO_CACHE_TOOLS:
        _cache_set(cache_key, result)

    return result
