"""Pydantic schemas for all data structures in the retail optimization system."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Enumerations ────────────────────────────────────────────────────────────

class DisruptionType(str, Enum):
    CARRIER_STRIKE = "carrier_strike"
    PORT_DELAY = "port_delay"
    SUPPLIER_BANKRUPTCY = "supplier_bankruptcy"


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    OK = "ok"


class ProductClass(str, Enum):
    """Determines elasticity model (symmetric vs asymmetric) and perishability rules."""
    DIAPER = "diaper"        # asymmetric elasticity
    DAIRY = "dairy"          # perishable
    PRODUCE = "produce"      # perishable
    TOBACCO = "tobacco"      # asymmetric elasticity
    ALCOHOL = "alcohol"      # asymmetric elasticity
    FORMULA = "formula"      # asymmetric elasticity
    STAPLE = "staple"
    TABLEWARE = "tableware"
    LINEN = "linen"
    GENERAL = "general"


class DataSource(str, Enum):
    OLAP = "OLAP"   # 24-hour batch — use for financial reporting
    OLTP = "OLTP"   # near real-time — use for operational decisions
    WMS = "WMS"     # warehouse management — 15-minute lag


class ReplenishmentPriority(str, Enum):
    STANDARD = "standard"
    EXPEDITED = "expedited"
    EMERGENCY = "emergency"


# ─── Core result wrapper ──────────────────────────────────────────────────────

class ToolResult(BaseModel):
    """Every tool call returns this structure, enabling provenance tracking."""
    data: Dict[str, Any] = {}
    error: Optional[str] = None
    provenance: DataSource = DataSource.OLTP
    freshness_minutes: int = 5
    is_stale: bool = False

    def ok(self) -> bool:
        return self.error is None


# ─── Demand ──────────────────────────────────────────────────────────────────

class WeeklyForecastPoint(BaseModel):
    week: int
    point_estimate: float
    ci_80_lower: float
    ci_80_upper: float
    ci_95_lower: float
    ci_95_upper: float


class DemandForecast(BaseModel):
    sku: str
    sku_name: str
    horizon_weeks: int
    current_accuracy_mape: float     # 0–1; e.g. 0.78 = 78%
    is_reliable: bool                # False if accuracy < FORECAST_ACCURACY_MINIMUM
    weekly_points: List[WeeklyForecastPoint]
    variable_contributions: Dict[str, float]  # variable → units/week impact
    key_drivers: List[str]
    accuracy_note: str


# ─── Inventory ────────────────────────────────────────────────────────────────

class InventoryRecord(BaseModel):
    location_type: str   # hq | dc | store
    location_id: str
    sku: str
    quantity_on_hand: int
    daily_velocity: float
    days_on_hand: float
    reorder_point: int
    safety_stock: int
    source: DataSource
    freshness_minutes: int


class StockoutRisk(BaseModel):
    sku: str
    location_id: str
    current_inventory: int
    daily_demand: float
    days_on_hand: float
    replenishment_lag_days: int
    days_to_stockout: float          # <0 means already out
    severity: SeverityLevel
    emergency_transfer_available: bool
    recommendation: str


# ─── Price Cascade ────────────────────────────────────────────────────────────

class POAdjustment(BaseModel):
    po_id: str
    current_units: int
    recommended_adjustment_units: int
    eta_days: int
    adjustable: bool


class PriceCascadeResult(BaseModel):
    sku: str
    sku_name: str
    old_price: float
    new_price: float
    price_change_pct: float
    elasticity_coefficient: float
    asymmetric: bool
    demand_change_pct: float
    old_weekly_demand_per_store: float
    new_weekly_demand_per_store: float
    total_stores: int
    po_adjustments: List[POAdjustment]
    excess_inventory_units: int
    carrying_cost_increase_usd: float
    gross_revenue_change_usd: float
    net_revenue_change_usd: float
    margin_change_usd: float
    affected_nodes: List[str]
    recommendations: List[str]


# ─── Supply Chain ─────────────────────────────────────────────────────────────

class CarrierInfo(BaseModel):
    carrier_id: str
    name: str
    status: str                    # active | strike | delayed
    cargo_types: List[str]
    regions_served: List[str]
    lead_time_days: int


class AlternateCarrierOption(BaseModel):
    carrier_id: str
    available: bool
    reason_unavailable: Optional[str] = None
    additional_lead_time_days: int = 0
    capacity_pct: int = 100
    cost_premium_pct: float = 0.0


class SupplyDisruptionResult(BaseModel):
    disruption_type: DisruptionType
    affected_entity: str
    affected_skus: List[str]
    duration_days: int
    stockout_risks: List[StockoutRisk]
    alternates_by_region: Dict[str, List[AlternateCarrierOption]]
    regional_gap_warning: Optional[str]
    revenue_at_risk_usd: float
    mitigation_plan: List[str]


# ─── Scenario Planning ────────────────────────────────────────────────────────

class ScenarioInput(BaseModel):
    name: str
    scenario_type: str             # price_change | promotion | supply_disruption | tariff
    start_date: str                # ISO date string
    horizon_days: int
    parameters: Dict[str, Any]


class ScenarioConflict(BaseModel):
    scenario_a: str
    scenario_b: str
    conflict_type: str             # demand_spike_supply_constrained | etc.
    severity: SeverityLevel
    description: str
    recommendation: str


class ScenarioComparisonRow(BaseModel):
    metric: str
    baseline: float
    scenario_values: Dict[str, float]  # scenario_name → value


# ─── Financial ────────────────────────────────────────────────────────────────

class RevenueImpact(BaseModel):
    sku: str
    gross_revenue_change_usd: float
    vendor_trade_offset_usd: float     # reduces the apparent cost of a promo
    net_revenue_change_usd: float
    margin_change_usd: float
    carrying_cost_change_usd: float
    tax_note: str                       # always flag as approximate


# ─── Agent Response ───────────────────────────────────────────────────────────

class AgentResponse(BaseModel):
    response_text: str
    tool_calls_made: List[Dict[str, Any]]   # [{name, input, result_summary}]
    iterations_used: int
    data_freshness_warnings: List[str]
    scenario_conflicts: List[ScenarioConflict] = []
    error: Optional[str] = None
