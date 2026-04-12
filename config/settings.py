"""Global configuration and settings."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Anthropic API — injected into os.environ by streamlit_app.py before this is read
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MODEL_ID: str = "claude-sonnet-4-6"
    MAX_TOKENS: int = 8096

    # Agentic loop — configurable per query complexity
    MAX_ITERATIONS_DEFAULT: int = 10
    MAX_ITERATIONS_COMPLEX: int = 20   # tariff shocks, multi-SKU multi-DC
    MAX_ITERATIONS_SIMPLE: int = 6     # single-SKU price queries

    # History summarization — condense at this many messages to stay under context window
    HISTORY_SUMMARIZE_THRESHOLD: int = 12

    # Data freshness (minutes since last upstream refresh)
    OLAP_FRESHNESS_MINUTES: int = 1440   # 24-hour batch
    WMS_FRESHNESS_MINUTES: int = 15      # warehouse management system
    OLTP_FRESHNESS_MINUTES: int = 5      # near real-time transactional

    # Forecast quality gates
    FORECAST_ACCURACY_WARNING: float = 0.70   # show yellow flag below this
    FORECAST_ACCURACY_MINIMUM: float = 0.60   # refuse to pass downstream below this

    # Confidence interval widening per 4-week horizon period (e.g. 15% wider each period)
    CI_WIDENING_PER_4W_PERIOD: float = 0.15

    # Replenishment chain latency (days)
    HQ_TO_DC_DAYS: int = 1
    DC_PICK_SHIP_DAYS: int = 2     # range 1-2, modelled at upper end for safety
    STORE_RECEIVE_SHELF_DAYS: int = 1
    TOTAL_REPLENISHMENT_LAG_DAYS: int = 4

    # Lead time variability — 30% of POs arrive 2-4 days late
    LEAD_TIME_LATE_PROBABILITY: float = 0.30
    LEAD_TIME_LATE_DAYS_EXTRA: int = 3   # midpoint of 2-4 day range

    # Carrying cost
    ANNUAL_CARRYING_RATE: float = 0.25   # 25% of inventory value per year

    # Perishable max days-of-supply cap
    PERISHABLE_MAX_DOS: int = 3   # trigger markdown recommendation above this


settings = Settings()
