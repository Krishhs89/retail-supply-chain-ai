"""
Multi-agent orchestrator for the retail supply chain optimization system.

Features:
• Configurable max_iterations by query complexity (detected from query content)
• Full agentic tool-use loop via Anthropic SDK
• Conversation history summarization at HISTORY_SUMMARIZE_THRESHOLD messages
• Data provenance tracking — flags OLAP vs WMS staleness to the user
• All tool calls routed through mock_executor with error wrapping
• Scenario conflict detection surfaced to the UI
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import anthropic

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config.settings import settings
from tools.tool_definitions import ALL_TOOLS
from tools import mock_executor

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a retail supply chain optimization AI assistant operating at Walmart scale.
You connect pricing, demand forecasting, inventory management, supply chain, and finance into one coherent
decision system — the connective tissue that was always missing.

== Core mandate ==
When a price changes, a carrier goes on strike, or a forecast shifts, you surface EVERY downstream
consequence and tell every team exactly what to do. Dollar impact, data freshness, confidence intervals —
all surfaced explicitly.

== Non-negotiables ==
1. CONFIDENCE INTERVALS: Never present a single forecast point as truth. Always state the CI range.
   If forecast accuracy < 60%, return is_reliable=false and DO NOT pass numbers to PO system.

2. DATA PROVENANCE: OLAP data is 24 hours old. WMS is 15 minutes old. OLTP is near real-time.
   When OLAP and WMS inventory numbers differ, call it out. Use WMS for operational decisions.

3. REPLENISHMENT LAG: HQ→DC→Store takes 3-4 days minimum (+ 30% chance of 2-4 extra days).
   If days-on-hand < lag, the shelf hits zero before replenishment arrives. Trigger emergency
   transfers proactively — do not wait for the shelf to go empty.

4. ASYMMETRIC ELASTICITY: For diapers, formula, tobacco, alcohol — a price increase suppresses
   demand more than the same decrease recovers it. Use product-class-specific elasticity.

5. SCENARIO CONFLICTS: Promotion + supply disruption = demand spike during shortage. This is
   dangerous. Detect conflicts and warn the user before both scenarios proceed simultaneously.

6. VENDOR TRADE DOLLARS: Always net vendor trade funding against retail revenue before reporting
   margin impact. A promo funded by vendor trade may be revenue-neutral or positive.

7. REGIONAL vs NATIONAL: Alternate carrier availability must be checked at the REGIONAL level.
   A carrier available nationally but absent from the affected region provides zero benefit.

8. VMI vs OWNED: Carrying cost applies only to owned inventory. If VMI percentage > 0,
   exclude that fraction from carrying cost calculations.

== Response format ==
Always close with:
  Actions: numbered list of recommended actions
  Dollar impact: range estimate (e.g. $2M–$4M revenue at risk)
  Data caveats: any OLAP data used that may be 24h stale

== Data sources available ==
- PRODUCTS: HUG48-3 (Huggies diapers), PAM72-5 (Pampers), MLK-GAL (milk, perishable),
  TAB-DIN (tableware), BLK-THR (blankets), CIG-PKT (cigarettes/tobacco)
- CARRIERS: TruckCo_A (tableware/linen, NW+MW), TruckCo_B (diapers, SE+MW — ON STRIKE),
  TruckCo_C (dairy, NW+SE), TruckCo_D (general/mixed, all regions)
- DCs: DC-NW (Seattle), DC-SE (Atlanta), DC-MW (Chicago), 10 stores each (30 total)
"""

# ─── Complexity Detection ─────────────────────────────────────────────────────

_COMPLEX_KEYWORDS = [
    "tariff", "multiple sku", "multi-sku", "all regions", "supplier bankruptcy",
    "port delay", "40 sku", "multi-dc", "all dc", "across all",
]
_SIMPLE_KEYWORDS = [
    "what is the price", "how many", "current inventory", "status of", "show me",
]


def _detect_max_iterations(query: str) -> int:
    q = query.lower()
    if any(k in q for k in _COMPLEX_KEYWORDS):
        return settings.MAX_ITERATIONS_COMPLEX
    if any(k in q for k in _SIMPLE_KEYWORDS):
        return settings.MAX_ITERATIONS_SIMPLE
    return settings.MAX_ITERATIONS_DEFAULT


# ─── History Summarization ────────────────────────────────────────────────────

def _summarize_history(client: anthropic.Anthropic, messages: List[Dict]) -> List[Dict]:
    """Condense older messages into a running summary to stay under context window."""
    if len(messages) < settings.HISTORY_SUMMARIZE_THRESHOLD:
        return messages

    # Keep last 4 messages verbatim; summarize everything before
    recent = messages[-4:]
    older = messages[:-4]

    older_text = "\n\n".join(
        f"[{m['role'].upper()}]: "
        + (m["content"] if isinstance(m["content"], str)
           else json.dumps(m["content"])[:500])
        for m in older
    )

    try:
        summary_resp = client.messages.create(
            model=settings.MODEL_ID,
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": (
                    "Summarize the following conversation history for a retail supply chain AI. "
                    "Capture key decisions made, SKUs discussed, scenarios analyzed, and any "
                    "open action items. Be concise but preserve all dollar figures and dates.\n\n"
                    + older_text
                )
            }]
        )
        summary_text = summary_resp.content[0].text
    except Exception as exc:
        logger.warning("History summarization failed: %s", exc)
        # Fall back to keeping all messages
        return messages

    return [
        {"role": "user", "content": f"[CONVERSATION SUMMARY — earlier turns]:\n{summary_text}"},
        {"role": "assistant", "content": "Understood. I have context from the earlier conversation."},
    ] + recent


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self):
        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Add it to your .env file or environment."
            )
        self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def run(
        self,
        query: str,
        history: List[Dict],
        max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the agentic loop for a query.

        Returns:
            dict with keys:
              response_text        — final LLM response
              tool_calls_made      — list of {name, input, result_summary}
              iterations_used      — number of agentic loop iterations
              data_freshness_warnings — list of provenance/staleness warnings
              error                — None or error string
        """
        max_iter = max_iterations or _detect_max_iterations(query)
        messages = list(history)  # copy

        # Summarize if history is long
        if len(messages) >= settings.HISTORY_SUMMARIZE_THRESHOLD:
            messages = _summarize_history(self.client, messages)

        messages.append({"role": "user", "content": query})

        tool_calls_made: List[Dict] = []
        freshness_warnings: List[str] = []
        iteration = 0

        try:
            response = self.client.messages.create(
                model=settings.MODEL_ID,
                max_tokens=settings.MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=ALL_TOOLS,
                messages=messages,
            )

            while response.stop_reason == "tool_use" and iteration < max_iter:
                iteration += 1

                # Process all tool calls in this response
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name = block.name
                    tool_input = block.input

                    logger.info("Iteration %d — calling tool: %s", iteration, tool_name)
                    result = mock_executor.execute(tool_name, tool_input)

                    # Track provenance warnings
                    if result.get("is_stale"):
                        freshness_warnings.append(
                            f"⚠ Tool '{tool_name}' returned stale data "
                            f"({result.get('provenance', '?')} — "
                            f"{result.get('freshness_minutes', '?')} min lag)."
                        )
                    if result.get("provenance") == "OLAP":
                        freshness_warnings.append(
                            f"ℹ '{tool_name}' uses OLAP data (24-hour lag). "
                            "Use WMS for operational decisions."
                        )

                    # Surface any freshness warnings inside the data
                    data = result.get("data", {})
                    if isinstance(data, dict):
                        for fw in data.get("freshness_warnings", []):
                            freshness_warnings.append(fw)

                    result_summary = {
                        "tool": tool_name,
                        "input": tool_input,
                        "result_summary": (
                            result.get("error") or
                            str(data)[:300] + "..." if len(str(data)) > 300 else str(data)
                        ),
                        "provenance": result.get("provenance", "OLTP"),
                        "freshness_minutes": result.get("freshness_minutes", 5),
                        "had_error": result.get("error") is not None,
                    }
                    tool_calls_made.append(result_summary)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

                # Feed results back
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

                response = self.client.messages.create(
                    model=settings.MODEL_ID,
                    max_tokens=settings.MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    tools=ALL_TOOLS,
                    messages=messages,
                )

            # Extract final text
            final_text = " ".join(
                block.text for block in response.content
                if hasattr(block, "text")
            )

            if iteration >= max_iter and response.stop_reason == "tool_use":
                final_text += (
                    f"\n\n⚠ Note: Reached max iterations ({max_iter}). "
                    "Some analysis may be incomplete. For complex multi-SKU queries, "
                    "consider increasing max iterations in the sidebar."
                )

            return {
                "response_text": final_text,
                "tool_calls_made": tool_calls_made,
                "iterations_used": iteration,
                "data_freshness_warnings": list(set(freshness_warnings)),  # deduplicate
                "updated_history": messages + [{"role": "assistant", "content": final_text}],
                "error": None,
            }

        except anthropic.AuthenticationError:
            return _error_response("Invalid API key. Check ANTHROPIC_API_KEY in your .env file.")
        except anthropic.RateLimitError:
            return _error_response("Rate limit reached. Wait a moment and retry.")
        except anthropic.APIConnectionError as exc:
            return _error_response(f"API connection error: {exc}")
        except Exception as exc:
            logger.exception("Unexpected orchestrator error")
            return _error_response(f"Unexpected error: {type(exc).__name__}: {exc}")

    def stream(
        self,
        query: str,
        history: List[Dict],
        max_iterations: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream the final response text token by token (after tool calls complete).
        Yields string chunks.
        """
        result = self.run(query, history, max_iterations)
        if result["error"]:
            yield f"Error: {result['error']}"
            return
        # Yield in 50-char chunks to simulate streaming
        text = result["response_text"]
        for i in range(0, len(text), 50):
            yield text[i:i + 50]


def _error_response(msg: str) -> Dict[str, Any]:
    return {
        "response_text": f"Error: {msg}",
        "tool_calls_made": [],
        "iterations_used": 0,
        "data_freshness_warnings": [],
        "updated_history": [],
        "error": msg,
    }
