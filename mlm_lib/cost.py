"""
cost.py — OpenRouter cost interception hook.

Patches the OpenAI SDK's sync and async completion methods once (idempotent)
to capture the ``cost`` field that OpenRouter embeds in ``usage``.

Usage
-----
    from cascade_lib.cost import COST_TRACKER

    COST_TRACKER.install()   # call once; safe to call multiple times
    COST_TRACKER.reset()     # clear before an API call
    # ... make call ...
    cost_dict = COST_TRACKER.pop_cost()   # returns cost and resets
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import openai
import openai.resources.chat.completions


class CostTracker:
    """
    Singleton that intercepts OpenAI SDK completion calls and reads the
    ``cost`` field returned by OpenRouter in the ``usage`` object.

    Thread safety: sufficient for sequential / async-gather use cases.
    For true multi-threaded workloads, add a threading.Lock around writes.
    """

    def __init__(self) -> None:
        self._d: Dict[str, float] = defaultdict(float)
        self._d.update({"last_cost": 0.0, "last_input": 0.0, "last_output": 0.0})
        self._installed: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Patch the OpenAI SDK. Idempotent — safe to call multiple times."""
        if self._installed:
            return

        orig_sync = openai.resources.chat.completions.Completions.create
        orig_async = openai.resources.chat.completions.AsyncCompletions.create
        tracker = self

        def _spy_sync(*args, **kwargs):
            response = orig_sync(*args, **kwargs)
            tracker._extract(response)
            return response

        async def _spy_async(*args, **kwargs):
            response = await orig_async(*args, **kwargs)
            tracker._extract(response)
            return response

        openai.resources.chat.completions.Completions.create = _spy_sync
        openai.resources.chat.completions.AsyncCompletions.create = _spy_async
        self._installed = True

    def reset(self) -> None:
        """Zero out all tracked values."""
        self._d["last_cost"] = 0.0
        self._d["last_input"] = 0.0
        self._d["last_output"] = 0.0

    def pop_cost(self) -> Dict[str, float]:
        """Return a cost dict and immediately reset the tracker."""
        cost = {
            "input_cost": 0.0,
            "output_cost": 0.0,
            "total_cost": float(self._d["last_cost"]),
        }
        self.reset()
        return cost

    @property
    def last_cost(self) -> float:
        return float(self._d["last_cost"])

    @property
    def last_input_tokens(self) -> int:
        return int(self._d["last_input"])

    @property
    def last_output_tokens(self) -> int:
        return int(self._d["last_output"])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _extract(self, response) -> None:
        if not (hasattr(response, "usage") and response.usage):
            return
        try:
            usage = response.usage.model_dump()
            cost = usage.get("cost")
            self._d["last_input"] = getattr(response.usage, "prompt_tokens", 0) or 0
            self._d["last_output"] = getattr(response.usage, "completion_tokens", 0) or 0
            self._d["last_cost"] = float(cost) if cost is not None else 0.0
        except Exception:
            pass


# Module-level singleton — import and use this directly.
COST_TRACKER = CostTracker()