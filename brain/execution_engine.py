from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional


class ExecutionEngine:
    """
    🔥 Upgraded Execution Engine

    Features:
    - deterministic execution (unchanged)
    - timeout safe
    - shared state across steps (NEW)
    - context passing (NEW)
    - backward compatible handlers
    """

    def execute(
        self,
        plan: List[Dict[str, Any]],
        handlers: Dict[str, Callable],
        timeout_seconds: float = 3.0,
        slow_step_threshold_seconds: float = 1.5,
        context: Optional[Dict[str, Any]] = None,   # 🔥 NEW
        state: Optional[Dict[str, Any]] = None,     # 🔥 NEW
    ) -> Dict[str, Any]:

        results: List[Dict[str, Any]] = []

        # 🔥 shared containers
        context = context or {}
        state = state or {}

        for node in plan:
            step = str(node.get("step", "")).strip()
            handler = handlers.get(step)

            if handler is None:
                results.append({
                    "step": step,
                    "ok": False,
                    "error": "missing handler"
                })
                continue

            try:
                started_at = perf_counter()

                # 🔥 wrapper to support both old + new handlers
                def _run():
                    try:
                        # NEW STYLE: handler(context, state)
                        return handler(context=context, state=state)
                    except TypeError:
                        try:
                            # MID STYLE: handler(state)
                            return handler(state=state)
                        except TypeError:
                            # OLD STYLE: handler()
                            return handler()

                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_run)
                    payload = future.result(timeout=timeout_seconds)

                elapsed = perf_counter() - started_at

                # 🔥 store last output in shared state
                state[step] = payload

                results.append({
                    "step": step,
                    "ok": True,
                    "payload": payload,
                    "latency_ms": round(elapsed * 1000.0, 2),
                    "slow": elapsed > slow_step_threshold_seconds,
                })

            except FutureTimeoutError:
                results.append({
                    "step": step,
                    "ok": False,
                    "error": f"timeout after {timeout_seconds}s",
                })
                break

            except Exception as exc:
                results.append({
                    "step": step,
                    "ok": False,
                    "error": str(exc)
                })
                break

        return {
            "steps": results,
            "success": all(r.get("ok") for r in results),
            "state": state,  # 🔥 NEW (very useful)
        }


execution_engine = ExecutionEngine()