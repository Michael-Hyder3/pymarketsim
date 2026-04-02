import argparse
import json
import os
import statistics
from typing import Dict, List, Tuple, Any, Optional
import urllib.request

from marketsim.tests.new_setting_tests.mixed_agent_simulation import run_mixed_agent_batch


def load_agent_data(path: str, tx_limit: Optional[int] = None) -> Dict[str, Any]:
    with open(path, "r") as handle:
        data = json.load(handle)

    participant = data.get("participant_private_values", {})
    agent_type = participant.get("agent_type")
    private_values = participant.get("buyer_values") or participant.get("seller_costs") or []

    transactions = data.get("transactions", [])
    if tx_limit is not None and tx_limit > 0:
        transactions = transactions[-tx_limit:]

    return {
        "agent_id": participant.get("agent_id"),
        "agent_type": agent_type,
        "private_values": [float(v) for v in private_values],
        "timesteps": data.get("timesteps"),
        "transactions": transactions,
    }


def build_prompt(agent_data: Dict[str, Any], last_result: Optional[Dict[str, Any]]) -> str:
    payload = {
        "agent_id": agent_data.get("agent_id"),
        "agent_type": agent_data.get("agent_type"),
        "private_values": agent_data.get("private_values"),
        "timesteps": agent_data.get("timesteps"),
        "transactions": agent_data.get("transactions"),
    }
    history_block = "None"
    if last_result is not None:
        history_block = json.dumps(
            {
                "shade_min": last_result.get("shade_min"),
                "shade_max": last_result.get("shade_max"),
                "avg_surplus": last_result.get("avg_surplus"),
                "avg_trades": last_result.get("avg_trades"),
                "efficiency": last_result.get("efficiency"),
            },
            indent=2,
        )

    return (
        "You are a Zero-Intelligence (ZI) agent in a continuous double auction.\n"
        "Your goal is to maximize surplus.\n"
        "You must output ONE shading range [shade_min, shade_max] in price units.\n\n"
        "Agent data (JSON):\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Last result (if any):\n"
        f"{history_block}\n\n"
        "Return JSON only with keys: shade_min, shade_max, rationale.\n"
    )


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No JSON object found in LLM response.")


def parse_shade(response_text: str, max_shade: float, fallback: Tuple[float, float]) -> Tuple[float, float, str]:
    try:
        obj = extract_json(response_text)
        shade_min = float(obj.get("shade_min"))
        shade_max = float(obj.get("shade_max"))
        if shade_min > shade_max:
            shade_min, shade_max = shade_max, shade_min
        shade_min = max(0.0, min(shade_min, max_shade))
        shade_max = max(0.0, min(shade_max, max_shade))
        rationale = str(obj.get("rationale", ""))
        return shade_min, shade_max, rationale
    except Exception:
        return fallback[0], fallback[1], "fallback"


def stub_suggest(agent_data: Dict[str, Any], max_shade: float, base_range: Tuple[float, float]) -> Tuple[float, float, str]:
    prices = [float(tx["price"]) for tx in agent_data.get("transactions", []) if "price" in tx]
    if len(prices) < 2:
        return base_range[0], base_range[1], "fallback"

    spread = statistics.pstdev(prices)
    shade_max = min(max_shade, max(10.0, spread * 0.1))
    shade_min = max(0.0, shade_max * 0.5)
    return shade_min, shade_max, "stub based on recent price spread"


def call_openai_compat(prompt: str, endpoint: str, api_key: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    return data["choices"][0]["message"]["content"]


def run_batch(
    shade_range: Tuple[float, float],
    target_side: str,
    target_count: int,
    base_range: Tuple[float, float],
    num_zi_buy: int,
    num_zi_sell: int,
    num_iterations: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    if target_side == "buyer":
        buyer_ranges = [list(base_range), list(shade_range)]
        buyer_counts = [max(0, num_zi_buy - target_count), target_count]
        seller_ranges = [list(base_range)]
        seller_counts = [num_zi_sell]
    else:
        buyer_ranges = [list(base_range)]
        buyer_counts = [num_zi_buy]
        seller_ranges = [list(base_range), list(shade_range)]
        seller_counts = [max(0, num_zi_sell - target_count), target_count]

    summaries = run_mixed_agent_batch(
        num_iterations=num_iterations,
        num_zi_buy=num_zi_buy,
        num_zi_sell=num_zi_sell,
        zi_buy_shade_ranges=buyer_ranges,
        zi_buy_shade_counts=buyer_counts,
        zi_sell_shade_ranges=seller_ranges,
        zi_sell_shade_counts=seller_counts,
        seed=seed,
    )

    zi_total_list = summaries.get("zi_total_surplus", [])
    avg_zi_total = sum(zi_total_list) / max(len(zi_total_list), 1) if zi_total_list else 0.0
    avg_surplus = avg_zi_total / max(num_zi_buy + num_zi_sell, 1)

    tx_list = summaries.get("total_transactions", [])
    avg_trades = sum(tx_list) / max(len(tx_list), 1) if tx_list else 0.0

    opt_list = summaries.get("optimal_surplus", [])
    optimal = sum(opt_list) / max(len(opt_list), 1) if opt_list else 0.0

    efficiency = (avg_zi_total / optimal * 100.0) if optimal > 0 else 0.0

    return {
        "avg_surplus": avg_surplus,
        "avg_trades": avg_trades,
        "efficiency": efficiency,
        "summaries": summaries,
    }


def llm_loop(args: argparse.Namespace) -> Dict[str, Any]:
    agent_data = load_agent_data(args.data, args.tx_limit)
    history: List[Dict[str, Any]] = []

    current = {
        "shade_min": args.base_shade_min,
        "shade_max": args.base_shade_max,
    }
    last_metrics: Optional[Dict[str, Any]] = None

    for i in range(args.iterations):
        prompt = build_prompt(agent_data, last_metrics)
        if args.llm_mode == "openai_compat":
            response_text = call_openai_compat(
                prompt,
                args.llm_endpoint,
                args.llm_api_key,
                args.llm_model,
            )
        else:
            shade_min, shade_max, rationale = stub_suggest(
                agent_data,
                args.max_shade,
                (args.base_shade_min, args.base_shade_max),
            )
            response_text = json.dumps(
                {
                    "shade_min": shade_min,
                    "shade_max": shade_max,
                    "rationale": rationale,
                }
            )

        shade_min, shade_max, rationale = parse_shade(
            response_text,
            args.max_shade,
            (args.base_shade_min, args.base_shade_max),
        )
        metrics = run_batch(
            shade_range=(shade_min, shade_max),
            target_side=args.target_side,
            target_count=args.target_count,
            base_range=(args.base_shade_min, args.base_shade_max),
            num_zi_buy=args.num_zi_buy,
            num_zi_sell=args.num_zi_sell,
            num_iterations=args.batch_runs,
            seed=args.seed,
        )

        iteration_record = {
            "iteration": i + 1,
            "shade_min": shade_min,
            "shade_max": shade_max,
            "rationale": rationale,
            "avg_surplus": metrics["avg_surplus"],
            "avg_trades": metrics["avg_trades"],
            "efficiency": metrics["efficiency"],
            "prompt": prompt,
            "llm_response": response_text,
        }
        history.append(iteration_record)

        if last_metrics is not None:
            improvement = abs(metrics["avg_surplus"] - last_metrics["avg_surplus"])
            if improvement < args.stop_improvement:
                break

        last_metrics = {
            "shade_min": shade_min,
            "shade_max": shade_max,
            "avg_surplus": metrics["avg_surplus"],
            "avg_trades": metrics["avg_trades"],
            "efficiency": metrics["efficiency"],
        }

    output = {
        "config": vars(args),
        "history": history,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative LLM shading loop for ZI agents.")
    parser.add_argument("--data", default="results/zi_only_transactions.json")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-runs", type=int, default=5)
    parser.add_argument("--target-side", choices=["buyer", "seller"], default="buyer")
    parser.add_argument("--target-count", type=int, default=1)
    parser.add_argument("--num-zi-buy", type=int, default=6)
    parser.add_argument("--num-zi-sell", type=int, default=6)
    parser.add_argument("--base-shade-min", type=float, default=0.0)
    parser.add_argument("--base-shade-max", type=float, default=500.0)
    parser.add_argument("--max-shade", type=float, default=500.0)
    parser.add_argument("--tx-limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stop-improvement", type=float, default=0.5)
    parser.add_argument("--output", default="results/llm_loop_history.json")

    parser.add_argument("--llm-mode", choices=["stub", "openai_compat"], default="stub")
    parser.add_argument("--llm-endpoint", default=os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.llm_mode == "openai_compat" and not args.llm_api_key:
        raise SystemExit("LLM_API_KEY is required for openai_compat mode.")
    output = llm_loop(args)
    last = output["history"][-1] if output["history"] else {}
    print("\nLLM LOOP COMPLETE")
    print("=" * 60)
    print(f"Iterations: {len(output['history'])}")
    print(f"Last shade: [{last.get('shade_min')}, {last.get('shade_max')}]")
    print(f"Last avg surplus: {last.get('avg_surplus')}")
    print(f"Last avg trades: {last.get('avg_trades')}")
    print(f"Last efficiency: {last.get('efficiency')}")
    print(f"History saved to: {args.output}")


if __name__ == "__main__":
    main()
