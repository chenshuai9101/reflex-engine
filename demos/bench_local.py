"""Local end-to-end latency benchmark for laya-mlx on this Mac (Apple M5)."""
import json
import time

import mlx.core as mx
import numpy as np

import laya_mlx as laya

STATE_EN = {
    "from": "user@example.com",
    "subject": "Duplicate charge on invoice #4411",
    "body": "We were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
}

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "other": "everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}

STATE_ZH = {"message": "发票4411被重复扣款，请今天退款。"}
QUESTIONS_ZH = {
    "department": {
        "type": "choice",
        "instructions": "这条消息该由哪个部门处理？",
        "criteria": {
            "billing": "发票、支付、退款",
            "technical": "故障、宕机、系统错误",
            "sales": "定价、新合同",
            "other": "其他",
        },
    }
}


def bench_predict(agent, state, questions, warmup=10, iters=200):
    for _ in range(warmup):
        agent.predict(state, questions)
    mx.synchronize()
    samples = []
    for _ in range(iters):
        mx.synchronize()
        t0 = time.perf_counter_ns()
        agent.predict(state, questions)
        mx.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    a = np.asarray(samples)
    return {
        "p50_ms": round(float(np.median(a)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "mean_ms": round(float(a.mean()), 3),
        "min_ms": round(float(a.min()), 3),
        "max_ms": round(float(a.max()), 3),
        "iters": iters,
    }


def load(path):
    t0 = time.perf_counter()
    agent = laya.load(path, dtype="float16")
    mx.synchronize()
    return agent, round(time.perf_counter() - t0, 2)


def main():
    print("=== device ===")
    print(" ", mx.device_info())
    print("  unified mem GB:", round(mx.metal.get_active_memory() if False else 0), "skip")
    import subprocess
    print("  hw.memsize:", subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())

    print("\n=== LOAD ===")
    en, load_en = load("./models/laya-mlx")
    print(f"  English 421M    load={load_en}s")
    multi, load_multi = load("./models/laya-multilingual-mlx")
    print(f"  Multilingual 322M load={load_multi}s")

    print("\n=== END-TO-END predict() latency (float16, warmup=10, iters=200) ===")

    mx.reset_peak_memory()
    r1 = bench_predict(en, STATE_EN, {"department": QUESTIONS["department"]})
    peak_en_1q = mx.get_peak_memory() / 1e6
    print(f"  EN  1 question (choice)   : P50={r1['p50_ms']}ms  P95={r1['p95_ms']}ms  mean={r1['mean_ms']}ms  peak={peak_en_1q:.0f}MiB")

    r3 = bench_predict(en, STATE_EN, QUESTIONS)
    peak_en_3q = mx.get_peak_memory() / 1e6
    print(f"  EN  3 questions (full)    : P50={r3['p50_ms']}ms  P95={r3['p95_ms']}ms  mean={r3['mean_ms']}ms  peak={peak_en_3q:.0f}MiB")

    mx.reset_peak_memory()
    rm1 = bench_predict(multi, STATE_EN, {"department": QUESTIONS["department"]})
    peak_multi = mx.get_peak_memory() / 1e6
    print(f"  ML  1 question (choice)   : P50={rm1['p50_ms']}ms  P95={rm1['p95_ms']}ms  mean={rm1['mean_ms']}ms  peak={peak_multi:.0f}MiB")

    rzh = bench_predict(multi, STATE_ZH, QUESTIONS_ZH)
    print(f"  ML  1 question (中文)     : P50={rzh['p50_ms']}ms  P95={rzh['p95_ms']}ms  mean={rzh['mean_ms']}ms")

    print("\n=== sample answers (sanity) ===")
    print("  EN choice:", json.dumps(en.predict(STATE_EN, {"department": QUESTIONS["department"]})["answers"], ensure_ascii=False))
    print("  EN 3q    :", json.dumps(en.predict(STATE_EN, QUESTIONS)["answers"], ensure_ascii=False))
    print("  ZH choice:", json.dumps(multi.predict(STATE_ZH, QUESTIONS_ZH)["answers"], ensure_ascii=False))


if __name__ == "__main__":
    main()
