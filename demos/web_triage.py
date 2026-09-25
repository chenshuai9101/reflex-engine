#!/usr/bin/env python3
"""医药咨询分诊：纯 Laya vs reflex-engine 闸门（同屏对比，网页演示）。

把"reflex-engine 相对纯 Laya 多出来的东西"摆出来看：

  纯 Laya      = 输入 ─▶ 概率分布（到此为止，你得自己决定拿这个概率干嘛）
  reflex-engine = 同一个概率 ─▶ gate.route 三段闸门 ─▶ 一个动作（自动路由/上抛复核/人工）
                  └▶ 修正回流：你点"认同/不同意"，累计校准偏差（置信度 vs 真实准确率）

场景选了医药分诊而不是贪吃蛇，因为这里没有确定性真值、判错的后果真实存在
（不良反应漏报 = 安全事故），gate 的阈值才有意义。

Run: ~/laya-mlx/.venv/bin/python web_triage.py  然后开 http://localhost:8000
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, "/Users/muyunye/reflex-engine")

from reflex.model.laya_model import LayaModel
from reflex.schema import load_decision_types
from reflex.gate import route

MODEL = "/Users/muyunye/laya-mlx/models/laya-multilingual-mlx"

MESSAGES = [
    "请问朗斯弗（TAS-102）的推荐剂量是多少？老年患者需要减量吗？",
    "帮我看看这个单子。",
    "患者服用爱斯万后出现严重皮疹、呼吸困难，是否应立即停药并上报？",
    "能便宜点吗？",
    "我上周订的货还没到，发票也开错了，麻烦帮我查一下。",
    "患者说有点不舒服。",
    "我是XX医院药剂科，想了解TAS-102的进院采购价格和付款条款。",
    "这个药怎么样？",
    "有情况，需要你们看一下。",
    "医生问了一些事情。",
    "请问你们在福建有没有经销商？我想代理你们的产品。",
    "客户那边有点意见。",
    "这个产品还行吗？",
    "帮我跟进一下。",
]

TEAM = {
    "medical_info": "医学部",
    "adverse_event": "药物警戒",
    "sales_lead": "销售",
    "customer_service": "客服",
}
LABEL = {
    "medical_info": "医学咨询",
    "adverse_event": "不良反应",
    "sales_lead": "销售商机",
    "customer_service": "客服/物流",
}
COLOR = {
    "medical_info": "#58a6ff",
    "adverse_event": "#f85149",
    "sales_lead": "#d29922",
    "customer_service": "#bc8cff",
}


class TriageEngine:
    def __init__(self):
        dts = load_decision_types("/Users/muyunye/reflex-engine/config/decisions.json")
        self.dt = dts["pharma_inquiry_triage"]
        self.model = LayaModel(
            MODEL, instructions="判断这条医药咨询属于哪一类。"
        )
        # 预热：吃掉首次编译开销
        self.model.decide_full({"text": "warmup"}, self.dt.options, self.dt.criteria)

        self.lock = threading.Lock()
        self.current = None  # {index, message, decision}
        self.stats = {
            "viewed": 0,
            "labelled": 0,
            "correct": 0,
            "routes": {"auto": 0, "llm": 0, "human": 0},
            "conf_sum": 0.0,
        }

    def next(self, index: int) -> dict:
        msg = MESSAGES[index % len(MESSAGES)]
        out = self.model.decide_full({"text": msg}, self.dt.options, self.dt.criteria)
        conf = out["confidence"]
        r = route(self.dt, conf)
        with self.lock:
            self.stats["viewed"] += 1
            self.stats["routes"][r.value] += 1
            self.current = {
                "index": index,
                "message": msg,
                "confidence": conf,
                "route": r.value,
            }
            stats = self._snapshot()
        return {
            "index": index,
            "message": msg,
            "laya": {
                "choice": out["action"],
                "confidence": conf,
                "probabilities": out["probabilities"],
                "act_probability": out["act_probability"],
            },
            "reflex": {
                "route": r.value,
                "tau_auto": self.dt.tau_auto,
                "tau_review": self.dt.tau_review,
                "action": self._action_text(out["action"], r.value),
            },
            "stats": stats,
        }

    def correct(self, verdict: str) -> dict:
        with self.lock:
            if self.current is None:
                return {"stats": self._snapshot(), "ok": False}
            self.stats["labelled"] += 1
            self.stats["conf_sum"] += self.current["confidence"]
            if verdict == "correct":
                self.stats["correct"] += 1
            return {"stats": self._snapshot(), "ok": True, "index": self.current["index"]}

    def _snapshot(self) -> dict:
        s = self.stats
        labelled = s["labelled"]
        accuracy = s["correct"] / labelled if labelled else None
        mean_conf = s["conf_sum"] / labelled if labelled else None
        gap = (mean_conf - accuracy) if (accuracy is not None) else None
        return {
            "viewed": s["viewed"],
            "labelled": labelled,
            "correct": s["correct"],
            "accuracy": accuracy,
            "mean_conf": mean_conf,
            "calibration_gap": gap,  # 正 = 过度自信
            "routes": dict(s["routes"]),
        }

    @staticmethod
    def _action_text(choice: str, r: str) -> str:
        if r == "auto":
            return f"自动路由到 → {TEAM.get(choice, choice)}"
        if r == "llm":
            return "上抛 LLM / 人工复核"
        return "人工处理"


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>医药分诊 · 纯 Laya vs reflex-engine</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e;
          --green:#3fb950; --red:#f85149; --yellow:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.6 -apple-system,"PingFang SC","SF Mono",ui-monospace,Menlo,monospace; }
  header { padding:14px 20px; border-bottom:1px solid var(--border);
           display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
  header h1 { font-size:18px; margin:0; font-weight:600; }
  header .tag { color:var(--muted); font-size:12px; }
  main { padding:20px; max-width:1080px; margin:0 auto; }
  .msg-card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:18px 20px; margin-bottom:18px; }
  .msg-card .k { color:var(--muted); font-size:12px; }
  .msg-card .txt { font-size:17px; margin:8px 0 14px; line-height:1.5; }
  .cols { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:760px){ .cols { grid-template-columns:1fr; } }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px 18px; }
  .card h2 { font-size:12px; margin:0 0 6px; color:var(--muted); font-weight:600;
             text-transform:uppercase; letter-spacing:.06em; }
  .card .sub { font-size:11px; color:var(--muted); margin-bottom:12px; }
  .choice { font-size:22px; font-weight:700; margin:6px 0 2px; }
  .conf-num { font-size:13px; color:var(--muted); }
  .bars { margin-top:12px; display:flex; flex-direction:column; gap:7px; }
  .bar-row { display:flex; align-items:center; gap:8px; font-size:12px; }
  .bar-row .nm { width:64px; text-align:right; color:var(--muted); }
  .bar { flex:1; height:14px; background:#21262d; border-radius:3px; overflow:hidden; }
  .bar > i { display:block; height:100%; border-radius:3px; transition:width .15s; }
  .bar-row .pc { width:40px; color:var(--muted); }
  .gate-track { position:relative; height:22px; background:#21262d; border-radius:4px; margin:14px 0 6px; }
  .gate-fill { position:absolute; left:0; top:0; bottom:0; border-radius:4px; }
  .thr { position:absolute; top:-4px; bottom:-4px; width:2px; background:#8b949e; }
  .thr .lbl { position:absolute; top:6px; left:4px; font-size:10px; color:var(--muted); white-space:nowrap; }
  .route { display:inline-block; padding:4px 12px; border-radius:6px; font-weight:700; font-size:14px; margin-top:8px; }
  .route.auto { background:rgba(63,185,80,.15); color:var(--green); border:1px solid var(--green); }
  .route.llm  { background:rgba(210,153,34,.15); color:var(--yellow); border:1px solid var(--yellow); }
  .route.human{ background:rgba(248,81,73,.15); color:var(--red); border:1px solid var(--red); }
  .action { margin-top:8px; font-size:14px; }
  .ctl { display:flex; gap:10px; margin:18px 0; align-items:center; flex-wrap:wrap; }
  button { background:#21262d; color:var(--text); border:1px solid var(--border);
           padding:9px 14px; border-radius:7px; cursor:pointer; font:inherit; }
  button:hover { border-color:var(--green); }
  button.primary { background:rgba(63,185,80,.15); border-color:var(--green); }
  button.danger { background:rgba(248,81,73,.12); border-color:var(--red); }
  .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-top:6px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:12px; text-align:center; }
  .stat .v { font-size:20px; font-weight:700; }
  .stat .k { font-size:11px; color:var(--muted); margin-top:2px; }
  .stat .v.bad { color:var(--red); } .stat .v.good { color:var(--green); }
  #toast { color:var(--muted); font-size:12px; margin-left:6px; }
</style>
</head>
<body>
<header>
  <h1>医药咨询分诊 · 纯 Laya vs reflex-engine</h1>
  <span class="tag">同一套 Laya 概率 → 左侧到此为止 · 右侧多出闸门 + 修正回流</span>
</header>
<main>
  <div class="msg-card">
    <div class="k">来了一条咨询 #<span id="idx">—</span></div>
    <div class="txt" id="msg">加载中…</div>
    <div class="ctl">
      <button class="primary" id="b-next">下一条 →</button>
      <button id="b-play">▶ 自动播放</button>
      <button class="danger" id="b-wrong">✗ 我会有不同路由</button>
      <button id="b-right">✓ 路由正确</button>
      <span id="toast"></span>
    </div>
  </div>

  <div class="cols">
    <div class="card">
      <h2>纯 Laya 输出</h2>
      <div class="sub">一个函数：输入 → 概率。到这里就结束了，你得自己决定拿概率干嘛。</div>
      <div class="choice" id="l-choice">—</div>
      <div class="conf-num">argmax 置信度 <b id="l-conf">—</b></div>
      <div class="bars" id="l-bars"></div>
    </div>

    <div class="card">
      <h2>reflex-engine 闸门</h2>
      <div class="sub">同一概率 → gate.route 按风险档阈值 → 一个动作；下面还能修正回流。</div>
      <div class="gate-track">
        <div class="gate-fill" id="g-fill" style="width:0; background:var(--green);"></div>
        <div class="thr" id="t-review" style="left:0%"><span class="lbl">复审 <span id="t-review-v"></span></span></div>
        <div class="thr" id="t-auto" style="left:0%"><span class="lbl">自动 <span id="t-auto-v"></span></span></div>
      </div>
      <div><span class="route" id="g-route">—</span></div>
      <div class="action" id="g-action"></div>
    </div>
  </div>

  <div style="margin-top:18px">
    <div class="card">
      <h2>修正回流 · 校准面板（纯 Laya 没有这一层）</h2>
      <div class="stats">
        <div class="stat"><div class="v" id="s-viewed">0</div><div class="k">已看</div></div>
        <div class="stat"><div class="v" id="s-labelled">0</div><div class="k">已标注</div></div>
        <div class="stat"><div class="v" id="s-acc">—</div><div class="k">准确率</div></div>
        <div class="stat"><div class="v" id="s-mconf">—</div><div class="k">平均置信度</div></div>
        <div class="stat"><div class="v" id="s-gap">—</div><div class="k">校准偏差</div></div>
        <div class="stat"><div class="v" id="s-auto">0</div><div class="k">自动</div></div>
        <div class="stat"><div class="v" id="s-llm">0</div><div class="k">上抛</div></div>
        <div class="stat"><div class="v" id="s-human">0</div><div class="k">人工</div></div>
      </div>
      <div class="sub" style="margin-top:10px">校准偏差 = 平均置信度 − 准确率。正值 = Laya 过度自信（这就是 reflex-engine 铁律"置信度必校准"要治的）。</div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let cur = null, playing = false, timer = null;

function bars(el, probs) {
  el.innerHTML = "";
  for (const [k, v] of Object.entries(probs || {})) {
    const color = {"medical_info":"#58a6ff","adverse_event":"#f85149","sales_lead":"#d29922","customer_service":"#bc8cff"}[k] || "#58a6ff";
    const name = {"medical_info":"医学咨询","adverse_event":"不良反应","sales_lead":"销售商机","customer_service":"客服/物流"}[k] || k;
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span class="nm">${name}</span><div class="bar"><i style="width:${(v*100).toFixed(1)}%;background:${color}"></i></div><span class="pc">${(v*100).toFixed(0)}%</span>`;
    el.appendChild(row);
  }
}

function render(f) {
  $("idx").textContent = f.index + 1;
  $("msg").textContent = f.message;
  const L = f.laya, R = f.reflex;
  $("l-choice").textContent = ({"medical_info":"医学咨询","adverse_event":"不良反应","sales_lead":"销售商机","customer_service":"客服/物流"}[L.choice]) || L.choice;
  $("l-conf").textContent = (100 * L.confidence).toFixed(1) + "%";
  bars($("l-bars"), L.probabilities);

  const c = Math.round(100 * L.confidence);
  $("g-fill").style.width = c + "%";
  $("g-fill").style.background = R.route === "auto" ? "#3fb950" : R.route === "llm" ? "#d29922" : "#f85149";
  $("t-review").style.left = (100 * R.tau_review) + "%"; $("t-review-v").textContent = Math.round(100*R.tau_review)+"%";
  $("t-auto").style.left = (100 * R.tau_auto) + "%"; $("t-auto-v").textContent = Math.round(100*R.tau_auto)+"%";
  const rt = $("g-route");
  rt.textContent = {auto:"AUTO 自动", llm:"上抛复核", human:"人工"}[R.route];
  rt.className = "route " + R.route;
  $("g-action").textContent = R.action;

  const s = f.stats;
  $("s-viewed").textContent = s.viewed;
  $("s-labelled").textContent = s.labelled;
  $("s-acc").textContent = s.accuracy == null ? "—" : (100*s.accuracy).toFixed(0)+"%";
  $("s-mconf").textContent = s.mean_conf == null ? "—" : (100*s.mean_conf).toFixed(0)+"%";
  const gap = $("s-gap");
  if (s.calibration_gap == null) { gap.textContent = "—"; gap.className="v"; }
  else { gap.textContent = (s.calibration_gap >= 0 ? "+" : "") + (100*s.calibration_gap).toFixed(0)+"%"; gap.className = "v " + (Math.abs(s.calibration_gap) < 0.1 ? "good" : "bad"); }
  $("s-auto").textContent = s.routes.auto;
  $("s-llm").textContent = s.routes.llm;
  $("s-human").textContent = s.routes.human;
}

async function loadNext() {
  const idx = cur ? cur.index + 1 : 0;
  try { cur = await (await fetch("/next?index=" + idx)).json(); render(cur); $("toast").textContent = ""; }
  catch(e) { $("toast").textContent = "加载失败"; }
}
async function correct(v) {
  if (!cur) return;
  const r = await (await fetch("/correct?verdict=" + v)).json();
  if (r.ok) { render({...cur, stats: r.stats}); $("toast").textContent = v === "correct" ? "已记录：认同" : "已记录：不同意见"; }
}
function togglePlay() {
  playing = !playing;
  $("b-play").textContent = playing ? "⏸ 暂停" : "▶ 自动播放";
  if (playing) { timer = setInterval(loadNext, 4000); } else { clearInterval(timer); }
}

$("b-next").onclick = loadNext;
$("b-play").onclick = togglePlay;
$("b-right").onclick = () => correct("correct");
$("b-wrong").onclick = () => correct("wrong");
loadNext();
</script>
</body>
</html>
"""


engine = TriageEngine()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        url = urlparse(self.path)
        if url.path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path == "/next":
            q = parse_qs(url.query)
            index = int((q.get("index") or ["0"])[0])
            body = json.dumps(engine.next(index), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path == "/correct":
            q = parse_qs(url.query)
            verdict = (q.get("verdict") or ["correct"])[0]
            body = json.dumps(engine.correct(verdict), ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def main():
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("医药分诊对比 → http://localhost:8000")
    print("左=纯 Laya 概率 · 右=reflex-engine 闸门 · 下=修正回流/校准。Ctrl-C 停止。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
