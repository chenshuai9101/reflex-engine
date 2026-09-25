#!/usr/bin/env python3
"""Laya plays Snake — watch it live in your browser.

Reuses laya-mlx's deterministic SnakeGame engine + LayaPolicy (local MLX, ~8ms per
decision, with the cycle safety shield). This file only adds a tiny HTTP server and a
canvas UI on top.

Run:
    ~/laya-mlx/.venv/bin/python web_snake.py
Then open http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy

DEFAULT_MODEL = "/Users/muyunye/laya-mlx/models/laya-multilingual-mlx"

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laya plays Snake</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e; --accent: #3fb950;
    --red: #f85149; --yellow: #d29922; --blue: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 -apple-system, "SF Mono", ui-monospace, Menlo, monospace; }
  header { padding: 14px 20px; border-bottom: 1px solid var(--border);
           display: flex; align-items: baseline; gap: 12px; }
  header h1 { font-size: 18px; margin: 0; font-weight: 600; }
  header .tag { color: var(--muted); font-size: 12px; }
  .badge { color: var(--accent); }
  .badge::before { content: "● "; animation: pulse 1.2s infinite; }
  @keyframes pulse { 50% { opacity: .35; } }
  main { display: flex; gap: 20px; padding: 20px; flex-wrap: wrap; align-items: flex-start; }
  #board-wrap { position: relative; border: 1px solid var(--border); border-radius: 8px;
                padding: 10px; background: var(--panel); }
  canvas { display: block; image-rendering: pixelated; }
  #overlay { position: absolute; inset: 10px; display: none; place-items: center;
             background: rgba(13,17,23,.72); border-radius: 4px;
             font-size: 22px; font-weight: 700; letter-spacing: 1px; color: var(--red); }
  #panel { flex: 1; min-width: 300px; max-width: 420px; display: flex; flex-direction: column; gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px; }
  .card h2 { font-size: 12px; margin: 0 0 10px; color: var(--muted); font-weight: 600;
             text-transform: uppercase; letter-spacing: .08em; }
  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .stat { text-align: center; }
  .stat .v { font-size: 20px; font-weight: 700; }
  .stat .k { font-size: 11px; color: var(--muted); }
  .row { display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; }
  .row .k { color: var(--muted); }
  .moves { display: flex; flex-direction: column; gap: 6px; }
  .move { display: flex; align-items: center; gap: 8px; font-size: 13px; }
  .move .name { width: 62px; text-align: right; color: var(--muted); }
  .move .name.sel { color: var(--accent); font-weight: 700; }
  .bar { flex: 1; height: 12px; background: #21262d; border-radius: 3px; overflow: hidden; }
  .bar > i { display: block; height: 100%; background: var(--blue); border-radius: 3px; transition: width .1s; }
  .move.sel .bar > i { background: var(--accent); }
  .move .pct { width: 44px; color: var(--muted); font-size: 12px; }
  .note { color: var(--muted); font-size: 12px; }
  .note b { color: var(--text); }
  button { background: #21262d; color: var(--text); border: 1px solid var(--border);
           padding: 7px 12px; border-radius: 6px; cursor: pointer; font: inherit; }
  button:hover { border-color: var(--accent); }
  .controls { display: flex; gap: 8px; align-items: center; }
  .controls .speed { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Laya plays Snake <span class="badge">live</span></h1>
  <span class="tag">local MLX · ~8ms/decision · cycle safety shield on</span>
</header>
<main>
  <div id="board-wrap">
    <canvas id="board" width="528" height="352"></canvas>
    <div id="overlay"></div>
  </div>
  <div id="panel">
    <div class="card">
      <h2>Game</h2>
      <div class="stats">
        <div class="stat"><div class="v" id="s-score">0</div><div class="k">score</div></div>
        <div class="stat"><div class="v" id="s-length">6</div><div class="k">length</div></div>
        <div class="stat"><div class="v" id="s-best">0</div><div class="k">best</div></div>
        <div class="stat"><div class="v" id="s-round">1</div><div class="k">round</div></div>
        <div class="stat"><div class="v" id="s-sps">0</div><div class="k">steps/s</div></div>
        <div class="stat"><div class="v" id="s-ticks">0</div><div class="k">ticks</div></div>
      </div>
      <div class="row" style="margin-top:8px"><span class="k">deaths</span><span id="s-deaths">0</span></div>
      <div class="row"><span class="k">safety interventions</span><span id="s-interv">0</span></div>
    </div>

    <div class="card">
      <h2>Laya decision</h2>
      <div class="row"><span class="k">executed</span><b id="d-exec">—</b></div>
      <div class="row"><span class="k">proposed</span><span id="d-prop">—</span></div>
      <div class="row"><span class="k">inference</span><span id="d-ms">—</span></div>
      <div class="row"><span class="k">dead-end risk</span><span id="d-risk">—</span></div>
      <div class="row"><span class="k">food reachable</span><span id="d-food">—</span></div>
      <div class="moves" id="moves" style="margin-top:10px"></div>
    </div>

    <div class="controls">
      <button id="b-pause">Pause</button>
      <button id="b-restart">Restart</button>
      <button id="b-slow">−</button>
      <span class="speed">speed <b id="speed-v">12</b></span>
      <button id="b-fast">+</button>
    </div>

    <div class="note">The snake <b>does not</b> die against itself or the walls by design: a
    Hamiltonian cycle keeps a safe route, and the shield only executes safe moves. When Laya
    proposes an unsafe move it is overridden — watch for the <b>interventions</b> counter.</div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const DIRS = ["UP", "DOWN", "LEFT", "RIGHT"];
let cell = 22, last = null;

function draw(f) {
  const g = f.game, cv = $("board"), ctx = cv.getContext("2d");
  cell = Math.min(528 / g.width, 352 / g.height);
  cv.width = cell * g.width; cv.height = cell * g.height;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, cv.width, cv.height);
  // grid
  ctx.strokeStyle = "#1c2128"; ctx.lineWidth = 1;
  for (let x = 0; x <= g.width; x++) { ctx.beginPath(); ctx.moveTo(x*cell,0); ctx.lineTo(x*cell, cv.height); ctx.stroke(); }
  for (let y = 0; y <= g.height; y++) { ctx.beginPath(); ctx.moveTo(0,y*cell); ctx.lineTo(cv.width, y*cell); ctx.stroke(); }
  // food
  if (g.food) {
    ctx.fillStyle = "#f85149";
    ctx.beginPath(); ctx.arc((g.food[0]+.5)*cell, (g.food[1]+.5)*cell, cell*.38, 0, 7); ctx.fill();
  }
  // snake body (tail -> head)
  const body = g.body;
  for (let i = body.length - 1; i >= 0; i--) {
    const t = body.length === 1 ? 1 : i / (body.length - 1);
    const g2 = Math.round(90 + 120 * t);
    ctx.fillStyle = i === 0 ? "#3fb950" : `rgb(46,${g2},74)`;
    ctx.fillRect(body[i][0]*cell+1, body[i][1]*cell+1, cell-2, cell-2);
  }
  // death overlay
  const ov = $("overlay");
  if (!g.alive || g.won) {
    ov.style.display = "grid";
    ov.textContent = g.won ? "WON" : ("HIT " + (g.death_reason || "").toUpperCase());
  } else ov.style.display = "none";

  // stats
  $("s-score").textContent = g.score;
  $("s-length").textContent = g.length;
  $("s-best").textContent = f.stats.best;
  $("s-round").textContent = f.stats.round;
  $("s-sps").textContent = f.stats.sps;
  $("s-ticks").textContent = g.ticks;
  $("s-deaths").textContent = f.stats.deaths;
  $("s-interv").textContent = f.stats.interventions;

  if (f.decision) {
    const d = f.decision;
    $("d-exec").textContent = d.executed;
    $("d-prop").textContent = d.proposed + (d.intervened ? "  ⚠ overridden" : "");
    $("d-prop").style.color = d.intervened ? "#d29922" : "";
    $("d-ms").textContent = d.inference_ms.toFixed(1) + " ms";
    $("d-risk").textContent = (100*d.dead_end_risk).toFixed(1) + "%";
    $("d-food").textContent = (100*d.food_reachable).toFixed(1) + "%";
    // probability bars
    const mv = $("moves"); mv.innerHTML = "";
    for (const dir of DIRS) {
      const p = d.probabilities[dir] || 0;
      const sel = dir === d.executed;
      const el = document.createElement("div");
      el.className = "move" + (sel ? " sel" : "");
      el.innerHTML = `<span class="name">${dir}</span><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div><span class="pct">${(p*100).toFixed(0)}%</span>`;
      mv.appendChild(el);
    }
  }
}

async function tick() {
  try {
    const r = await fetch("/state");
    const f = await r.json();
    if (f) { draw(f); last = f; }
  } catch (e) {}
}

async function ctl(action) {
  await fetch("/control?action=" + action);
}

$("b-pause").onclick = () => ctl("pause");
$("b-restart").onclick = () => ctl("restart");
$("b-slow").onclick = async () => { await ctl("slow"); };
$("b-fast").onclick = async () => { await ctl("fast"); };

setInterval(tick, 90);
tick();
</script>
</body>
</html>
"""


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.frame = None

    def set(self, frame):
        with self._lock:
            self.frame = frame

    def get(self):
        with self._lock:
            return self.frame


class GameRunner(threading.Thread):
    def __init__(self, state: SharedState, args: argparse.Namespace):
        super().__init__(daemon=True)
        self.state = state
        self.args = args
        self._lock = threading.Lock()
        self._paused = False
        self._restart = False
        self.speed = args.fps

    # -- control surface (called from HTTP thread) -------------------------
    def pause(self):
        with self._lock:
            self._paused = not self._paused
        return self._paused

    def request_restart(self):
        with self._lock:
            self._restart = True

    def bump_speed(self, delta: float):
        with self._lock:
            self.speed = max(1.0, min(240.0, self.speed + delta))
        return self.speed

    # -- game loop ----------------------------------------------------------
    def run(self):
        policy = LayaPolicy(self.args.model, guarded=True, prompt="compact", optimize=False)
        round_num = 1
        game = self._new_game(round_num)
        # warmup so the first visible decision isn't the compile-cost spike
        for _ in range(4):
            d = policy.decide(game)
            if not game.alive:
                break
            game.step(d.executed)

        best = interventions = deaths = 0
        timestamps = deque(maxlen=60)
        interval = 1.0 / self.speed

        while True:
            with self._lock:
                if self._restart:
                    self._restart = False
                    round_num += 1
                    game = self._new_game(round_num)
                paused = self._paused
                speed = self.speed
            if paused:
                time.sleep(0.05)
                continue
            interval = 1.0 / speed

            if not game.alive or game.won:
                # hold the final frame, then start a fresh round
                self.state.set(self._frame(game, None, round_num, best, interventions, deaths, timestamps))
                time.sleep(1.4)
                deaths += int(not game.alive)
                round_num += 1
                game = self._new_game(round_num)
                continue

            started = time.perf_counter()
            decision = policy.decide(game)
            interventions += int(decision.intervened)
            self.state.set(
                self._frame(game, decision, round_num, best, interventions, deaths, timestamps)
            )

            game.step(decision.executed)
            best = max(best, game.score)
            timestamps.append(time.perf_counter())

            if not game.alive or game.won:
                # publish the death frame immediately, then restart next loop
                self.state.set(
                    self._frame(game, decision, round_num, best, interventions, deaths, timestamps)
                )

            elapsed = time.perf_counter() - started
            if elapsed < interval:
                time.sleep(interval - elapsed)

    def _new_game(self, round_num: int) -> SnakeGame:
        a = self.args
        return SnakeGame(a.width, a.height, a.seed + round_num - 1, a.initial_length)

    @staticmethod
    def _frame(game, decision, round_num, best, interventions, deaths, timestamps):
        sps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
        return {
            "game": game.snapshot(),
            "decision": decision.to_dict() if decision else None,
            "stats": {
                "round": round_num,
                "best": best,
                "interventions": interventions,
                "deaths": deaths,
                "sps": round(sps, 1),
            },
        }


class Handler(BaseHTTPRequestHandler):
    runner: GameRunner = None

    def log_message(self, *args):  # silence default logging
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
        if url.path == "/state":
            frame = self.runner.state.get()
            body = (json.dumps(frame) if frame else "null").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path == "/control":
            q = parse_qs(url.query)
            action = (q.get("action") or [""])[0]
            if action == "pause":
                self.runner.pause()
            elif action == "restart":
                self.runner.request_restart()
            elif action == "slow":
                self.runner.bump_speed(-2)
            elif action == "fast":
                self.runner.bump_speed(+2)
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Laya plays Snake, live in the browser")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-length", type=int, default=6)
    args = parser.parse_args()

    state = SharedState()
    runner = GameRunner(state, args)
    Handler.runner = runner
    runner.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Laya plays Snake → http://localhost:{args.port}")
    print("Controls: Pause / Restart / speed +− in the browser. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
