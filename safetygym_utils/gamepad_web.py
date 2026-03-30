from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

from .gamepad import (
    DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
    DEFAULT_SAFETY_GAMEPAD_WEB_PORT,
    GAMEPAD_AXIS_OPTIONS,
    GAMEPAD_BUTTON_OPTIONS,
    GAMEPAD_CONTROL_MODE_OPTIONS,
    GamepadMappingConfig,
    PygameGamepadController,
    gamepad_mapping_config_from_payload,
    gamepad_mapping_config_to_dict,
    load_gamepad_mapping_config,
    save_gamepad_mapping_config,
)

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SafetyGym Gamepad Config</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #17171c; color: #ececf1; }
    h1 { margin: 0 0 8px; }
    .sub { color: #b8bac7; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 16px; }
    .card { background: #26262f; border: 1px solid #50526a; border-radius: 10px; padding: 16px; }
    label { display: block; font-size: 14px; color: #d6d8e3; margin-bottom: 6px; }
    input, select { width: 100%; box-sizing: border-box; padding: 10px 12px; border-radius: 8px; border: 1px solid #666980; background: #13131a; color: #f3f4fb; }
    input[type="checkbox"] { width: auto; margin-right: 8px; }
    .check { display: flex; align-items: center; min-height: 42px; }
    .row { margin-bottom: 14px; }
    .footer { margin-top: 18px; color: #c9cad6; }
    .ok { color: #92d36e; }
    .muted { color: #a8aab8; }
    .bad { color: #ff8f8f; }
    code { background: #111217; padding: 2px 6px; border-radius: 5px; }
    .pill-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .pill { border: 1px solid #666980; border-radius: 999px; padding: 4px 8px; font-size: 12px; color: #a8aab8; }
    .pill.on { background: #e5c24e; color: #111217; border-color: #e5c24e; }
    canvas { width: 100%; max-width: 100%; background:#13131a;border-radius:8px;border:1px solid #666980; }
    .mono { font-family: monospace; }
  </style>
</head>
<body>
  <h1>SafetyGym Gamepad Config</h1>
  <div class="sub">Changes autosave to the shared gamepad mapping profile used by the monitor, config tool, eval, and training.</div>
  <div id="status" class="footer muted">Loading…</div>
  <div class="grid" id="form-root"></div>
  <div class="grid" style="margin-top:16px; grid-template-columns: 1.1fr 0.9fr 1fr;">
    <div class="card">
      <h3 style="margin-top:0">Live Device State</h3>
      <div id="live-meta" class="muted">waiting…</div>
      <canvas id="axes-canvas" width="420" height="200"></canvas>
      <div id="axes-lines" style="margin-top:10px; line-height:1.5"></div>
      <div id="buttons" class="pill-wrap"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Mapped Action Preview</h3>
      <div id="preview-lines" class="mono" style="line-height:1.6"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Runtime</h3>
      <div id="runtime-meta" class="muted">waiting…</div>
      <div id="runtime-lines" style="margin-top:10px; line-height:1.5"></div>
    </div>
  </div>
  <div class="footer">
    Profile file: <code id="profile-path"></code>
  </div>
  <script>
    const FIELD_GROUPS = [
      [
        { key: "device_index", label: "Device Index", type: "number", step: "1", min: "0" },
        { key: "control_mode", label: "Control Mode", type: "select", optionsKey: "control_modes" },
        { key: "action_scale", label: "Action Scale", type: "number", step: "0.1", min: "0.0" },
        { key: "deadzone", label: "Deadzone", type: "number", step: "0.01", min: "0.0", max: "0.95" },
        { key: "require_gate_button", label: "Require Gate Button", type: "checkbox" },
        { key: "gate_button", label: "Gate Button", type: "select", optionsKey: "button_options" }
      ],
      [
        { key: "throttle_axis", label: "Throttle Axis", type: "select", optionsKey: "axis_options" },
        { key: "steer_axis", label: "Steer Axis", type: "select", optionsKey: "axis_options" },
        { key: "left_wheel_axis", label: "Left Wheel Axis", type: "select", optionsKey: "axis_options" },
        { key: "right_wheel_axis", label: "Right Wheel Axis", type: "select", optionsKey: "axis_options" },
        { key: "planar_forward_axis", label: "Planar Forward Axis", type: "select", optionsKey: "axis_options" },
        { key: "planar_turn_axis", label: "Planar Turn Axis", type: "select", optionsKey: "axis_options" }
      ],
      [
        { key: "wheel_command_limit", label: "Wheel Command Limit", type: "number", step: "0.1", min: "0.1" },
        { key: "invert_throttle", label: "Invert Throttle", type: "checkbox" },
        { key: "invert_steer", label: "Invert Steer", type: "checkbox" },
        { key: "invert_left_wheel", label: "Invert Left Wheel", type: "checkbox" },
        { key: "invert_right_wheel", label: "Invert Right Wheel", type: "checkbox" },
        { key: "invert_planar_forward", label: "Invert Planar Forward", type: "checkbox" },
        { key: "invert_planar_turn", label: "Invert Planar Turn", type: "checkbox" }
      ]
    ];

    let options = null;
    let config = null;

    function setStatus(text, kind="muted") {
      const el = document.getElementById("status");
      el.textContent = text;
      el.className = `footer ${kind}`;
    }

    function fieldNode(field) {
      const wrap = document.createElement("div");
      wrap.className = "row";
      if (field.type === "checkbox") {
        const label = document.createElement("label");
        label.className = "check";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!config[field.key];
        input.onchange = () => saveField(field, input.checked);
        label.appendChild(input);
        label.appendChild(document.createTextNode(field.label));
        wrap.appendChild(label);
        return wrap;
      }
      const label = document.createElement("label");
      label.textContent = field.label;
      wrap.appendChild(label);
      let input;
      if (field.type === "select") {
        input = document.createElement("select");
        for (const opt of options[field.optionsKey]) {
          const node = document.createElement("option");
          node.value = opt;
          node.textContent = opt;
          if (String(config[field.key]) === String(opt)) node.selected = true;
          input.appendChild(node);
        }
      } else {
        input = document.createElement("input");
        input.type = "number";
        input.value = Number(config[field.key] ?? 0);
        if (field.step) input.step = field.step;
        if (field.min) input.min = field.min;
        if (field.max) input.max = field.max;
      }
      input.onchange = () => saveField(field, input.value);
      wrap.appendChild(input);
      return wrap;
    }

    function render() {
      const root = document.getElementById("form-root");
      root.innerHTML = "";
      FIELD_GROUPS.forEach(group => {
        const card = document.createElement("div");
        card.className = "card";
        group.forEach(field => card.appendChild(fieldNode(field)));
        root.appendChild(card);
      });
      document.getElementById("profile-path").textContent = options.profile_path;
    }

    async function saveField(field, value) {
      if (!config) return;
      config[field.key] = value;
      try {
        const res = await fetch("/api/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config)
        });
        config = await res.json();
        render();
        setStatus("Saved mapping profile", "ok");
      } catch (err) {
        console.error(err);
        setStatus(`Save failed: ${err}`, "bad");
      }
    }

    function drawAxes(live) {
      const canvas = document.getElementById("axes-canvas");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#666980";
      ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
      const axes = live.state?.named_axes || {};
      const names = options.axis_options || [];
      const barW = Math.max(20, Math.floor((canvas.width - 30) / Math.max(1, names.length)));
      names.forEach((name, idx) => {
        const x = 15 + idx * barW;
        const h = canvas.height - 40;
        const y0 = 20;
        const mid = y0 + h / 2;
        ctx.strokeStyle = "#50526a";
        ctx.beginPath();
        ctx.moveTo(x + barW / 2, y0);
        ctx.lineTo(x + barW / 2, y0 + h);
        ctx.stroke();
        const raw = Number(axes[name] || 0);
        const clamped = name === "lt" || name === "rt" ? Math.max(0, Math.min(1, raw)) : Math.max(-1, Math.min(1, raw));
        let top = mid;
        let height = 0;
        if (name === "lt" || name === "rt") {
          height = clamped * h;
          top = y0 + h - height;
        } else {
          height = Math.abs(clamped) * (h / 2);
          top = clamped >= 0 ? mid - height : mid;
        }
        ctx.fillStyle = "#e5c24e";
        ctx.fillRect(x + 4, top, barW - 8, Math.max(2, height));
        ctx.fillStyle = "#d6d8e3";
        ctx.font = "12px Arial";
        ctx.fillText(name, x + 2, canvas.height - 10);
      });
    }

    function renderLive(live) {
      const st = live.state || {};
      const connected = !!st.connected;
      const transport = live.transport || {};
      document.getElementById("live-meta").textContent =
        `scheme=${live.control_scheme} connected=${connected ? 1 : 0} device=${st.device_index ?? -1} name=${st.name || "-"} ` +
        `transport=${transport.connected ? "connected" : "local/disconnected"} ${transport.target || ""}`;
      const axes = st.named_axes || {};
      document.getElementById("axes-lines").innerHTML =
        Object.entries(axes).map(([k, v]) => `<div>${k}: ${Number(v).toFixed(3)}</div>`).join("");
      const buttonsWrap = document.getElementById("buttons");
      buttonsWrap.innerHTML = "";
      for (const name of options.button_options) {
        const pill = document.createElement("div");
        pill.className = "pill" + ((st.buttons || {})[name] ? " on" : "");
        pill.textContent = name;
        buttonsWrap.appendChild(pill);
      }
      const action = live.preview_action || [];
      document.getElementById("preview-lines").innerHTML =
        `<div>preview_action=${JSON.stringify(action)}</div>` +
        `<div>mode=${config.control_mode}</div>` +
        `<div>gate=${config.require_gate_button ? config.gate_button : "off"}</div>`;
      const runtime = live.runtime || {};
      document.getElementById("runtime-meta").textContent = runtime.summary || "no runtime connected";
      document.getElementById("runtime-lines").innerHTML =
        Object.entries(runtime.values || {}).map(([k, v]) => `<div>${k}: ${v}</div>`).join("");
      drawAxes(live);
      if (transport.error) {
        setStatus(`Gamepad transport error: ${transport.error}`, "bad");
      } else {
        setStatus(connected ? "Gamepad connected" : "Waiting for gamepad", connected ? "ok" : "muted");
      }
    }

    async function refreshLive() {
      try {
        const res = await fetch("/api/live");
        renderLive(await res.json());
      } catch (err) {
        console.error(err);
        setStatus(`Live refresh failed: ${err}`, "bad");
      }
    }

    async function init() {
      options = await (await fetch("/api/options")).json();
      config = await (await fetch("/api/config")).json();
      render();
      refreshLive();
      setInterval(refreshLive, 150);
    }
    init();
  </script>
</body>
</html>
"""


class _GamepadConfigHandler(BaseHTTPRequestHandler):
    controller: PygameGamepadController
    config_path: Path
    runtime_provider: Optional[Callable[[], dict[str, Any]]] = None

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(size) if size > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/options":
            self._send_json(
                {
                    "profile_path": str(self.config_path),
                    "control_modes": list(GAMEPAD_CONTROL_MODE_OPTIONS),
                    "axis_options": list(GAMEPAD_AXIS_OPTIONS),
                    "button_options": list(GAMEPAD_BUTTON_OPTIONS),
                }
            )
            return
        if self.path == "/api/config":
            config, _ = load_gamepad_mapping_config(self.config_path)
            self._send_json(gamepad_mapping_config_to_dict(config))
            return
        if self.path == "/api/live":
            payload = self.controller.live_payload()
            if self.runtime_provider is not None:
                try:
                    payload["runtime"] = dict(self.runtime_provider() or {})
                except Exception as exc:
                    payload["runtime"] = {"summary": f"runtime error: {exc}", "values": {}}
            self._send_json(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/config":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        payload = self._read_json()
        config = gamepad_mapping_config_from_payload(payload)
        save_gamepad_mapping_config(config, self.config_path)
        self._send_json(gamepad_mapping_config_to_dict(config))


class GamepadWebServer:
    def __init__(
        self,
        *,
        controller: PygameGamepadController,
        config_path: Path | str = DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH,
        port: int = DEFAULT_SAFETY_GAMEPAD_WEB_PORT,
        runtime_provider: Optional[Callable[[], dict[str, Any]]] = None,
    ):
        self.controller = controller
        self.config_path = Path(config_path).expanduser()
        self.port = int(port)
        self.runtime_provider = runtime_provider
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        handler = type("GamepadConfigHandler", (_GamepadConfigHandler,), {})
        handler.controller = self.controller
        handler.config_path = self.config_path
        handler.runtime_provider = self.runtime_provider
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._thread = Thread(target=self._httpd.serve_forever, name="gamepad-web-ui", daemon=True)
        self._thread.start()

    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


def build_gamepad_web_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--gamepad_config_path", type=str, default=str(DEFAULT_SAFETY_GAMEPAD_CONFIG_PATH))
    p.add_argument("--web_port", type=int, default=DEFAULT_SAFETY_GAMEPAD_WEB_PORT)
    return p
