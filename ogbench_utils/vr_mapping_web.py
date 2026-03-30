from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Optional

from .vr_teleop import (
    DEFAULT_VR_CACHE_PATH,
    DEFAULT_VR_MAPPING_PATH,
    DEFAULT_VR_MAPPING_WEB_PORT,
    DEFAULT_VR_PORT,
    VR_GATE_BUTTON_OPTIONS,
    VR_OPTIONAL_BUTTON_OPTIONS,
    VR_GRIPPER_AXIS_OPTIONS,
    VR_HAND_OPTIONS,
    VRManipActionMapper,
    VRManipMappingConfig,
    VR_MOTION_CONTROL_OPTIONS,
    VR_GRIPPER_CONTROL_OPTIONS,
    VR_POSITION_SOURCE_OPTIONS,
    VR_ROTATION_SOURCE_OPTIONS,
    VRRawStateClient,
    VRRawStateServer,
    controller_axis_value,
    controller_named_button_states,
    controller_named_pressed_buttons,
    extract_controller_state,
    load_vr_mapping_config,
    quaternion_wxyz_to_yaw,
    resolve_cached_endpoint,
    save_vr_mapping_config,
    vr_mapping_config_from_payload,
    vr_mapping_config_to_dict,
)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>VR Mapping Editor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #17171c; color: #ececf1; }
    h1 { margin: 0 0 8px; }
    .sub { color: #b8bac7; margin-bottom: 20px; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 16px; }
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
    .legend { display: flex; gap: 12px; margin-top: 8px; font-size: 12px; color: #c9cad6; }
    .legend span::before { content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
    .legend .x::before { background: #ff7b72; }
    .legend .y::before { background: #79c0ff; }
    .legend .z::before { background: #7ee787; }
  </style>
</head>
<body>
  <h1>VR Mapping Editor</h1>
  <div class="sub">Changes autosave to the shared mapping profile used by the monitor, eval, and calibration tools.</div>
  <div id="status" class="footer muted">Loading…</div>
  <div class="grid" id="form-root"></div>
  <div class="grid" style="margin-top:16px; grid-template-columns: 1.15fr 1fr 1fr;">
    <div class="card">
      <h3 style="margin-top:0">Live Mapping Preview</h3>
      <div id="live-meta" class="muted">waiting…</div>
      <div id="preview-lines" style="margin-top:10px; line-height:1.5"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Left Controller</h3>
      <canvas id="left-pad" width="180" height="180" style="background:#13131a;border-radius:8px;border:1px solid #666980"></canvas>
      <canvas id="left-xyz" width="300" height="120" style="margin-top:12px;background:#13131a;border-radius:8px;border:1px solid #666980"></canvas>
      <div class="legend"><span class="x">X</span><span class="y">Y</span><span class="z">Z</span></div>
      <div id="left-lines" style="margin-top:10px; line-height:1.5"></div>
      <div id="left-buttons" class="pill-wrap"></div>
    </div>
    <div class="card">
      <h3 style="margin-top:0">Right Controller</h3>
      <canvas id="right-pad" width="180" height="180" style="background:#13131a;border-radius:8px;border:1px solid #666980"></canvas>
      <canvas id="right-xyz" width="300" height="120" style="margin-top:12px;background:#13131a;border-radius:8px;border:1px solid #666980"></canvas>
      <div class="legend"><span class="x">X</span><span class="y">Y</span><span class="z">Z</span></div>
      <div id="right-lines" style="margin-top:10px; line-height:1.5"></div>
      <div id="right-buttons" class="pill-wrap"></div>
    </div>
  </div>
  <div class="card" style="margin-top:16px;">
    <h3 style="margin-top:0">Calibration Runtime</h3>
    <div id="runtime-meta" class="muted">waiting…</div>
    <div id="runtime-lines" style="margin-top:10px; line-height:1.5"></div>
  </div>
  <div class="footer">
    Profile file: <code id="profile-path"></code><br>
    The live preview uses the current saved profile directly. The runtime card is populated when a calibration tool is connected.
  </div>
  <script>
    const FIELD_GROUPS = [
      [
        { key: "hand", label: "Controller Hand", type: "select", optionsKey: "hands" },
        { key: "gate_button", label: "Intervention / Gate Button", type: "select", optionsKey: "gate_buttons" },
        { key: "gripper_mirror_toggle_button", label: "Gripper Mirror Toggle Button", type: "select", optionsKey: "optional_buttons" },
        { key: "require_gate", label: "Require Gate For Motion", type: "checkbox" },
        { key: "mirror_gripper_when_inactive", label: "Mirror Gripper While Gate Off", type: "checkbox" }
      ],
      [
        { key: "action_x_source", label: "Robot X Source", type: "select", optionsKey: "position_sources" },
        { key: "action_y_source", label: "Robot Y Source", type: "select", optionsKey: "position_sources" },
        { key: "action_z_source", label: "Robot Z Source", type: "select", optionsKey: "position_sources" },
        { key: "rotation_source", label: "Robot Rot Source", type: "select", optionsKey: "rotation_sources" },
        { key: "motion_control_mode", label: "Motion Control Mode", type: "select", optionsKey: "motion_control_modes" },
        { key: "trigger_axis", label: "Gripper Axis", type: "select", optionsKey: "gripper_axes" }
      ],
      [
        { key: "position_gain", label: "XYZ Gain", type: "number", step: "1", min: "0.1" },
        { key: "position_response_gain", label: "XYZ Response Gain", type: "number", step: "0.1", min: "0.0" },
        { key: "position_feedforward", label: "XYZ Feedforward", type: "number", step: "0.1", min: "0.0" },
        { key: "yaw_gain", label: "Yaw Gain", type: "number", step: "0.1", min: "0.1" },
        { key: "yaw_response_gain", label: "Yaw Response Gain", type: "number", step: "0.1", min: "0.0" },
        { key: "yaw_feedforward", label: "Yaw Feedforward", type: "number", step: "0.1", min: "0.0" },
        { key: "gripper_gain", label: "Gripper Gain", type: "number", step: "0.1", min: "0.1" },
        { key: "gripper_control_mode", label: "Gripper Control Mode", type: "select", optionsKey: "gripper_control_modes" },
        { key: "binary_gripper", label: "Binary Gripper", type: "checkbox" }
      ],
      [
        { key: "trigger_close_threshold", label: "Trigger Close Threshold", type: "number", step: "0.01", min: "0.0", max: "1.0" },
        { key: "trigger_open_threshold", label: "Trigger Open Threshold", type: "number", step: "0.01", min: "0.0", max: "1.0" },
        { key: "invert_yaw", label: "Invert Yaw", type: "checkbox" },
        { key: "invert_x", label: "Invert X", type: "checkbox" },
        { key: "invert_y", label: "Invert Y", type: "checkbox" },
        { key: "invert_z", label: "Invert Z", type: "checkbox" },
        { key: "invert_gripper", label: "Invert Gripper", type: "checkbox" }
      ]
    ];

    let options = null;
    let config = null;
    let liveTimer = null;
    const historyByHand = { left: [], right: [] };

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

    function drawPad(canvasId, controller) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#666980";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      ctx.strokeStyle = "#50526a";
      ctx.beginPath();
      ctx.moveTo(10, cy);
      ctx.lineTo(canvas.width - 10, cy);
      ctx.moveTo(cx, 10);
      ctx.lineTo(cx, canvas.height - 10);
      ctx.stroke();
      const joy = controller?.joystick || { x: 0, y: 0 };
      const pad = controller?.trackpad || { x: 0, y: 0 };
      const r = canvas.width * 0.28;
      ctx.fillStyle = "#e5c24e";
      ctx.beginPath();
      ctx.arc(cx + joy.x * r, cy - joy.y * r, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#5ea0e5";
      ctx.beginPath();
      ctx.arc(cx + pad.x * r, cy - pad.y * r, 5, 0, Math.PI * 2);
      ctx.fill();
      if (controller?.tracked) {
        const zRaw = controller.position?.[2] ?? 0;
        const zFrac = Math.max(0, Math.min(1, zRaw / 1.6));
        ctx.fillStyle = "#84aa78";
        ctx.fillRect(canvas.width - 20, canvas.height - 10 - zFrac * (canvas.height - 20), 8, zFrac * (canvas.height - 20));
      }
    }

    function drawXYZHistory(canvasId, hand, controller) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const history = historyByHand[hand];
      if (controller?.tracked) {
        history.push([
          Number(controller.position?.[0] ?? 0),
          Number(controller.position?.[1] ?? 0),
          Number(controller.position?.[2] ?? 0),
        ]);
        while (history.length > 120) history.shift();
      }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "#666980";
      ctx.strokeRect(0.5, 0.5, canvas.width - 1, canvas.height - 1);
      if (history.length < 2) return;
      const vals = history.flat();
      const minV = Math.min(...vals);
      const maxV = Math.max(...vals);
      const span = Math.max(1e-6, maxV - minV);
      const colors = ["#ff7b72", "#79c0ff", "#7ee787"];
      for (let axis = 0; axis < 3; axis++) {
        ctx.strokeStyle = colors[axis];
        ctx.lineWidth = 2;
        ctx.beginPath();
        history.forEach((item, idx) => {
          const x = 8 + idx * ((canvas.width - 16) / Math.max(1, history.length - 1));
          const y = canvas.height - 8 - ((item[axis] - minV) / span) * (canvas.height - 16);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    }

    function renderButtonPills(rootId, states) {
      const root = document.getElementById(rootId);
      root.innerHTML = "";
      for (const name of options.gate_buttons) {
        const pill = document.createElement("div");
        pill.className = `pill ${states?.[name] ? "on" : ""}`;
        pill.textContent = name;
        root.appendChild(pill);
      }
    }

    function linesHtml(lines) {
      return lines.map(x => `<div>${x}</div>`).join("");
    }

    async function refreshLive() {
      try {
        const live = await (await fetch("/api/live")).json();
        document.getElementById("live-meta").textContent =
          `link=${live.link_state} peer=${live.last_remote || "-"} error=${live.last_error || "-"}${live.live_error ? " preview=" + live.live_error : ""}`;
        document.getElementById("preview-lines").innerHTML = linesHtml([
          `raw_dxyz=[${live.preview.raw_delta_position.map(v => Number(v).toFixed(4)).join(", ")}] drot=${Number(live.preview.raw_delta_yaw).toFixed(4)}`,
          `mapped_action=[${live.preview.mapped_action.map(v => Number(v).toFixed(3)).join(", ")}]`,
          `mapping: x<-${live.mapping.action_x_source}  y<-${live.mapping.action_y_source}  z<-${live.mapping.action_z_source}  rot<-${live.mapping.rotation_source}`,
          `motion_mode=${live.mapping.motion_control_mode} gate=${live.mapping.gate_button} require_gate=${live.mapping.require_gate ? 1 : 0}`,
          `grip_mirror_default=${live.mapping.mirror_gripper_when_inactive ? 1 : 0} runtime=${live.preview.gripper_mirror_enabled ? 1 : 0} toggle_button=${live.mapping.gripper_mirror_toggle_button}`,
          `xyz_gain=${Number(live.mapping.position_gain).toFixed(2)} resp=${Number(live.mapping.position_response_gain).toFixed(2)} ff=${Number(live.mapping.position_feedforward).toFixed(2)}`,
          `yaw_gain=${Number(live.mapping.yaw_gain).toFixed(2)} resp=${Number(live.mapping.yaw_response_gain).toFixed(2)} ff=${Number(live.mapping.yaw_feedforward).toFixed(2)}`,
          `gripper_axis=${live.mapping.trigger_axis} gripper_mode=${live.mapping.gripper_control_mode} invert_gripper=${live.mapping.invert_gripper ? 1 : 0}`,
        ]);
        for (const hand of ["left", "right"]) {
          const ctrl = live.controllers[hand];
          drawPad(`${hand}-pad`, ctrl);
          drawXYZHistory(`${hand}-xyz`, hand, ctrl);
          renderButtonPills(`${hand}-buttons`, ctrl.button_states);
          document.getElementById(`${hand}-lines`).innerHTML = linesHtml([
            `connected=${ctrl.connected ? 1 : 0} tracked=${ctrl.tracked ? 1 : 0} device=${ctrl.device_index}`,
            `pos=[${ctrl.position.map(v => Number(v).toFixed(3)).join(", ")}] yaw=${Number(ctrl.yaw).toFixed(2)}`,
            `trigger=${Number(ctrl.trigger).toFixed(3)} buttons=${ctrl.buttons.join(",") || "-"}`,
            `joy=[${Number(ctrl.joystick.x).toFixed(2)}, ${Number(ctrl.joystick.y).toFixed(2)}] pad=[${Number(ctrl.trackpad.x).toFixed(2)}, ${Number(ctrl.trackpad.y).toFixed(2)}]`,
          ]);
        }
        const runtime = live.runtime || {};
        document.getElementById("runtime-meta").textContent =
          runtime.available ? `episode=${runtime.episode} step=${runtime.step} reset_count=${runtime.reset_count}` : "no calibration runtime connected";
        document.getElementById("runtime-lines").innerHTML = runtime.available ? linesHtml(runtime.lines || []) : "";
      } catch (err) {
        document.getElementById("live-meta").textContent = `live fetch failed: ${err}`;
      }
    }

    async function saveField(field, value) {
      const next = { ...config };
      if (field.type === "checkbox") next[field.key] = !!value;
      else if (field.type === "number") next[field.key] = Number(value);
      else next[field.key] = String(value);
      const resp = await fetch("/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next),
      });
      config = await resp.json();
      setStatus("Saved", "ok");
      render();
    }

    async function init() {
      options = await (await fetch("/api/options")).json();
      config = await (await fetch("/api/config")).json();
      render();
      setStatus("Ready", "ok");
      await refreshLive();
      liveTimer = window.setInterval(refreshLive, 200);
    }

    init().catch(err => setStatus("Failed to load: " + err, "bad"));
  </script>
</body>
</html>
"""


def create_vr_source(
    *,
    vr_mode: str,
    vr_host: str,
    vr_port: int,
    cache_path: Path | str,
    reconnect_seconds: float,
) -> Any:
    cache_path = Path(cache_path)
    if str(vr_mode) == "listen":
        listen_port = int(vr_port) if int(vr_port) > 0 else DEFAULT_VR_PORT
        source = VRRawStateServer(host=str(vr_host).strip() or "0.0.0.0", port=listen_port)
        source.start()
        return source

    host, port, _ = resolve_cached_endpoint(
        vr_host,
        vr_port,
        cache_path=cache_path,
        default_host="127.0.0.1",
        default_port=DEFAULT_VR_PORT,
    )
    source = VRRawStateClient(
        host=host,
        port=port,
        reconnect_seconds=float(reconnect_seconds),
        cache_path=cache_path,
        save_cache=True,
    )
    source.start()
    return source


class VRMappingWebServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        mapping_path: Path | str = DEFAULT_VR_MAPPING_PATH,
        source: Any,
        action_dim: int = 5,
        runtime_status_provider: Optional[Callable[[], dict[str, Any]]] = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.mapping_path = Path(mapping_path)
        self.source = source
        self.action_dim = int(action_dim)
        self.runtime_status_provider = runtime_status_provider
        self._server = ThreadingHTTPServer((self.host, self.port), self._make_handler())
        self._thread: Optional[Thread] = None
        self._current_cfg = load_vr_mapping_config(self.mapping_path) or VRManipMappingConfig()
        self._current_cfg_dict = vr_mapping_config_to_dict(self._current_cfg)
        self._live_mapper = VRManipActionMapper(action_dim=self.action_dim, config=self._current_cfg)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _maybe_refresh_mapper(self) -> tuple[VRManipMappingConfig, dict[str, Any]]:
        cfg = load_vr_mapping_config(self.mapping_path) or VRManipMappingConfig()
        cfg_dict = vr_mapping_config_to_dict(cfg)
        if cfg_dict != self._current_cfg_dict:
            self._current_cfg = cfg
            self._current_cfg_dict = cfg_dict
            self._live_mapper = VRManipActionMapper(action_dim=self.action_dim, config=self._current_cfg)
        return self._current_cfg, self._current_cfg_dict

    @staticmethod
    def _json_bytes(payload: dict[str, Any]) -> bytes:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def _summarize_controller(self, sample: Optional[dict[str, Any]], hand: str) -> dict[str, Any]:
        ctrl = extract_controller_state(sample, hand)
        pose = ctrl.get("pose") if isinstance(ctrl, dict) else None
        position = [0.0, 0.0, 0.0]
        yaw = 0.0
        if isinstance(pose, dict):
            try:
                position = [float(v) for v in pose.get("position", [0.0, 0.0, 0.0])[:3]]
            except Exception:
                position = [0.0, 0.0, 0.0]
            quat = pose.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
            try:
                yaw = float(quaternion_wxyz_to_yaw(quat))
            except Exception:
                yaw = 0.0
        return {
            "connected": bool(isinstance(ctrl, dict) and ctrl.get("connected", False)),
            "tracked": bool(isinstance(ctrl, dict) and ctrl.get("tracked", False)),
            "device_index": int(ctrl.get("device_index", -1)) if isinstance(ctrl, dict) else -1,
            "position": position,
            "yaw": yaw,
            "trigger": float(controller_axis_value(ctrl, "trigger")),
            "trackpad": {
                "x": float(controller_axis_value(ctrl, "trackpad_x")),
                "y": float(controller_axis_value(ctrl, "trackpad_y")),
            },
            "joystick": {
                "x": float(controller_axis_value(ctrl, "joystick_x")),
                "y": float(controller_axis_value(ctrl, "joystick_y")),
            },
            "buttons": controller_named_pressed_buttons(ctrl),
            "button_states": controller_named_button_states(ctrl),
        }

    def _runtime_payload(self) -> dict[str, Any]:
        if self.runtime_status_provider is None:
            return {"available": False, "lines": []}
        try:
            payload = self.runtime_status_provider()
        except Exception as exc:
            return {"available": False, "lines": [f"runtime provider error: {exc}"]}
        return payload if isinstance(payload, dict) else {"available": False, "lines": []}

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/":
                    body = HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/options":
                    payload = {
                        "hands": list(VR_HAND_OPTIONS),
                        "position_sources": list(VR_POSITION_SOURCE_OPTIONS),
                        "rotation_sources": [
                            item
                            for item in VR_ROTATION_SOURCE_OPTIONS
                            if item == "none" or item.startswith("global_") or item.startswith("-global_")
                        ],
                        "gate_buttons": list(VR_GATE_BUTTON_OPTIONS),
                        "optional_buttons": list(VR_OPTIONAL_BUTTON_OPTIONS),
                        "gripper_axes": list(VR_GRIPPER_AXIS_OPTIONS),
                        "motion_control_modes": list(VR_MOTION_CONTROL_OPTIONS),
                        "gripper_control_modes": list(VR_GRIPPER_CONTROL_OPTIONS),
                        "profile_path": str(outer.mapping_path),
                    }
                    body = outer._json_bytes(payload)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/config":
                    cfg = load_vr_mapping_config(outer.mapping_path) or VRManipMappingConfig()
                    body = outer._json_bytes(vr_mapping_config_to_dict(cfg))
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path == "/api/live":
                    cfg, cfg_dict = outer._maybe_refresh_mapper()
                    snapshot = outer.source.snapshot()
                    latest = snapshot.get("latest_sample")
                    live_error = ""
                    try:
                        preview_action, preview_diag = outer._live_mapper.map_sample(latest)
                        preview_values = [float(v) for v in preview_action.tolist()]
                    except Exception as exc:
                        live_error = str(exc)
                        preview_diag = {
                            "raw_delta_position": [0.0, 0.0, 0.0],
                            "raw_delta_yaw": 0.0,
                        }
                        preview_values = [0.0] * 5
                    payload = {
                        "link_state": "CONNECTED" if bool(snapshot.get("connected", snapshot.get("connected_clients", 0))) else "DISCONNECTED",
                        "last_error": str(snapshot.get("last_error", "") or ""),
                        "last_remote": str(snapshot.get("last_client", "-") or "-"),
                        "live_error": live_error,
                        "mapping": cfg_dict,
                        "preview": {
                            "raw_delta_position": list(preview_diag.get("raw_delta_position", [0.0, 0.0, 0.0])),
                            "raw_delta_yaw": float(preview_diag.get("raw_delta_yaw", preview_diag.get("raw_delta_rotation", 0.0))),
                            "mapped_action": preview_values,
                            "gripper_mirror_enabled": bool(preview_diag.get("gripper_mirror_enabled", cfg.mirror_gripper_when_inactive)),
                        },
                        "controllers": {
                            "left": outer._summarize_controller(latest, "left"),
                            "right": outer._summarize_controller(latest, "right"),
                        },
                        "runtime": outer._runtime_payload(),
                    }
                    body = outer._json_bytes(payload)
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/config":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except Exception:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
                    return
                cfg = vr_mapping_config_from_payload(payload if isinstance(payload, dict) else {})
                save_vr_mapping_config(cfg, outer.mapping_path)
                body = outer._json_bytes(vr_mapping_config_to_dict(cfg))
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        return Handler

    def start_background(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._server.serve_forever, daemon=True, name="vr-mapping-web")
        self._thread.start()

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def build_mapping_web_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_VR_MAPPING_WEB_PORT)
    p.add_argument("--mapping_path", type=str, default=str(DEFAULT_VR_MAPPING_PATH))
    p.add_argument("--vr_mode", type=str, default="connect", choices=["connect", "listen"])
    p.add_argument("--vr_host", type=str, default="")
    p.add_argument("--vr_port", type=int, default=0)
    p.add_argument("--vr_cache_path", type=str, default=str(DEFAULT_VR_CACHE_PATH))
    p.add_argument("--vr_reconnect_seconds", type=float, default=2.0)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_mapping_web_arg_parser(
        "Local web UI for editing the shared VR mapping profile."
    )
    args = parser.parse_args(argv)
    source = create_vr_source(
        vr_mode=args.vr_mode,
        vr_host=args.vr_host,
        vr_port=args.vr_port,
        cache_path=args.vr_cache_path,
        reconnect_seconds=args.vr_reconnect_seconds,
    )
    server = VRMappingWebServer(
        host=args.host,
        port=args.port,
        mapping_path=args.mapping_path,
        source=source,
    )
    print(f"vr mapping web ui listening at {server.url}", flush=True)
    print(f"mapping profile path: {Path(args.mapping_path)}", flush=True)
    print(source.banner_text(), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        source.close()
    return 0
