from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Any

import numpy as np


def resolve_env_render_mode(render_mode: str | None) -> str | None:
    mode = str(render_mode or "none").strip().lower()
    if mode == "pygame":
        return "rgb_array"
    if mode in {"topdown", "none"}:
        return "none"
    return mode


def wants_external_viewer(render_mode: str | None) -> bool:
    return str(render_mode or "").strip().lower() in {"pygame", "topdown"}


def build_external_viewer(
    *,
    render_mode: str | None,
    title: str,
    draw_hz: float,
    scale: float,
):
    mode = str(render_mode or "").strip().lower()
    if mode == "pygame":
        return PygameRGBArrayViewer(title=title, draw_hz=draw_hz, scale=scale)
    if mode == "topdown":
        return PygameTopDownViewer(title=title, draw_hz=draw_hz, scale=scale)
    return None


def _has_graphical_display() -> bool:
    if os.name == "nt":
        return True
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


class PygameRGBArrayViewer:
    def __init__(
        self,
        *,
        title: str = "SafetyGym Viewer",
        draw_hz: float = 20.0,
        scale: float = 1.0,
    ) -> None:
        self.title = str(title)
        self.draw_hz = float(max(0.0, draw_hz))
        self.scale = float(max(0.1, scale))
        self._pygame = None
        self._screen = None
        self._clock = None
        self._last_draw_ts = 0.0
        self._frame_shape: tuple[int, int] | None = None

    def _ensure(self, frame: np.ndarray) -> bool:
        if self._pygame is not None:
            return True
        if not _has_graphical_display():
            return False
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        frame_h, frame_w = int(frame.shape[0]), int(frame.shape[1])
        self._frame_shape = (frame_w, frame_h)
        win_w = max(64, int(round(frame_w * self.scale)))
        win_h = max(64, int(round(frame_h * self.scale)))

        pygame.init()
        pygame.display.set_caption(self.title)
        self._screen = pygame.display.set_mode((win_w, win_h))
        self._clock = pygame.time.Clock()
        self._pygame = pygame
        return True

    def _should_draw(self) -> bool:
        now = time.perf_counter()
        if self._last_draw_ts <= 0.0:
            self._last_draw_ts = now
            return True
        if self.draw_hz <= 0.0:
            self._last_draw_ts = now
            return True
        if (now - self._last_draw_ts) >= (1.0 / self.draw_hz):
            self._last_draw_ts = now
            return True
        return False

    def draw_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return
        arr = np.asarray(arr[..., :3], dtype=np.uint8)
        if not self._ensure(arr):
            return

        pygame = self._pygame
        screen = self._screen
        clock = self._clock
        assert pygame is not None and screen is not None and clock is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        surface = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
        if self.scale != 1.0 or surface.get_size() != screen.get_size():
            surface = pygame.transform.smoothscale(surface, screen.get_size())
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(120)

    def draw_env(self, env) -> None:
        if not self._should_draw():
            return
        try:
            frame = env.render()
        except Exception:
            return
        self.draw_frame(frame)

    def close(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._frame_shape = None


def _unwrap_env(env: Any) -> Any:
    try:
        return env.unwrapped
    except Exception:
        return env


def _as_xy_bounds(extents: Any) -> tuple[float, float, float, float] | None:
    try:
        arr = np.asarray(extents, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size != 4 or not np.isfinite(arr).all():
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def _iter_xy_positions(value: Any) -> list[np.ndarray]:
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return []
    if arr.ndim == 1 and arr.size >= 2 and np.isfinite(arr[:2]).all():
        return [arr[:2].astype(np.float32, copy=False)]
    if arr.ndim == 2 and arr.shape[1] >= 2:
        out: list[np.ndarray] = []
        for row in arr:
            if np.isfinite(row[:2]).all():
                out.append(row[:2].astype(np.float32, copy=False))
        return out
    return []


def _body_pose_xy(task: Any, body_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        body = task.data.body(str(body_name))
        pos = np.asarray(body.xpos, dtype=np.float32).reshape(-1)
        mat = np.asarray(body.xmat, dtype=np.float32).reshape(3, 3)
        if pos.size < 2 or not np.isfinite(pos[:2]).all() or not np.isfinite(mat).all():
            return None
        return pos[:2].astype(np.float32, copy=False), mat
    except Exception:
        return None


def _strip_numeric_suffix(name: str) -> str:
    out = str(name)
    while out and out[-1].isdigit():
        out = out[:-1]
    for suffix in ("obj", "mocap"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out or str(name)


def _shape_spec_from_world_config(task: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cfg = getattr(getattr(task, "world_info", None), "world_config_dict", None)
    if not isinstance(cfg, dict):
        return out
    for section in ("geoms", "free_geoms"):
        items = cfg.get(section, {})
        if not isinstance(items, dict):
            continue
        for body_name, body_cfg in items.items():
            if not isinstance(body_cfg, dict):
                continue
            geoms = body_cfg.get("geoms", [])
            if not geoms:
                continue
            geom = geoms[0]
            geom_type = str(geom.get("type", "cylinder")).lower()
            size = np.asarray(geom.get("size", [0.1, 0.1, 0.1]), dtype=np.float32).reshape(-1)
            out.append(
                {
                    "body_name": str(body_name),
                    "label": _strip_numeric_suffix(str(body_name)).lower(),
                    "geom_type": geom_type,
                    "size": size,
                }
            )
    return out


def extract_safety_topdown_state(env: Any) -> dict[str, Any]:
    base = _unwrap_env(env)
    task = getattr(base, "task", None)
    if task is None:
        return {"available": False}

    extents = _as_xy_bounds(getattr(getattr(task, "placements_conf", None), "extents", None))
    layout = getattr(getattr(task, "world_info", None), "layout", None)
    if extents is None and isinstance(layout, dict) and layout:
        pts = []
        for value in layout.values():
            pts.extend(_iter_xy_positions(value))
        if pts:
            stack = np.stack(pts, axis=0)
            pad = 0.6
            extents = (
                float(np.min(stack[:, 0]) - pad),
                float(np.min(stack[:, 1]) - pad),
                float(np.max(stack[:, 0]) + pad),
                float(np.max(stack[:, 1]) + pad),
            )
    if extents is None:
        extents = (-2.5, -2.5, 2.5, 2.5)

    agent_xy = None
    agent_forward = None
    agent_lateral = None
    try:
        agent_pos = np.asarray(task.agent.pos, dtype=np.float32).reshape(-1)
        if agent_pos.size >= 2 and np.isfinite(agent_pos[:2]).all():
            agent_xy = [float(agent_pos[0]), float(agent_pos[1])]
        agent_mat = np.asarray(task.agent.mat, dtype=np.float32).reshape(3, 3)
        if np.isfinite(agent_mat).all():
            lateral = agent_mat[:2, 0].astype(np.float32, copy=False)
            forward = (-agent_mat[:2, 1]).astype(np.float32, copy=False)
            if np.linalg.norm(forward) > 1e-6:
                forward = forward / np.linalg.norm(forward)
            if np.linalg.norm(lateral) > 1e-6:
                lateral = lateral / np.linalg.norm(lateral)
            agent_forward = [float(forward[0]), float(forward[1])]
            agent_lateral = [float(lateral[0]), float(lateral[1])]
    except Exception:
        pass

    objects: list[dict[str, Any]] = []
    color_map = {
        "goal": (60, 210, 110),
        "hazards": (220, 80, 80),
        "pillars": (245, 190, 70),
        "buttons": (70, 150, 245),
        "vases": (210, 120, 245),
        "gremlins": (255, 120, 180),
    }
    for spec in _shape_spec_from_world_config(task):
        pose = _body_pose_xy(task, spec["body_name"])
        if pose is None:
            continue
        xy, mat = pose
        name = str(spec["label"])
        size_arr = np.asarray(spec["size"], dtype=np.float32).reshape(-1)
        color = color_map.get(name, (170, 170, 185))
        is_goal = name == "goal"
        is_hazard = name in {"hazard", "pillar", "vase", "hazards", "pillars", "vases"}
        objects.append(
            {
                "name": name,
                "xy": [float(xy[0]), float(xy[1])],
                "geom_type": str(spec["geom_type"]),
                "size": size_arr.tolist(),
                "lateral": [float(mat[0, 0]), float(mat[1, 0])],
                "forward": [float(mat[0, 1]), float(mat[1, 1])],
                "color": color,
                "is_goal": is_goal,
                "is_hazard": is_hazard,
            }
        )

    goal_met = False
    try:
        goal_met = bool(getattr(task, "goal_achieved", False))
    except Exception:
        goal_met = False
    return {
        "available": agent_xy is not None,
        "bounds": [float(v) for v in extents],
        "agent_xy": agent_xy,
        "agent_forward": agent_forward,
        "agent_lateral": agent_lateral,
        "goal_distance": float(getattr(task, "dist_goal", lambda: float("nan"))()),
        "goal_met": goal_met,
        "objects": objects,
    }


class PygameTopDownViewer:
    def __init__(
        self,
        *,
        title: str = "SafetyGym Top-Down",
        draw_hz: float = 20.0,
        scale: float = 1.0,
        width: int = 760,
        height: int = 760,
    ) -> None:
        self.title = str(title)
        self.draw_hz = float(max(0.0, draw_hz))
        self.scale = float(max(0.5, scale))
        self.width = int(width)
        self.height = int(height)
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._last_draw_ts = 0.0
        self._trail: deque[tuple[float, float]] = deque(maxlen=200)

    def _ensure(self) -> bool:
        if self._pygame is not None:
            return True
        if not _has_graphical_display():
            return False
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        pygame.init()
        win_w = max(160, int(round(self.width * self.scale)))
        win_h = max(160, int(round(self.height * self.scale)))
        pygame.display.set_caption(self.title)
        self._screen = pygame.display.set_mode((win_w, win_h))
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 26)
        self._small_font = pygame.font.Font(None, 20)
        self._pygame = pygame
        return True

    def _should_draw(self) -> bool:
        now = time.perf_counter()
        if self._last_draw_ts <= 0.0:
            self._last_draw_ts = now
            return True
        if self.draw_hz <= 0.0:
            self._last_draw_ts = now
            return True
        if (now - self._last_draw_ts) >= (1.0 / self.draw_hz):
            self._last_draw_ts = now
            return True
        return False

    @staticmethod
    def _to_screen(
        xy: tuple[float, float] | list[float],
        *,
        bounds: tuple[float, float, float, float],
        rect: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        xmin, ymin, xmax, ymax = bounds
        rx, ry, rw, rh = rect
        span_x = max(1e-6, xmax - xmin)
        span_y = max(1e-6, ymax - ymin)
        fx = (float(xy[0]) - xmin) / span_x
        fy = (float(xy[1]) - ymin) / span_y
        px = rx + int(np.clip(fx, 0.0, 1.0) * rw)
        py = ry + rh - int(np.clip(fy, 0.0, 1.0) * rh)
        return px, py

    def draw_env(self, env) -> None:
        if not self._should_draw():
            return
        state = extract_safety_topdown_state(env)
        if not state.get("available", False):
            return
        if not self._ensure():
            return

        pygame = self._pygame
        screen = self._screen
        clock = self._clock
        font = self._font
        small_font = self._small_font
        assert pygame is not None and screen is not None and clock is not None and font is not None and small_font is not None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass

        screen.fill((18, 20, 24))
        win_w, win_h = screen.get_size()
        world_rect = (24, 64, max(100, win_w - 48), max(100, win_h - 96))
        bounds = tuple(float(v) for v in state["bounds"])

        pygame.draw.rect(screen, (38, 42, 50), world_rect)
        pygame.draw.rect(screen, (92, 98, 112), world_rect, width=1)

        title_surf = font.render("SafetyGym Top-Down", True, (232, 232, 232))
        screen.blit(title_surf, (24, 18))
        dist_txt = f"goal_dist={float(state.get('goal_distance', float('nan'))):.3f}"
        goal_txt = f"goal_met={1 if bool(state.get('goal_met', False)) else 0}"
        screen.blit(small_font.render(dist_txt, True, (190, 190, 190)), (260, 22))
        screen.blit(small_font.render(goal_txt, True, (190, 190, 190)), (430, 22))

        # Grid.
        xmin, ymin, xmax, ymax = bounds
        step = 1.0
        gx = math.floor(xmin)
        while gx <= math.ceil(xmax):
            p0 = self._to_screen((gx, ymin), bounds=bounds, rect=world_rect)
            p1 = self._to_screen((gx, ymax), bounds=bounds, rect=world_rect)
            pygame.draw.line(screen, (44, 48, 58), p0, p1, width=1)
            gx += step
        gy = math.floor(ymin)
        while gy <= math.ceil(ymax):
            p0 = self._to_screen((xmin, gy), bounds=bounds, rect=world_rect)
            p1 = self._to_screen((xmax, gy), bounds=bounds, rect=world_rect)
            pygame.draw.line(screen, (44, 48, 58), p0, p1, width=1)
            gy += step

        pixels_per_unit = min(world_rect[2] / max(1e-6, xmax - xmin), world_rect[3] / max(1e-6, ymax - ymin))

        def _world_poly_to_screen(points_xy: list[tuple[float, float]]) -> list[tuple[int, int]]:
            return [self._to_screen(p, bounds=bounds, rect=world_rect) for p in points_xy]

        def _draw_oriented_box(center_xy, half_x, half_y, axis_x, axis_y, color, *, outline=(255, 255, 255)):
            ax = np.asarray(axis_x, dtype=np.float32)
            ay = np.asarray(axis_y, dtype=np.float32)
            if np.linalg.norm(ax) <= 1e-6:
                ax = np.asarray([1.0, 0.0], dtype=np.float32)
            else:
                ax = ax / np.linalg.norm(ax)
            if np.linalg.norm(ay) <= 1e-6:
                ay = np.asarray([0.0, 1.0], dtype=np.float32)
            else:
                ay = ay / np.linalg.norm(ay)
            center = np.asarray(center_xy, dtype=np.float32)
            corners = [
                center - ax * half_x - ay * half_y,
                center + ax * half_x - ay * half_y,
                center + ax * half_x + ay * half_y,
                center - ax * half_x + ay * half_y,
            ]
            pts = _world_poly_to_screen([(float(p[0]), float(p[1])) for p in corners])
            pygame.draw.polygon(screen, color, pts)
            pygame.draw.polygon(screen, outline, pts, width=1)

        # Static/dynamic objects.
        for obj in state.get("objects", []):
            center = self._to_screen(obj["xy"], bounds=bounds, rect=world_rect)
            color = tuple(int(c) for c in obj.get("color", (160, 160, 160)))
            geom_type = str(obj.get("geom_type", "cylinder")).lower()
            size_arr = np.asarray(obj.get("size", [0.15]), dtype=np.float32).reshape(-1)
            if geom_type == "box":
                half_x = float(size_arr[0]) if size_arr.size > 0 else 0.1
                half_y = float(size_arr[1]) if size_arr.size > 1 else half_x
                _draw_oriented_box(
                    obj["xy"],
                    half_x,
                    half_y,
                    obj.get("lateral", [1.0, 0.0]),
                    obj.get("forward", [0.0, 1.0]),
                    color,
                )
            else:
                radius_px = max(3, int(float(size_arr[0] if size_arr.size > 0 else 0.15) * pixels_per_unit))
                if obj.get("is_goal", False):
                    pygame.draw.circle(screen, color, center, radius_px, width=2)
                elif obj.get("is_hazard", False):
                    pygame.draw.circle(screen, color, center, radius_px)
                    pygame.draw.circle(screen, (255, 255, 255), center, radius_px, width=1)
                else:
                    pygame.draw.circle(screen, color, center, radius_px)

        # Agent trail and heading.
        agent_xy = state.get("agent_xy")
        if agent_xy is not None:
            self._trail.append((float(agent_xy[0]), float(agent_xy[1])))
            if len(self._trail) >= 2:
                pts = [self._to_screen(p, bounds=bounds, rect=world_rect) for p in self._trail]
                pygame.draw.lines(screen, (90, 190, 255), False, pts, width=2)

            forward = state.get("agent_forward", [0.0, 1.0])
            lateral = state.get("agent_lateral", [1.0, 0.0])
            # Approximate footprint from car.xml extents.
            _draw_oriented_box(
                agent_xy,
                0.10,
                0.1675,
                lateral,
                forward,
                (235, 235, 245),
                outline=(60, 120, 255),
            )
            fwd = np.asarray(forward, dtype=np.float32)
            if np.linalg.norm(fwd) > 1e-6:
                fwd = fwd / np.linalg.norm(fwd)
            nose_xy = (
                float(agent_xy[0] + fwd[0] * 0.22),
                float(agent_xy[1] + fwd[1] * 0.22),
            )
            center = self._to_screen(agent_xy, bounds=bounds, rect=world_rect)
            tip = self._to_screen(nose_xy, bounds=bounds, rect=world_rect)
            pygame.draw.line(screen, (40, 110, 255), center, tip, width=3)

        screen.blit(small_font.render("blue line: forward, white box: car footprint", True, (170, 182, 210)), (24, win_h - 26))
        pygame.display.flip()
        clock.tick(120)

    def close(self) -> None:
        pygame = self._pygame
        if pygame is not None:
            pygame.display.quit()
            pygame.quit()
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._small_font = None
        self._trail.clear()
