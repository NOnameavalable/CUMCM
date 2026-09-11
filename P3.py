"""CUMCM 2026 B 题问题 3：多干扰源的搜索、定位与清除框架。

总体思路
--------
1. 用原点和半径 ``1800*cos(30°)`` 的正六边形顶点构成七点覆盖。
   对尚未发现的频道完成七点扫描，可利用最小接收半径 1000 m 保证不漏检。
2. 每个已发现频道维护“目标位置—未知接收半径”联合后验；更新规则沿用
   :mod:`P2` 的离散状态与观测相容性判定。
3. 两次及以上方位观测可调用 :mod:`P1` 做连续几何结果核验；实际清除点使用
   :mod:`P2` 的保守最小包围圆，半径不超过 20 m 时才执行清除。
4. 控制器始终串行调用模拟器，并为网络重试复用同一个 request_id。

默认运行本文件只执行离线自检，不会连接模拟器。正式联调需显式使用：

``python P3.py --run --robot-id <参赛队号>``

本文件是可运行、可继续扩展的工程框架。正式测试前应先用练习环境调节网格
精度、动作优先级和现实运行时间预算。
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from P1 import solve_problem_1
from P2 import (
    EnclosingCircle,
    Observation as P2Observation,
    Problem2Config,
    StateSample,
    _build_first_feasible_region,
    _expand_receive_radius_states,
    _is_observation_compatible,
    _minimum_enclosing_circle,
    _sample_target_cells,
    _state_arrays,
    _validate_inputs,
    solve_problem_2,
)
from utils import Point


ObservationKind = Literal["direction", "near", "no_signal"]
ActionKind = Literal["measure", "clear", "exit"]


class SimulatorError(RuntimeError):
    """模拟器连接、HTTP 或业务响应错误。"""


@dataclass(frozen=True)
class Problem3Config:
    """问题 3 的物理参数、算法精度和运行保护参数。"""

    target_radius: float = 1800.0
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    bearing_error_deg: float = 1.0
    near_radius: float = 5.0
    clear_radius: float = 20.0
    channel_count: int = 20
    dog_speed: float = 5.0
    measure_time: float = 5.0
    switch_time: float = 1.0
    successful_clear_time: float = 5.0
    failed_clear_time: float = 3.0
    max_actions: int = 2000
    request_timeout_s: float = 5.0
    request_retries: int = 3
    real_time_reserve_s: float = 10.0
    localization_baseline_m: float = 500.0

    # P3 直接复用 P2 的联合状态模型。练习时可逐步减小网格间距。
    p2: Problem2Config = field(
        default_factory=lambda: Problem2Config(
            target_grid_spacing=30.0,
            candidate_grid_spacing=120.0,
            refined_grid_spacing=30.0,
            refinement_radius=180.0,
            receive_radius_sample_count=7,
            optimization_mode="robust",
        )
    )

    def __post_init__(self) -> None:
        if self.channel_count != 20:
            raise ValueError("题目规定频道数必须为 20。")
        if self.target_radius <= 0.0 or self.receive_radius_min <= 0.0:
            raise ValueError("目标区域和接收半径必须为正数。")
        if self.receive_radius_min > self.receive_radius_max:
            raise ValueError("接收半径下限不能大于上限。")
        if self.near_radius > self.clear_radius:
            raise ValueError("near 半径不能大于清除半径。")
        if self.max_actions < 1 or self.request_retries < 1:
            raise ValueError("动作上限和重试次数至少为 1。")


@dataclass(frozen=True)
class ObservationRecord:
    """某频道的一次检测记录。"""

    position: Point
    kind: ObservationKind
    bearing_deg: float | None
    virtual_time_s: float
    coverage_index: int | None = None


@dataclass(frozen=True)
class ConservativeCircle:
    """离散点最小包围圆加网格单元半径后的保守清除圆。"""

    center: Point
    sample_radius: float
    conservative_radius: float


@dataclass
class ChannelBelief:
    """一个频道的状态、观测历史和联合后验。"""

    channel: int
    observations: list[ObservationRecord] = field(default_factory=list)
    states: tuple[StateSample, ...] = ()
    coverage_visited: set[int] = field(default_factory=set)
    source_detected: bool = False
    cleared: bool = False
    pending_near_position: Point | None = None
    failed_clear_points: list[Point] = field(default_factory=list)

    @property
    def direction_observations(self) -> list[ObservationRecord]:
        return [item for item in self.observations if item.kind == "direction"]

    def mark_coverage(self, index: int) -> None:
        self.coverage_visited.add(index)

    def coverage_complete(self, point_count: int) -> bool:
        return len(self.coverage_visited) == point_count

    def certified_complete(self, point_count: int) -> bool:
        """已清除，或完整覆盖扫描后始终未发现该频道。"""
        return self.cleared or (
            not self.source_detected and self.coverage_complete(point_count)
        )

    def _initialize_states(
        self,
        first: ObservationRecord,
        config: Problem3Config,
    ) -> tuple[StateSample, ...]:
        assert first.kind == "direction" and first.bearing_deg is not None
        first_point = _validate_inputs(
            (first.position.x, first.position.y), first.bearing_deg
        )
        region = _build_first_feasible_region(first_point, first.bearing_deg, config.p2)
        cells = _sample_target_cells(region, config.p2)
        states = _expand_receive_radius_states(cells, first_point, config.p2)
        if not states:
            raise RuntimeError(f"频道 {self.channel} 的首次观测未生成可行状态。")
        return tuple(states)

    @staticmethod
    def _as_p2_observation(record: ObservationRecord) -> P2Observation:
        return P2Observation(record.kind, record.bearing_deg)

    def _filter_states(
        self,
        states: Sequence[StateSample],
        record: ObservationRecord,
        config: Problem3Config,
    ) -> tuple[StateSample, ...]:
        arrays = _state_arrays(states)
        mask = _is_observation_compatible(
            arrays,
            record.position,
            self._as_p2_observation(record),
            config.p2,
        )
        return tuple(state for state, keep in zip(states, mask) if bool(keep))

    def update(self, record: ObservationRecord, config: Problem3Config) -> None:
        """加入一次观测并更新后验；离散后验坍缩时显式报错。"""
        self.observations.append(record)
        if record.coverage_index is not None:
            self.mark_coverage(record.coverage_index)

        if record.kind == "near":
            self.source_detected = True
            self.pending_near_position = record.position
            return
        if record.kind == "direction":
            self.source_detected = True

        if not self.source_detected:
            # 尚未发现信号时只保存 no_signal 历史；首次 direction 后统一重放。
            return

        if not self.states:
            first_direction = self.direction_observations[0]
            posterior = self._initialize_states(first_direction, config)
            for old_record in self.observations:
                if old_record is first_direction or old_record.kind == "near":
                    continue
                posterior = self._filter_states(posterior, old_record, config)
                if not posterior:
                    raise RuntimeError(
                        f"频道 {self.channel} 的离散后验为空；请减小 P2 网格间距。"
                    )
            self.states = posterior
            return

        posterior = self._filter_states(self.states, record, config)
        if not posterior:
            raise RuntimeError(
                f"频道 {self.channel} 的新观测使离散后验为空；请减小 P2 网格间距。"
            )
        self.states = posterior

    def conservative_circle(self, config: Problem3Config) -> ConservativeCircle | None:
        if self.pending_near_position is not None:
            return ConservativeCircle(
                self.pending_near_position,
                config.near_radius,
                config.near_radius,
            )
        if not self.states:
            return None
        targets: dict[int, tuple[Point, float]] = {}
        for state in self.states:
            targets[state.target_index] = (state.target, state.cell_radius)
        points = [item[0] for item in targets.values()]
        cell_radius = max((item[1] for item in targets.values()), default=0.0)
        circle: EnclosingCircle = _minimum_enclosing_circle(points, config.p2.random_seed)
        return ConservativeCircle(
            circle.center,
            circle.radius,
            circle.radius + cell_radius,
        )

    def can_clear(self, config: Problem3Config) -> bool:
        circle = self.conservative_circle(config)
        return circle is not None and circle.conservative_radius <= config.clear_radius

    def register_clear_failure(self, point: Point, config: Problem3Config) -> None:
        """清除失败意味着目标与该点距离严格大于 20 m。"""
        self.pending_near_position = None
        self.failed_clear_points.append(point)
        if self.states:
            posterior = tuple(
                state
                for state in self.states
                if state.target.distance_to(point) > config.clear_radius
            )
            if posterior:
                self.states = posterior

    def p1_geometry_summary(self) -> tuple[float, bool] | None:
        """用 P1 对两次以上方向观测进行连续几何核验。"""
        data = [
            (item.position.x, item.position.y, float(item.bearing_deg))
            for item in self.direction_observations
            if item.bearing_deg is not None
        ]
        if len(data) < 2:
            return None
        try:
            return solve_problem_1(data)
        except ValueError:
            return None


class CoveragePlanner:
    """构造并管理原点加正六边形顶点的保证覆盖路线。"""

    def __init__(self, config: Problem3Config) -> None:
        self.config = config
        rho = config.target_radius * math.cos(math.pi / 6.0)
        self.points = [Point(0.0, 0.0)] + [
            Point(rho * math.cos(k * math.pi / 3.0), rho * math.sin(k * math.pi / 3.0))
            for k in range(6)
        ]

    def channels_for_point(
        self,
        point_index: int,
        beliefs: dict[int, ChannelBelief],
    ) -> list[int]:
        channels = [
            channel
            for channel, belief in beliefs.items()
            if not belief.source_detected
            and not belief.cleared
            and point_index not in belief.coverage_visited
        ]
        # 蛇形频道顺序可使相邻覆盖点的首频道等于上一点的末频道。
        return sorted(channels, reverse=bool(point_index % 2))

    def next_scan(
        self,
        beliefs: dict[int, ChannelBelief],
    ) -> tuple[int, Point, int] | None:
        for index, point in enumerate(self.points):
            channels = self.channels_for_point(index, beliefs)
            if channels:
                return index, point, channels[0]
        return None

    def verify_dense(self, radial_steps: int = 90, angular_steps: int = 720) -> float:
        """离线抽样返回目标圆内点到最近覆盖点的最大距离。"""
        maximum = 0.0
        for radius in np.linspace(0.0, self.config.target_radius, radial_steps + 1):
            for angle in np.linspace(0.0, 2.0 * math.pi, angular_steps, endpoint=False):
                probe = Point(radius * math.cos(angle), radius * math.sin(angle))
                maximum = max(maximum, min(probe.distance_to(p) for p in self.points))
        return maximum


class LocalizationPlanner:
    """把 P2 的第二点策略扩展为多次观测下的局部定位策略。"""

    def __init__(self, config: Problem3Config) -> None:
        self.config = config

    def recommend(self, belief: ChannelBelief, current: Point) -> Point:
        circle = belief.conservative_circle(self.config)
        if circle is not None and circle.conservative_radius <= self.config.clear_radius:
            return circle.center

        directions = belief.direction_observations
        if len(directions) == 1:
            first = directions[0]
            assert first.bearing_deg is not None
            try:
                result = solve_problem_2(
                    (first.position.x, first.position.y),
                    first.bearing_deg,
                    self.config.p2,
                )
                return result.best_candidate.point
            except (RuntimeError, ValueError):
                pass

        if circle is None:
            raise RuntimeError(f"频道 {belief.channel} 尚无足够信息生成定位点。")

        # 多次观测后在后验圆心两侧构造近似正交基线，并优先选择较近的一侧。
        reference = directions[-1].position if directions else current
        dx, dy = circle.center.x - reference.x, circle.center.y - reference.y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            dx, dy, length = 1.0, 0.0, 1.0
        px, py = -dy / length, dx / length
        baseline = min(
            self.config.localization_baseline_m,
            max(120.0, 2.0 * circle.conservative_radius),
        )
        candidates = [
            Point(circle.center.x + sign * baseline * px, circle.center.y + sign * baseline * py)
            for sign in (-1.0, 1.0)
        ]
        return min(candidates, key=current.distance_to)


@dataclass(frozen=True)
class PlannedAction:
    kind: ActionKind
    point: Point | None = None
    channel: int | None = None
    coverage_index: int | None = None
    reason: str = ""


class GlobalScheduler:
    """在清除、保证覆盖扫描和补充定位之间选择下一动作。"""

    def __init__(
        self,
        config: Problem3Config,
        coverage: CoveragePlanner,
        localization: LocalizationPlanner,
    ) -> None:
        self.config = config
        self.coverage = coverage
        self.localization = localization

    def next_action(
        self,
        beliefs: dict[int, ChannelBelief],
        current: Point,
    ) -> PlannedAction:
        clearable: list[tuple[float, int, Point]] = []
        for channel, belief in beliefs.items():
            if belief.cleared or not belief.can_clear(self.config):
                continue
            circle = belief.conservative_circle(self.config)
            assert circle is not None
            clearable.append((current.distance_to(circle.center), channel, circle.center))
        if clearable:
            _, channel, point = min(clearable)
            return PlannedAction("clear", point, channel, reason="后验已进入20米清除圆")

        scan = self.coverage.next_scan(beliefs)
        if scan is not None:
            index, point, channel = scan
            return PlannedAction(
                "measure", point, channel, index, "完成最小接收半径保证覆盖"
            )

        active = [
            belief
            for belief in beliefs.values()
            if belief.source_detected and not belief.cleared
        ]
        if active:
            belief = min(
                active,
                key=lambda item: current.distance_to(
                    item.conservative_circle(self.config).center
                    if item.conservative_circle(self.config) is not None
                    else current
                ),
            )
            point = self.localization.recommend(belief, current)
            return PlannedAction(
                "measure", point, belief.channel, reason="缩小已发现频道的定位后验"
            )

        if all(
            belief.certified_complete(len(self.coverage.points))
            for belief in beliefs.values()
        ):
            return PlannedAction("exit", reason="全部频道均已清除或完成无信号覆盖证明")
        raise RuntimeError("调度器没有可执行动作，但任务尚未满足严格判停条件。")


class SimulatorClient:
    """严格串行、支持幂等重试的四接口客户端。"""

    def __init__(
        self,
        robot_id: str,
        base_url: str = "http://127.0.0.1:2026",
        timeout_s: float = 5.0,
        retries: int = 3,
        log_path: Path | None = None,
    ) -> None:
        if not robot_id:
            raise ValueError("robot_id 不能为空。")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.retries = retries
        self.log_path = log_path
        self.session_prefix = uuid.uuid4().hex[:10]
        self.sequence = 0
        self.virtual_time_s = 0.0
        self.position = Point(0.0, 0.0)
        self.current_channel = 1
        self.records: list[dict[str, Any]] = []

    def _request_id(self, action: str) -> str:
        self.sequence += 1
        return f"{self.session_prefix}-{action}-{self.sequence}"

    def _base(self, request_id: str) -> dict[str, Any]:
        return {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": request_id,
        }

    def _action_payload(self, action: str, point: Point, channel: int) -> dict[str, Any]:
        payload = self._base(self._request_id(action))
        payload["position"] = {"x": float(point.x), "y": float(point.y)}
        payload["channel"] = int(channel)
        return payload

    def _record(self, path: str, payload: dict[str, Any], response: dict[str, Any]) -> None:
        record = {
            "path": path,
            "request": payload,
            "response": response,
            "wall_time_s": time.time(),
        }
        self.records.append(record)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries):
            request = Request(
                self.base_url + path,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as http_response:
                    response = json.loads(http_response.read().decode("utf-8"))
                if response.get("accepted") is not True:
                    raise SimulatorError(f"{path} 未被接受：{response}")
                self.virtual_time_s = float(response["virtual_time_s"])
                self._record(path, payload, response)
                return response
            except HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                raise SimulatorError(f"{path} HTTP {exc.code}: {detail}") from exc
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(0.2 * (attempt + 1))
        raise SimulatorError(f"{path} 在 {self.retries} 次幂等重试后失败。") from last_error

    def enter(self) -> dict[str, Any]:
        request_id = self._request_id("enter")
        return self._post("/enter", self._base(request_id))

    def measure(self, point: Point, channel: int) -> dict[str, Any]:
        response = self._post(
            "/measure", self._action_payload("measure", point, channel)
        )
        self.position = point
        self.current_channel = channel
        return response

    def clear(self, point: Point, channel: int) -> dict[str, Any]:
        response = self._post("/clear", self._action_payload("clear", point, channel))
        self.position = point
        return response

    def exit(self) -> dict[str, Any]:
        request_id = self._request_id("exit")
        return self._post("/exit", self._base(request_id))


class Problem3Controller:
    """问题 3 主控制循环。"""

    def __init__(self, client: SimulatorClient, config: Problem3Config | None = None) -> None:
        self.client = client
        self.config = config or Problem3Config()
        self.coverage = CoveragePlanner(self.config)
        self.localization = LocalizationPlanner(self.config)
        self.scheduler = GlobalScheduler(self.config, self.coverage, self.localization)
        self.beliefs = {
            channel: ChannelBelief(channel)
            for channel in range(1, self.config.channel_count + 1)
        }
        self.wall_start_s = 0.0
        self.real_deadline_s = math.inf
        self.actions_executed = 0

    def _check_real_time(self) -> None:
        if time.monotonic() >= self.real_deadline_s - self.config.real_time_reserve_s:
            raise SimulatorError("现实时间即将耗尽，已停止产生新动作。")

    def _handle_measure(self, action: PlannedAction) -> None:
        assert action.point is not None and action.channel is not None
        response = self.client.measure(action.point, action.channel)
        kind = response.get("measure_result")
        if kind not in ("direction", "near", "no_signal"):
            raise SimulatorError(f"未知 measure_result：{kind!r}")
        bearing = float(response["svd_deg"]) if kind == "direction" else None
        record = ObservationRecord(
            position=action.point,
            kind=kind,
            bearing_deg=bearing,
            virtual_time_s=float(response["virtual_time_s"]),
            coverage_index=action.coverage_index,
        )
        self.beliefs[action.channel].update(record, self.config)

    def _handle_clear(self, action: PlannedAction) -> None:
        assert action.point is not None and action.channel is not None
        belief = self.beliefs[action.channel]
        response = self.client.clear(action.point, action.channel)
        result = response.get("clear_result")
        if result == "success":
            belief.cleared = True
            belief.pending_near_position = None
            return
        if result == "no_target_in_range":
            belief.register_clear_failure(action.point, self.config)
            return
        raise SimulatorError(f"未知 clear_result：{result!r}")

    def run(self) -> dict[str, Any]:
        """进入测试、执行动作直到严格判停，并返回统计摘要。"""
        self.wall_start_s = time.monotonic()
        enter_response = self.client.enter()
        remaining = float(enter_response["remaining_real_duration_s"])
        self.real_deadline_s = time.monotonic() + remaining

        while self.actions_executed < self.config.max_actions:
            self._check_real_time()
            action = self.scheduler.next_action(self.beliefs, self.client.position)
            if action.kind == "exit":
                exit_response = self.client.exit()
                return self.summary(exit_response.get("exit_reason", "user_exit"))
            if action.kind == "measure":
                self._handle_measure(action)
            elif action.kind == "clear":
                self._handle_clear(action)
            self.actions_executed += 1

        raise RuntimeError(f"达到动作上限 {self.config.max_actions}，任务尚未完成。")

    def summary(self, exit_reason: str) -> dict[str, Any]:
        cleared = sum(belief.cleared for belief in self.beliefs.values())
        elapsed = time.monotonic() - self.wall_start_s
        return {
            "exit_reason": exit_reason,
            "cleared_count": cleared,
            "cleared_channels": [
                channel for channel, belief in self.beliefs.items() if belief.cleared
            ],
            "virtual_time_s": self.client.virtual_time_s,
            "average_localize_clear_time_s": (
                self.client.virtual_time_s / cleared if cleared else None
            ),
            "program_runtime_s": elapsed,
            "actions_executed": self.actions_executed,
        }


def run_offline_check(config: Problem3Config) -> dict[str, Any]:
    """验证覆盖构造和核心参数；不会访问模拟器。"""
    planner = CoveragePlanner(config)
    maximum_distance = planner.verify_dense()
    rho = config.target_radius * math.cos(math.pi / 6.0)
    route_distance = 6.0 * rho
    full_scan_time = len(planner.points) * (
        config.channel_count * config.measure_time
        + (config.channel_count - 1) * config.switch_time
    )
    return {
        "coverage_points": [(round(p.x, 3), round(p.y, 3)) for p in planner.points],
        "sampled_max_nearest_distance_m": maximum_distance,
        "coverage_guaranteed": maximum_distance <= config.receive_radius_min + 1e-6,
        "base_route_distance_m": route_distance,
        "base_route_time_s": route_distance / config.dog_speed,
        "worst_full_channel_scan_time_s": full_scan_time,
        "worst_discovery_stage_time_s": route_distance / config.dog_speed + full_scan_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CUMCM 2026 B 题问题 3 控制框架")
    parser.add_argument("--run", action="store_true", help="连接本机模拟器并执行")
    parser.add_argument("--robot-id", help="当前登录模拟器的参赛队号")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path, default=Path("P3_run_log.jsonl"))
    args = parser.parse_args()

    config = Problem3Config()
    if not args.run:
        print(json.dumps(run_offline_check(config), ensure_ascii=False, indent=2))
        print("离线自检完成；未连接模拟器。正式运行请添加 --run --robot-id <参赛队号>。")
        return
    if not args.robot_id:
        parser.error("使用 --run 时必须同时提供 --robot-id。")

    client = SimulatorClient(
        robot_id=args.robot_id,
        base_url=args.base_url,
        timeout_s=config.request_timeout_s,
        retries=config.request_retries,
        log_path=args.log,
    )
    controller = Problem3Controller(client, config)
    result = controller.run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
