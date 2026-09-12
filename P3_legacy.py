"""P3 重构前的实现备份；仅供迁移旧算法函数时参考。

CUMCM 2026 B 题问题 3：多干扰源的搜索、定位与清除框架。

总体思路
--------
1. 用原点和半径 ``1800*cos(30°)`` 的正六边形顶点构成七点覆盖。
   对尚未发现的频道完成七点扫描，可利用最小接收半径 1000 m 保证不漏检。
2. 每个已发现频道维护“目标位置—未知接收半径”联合后验；更新规则沿用
   :mod:`P2` 的离散状态与观测相容性判定。
3. 第一次方位观测后只调用一次 :mod:`P2` 选择第二检测点；两次及以上方位
   观测调用 :mod:`P1` 构造连续交会区域。区域不能清除时，到其最小包围圆
   圆心继续检测；半径不超过 20 m 时在圆心清除。
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Sequence, Any

import numpy as np

from P1 import Problem1Geometry, solve_problem_1_geometry
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
from simulator import SimulatorClient, SimulatorError
from utils import Point


ObservationKind = Literal["direction", "near", "no_signal"]
ActionKind = Literal["measure", "clear", "exit"]


# ============================================================
# 作用：集中保存问题 3 的物理参数、离散精度和运行保护参数。
# ============================================================
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

    # ============================================================
    # 作用：检查配置是否符合题目物理规则和程序运行要求。
    # ============================================================
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


# ============================================================
# 作用：保存某频道的一次测量位置、结果、时间和覆盖点编号。
# ============================================================
@dataclass(frozen=True)
class ObservationRecord:
    """某频道的一次检测记录。"""

    position: Point
    kind: ObservationKind
    bearing_deg: float | None
    virtual_time_s: float
    coverage_index: int | None = None


# ============================================================
# 作用：统一表示定位区域的圆心、样本半径和保守覆盖半径。
# ============================================================
@dataclass(frozen=True)
class ConservativeCircle:
    """离散点最小包围圆加网格单元半径后的保守清除圆。"""

    center: Point
    sample_radius: float
    conservative_radius: float


# ============================================================
# 作用：维护一个频道的全部观测、候选状态、P1 几何和清除状态。
# ============================================================
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
    clear_blocked_until_measurement: bool = False

    # ============================================================
    # 作用：从全部观测中提取可用于 P1/P2 的方向观测。
    # ============================================================
    @property
    def direction_observations(self) -> list[ObservationRecord]:
        return [item for item in self.observations if item.kind == "direction"]

    # ============================================================
    # 作用：记录该频道已经在哪个七点覆盖位置完成检测。
    # ============================================================
    def mark_coverage(self, index: int) -> None:
        self.coverage_visited.add(index)

    # ============================================================
    # 作用：判断该频道是否已经完成全部七点扫描。
    # ============================================================
    def coverage_complete(self, point_count: int) -> bool:
        return len(self.coverage_visited) == point_count

    # ============================================================
    # 作用：判断频道已经清除，或已通过七点无信号证明不存在目标。
    # ============================================================
    def certified_complete(self, point_count: int) -> bool:
        """已清除，或完整覆盖扫描后始终未发现该频道。"""
        return self.cleared or (
            not self.source_detected and self.coverage_complete(point_count)
        )

    # ============================================================
    # 作用：首次获得方向观测后，调用 P2 内部模型建立联合候选状态。
    # ============================================================
    def _initialize_states(
        self,
        first: ObservationRecord,
        config: Problem3Config,
        p2_config: Problem2Config | None = None,
    ) -> tuple[StateSample, ...]:
        assert first.kind == "direction" and first.bearing_deg is not None
        active_p2 = p2_config or config.p2
        first_point = _validate_inputs(
            (first.position.x, first.position.y), first.bearing_deg
        )
        region = _build_first_feasible_region(first_point, first.bearing_deg, active_p2)
        cells = _sample_target_cells(region, active_p2)
        states = _expand_receive_radius_states(cells, first_point, active_p2)
        if not states:
            raise RuntimeError(f"频道 {self.channel} 的首次观测未生成可行状态。")
        return tuple(states)

    # ============================================================
    # 作用：把 P3 测量记录转换成 P2 的观测对象。
    # ============================================================
    @staticmethod
    def _as_p2_observation(record: ObservationRecord) -> P2Observation:
        return P2Observation(record.kind, record.bearing_deg)

    # ============================================================
    # 作用：按新观测筛除不相容的“位置—接收半径”联合状态。
    # ============================================================
    def _filter_states(
        self,
        states: Sequence[StateSample],
        record: ObservationRecord,
        config: Problem3Config,
        p2_config: Problem2Config | None = None,
    ) -> tuple[StateSample, ...]:
        arrays = _state_arrays(states)
        active_p2 = p2_config or config.p2
        mask = _is_observation_compatible(
            arrays,
            record.position,
            self._as_p2_observation(record),
            active_p2,
        )
        return tuple(state for state, keep in zip(states, mask) if bool(keep))

    # ============================================================
    # 作用：将清除失败转化为“目标在失败点 20 米圆外”的约束。
    # ============================================================
    def _apply_failed_clear_constraints(
        self,
        states: Sequence[StateSample],
        config: Problem3Config,
    ) -> tuple[StateSample, ...]:
        """保留单元中仍可能位于所有失败清除圆之外的状态。"""
        return tuple(
            state
            for state in states
            if all(
                state.target.distance_to(point) + state.cell_radius
                > config.clear_radius + 1e-9
                for point in self.failed_clear_points
            )
        )

    # ============================================================
    # 作用：离散后验坍缩时，用更细网格重放全部历史观测。
    # ============================================================
    def _rebuild_states(self, config: Problem3Config) -> tuple[StateSample, ...]:
        """用更细的位置和接收半径网格重放全部历史约束。"""
        directions = self.direction_observations
        if not directions:
            return ()
        first = directions[0]
        for spacing_factor, radius_factor in ((0.5, 2), (0.25, 4)):
            refined_p2 = replace(
                config.p2,
                target_grid_spacing=config.p2.target_grid_spacing * spacing_factor,
                receive_radius_sample_count=max(
                    config.p2.receive_radius_sample_count * radius_factor,
                    config.p2.receive_radius_sample_count + 1,
                ),
            )
            posterior = self._initialize_states(first, config, refined_p2)
            for old_record in self.observations:
                if old_record is first or old_record.kind == "near":
                    continue
                posterior = self._filter_states(
                    posterior, old_record, config, refined_p2
                )
                if not posterior:
                    break
            posterior = self._apply_failed_clear_constraints(posterior, config)
            if posterior:
                return posterior
        return ()

    # ============================================================
    # 作用：加入一次新观测并更新频道发现状态和联合后验。
    # ============================================================
    def update(self, record: ObservationRecord, config: Problem3Config) -> None:
        """加入一次观测并更新后验；离散后验坍缩时显式报错。"""
        self.observations.append(record)
        if record.coverage_index is not None:
            self.mark_coverage(record.coverage_index)

        if record.kind == "near":
            self.source_detected = True
            self.pending_near_position = record.position
            self.clear_blocked_until_measurement = False
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
                    break
            posterior = self._apply_failed_clear_constraints(posterior, config)
            if not posterior:
                posterior = self._rebuild_states(config)
            if not posterior:
                # 两条以上示向度已经能由 P1 给出连续定位区域时，离散网格
                # 采不到狭小交集不代表观测矛盾。保留空离散集，后续由 P1
                # 的连续区域圆心和半径负责补测与清除判断。
                if self.p1_geometry_result(config) is not None:
                    self.states = ()
                    self.clear_blocked_until_measurement = False
                    return
                raise RuntimeError(
                    f"频道 {self.channel} 无法从历史观测重建后验；需要更细网格。"
                )
            self.states = posterior
            self.clear_blocked_until_measurement = False
            return

        posterior = self._filter_states(self.states, record, config)
        if not posterior:
            posterior = self._rebuild_states(config)
        posterior = self._apply_failed_clear_constraints(posterior, config)
        if not posterior:
            # P1 连续交会区域非空时，以连续几何结果为准，不能把离散采样
            # 漏点误判为真实观测不相容。
            if self.p1_geometry_result(config) is not None:
                self.states = ()
                self.clear_blocked_until_measurement = False
                return
            raise RuntimeError(
                f"频道 {self.channel} 的新观测与历史约束不相容；需要更细网格。"
            )
        self.states = posterior
        self.clear_blocked_until_measurement = False

    # ============================================================
    # 作用：给出当前定位区域的保守覆盖圆，供补测和清除决策使用。
    # ============================================================
    def conservative_circle(self, config: Problem3Config) -> ConservativeCircle | None:
        """优先返回 P1 连续区域覆盖圆；仅一次测向时使用 P2 离散后验圆。"""
        if self.pending_near_position is not None:
            return ConservativeCircle(
                self.pending_near_position,
                config.near_radius,
                config.near_radius,
            )
        p1_circle = self.p1_localization_circle(config)
        if p1_circle is not None:
            return p1_circle
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

    # ============================================================
    # 作用：调用 P1 构造两次及以上示向度的连续交会定位区域。
    # ============================================================
    def p1_geometry_result(
        self,
        config: Problem3Config,
    ) -> Problem1Geometry | None:
        data = [
            (item.position.x, item.position.y, float(item.bearing_deg))
            for item in self.direction_observations
            if item.bearing_deg is not None
        ]
        if len(data) < 2:
            return None
        try:
            return solve_problem_1_geometry(data, config.bearing_error_deg)
        except ValueError:
            return None

    # ============================================================
    # 作用：计算 P1 定位区域的最小包围圆，圆心作为第三及后续检测点。
    # ============================================================
    def p1_localization_circle(
        self,
        config: Problem3Config,
    ) -> ConservativeCircle | None:
        geometry = self.p1_geometry_result(config)
        if geometry is None:
            return None
        circle = _minimum_enclosing_circle(
            geometry.polygon.vertices,
            config.p2.random_seed,
        )
        return ConservativeCircle(
            center=circle.center,
            sample_radius=circle.radius,
            conservative_radius=circle.radius,
        )

    # ============================================================
    # 作用：检查当前定位区域能否被半径 20 米的圆安全覆盖。
    # ============================================================
    def can_clear(self, config: Problem3Config) -> bool:
        if self.clear_blocked_until_measurement:
            return False
        circle = self.conservative_circle(config)
        if circle is None or circle.conservative_radius > config.clear_radius:
            return False
        return all(
            circle.center.distance_to(point) > 1e-6
            for point in self.failed_clear_points
        )

    # ============================================================
    # 作用：记录清除失败、禁止原点重试，并强制先进行新测量。
    # ============================================================
    def register_clear_failure(self, point: Point, config: Problem3Config) -> None:
        """清除失败意味着目标与该点距离严格大于 20 m。"""
        self.pending_near_position = None
        if not any(point.distance_to(old) <= 1e-6 for old in self.failed_clear_points):
            self.failed_clear_points.append(point)
        self.clear_blocked_until_measurement = True
        if self.states:
            posterior = self._apply_failed_clear_constraints(self.states, config)
            if not posterior:
                posterior = self._rebuild_states(config)
            self.states = posterior

    # ============================================================
    # 作用：向日志或论文摘要提供 P1 的区域直径和直径圆覆盖结论。
    # ============================================================
    def p1_geometry_summary(
        self,
        config: Problem3Config,
    ) -> tuple[float, bool] | None:
        geometry = self.p1_geometry_result(config)
        if geometry is None:
            return None
        return geometry.diameter, geometry.diameter_circle_covers


# ============================================================
# 作用：生成原点加正六边形顶点的七点保证覆盖路线。
# ============================================================
class CoveragePlanner:
    """构造并管理原点加正六边形顶点的保证覆盖路线。"""

    # ============================================================
    # 作用：根据目标圆半径计算七个覆盖检测点。
    # ============================================================
    def __init__(self, config: Problem3Config) -> None:
        self.config = config
        rho = config.target_radius * math.cos(math.pi / 6.0)
        self.points = [Point(0.0, 0.0)] + [
            Point(rho * math.cos(k * math.pi / 3.0), rho * math.sin(k * math.pi / 3.0))
            for k in range(6)
        ]

    # ============================================================
    # 作用：确定某覆盖点仍需检测的未知频道，并使用蛇形频道顺序。
    # ============================================================
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

    # ============================================================
    # 作用：返回下一项尚未完成的“覆盖点—频道”扫描任务。
    # ============================================================
    def next_scan(
        self,
        beliefs: dict[int, ChannelBelief],
    ) -> tuple[int, Point, int] | None:
        for index, point in enumerate(self.points):
            channels = self.channels_for_point(index, beliefs)
            if channels:
                return index, point, channels[0]
        return None

    # ============================================================
    # 作用：用密集抽样复核七点布局的最大最近距离不超过 1000 米。
    # ============================================================
    def verify_dense(self, radial_steps: int = 90, angular_steps: int = 720) -> float:
        """离线抽样返回目标圆内点到最近覆盖点的最大距离。"""
        maximum = 0.0
        for radius in np.linspace(0.0, self.config.target_radius, radial_steps + 1):
            for angle in np.linspace(0.0, 2.0 * math.pi, angular_steps, endpoint=False):
                probe = Point(radius * math.cos(angle), radius * math.sin(angle))
                maximum = max(maximum, min(probe.distance_to(p) for p in self.points))
        return maximum


# ============================================================
# 作用：选择单频道的第二检测点及第三次以后的区域中心检测点。
# ============================================================
class LocalizationPlanner:
    """第一次测向后调用 P2；此后返回 P1 定位区域的中心。"""

    # ============================================================
    # 作用：保存定位阶段所使用的问题配置。
    # ============================================================
    def __init__(self, config: Problem3Config) -> None:
        self.config = config

    # ============================================================
    # 作用：清除失败后换点补测，避免在同一失败位置形成死循环。
    # ============================================================
    def _recovery_probe(self, belief: ChannelBelief, current: Point) -> Point:
        """清除失败后横向移动再测一次，避免重复访问失败点。"""
        anchor = belief.failed_clear_points[-1]
        directions = belief.direction_observations
        angle = math.radians(
            directions[-1].bearing_deg
            if directions and directions[-1].bearing_deg is not None
            else 0.0
        )
        step = min(120.0, max(60.0, 2.0 * self.config.clear_radius))
        perpendicular = Point(-math.sin(angle), math.cos(angle))
        candidates = [
            Point(
                anchor.x + sign * step * perpendicular.x,
                anchor.y + sign * step * perpendicular.y,
            )
            for sign in (-1.0, 1.0)
        ]
        return min(candidates, key=current.distance_to)

    # ============================================================
    # 作用：首次示向后调用 P2；两次及以上示向后返回 P1 区域中心。
    # ============================================================
    def recommend(self, belief: ChannelBelief, current: Point) -> Point:
        if belief.clear_blocked_until_measurement and belief.failed_clear_points:
            return self._recovery_probe(belief, current)

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

        # 第二次及后续示向后不再调用 P2，直接到 P1 连续定位区域中心检测。
        p1_circle = belief.p1_localization_circle(self.config)
        if p1_circle is not None:
            return p1_circle.center

        circle = belief.conservative_circle(self.config)
        if circle is None:
            raise RuntimeError(f"频道 {belief.channel} 尚无足够信息生成定位点。")
        return circle.center


# ============================================================
# 作用：统一描述调度器输出的测量、清除或退出动作。
# ============================================================
@dataclass(frozen=True)
class PlannedAction:
    kind: ActionKind
    point: Point | None = None
    channel: int | None = None
    coverage_index: int | None = None
    reason: str = ""


# ============================================================
# 作用：在立即清除、七点搜索、补充定位和退出之间选择下一动作。
# ============================================================
class GlobalScheduler:
    """在清除、保证覆盖扫描和补充定位之间选择下一动作。"""

    # ============================================================
    # 作用：装配覆盖搜索器和单频道定位器。
    # ============================================================
    def __init__(
        self,
        config: Problem3Config,
        coverage: CoveragePlanner,
        localization: LocalizationPlanner,
    ) -> None:
        self.config = config
        self.coverage = coverage
        self.localization = localization

    # ============================================================
    # 作用：按“可清除→覆盖搜索→中心补测→退出”的优先级调度。
    # ============================================================
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


# ============================================================
# 作用：执行模拟器主循环并把反馈写回各频道状态。
# ============================================================
class Problem3Controller:
    """问题 3 主控制循环。"""

    # ============================================================
    # 作用：创建覆盖规划器、定位规划器、调度器和 20 个频道状态。
    # ============================================================
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

    # ============================================================
    # 作用：在现实运行时间耗尽前停止产生新动作。
    # ============================================================
    def _check_real_time(self) -> None:
        if time.monotonic() >= self.real_deadline_s - self.config.real_time_reserve_s:
            raise SimulatorError("现实时间即将耗尽，已停止产生新动作。")

    # ============================================================
    # 作用：执行一次测量，将模拟器反馈转换为观测并更新频道后验。
    # ============================================================
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

    # ============================================================
    # 作用：执行一次清除，并处理成功或失败约束。
    # ============================================================
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

    # ============================================================
    # 作用：进入测试并循环执行调度动作，完成后主动退出。
    # ============================================================
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

    # ============================================================
    # 作用：汇总清除数量、频道、虚拟时间和程序运行时间。
    # ============================================================
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


# ============================================================
# 作用：离线检查七点覆盖距离及发现阶段的基础时间。
# ============================================================
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


# ============================================================
# 作用：解析命令行参数，选择离线检查或连接模拟器运行。
# ============================================================
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
