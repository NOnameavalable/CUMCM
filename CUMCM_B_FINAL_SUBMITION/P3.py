"""CUMCM 2026 B 题问题 3 的状态、任务与控制器结构。

覆盖、定位、多余区域清除以及模拟器主循环等算法函数统一放在文件末尾的
“算法函数区”。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from python_tsp.heuristics import solve_tsp_local_search

from CUMCM_B_FINAL_SUBMITION.P1 import Problem1Geometry, solve_problem_1_geometry
from CUMCM_B_FINAL_SUBMITION.P2 import (
    CandidateMetrics,
    Problem2Config,
    Problem2Result,
    evaluate_problem_2_candidate,
)
from CUMCM_B_FINAL_SUBMITION.P2_cache import get_problem_2_result
from CUMCM_B_FINAL_SUBMITION.simulator import SimulatorClient, SimulatorError
from CUMCM_B_FINAL_SUBMITION.utils import Point, Region, point_key


ObservationKind = Literal["direction", "near", "no_signal"]
ActionKind = Literal["measure", "clear", "exit"]
P2PlanningMode = Literal["legacy_single", "bilateral_independent", "bilateral_shared"]


# 类定义
@dataclass(frozen=True)
class Problem3Config:
    """集中保存问题 3 使用的物理常量、算法配置和运行保护参数。

    Attributes:
        target_radius: 干扰源目标圆域的半径，单位为米。
        receive_radius_min: 干扰源有效接收半径的题设下限，单位为米。
        receive_radius_max: 干扰源有效接收半径的题设上限，单位为米。
        bearing_error_deg: 测向机示向度的最大绝对误差，单位为度。
        near_radius: 测量结果变为 ``near`` 的距离阈值，单位为米。
        clear_radius: 光学精确定位和清除能够覆盖的半径，单位为米。
        channel_count: 测向机支持的频道总数，题目规定为 20。
        source_count_max: 一次测试中干扰源数量的已知上限。
        dog_speed: 机器狗沿直线移动的速度，单位为米每秒。
        switch_time: 两次测量频道不同时的频道切换耗时，单位为秒。
        max_actions: 单次运行允许执行的最大任务数，用于防止死循环。
        request_timeout_s: 单次模拟器 HTTP 请求的超时时间，单位为秒。
        request_retries: 网络异常时同一幂等请求允许尝试的总次数。
        real_time_reserve_s: 现实截止时间前停止生成新动作的安全余量。
        point_precision: 比较任务点或清除失败点时使用的坐标去重精度。
        p2: 问题 2 第二检测点算法使用的完整配置。
    """

    target_radius: float = 1800.0
    receive_radius_min: float = 1000.0
    receive_radius_max: float = 1500.0
    bearing_error_deg: float = 1.0
    near_radius: float = 5.0
    clear_radius: float = 20.0
    channel_count: int = 20
    source_count_max: int = 16
    dog_speed: float = 5.0
    switch_time: float = 1.0
    max_actions: int = 2000
    request_timeout_s: float = 5.0
    request_retries: int = 3
    real_time_reserve_s: float = 10.0
    point_precision: float = 1e-6
    route_search_time_s: float = 0.1
    p2_planning_mode: P2PlanningMode = "bilateral_shared"
    shared_planning_time_s: float = 1.0
    shared_grid_spacing: float = 100.0
    shared_refined_spacing: float = 25.0
    shared_refinement_radius: float = 100.0
    shared_min_radius_reduction: float = 0.05
    p2: Problem2Config = field(default_factory=Problem2Config)

    def __post_init__(self) -> None:
        """校验各配置项之间的物理关系和允许范围。

        Raises:
            ValueError: 频道数、半径、速度、重试次数或坐标精度不符合要求。
        """
        if self.channel_count != 20:
            raise ValueError("题目规定频道数必须为 20。")
        if not 1 <= self.source_count_max <= self.channel_count:
            raise ValueError("干扰源数量上限必须位于频道数范围内。")
        if self.target_radius <= 0.0 or self.receive_radius_min <= 0.0:
            raise ValueError("目标区域和接收半径必须为正数。")
        if self.receive_radius_min > self.receive_radius_max:
            raise ValueError("接收半径下限不能大于上限。")
        if not 0.0 < self.near_radius <= self.clear_radius:
            raise ValueError("near 半径必须为正且不能大于清除半径。")
        if self.dog_speed <= 0.0 or self.switch_time < 0.0:
            raise ValueError("移动速度必须为正，频道切换时间不能为负。")
        if self.max_actions < 1 or self.request_retries < 1:
            raise ValueError("动作上限和重试次数至少为 1。")
        if self.point_precision <= 0.0:
            raise ValueError("坐标去重精度必须为正数。")
        if not math.isfinite(self.route_search_time_s) or self.route_search_time_s <= 0:
            raise ValueError("路径规划时间预算必须为有限正数。")
        if self.p2_planning_mode not in (
            "legacy_single", "bilateral_independent", "bilateral_shared"
        ):
            raise ValueError("未知 P2 规划模式。")
        if self.shared_planning_time_s <= 0 or self.shared_grid_spacing <= 0:
            raise ValueError("共享规划预算和网格间距必须为正数。")
        if self.shared_refined_spacing <= 0 or self.shared_refinement_radius <= 0:
            raise ValueError("共享加密间距和半径必须为正数。")
        if not 0.0 <= self.shared_min_radius_reduction < 1.0:
            raise ValueError("共享点最小半径缩减比例必须位于 [0, 1)。")


class TaskType(str, Enum):
    """任务池中可能出现的检测、清除和退出任务类型。

    Attributes:
        COVERAGE_MEASURE: 在七点覆盖位置检测一个尚未发现的频道。
        SECOND_MEASURE: 执行 P2 为某频道生成的第二次测量。
        FOLLOW_UP_MEASURE: 候选区域直径较大时执行第三次或后续测量。
        CENTER_MEASURE: 在直径圆圆心进行精细排查测量。
        CENTER_CLEAR: 在能够保证覆盖时前往直径圆圆心清除。
        RESIDUAL_CLEAR: 进入直径圆之外的多余候选区域直接清除。
        EXIT: 所有频道处理完成后主动退出模拟器。
    """

    COVERAGE_MEASURE = "coverage_measure"
    SECOND_MEASURE = "second_measure"
    FOLLOW_UP_MEASURE = "follow_up_measure"
    CENTER_MEASURE = "center_measure"
    CENTER_CLEAR = "center_clear"
    RESIDUAL_CLEAR = "residual_clear"
    EXIT = "exit"

    @property
    def is_measurement(self) -> bool:
        """判断该任务是否需要调用模拟器的 ``/measure`` 接口。

        Returns:
            测量类任务返回 ``True``，清除和退出任务返回 ``False``。
        """
        return self in {
            TaskType.COVERAGE_MEASURE,
            TaskType.SECOND_MEASURE,
            TaskType.FOLLOW_UP_MEASURE,
            TaskType.CENTER_MEASURE,
        }

    @property
    def action_kind(self) -> ActionKind:
        """把细分任务类型映射为模拟器支持的基础动作类型。

        Returns:
            ``measure``、``clear`` 或 ``exit`` 中的一个。
        """
        if self.is_measurement:
            return "measure"
        if self is TaskType.EXIT:
            return "exit"
        return "clear"


@dataclass(frozen=True)
class ObservationRecord:
    """保存某频道一次测量的请求位置和模拟器响应。

    Attributes:
        position: 执行本次测量时机器狗所在的全局坐标。
        kind: 测量结果，取 ``direction``、``near`` 或 ``no_signal``。
        bearing_deg: ``direction`` 结果对应的示向度；其他结果为 ``None``。
        virtual_time_s: 本次测量完成后的模拟器虚拟时刻，单位为秒。
        coverage_index: 本次测量对应的七点覆盖编号；定位测量时为 ``None``。
    """

    position: Point
    kind: ObservationKind
    bearing_deg: float | None
    virtual_time_s: float
    coverage_index: int | None = None


@dataclass
class ChannelBelief:
    """保存单个频道的观测事实、连续候选区域和清除状态。

    该类只描述“这个频道当前已知什么”。任务的生成、删除和全局访问顺序由
    :class:`TaskPlanner` 负责。

    Attributes:
        channel: 频道编号，范围为 1 到 20。
        observations: 该频道按执行顺序保存的全部测量记录。
        coverage_visited: 已经对该频道完成测量的七点覆盖编号集合。
        source_detected: 是否曾经在该频道得到 ``direction`` 或 ``near``。
        cleared: 该频道对应的干扰源是否已经清除成功。
        second_measure_point: P2 为该频道计算的第二检测点。
        second_measure_fallback_point: 首选点无信号时使用的保证接收候选点。
        second_measure_completed: P2 第二检测任务是否已经执行。
        candidate_region: 当前所有有效示向约束形成的连续候选区域。
        p1_geometry: P1 最近一次计算的直径、端点和直径圆覆盖结果。
        pending_center_clear_point: 圆心测量真实完成后等待清除的原圆心。
        remaining_clear_region: 圆心清除失败后需要直接进入清除的多余区域。
        remaining_clear_point: 已从多余区域中选定的直接清除位置。
        failed_clear_points: 已调用 ``/clear`` 但没有发现目标的位置集合。
        revision: 状态版本号；任务通过它判断自己是否已经过期。
    """

    channel: int
    observations: list[ObservationRecord] = field(default_factory=list)
    coverage_visited: set[int] = field(default_factory=set)
    source_detected: bool = False
    cleared: bool = False
    second_measure_point: Point | None = None
    second_measure_fallback_point: Point | None = None
    second_measure_completed: bool = False
    p2_result: Problem2Result | None = field(default=None, repr=False)
    p2_result_revision: int = -1
    shared_candidates_disabled: bool = False
    candidate_region: Region | None = None
    p1_geometry: Problem1Geometry | None = None
    pending_center_clear_point: Point | None = None
    center_clear_allows_residual: bool = False
    remaining_clear_region: Region | None = None
    remaining_clear_point: Point | None = None
    failed_clear_points: list[Point] = field(default_factory=list)
    revision: int = 0

    @property
    def direction_observations(self) -> list[ObservationRecord]:
        """提取能够作为 P1 输入的全部方向观测。

        Returns:
            按原测量顺序排列、结果类型为 ``direction`` 的记录列表。
        """
        return [record for record in self.observations if record.kind == "direction"]

    def register_observation(self, record: ObservationRecord) -> None:
        """登记一次测量事实并递增频道状态版本。

        本方法更新覆盖访问记录和是否发现信号，不负责计算 P1/P2 结果，也不
        直接创建后续任务。

        Args:
            record: 已校验的模拟器测量记录。
        """
        self.observations.append(record)
        if record.coverage_index is not None:
            self.coverage_visited.add(record.coverage_index)
        if record.kind in ("direction", "near"):
            self.source_detected = True
        self.revision += 1

    def register_clear_success(self) -> None:
        """记录清除成功并使该频道所有后续清除区域失效。"""
        self.cleared = True
        self.pending_center_clear_point = None
        self.center_clear_allows_residual = False
        self.remaining_clear_region = None
        self.remaining_clear_point = None
        self.revision += 1

    def register_clear_failure(self, point: Point) -> None:
        """记录一次清除失败的位置并递增状态版本。

        重复位置会按照 ``point_key`` 的坐标精度去重。

        Args:
            point: 本次未能清除目标的圆心位置。
        """
        key = point_key(point)
        if all(point_key(old) != key for old in self.failed_clear_points):
            self.failed_clear_points.append(point)
        self.revision += 1

    def coverage_complete(self, point_count: int) -> bool:
        """判断该频道是否已经访问全部保证覆盖点。

        Args:
            point_count: 当前覆盖方案包含的检测点总数。

        Returns:
            已访问编号数量等于覆盖点总数时返回 ``True``。
        """
        return len(self.coverage_visited) == point_count

    def certified_complete(self, point_count: int) -> bool:
        """判断该频道是否已经得到可用于退出的终态证明。

        Args:
            point_count: 当前覆盖方案包含的检测点总数。

        Returns:
            干扰源已清除，或从未发现信号且完成全部覆盖测量时返回 ``True``。
        """
        return self.cleared or (
            not self.source_detected and self.coverage_complete(point_count)
        )


@dataclass(frozen=True)
class RouteTask:
    """描述任务池中的一个待访问位置及需要执行的动作。

    Attributes:
        task_id: 当前运行内唯一的任务标识，用于删除和替换任务。
        task_type: 任务的具体业务类型。
        point: 机器狗需要到达的位置；退出任务可以为 ``None``。
        channel: 测量或清除的目标频道；退出任务可以为 ``None``。
        coverage_index: 七点覆盖任务对应的点编号，其他任务为 ``None``。
        channel_revision: 创建任务时的频道版本，用于淘汰过期任务。
        reason: 任务产生原因，供日志、调试和论文演练分析使用。
    """

    task_id: str
    task_type: TaskType
    point: Point | None
    channel: int | None
    coverage_index: int | None = None
    channel_revision: int = 0
    reason: str = ""

    @property
    def action_kind(self) -> ActionKind:
        """返回向模拟器发送该任务时使用的基础动作类型。

        Returns:
            ``measure``、``clear`` 或 ``exit``。
        """
        return self.task_type.action_kind


@dataclass
class SharedCandidate:
    """统一世界坐标中的共享二测候选位置。"""

    point: Point
    channels: set[int]
    metrics: dict[int, CandidateMetrics]
    revisions: dict[int, int]


class TaskPlanner:
    """维护动态任务池，用 python-tsp 优化开放路线并滚动执行第一项。

    TSP 优化位置间的移动时间。同位置先清除，再优先检测当前频道；
    清除不改变测向机频道。频道切换不伪装成普通的两点距离。

    Attributes:
        config: 问题 3 的统一配置。
        coverage_points: 保证覆盖目标圆域的固定七个测量点。
        tasks: 以任务 ID 为键保存的全部当前有效待执行任务。
        _sequence: 生成运行内唯一任务 ID 使用的递增序号。
    """

    def __init__(self, config: Problem3Config) -> None:
        """初始化空任务池并预先生成七点覆盖坐标。

        Args:
            config: 问题 3 的统一配置。
        """
        self.config = config
        self.coverage_points = generate_coverage_points(config)
        self.tasks: dict[str, RouteTask] = {}
        self._sequence = 0
        self._route_keys: list[tuple] = []

    def next_task_id(self, channel: int | None, task_type: TaskType) -> str:
        """生成一个带频道、任务类型和递增序号的唯一任务 ID。

        Args:
            channel: 任务所属频道；全局任务传入 ``None``。
            task_type: 需要创建的任务类型。

        Returns:
            当前 ``TaskPlanner`` 实例内唯一的字符串标识。
        """
        self._sequence += 1
        channel_text = "global" if channel is None else str(channel)
        return f"{channel_text}:{task_type.value}:{self._sequence}"

    def add_task(self, task: RouteTask) -> None:
        """把任务加入任务池，相同 ID 的旧任务会被替换。

        Args:
            task: 需要登记的动态访问任务。
        """
        self.tasks[task.task_id] = task

    def remove_task(self, task_id: str) -> None:
        """按 ID 删除任务；任务不存在时保持任务池不变。

        Args:
            task_id: 需要删除的任务标识。
        """
        self.tasks.pop(task_id, None)

    def remove_channel_tasks(self, channel: int) -> None:
        """删除属于指定频道的全部待执行任务。

        Args:
            channel: 需要清空任务的频道编号。
        """
        self.tasks = {
            task_id: task
            for task_id, task in self.tasks.items()
            if task.channel != channel
        }

    def replace_channel_tasks(self, channel: int, tasks: list[RouteTask]) -> None:
        """用一组新任务完整替换指定频道的旧任务。

        Args:
            channel: 需要重建任务的频道编号。
            tasks: 替换后属于该频道的全部任务。

        Raises:
            ValueError: 新任务中包含其他频道的任务。
        """
        for task in tasks:
            if task.channel != channel:
                raise ValueError("替换频道任务时发现不属于该频道的任务。")
        self.remove_channel_tasks(channel)
        for task in tasks:
            self.add_task(task)

    def discard_stale_tasks(self, beliefs: dict[int, ChannelBelief]) -> None:
        """根据频道版本和清除状态删除已经失效的任务。

        Args:
            beliefs: 频道编号到最新频道状态的映射。
        """
        self.tasks = {
            task_id: task
            for task_id, task in self.tasks.items()
            if task.channel is None
            or (
                task.channel in beliefs
                and not beliefs[task.channel].cleared
                and task.channel_revision == beliefs[task.channel].revision
            )
        }

    def plan_route(
        self,
        current_position: Point,
        current_channel: int,
    ) -> list[RouteTask]:
        """规划全部当前有效任务，返回开放路线，不提前推进频道状态。

        同坐标任务合并为一个位置。节点0固定为当前位置，矩阵第0列
        置零，免除回程费用。以上轮仍有效的位置顺序热启动2-opt，
        并尝试反向初始顺序；两次搜索共享预算。结果是近似解。
        在当前位置的任务先执行，远处任务每次反馈后重新规划。
        """
        if not self.tasks:
            raise RuntimeError("任务池为空，无法选择下一任务。")
        non_exit = [task for task in self.tasks.values() if task.action_kind != "exit"]
        if not non_exit:
            self._route_keys = []
            return sorted(self.tasks.values(), key=lambda task: task.task_id)

        groups: dict[tuple, list[RouteTask]] = {}
        for task in non_exit:
            assert task.point is not None and task.channel is not None
            key = point_key(task.point, self.config.point_precision)
            groups.setdefault(key, []).append(task)
        current_key = point_key(current_position, self.config.point_precision)
        remote = set(groups) - {current_key}
        keys = [key for key in self._route_keys if key in remote]
        keys.extend(sorted(remote - set(keys)))

        if len(keys) > 1:
            points = [current_position] + [groups[key][0].point for key in keys]
            coords = np.array([(point.x, point.y) for point in points])
            matrix = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
            matrix /= self.config.dog_speed
            matrix[:, 0] = 0.0
            initial = list(range(len(points)))
            best_order = initial
            best_cost = float(matrix[initial, initial[1:] + [0]].sum())
            for seed in (initial, [0] + initial[:0:-1]):
                order, cost = solve_tsp_local_search(
                    matrix,
                    x0=seed,
                    perturbation_scheme="two_opt",
                    max_processing_time=self.config.route_search_time_s / 2.0,
                )
                if cost < best_cost:
                    best_order, best_cost = order, cost
            keys = [keys[index - 1] for index in best_order[1:]]

        self._route_keys = keys
        visit_keys = ([current_key] if current_key in groups else []) + keys
        route = []
        for key in visit_keys:
            tasks = sorted(groups[key], key=lambda task: (
                task.task_type.is_measurement,
                task.channel != current_channel,
                task.channel,
                task.task_id,
            ))
            route.extend(tasks)
            for task in tasks:
                if task.task_type.is_measurement:
                    current_channel = task.channel
        return route

    def choose_next(
        self,
        current_position: Point,
        current_channel: int,
    ) -> RouteTask:
        """持续处理当前点的访问批次，批次耗尽后才调用TSP。

        每次从最新任务池提取原地任务，不缓存任务对象，保证反馈产生的
        原地清除立即加入、被取消的任务不会继续执行。调用方在每个动作
        前负责清理失效任务和检查时间、退出条件。
        """
        current_key = point_key(current_position, self.config.point_precision)
        local_tasks = [
            task for task in self.tasks.values()
            if task.point is not None
            and task.action_kind != "exit"
            and point_key(task.point, self.config.point_precision) == current_key
        ]
        if local_tasks:
            return min(local_tasks, key=lambda task: (
                task.task_type.is_measurement,
                task.channel != current_channel,
                task.channel,
                task.task_id,
            ))
        return self.plan_route(current_position, current_channel)[0]


class Problem3Controller:
    """连接模拟器、20 个频道状态和动态任务规划器。

    Attributes:
        client: 负责串行 HTTP 请求、幂等重试和位置维护的模拟器客户端。
        config: 问题 3 的统一配置。
        beliefs: 频道编号到该频道最新状态的映射。
        task_planner: 保存全部待执行任务并选择下一任务的规划器。
        wall_start_s: 成功进入测试前记录的本机单调时钟时刻。
        real_deadline_s: 根据 ``/enter`` 响应计算出的本次现实截止时刻。
        actions_executed: 本次运行已经完成的测量和清除任务数量。
    """

    def __init__(
        self,
        client: SimulatorClient,
        config: Problem3Config | None = None,
    ) -> None:
        """创建频道状态、任务规划器和运行计数器。

        Args:
            client: 已配置参赛队号和模拟器地址的客户端。
            config: 可选的问题配置；省略时使用默认配置。
        """
        self.client = client
        self.config = config or Problem3Config()
        self.beliefs = {
            channel: self.create_channel_belief(channel)
            for channel in range(1, self.config.channel_count + 1)
        }
        self.task_planner = TaskPlanner(self.config)
        self.wall_start_s = 0.0
        self.real_deadline_s = math.inf
        self.actions_executed = 0
        self.measurement_actions = 0
        self.second_measurement_actions = 0
        self.movement_distance_m = 0.0
        self.shared_planning_time_s = 0.0
        self.shared_planned_distance_saved_m = 0.0
        self.shared_assignment_events: list[dict[str, Any]] = []

    def create_channel_belief(self, channel: int) -> ChannelBelief:
        """创建频道状态；子类可扩展状态而无需复制控制器初始化。"""
        return ChannelBelief(channel)

    def refresh_channel(self, belief: ChannelBelief) -> None:
        """根据频道状态重建任务。"""
        refresh_channel_tasks(belief, self.task_planner, self.config)

    def process_measurement(
        self,
        belief: ChannelBelief,
        task: RouteTask,
        record: ObservationRecord,
    ) -> None:
        """处理P3测量后的策略状态。"""
        if task.task_type is TaskType.SECOND_MEASURE:
            fallback = belief.second_measure_fallback_point
            if record.kind == "no_signal" and "共享" in task.reason:
                belief.shared_candidates_disabled = True
                anomaly = {
                    "type": "shared_guaranteed_signal_miss",
                    "channel": task.channel,
                    "point": (task.point.x, task.point.y),
                }
                events = getattr(self, "shared_assignment_events", None)
                if events is not None:
                    events.append(anomaly)
                recorder = getattr(self.client, "record_planning_event", None)
                if callable(recorder):
                    recorder(anomaly)
            if (
                record.kind == "no_signal"
                and fallback is not None
                and point_key(fallback, self.config.point_precision)
                != point_key(task.point, self.config.point_precision)
            ):
                belief.second_measure_point = fallback
                belief.second_measure_fallback_point = None
                belief.second_measure_completed = False
            else:
                belief.second_measure_completed = True
        elif task.task_type in (TaskType.CENTER_MEASURE, TaskType.FOLLOW_UP_MEASURE):
            if record.kind == "no_signal":
                raise RuntimeError(
                    f"频道 {task.channel} 在 P1 候选区域直径圆圆心未收到信号，"
                    "与有效接收距离约束矛盾。"
                )
            if task.task_type is TaskType.CENTER_MEASURE:
                belief.pending_center_clear_point = task.point
                belief.center_clear_allows_residual = True

    def process_clear_failure(
        self,
        belief: ChannelBelief,
        task: RouteTask,
    ) -> None:
        """处理P3清除失败后的残余区域。"""
        if task.task_type is TaskType.RESIDUAL_CLEAR:
            raise RuntimeError(
                f"频道 {task.channel} 在多余区域直接清除失败；"
                "当前简化策略未定义后续恢复。"
            )
        if not belief.center_clear_allows_residual:
            raise RuntimeError(
                f"频道 {task.channel} 的保证清除失败，与候选区域约束矛盾。"
            )
        if belief.candidate_region is None:
            raise RuntimeError("圆心清除失败时缺少候选区域。")
        remaining, clear_point = select_residual_clear_point(
            belief.candidate_region,
            task.point,
            self.config.clear_radius,
        )
        belief.pending_center_clear_point = None
        belief.center_clear_allows_residual = False
        belief.remaining_clear_region = remaining
        belief.remaining_clear_point = clear_point

    def optimize_shared_measurements(self) -> dict[str, Any]:
        """执行P3共享二测规划；P4可替换候选评价模型。"""
        return optimize_shared_second_measurements(
            self.beliefs,
            self.task_planner,
            self.client.position,
            self.client.current_channel,
        )

    def _check_real_time(self) -> None:
        """检查剩余现实时间是否仍足以安全发起下一动作。

        Raises:
            SimulatorError: 当前时刻已经进入截止时间前的安全保留区。
        """
        if time.monotonic() >= self.real_deadline_s - self.config.real_time_reserve_s:
            raise SimulatorError("现实时间即将耗尽，已停止产生新动作。")

    def _handle_measure(self, task: RouteTask) -> None:
        """执行测量任务、登记响应并触发该频道任务重建。

        Args:
            task: 位置和频道均完整的测量任务。

        Raises:
            AssertionError: 任务缺少位置或频道。
            SimulatorError: 模拟器返回未知的测量结果。
            RuntimeError: P1 圆心测量结果与有效接收距离约束矛盾。
        """
        assert task.point is not None and task.channel is not None
        movement = getattr(self.client, "position", task.point).distance_to(task.point)
        response = self.client.measure(task.point, task.channel)
        self.movement_distance_m = getattr(self, "movement_distance_m", 0.0) + movement
        self.measurement_actions = getattr(self, "measurement_actions", 0) + 1
        if task.task_type is TaskType.SECOND_MEASURE:
            self.second_measurement_actions = getattr(
                self, "second_measurement_actions", 0
            ) + 1
        kind = response.get("measure_result")
        if kind not in ("direction", "near", "no_signal"):
            raise SimulatorError(f"未知 measure_result：{kind!r}")
        record = ObservationRecord(
            position=task.point,
            kind=kind,
            bearing_deg=float(response["svd_deg"]) if kind == "direction" else None,
            virtual_time_s=float(response["virtual_time_s"]),
            coverage_index=task.coverage_index,
        )
        belief = self.beliefs[task.channel]
        belief.register_observation(record)
        if isinstance(self, Problem3Controller):
            self.process_measurement(belief, task, record)
        else:
            Problem3Controller.process_measurement(self, belief, task, record)
        self.task_planner.remove_task(task.task_id)
        if isinstance(self, Problem3Controller):
            self.refresh_channel(belief)
        else:
            refresh_channel_tasks(belief, self.task_planner, self.config)
        if belief.p2_result is not None and belief.p2_result_revision == belief.revision:
            recorder = getattr(self.client, "record_planning_event", None)
            if callable(recorder):
                recorder({
                    "type": "p2_result",
                    "channel": belief.channel,
                    "source": belief.p2_result.result_source,
                    "first_position": (belief.p2_result.first_position.x, belief.p2_result.first_position.y),
                    "bearing_deg": belief.p2_result.first_bearing_deg,
                })

    def _handle_clear(self, task: RouteTask) -> None:
        """执行清除任务并处理成功或失败后的频道状态。

        清除成功时删除该频道所有剩余任务；清除失败时记录失败圆心并重新生成
        该频道任务。

        Args:
            task: 位置和频道均完整的清除任务。

        Raises:
            AssertionError: 任务缺少位置或频道。
            SimulatorError: 模拟器返回未知的清除结果。
            RuntimeError: 保证清除失败，或多余区域清除失败且无恢复策略。
        """
        assert task.point is not None and task.channel is not None
        movement = getattr(self.client, "position", task.point).distance_to(task.point)
        response = self.client.clear(task.point, task.channel)
        self.movement_distance_m = getattr(self, "movement_distance_m", 0.0) + movement
        belief = self.beliefs[task.channel]
        result = response.get("clear_result")
        self.task_planner.remove_task(task.task_id)
        if result == "success":
            belief.register_clear_success()
            self.task_planner.remove_channel_tasks(task.channel)
            return
        if result == "no_target_in_range":
            belief.register_clear_failure(task.point)
            if isinstance(self, Problem3Controller):
                self.process_clear_failure(belief, task)
                self.refresh_channel(belief)
            else:
                Problem3Controller.process_clear_failure(self, belief, task)
                refresh_channel_tasks(belief, self.task_planner, self.config)
            return
        raise SimulatorError(f"未知 clear_result：{result!r}")

    def run(self) -> dict[str, Any]:
        """运行问题 3 主循环。

        Returns:
            本次测试的清除数量、虚拟时间和现实运行时间统计。

        Raises:
            RuntimeError: 动作达到上限或任务池意外为空。
        """
        return run_controller(self)

    def summary(self, exit_reason: str) -> dict[str, Any]:
        """汇总当前运行的清除结果和时间统计。

        Args:
            exit_reason: 模拟器返回或控制器记录的退出原因。

        Returns:
            包含退出原因、已清除频道、虚拟时间、平均时间和动作数的字典。
        """
        cleared = sum(belief.cleared for belief in self.beliefs.values())
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
            "program_runtime_s": time.monotonic() - self.wall_start_s,
            "actions_executed": self.actions_executed,
            "measurement_actions": self.measurement_actions,
            "second_measurement_actions": self.second_measurement_actions,
            "movement_distance_m": self.movement_distance_m,
            "movement_time_s": self.movement_distance_m / self.config.dog_speed,
            "shared_planning_time_s": self.shared_planning_time_s,
            "shared_planned_distance_saved_m": self.shared_planned_distance_saved_m,
            "shared_assignment_events": self.shared_assignment_events,
        }


# =============================================================================
# 1. 全局搜索与覆盖算法
# =============================================================================


def generate_coverage_points(config: Problem3Config) -> list[Point]:
    """生成原点及以内接正六边形各边为弦的检测圆内侧圆心。

    检测圆使用最小接收半径，沿原边中点方向向内偏移。
    r >= R/2 才能以边长 R 为弦；r >= R 时原点已能覆盖整个
    目标圆域，直接返回原点，避免产生负径向坐标。
    """
    radius = config.target_radius
    receive_radius = config.receive_radius_min
    if receive_radius < radius / 2.0:
        raise ValueError("最小接收半径不足，无法以内接正六边形边为弦。")
    if receive_radius >= radius:
        return [Point(0.0, 0.0)]
    offset = math.sqrt(receive_radius ** 2 - (radius / 2.0) ** 2)
    rho = radius * math.cos(math.pi / 6.0) - offset
    return [Point(0.0, 0.0)] + [
        Point(
            rho * math.cos(index * math.pi / 3.0),
            rho * math.sin(index * math.pi / 3.0),
        )
        for index in range(6)
    ]


# =============================================================================
# 2. 频道定位与任务生成算法
# =============================================================================


def refresh_channel_tasks(
    belief: ChannelBelief,
    planner: TaskPlanner,
    config: Problem3Config,
) -> None:
    """根据频道观测阶段，用 P2 或 P1 重建该频道的待执行任务。

    第一次方向观测后由 P2 生成第二检测点；两次及以上方向观测由
    P1 计算连续候选区域，并根据直径和直径圆覆盖性生成清除、圆心
    测量或后续测量任务。小直径但直径圆未覆盖时，圆心测量完成后
    才创建原圆心清除任务。
    """

    def make_task(task_type: TaskType, point: Point, reason: str) -> RouteTask:
        return RouteTask(
            task_id=planner.next_task_id(belief.channel, task_type),
            task_type=task_type,
            point=point,
            channel=belief.channel,
            channel_revision=belief.revision,
            reason=reason,
        )

    if belief.cleared:
        planner.remove_channel_tasks(belief.channel)
        return

    latest = belief.observations[-1] if belief.observations else None

    # near 已把目标限制在当前检测点 5 米内，可以立即清除。
    if latest is not None and latest.kind == "near":
        belief.pending_center_clear_point = None
        belief.center_clear_allows_residual = False
        belief.remaining_clear_region = None
        belief.remaining_clear_point = None
        planner.replace_channel_tasks(
            belief.channel,
            [make_task(TaskType.CENTER_CLEAR, latest.position, "near结果保证可清除")],
        )
        return

    # 原圆心清除失败后，直接进入多余区域选择的位置清除。
    if belief.remaining_clear_region is not None:
        if belief.remaining_clear_region.is_empty:
            raise RuntimeError("圆心清除失败后多余区域为空，与清除结果矛盾。")
        if belief.remaining_clear_point is None:
            raise RuntimeError("多余区域存在，但缺少直接清除位置。")
        planner.replace_channel_tasks(
            belief.channel,
            [
                make_task(
                    TaskType.RESIDUAL_CLEAR,
                    belief.remaining_clear_point,
                    "进入多余区域直接清除",
                )
            ],
        )
        return

    directions = belief.direction_observations
    if not directions:
        coverage_tasks = [
            RouteTask(
                task_id=planner.next_task_id(
                    belief.channel, TaskType.COVERAGE_MEASURE
                ),
                task_type=TaskType.COVERAGE_MEASURE,
                point=point,
                channel=belief.channel,
                coverage_index=index,
                channel_revision=belief.revision,
                reason="完成最小接收半径保证覆盖",
            )
            for index, point in enumerate(planner.coverage_points)
            if index not in belief.coverage_visited
        ]
        planner.replace_channel_tasks(belief.channel, coverage_tasks)
        return

    if len(directions) == 1:
        if belief.second_measure_completed:
            raise RuntimeError(
                f"频道 {belief.channel} 的第二检测未得到方向信息，"
                "当前 P2 接口无法据此继续生成检测点。"
            )
        first = directions[0]
        assert first.bearing_deg is not None
        if belief.second_measure_point is None:
            p2_config = replace(
                config.p2,
                bilateral_search=config.p2_planning_mode != "legacy_single",
            )
            p2_result = get_problem_2_result(
                (first.position.x, first.position.y),
                first.bearing_deg,
                p2_config,
            )
            belief.p2_result = p2_result
            belief.p2_result_revision = belief.revision
            belief.second_measure_point = p2_result.best_candidate.point
            best_key = point_key(
                belief.second_measure_point, config.point_precision
            )
            belief.second_measure_fallback_point = next(
                (
                    candidate.point
                    for candidate in p2_result.candidates
                    if candidate.guaranteed_signal
                    and point_key(candidate.point, config.point_precision) != best_key
                ),
                None,
            )
            belief.candidate_region = p2_result.first_feasible_region
        planner.replace_channel_tasks(
            belief.channel,
            [
                make_task(
                    TaskType.SECOND_MEASURE,
                    belief.second_measure_point,
                    "P2生成的第二检测点",
                )
            ],
        )
        return

    detection_data = [
        (record.position.x, record.position.y, float(record.bearing_deg))
        for record in directions
        if record.bearing_deg is not None
    ]
    geometry = solve_problem_1_geometry(
        detection_data,
        error_deg=config.bearing_error_deg,
        target_radius=config.target_radius,
        receive_radius_max=config.receive_radius_max,
        near_radius=config.near_radius,
    )
    belief.p1_geometry = geometry
    belief.candidate_region = geometry.polygon

    # 小直径非覆盖分支的圆心测量已完成。候选区域先由新方向更新，
    # 随后仍回到原圆心尝试清除。
    if belief.pending_center_clear_point is not None:
        planner.replace_channel_tasks(
            belief.channel,
            [
                make_task(
                    TaskType.CENTER_CLEAR,
                    belief.pending_center_clear_point,
                    "圆心测量后尝试清除原圆心",
                )
            ],
        )
        return

    center = geometry.diameter_circle_center
    diameter_limit = 2.0 * config.clear_radius
    if geometry.diameter <= diameter_limit + 1e-9:
        if geometry.diameter_circle_covers:
            belief.center_clear_allows_residual = False
            task_type = TaskType.CENTER_CLEAR
            reason = "直径不超过40米且直径圆覆盖，直接到圆心清除"
        else:
            belief.center_clear_allows_residual = True
            task_type = TaskType.CENTER_MEASURE
            reason = "直径不超过40米但直径圆未覆盖，在圆心继续测量"
    else:
        belief.center_clear_allows_residual = False
        task_type = TaskType.FOLLOW_UP_MEASURE
        reason = "候选区域直径超过40米，在直径圆圆心继续测量"

    planner.replace_channel_tasks(
        belief.channel,
        [make_task(task_type, center, reason)],
    )


# =============================================================================
# 3. 精细排查与多余区域清除算法
# =============================================================================


def select_residual_clear_point(
    region: Region,
    excluded_center: Point,
    clear_radius: float,
) -> tuple[Region, Point]:
    """扣除已失败的 20 米清除圆，并选择多余区域内的直接清除位置。"""
    if region.is_empty:
        raise ValueError("候选区域不能为空。")
    if clear_radius <= 0.0:
        raise ValueError("清除半径必须为正数。")
    remaining = region.difference(Region.disk(excluded_center, clear_radius))
    if remaining.is_empty:
        raise RuntimeError("清除失败后没有剩余候选区域。")
    return remaining, remaining.representative_point()


# =============================================================================
# 4. 动态寻路辅助算法
# 当前由 TaskPlanner.plan_route() 和 TaskPlanner.choose_next() 实现
# =============================================================================


def _route_distance(tasks: list[RouteTask], start: Point) -> float:
    points = [start] + [task.point for task in tasks if task.point is not None]
    return sum(first.distance_to(second) for first, second in zip(points, points[1:]))


def _sample_route_points(route: list[RouteTask], start: Point, spacing: float) -> list[Point]:
    result = [start]
    previous = start
    for task in route:
        if task.point is None:
            continue
        distance = previous.distance_to(task.point)
        for index in range(1, int(distance // spacing) + 1):
            ratio = min(1.0, index * spacing / distance) if distance else 1.0
            result.append(Point(
                previous.x + ratio * (task.point.x - previous.x),
                previous.y + ratio * (task.point.y - previous.y),
            ))
        result.append(task.point)
        previous = task.point
    return result


def _candidate_is_useful(
    belief: ChannelBelief,
    metrics: CandidateMetrics,
    config: Problem3Config,
) -> bool:
    result = belief.p2_result
    if result is None or not metrics.guaranteed_signal:
        return False
    limit = result.initial_worst_radius * (1.0 - config.shared_min_radius_reduction)
    if metrics.worst_radius > limit + 1e-9:
        return False
    key = point_key(metrics.point, config.point_precision)
    return all(
        point_key(record.position, config.point_precision) != key
        for record in belief.observations
    )


def build_shared_candidate_pool(
    beliefs: dict[int, ChannelBelief],
    planner: TaskPlanner,
    current_position: Point,
    current_channel: int,
    deadline_s: float = math.inf,
) -> list[SharedCandidate]:
    """用统一粗网格和现有路线构造待精确复评的共享点池。"""
    config = planner.config
    active = {
        channel: belief
        for channel, belief in beliefs.items()
        if not belief.cleared
        and not belief.second_measure_completed
        and not belief.shared_candidates_disabled
        and len(belief.direction_observations) == 1
        and belief.p2_result is not None
        and belief.p2_result_revision == belief.revision
    }
    if not active:
        return []

    route = planner.plan_route(current_position, current_channel)
    if time.monotonic() >= deadline_s:
        return []
    supplemental = _sample_route_points(route, current_position, config.shared_grid_spacing)
    supplemental.extend(
        task.point for task in planner.tasks.values() if task.point is not None
    )
    supplemental.extend(
        belief.p2_result.best_candidate.point for belief in active.values()
        if belief.p2_result is not None
    )

    all_candidates = [
        candidate
        for belief in active.values()
        for candidate in belief.p2_result.candidates  # type: ignore[union-attr]
    ]
    if not all_candidates:
        return []
    spacing = config.shared_grid_spacing
    min_x = math.floor(min(item.point.x for item in all_candidates) / spacing) * spacing
    max_x = math.ceil(max(item.point.x for item in all_candidates) / spacing) * spacing
    min_y = math.floor(min(item.point.y for item in all_candidates) / spacing) * spacing
    max_y = math.ceil(max(item.point.y for item in all_candidates) / spacing) * spacing
    grid = [
        Point(float(x), float(y))
        for x in np.arange(min_x, max_x + spacing * 0.5, spacing)
        for y in np.arange(min_y, max_y + spacing * 0.5, spacing)
    ]
    points = grid + supplemental

    channel_arrays: dict[int, tuple[np.ndarray, list[CandidateMetrics]]] = {}
    for channel, belief in active.items():
        candidates = belief.p2_result.candidates  # type: ignore[union-attr]
        coordinates = np.array([(item.point.x, item.point.y) for item in candidates])
        channel_arrays[channel] = coordinates, candidates

    coarse: dict[tuple, SharedCandidate] = {}
    max_interpolation_distance = spacing * math.sqrt(2.0) * 0.55
    for point in points:
        if time.monotonic() >= deadline_s:
            break
        metrics: dict[int, CandidateMetrics] = {}
        for channel, belief in active.items():
            coordinates, candidates = channel_arrays[channel]
            squared = np.sum((coordinates - np.array([point.x, point.y])) ** 2, axis=1)
            index = int(np.argmin(squared))
            nearest = candidates[index]
            if math.sqrt(float(squared[index])) <= max_interpolation_distance and _candidate_is_useful(
                belief, nearest, config
            ):
                metrics[channel] = nearest
        key = point_key(point, spacing / 2.0)
        if metrics:
            old = coarse.get(key)
            if old is None or len(metrics) > len(old.channels):
                coarse[key] = SharedCandidate(
                    point=point,
                    channels=set(metrics),
                    metrics=metrics,
                    revisions={channel: active[channel].revision for channel in metrics},
                )

    route_points = [current_position] + [task.point for task in route if task.point is not None]
    def insertion_cost(point: Point) -> float:
        return min(
            first.distance_to(point) + point.distance_to(second) - first.distance_to(second)
            for first, second in zip(route_points, route_points[1:])
        ) if len(route_points) > 1 else current_position.distance_to(point)

    grouped: list[list[SharedCandidate]] = []
    by_channels: dict[frozenset[int], list[SharedCandidate]] = {}
    for candidate in coarse.values():
        by_channels.setdefault(frozenset(candidate.channels), []).append(candidate)
    for same_channels in by_channels.values():
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, candidate in enumerate(same_channels):
            cell = (math.floor(candidate.point.x / spacing), math.floor(candidate.point.y / spacing))
            buckets.setdefault(cell, []).append(index)
        unseen = set(range(len(same_channels)))
        while unseen:
            stack = [unseen.pop()]
            component: list[SharedCandidate] = []
            while stack:
                index = stack.pop()
                item = same_channels[index]
                component.append(item)
                cell = (math.floor(item.point.x / spacing), math.floor(item.point.y / spacing))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for neighbor in buckets.get((cell[0] + dx, cell[1] + dy), []):
                            if neighbor in unseen and item.point.distance_to(
                                same_channels[neighbor].point
                            ) <= spacing * math.sqrt(2.0) + 1e-9:
                                unseen.remove(neighbor)
                                stack.append(neighbor)
            grouped.append(component)

    selected: dict[tuple, SharedCandidate] = {}
    for items in grouped:
        choices = [
            min(items, key=lambda item: sum(
                metric.worst_radius / active[channel].p2_result.initial_worst_radius
                for channel, metric in item.metrics.items()
            )),
            min(items, key=lambda item: item.point.distance_to(current_position)),
            min(items, key=lambda item: insertion_cost(item.point)),
        ]
        for item in choices:
            selected[point_key(item.point, config.point_precision)] = item
    return list(selected.values())


def optimize_shared_second_measurements(
    beliefs: dict[int, ChannelBelief],
    planner: TaskPlanner,
    current_position: Point,
    current_channel: int,
) -> dict[str, Any]:
    """在现实时间预算内，把独立二测任务迁入可缩短路线的共享位置。"""
    started = time.monotonic()
    config = planner.config
    if config.p2_planning_mode != "bilateral_shared":
        return {"elapsed_s": 0.0, "saved_distance_m": 0.0, "assignments": {}}
    if not any(
        len(belief.direction_observations) == 1
        and not belief.second_measure_completed
        and belief.p2_result is not None
        and belief.p2_result_revision == belief.revision
        and not belief.shared_candidates_disabled
        for belief in beliefs.values()
    ):
        return {"elapsed_s": 0.0, "saved_distance_m": 0.0, "assignments": {}}

    deadline = started + config.shared_planning_time_s
    base_tasks = dict(planner.tasks)
    base_route = planner.plan_route(current_position, current_channel)
    base_keys = list(planner._route_keys)
    base_distance = _route_distance(base_route, current_position)
    if time.monotonic() >= deadline:
        elapsed = time.monotonic() - started
        return {
            "elapsed_s": elapsed,
            "budget_overrun_s": max(0.0, elapsed - config.shared_planning_time_s),
            "saved_distance_m": 0.0,
            "assignments": {},
        }
    pool = build_shared_candidate_pool(
        beliefs, planner, current_position, current_channel, deadline
    )
    def rough_distance(item: SharedCandidate) -> float:
        trial_route = [
            replace(task, point=item.point)
            if task.task_type is TaskType.SECOND_MEASURE and task.channel in item.channels
            else task
            for task in base_route
        ]
        return _route_distance(trial_route, current_position)
    pool.sort(key=lambda item: (
        rough_distance(item),
        -len(item.channels),
        current_position.distance_to(item.point),
    ))
    best_tasks = base_tasks
    best_distance = base_distance
    best_quality_delta = 0.0
    best_assignments: dict[int, Point] = {}

    offsets = [(0.0, 0.0)]
    step = config.shared_refined_spacing
    radius = config.shared_refinement_radius
    for dx in np.arange(-radius, radius + step * 0.5, step):
        for dy in np.arange(-radius, radius + step * 0.5, step):
            if dx * dx + dy * dy <= radius * radius + 1e-9 and (dx or dy):
                offsets.append((float(dx), float(dy)))

    for coarse in pool:
        for dx, dy in offsets:
            if time.monotonic() - started >= config.shared_planning_time_s:
                break
            point = Point(coarse.point.x + dx, coarse.point.y + dy)
            exact: dict[int, CandidateMetrics] = {}
            for channel in coarse.channels:
                if time.monotonic() - started >= config.shared_planning_time_s:
                    break
                belief = beliefs[channel]
                result = belief.p2_result
                if result is None or belief.p2_result_revision != belief.revision:
                    continue
                metric = next((
                    item for item in result.candidates
                    if point_key(item.point, config.point_precision)
                    == point_key(point, config.point_precision)
                ), None)
                if metric is None:
                    metric = evaluate_problem_2_candidate(result, point)
                if _candidate_is_useful(belief, metric, config):
                    exact[channel] = metric
            existing_key = point_key(point, config.point_precision) in {
                point_key(task.point, config.point_precision)
                for task in base_tasks.values() if task.point is not None
            }
            if len(exact) < 2 and not (len(exact) == 1 and existing_key):
                continue

            trial = dict(base_tasks)
            assignments: dict[int, Point] = {}
            for task_id, task in trial.items():
                if task.task_type is TaskType.SECOND_MEASURE and task.channel in exact:
                    trial[task_id] = replace(task, point=point, reason="共享候选点二次检测")
                    assignments[task.channel] = point  # type: ignore[index]
            if not assignments:
                continue
            planner.tasks = trial
            planner._route_keys = list(base_keys)
            try:
                trial_route = planner.plan_route(current_position, current_channel)
            finally:
                planner.tasks = base_tasks
                planner._route_keys = list(base_keys)
            distance = _route_distance(trial_route, current_position)
            quality_delta = 0.0
            for channel in assignments:
                result = beliefs[channel].p2_result
                assert result is not None
                old_task = next(
                    task for task in base_tasks.values()
                    if task.task_type is TaskType.SECOND_MEASURE and task.channel == channel
                )
                old_metric = min(
                    result.candidates,
                    key=lambda item: item.point.distance_to(old_task.point),  # type: ignore[arg-type]
                )
                quality_delta += (
                    exact[channel].worst_radius - old_metric.worst_radius
                ) / result.initial_worst_radius
            if distance < best_distance - 1e-7 or (
                abs(distance - best_distance) <= 1e-7
                and quality_delta < best_quality_delta - 1e-12
            ):
                best_tasks = trial
                best_distance = distance
                best_quality_delta = quality_delta
                best_assignments = assignments
        if time.monotonic() - started >= config.shared_planning_time_s:
            break

    planner.tasks = best_tasks
    planner._route_keys = list(base_keys)
    planner.plan_route(current_position, current_channel)
    for channel, point in best_assignments.items():
        beliefs[channel].second_measure_point = point
    elapsed = time.monotonic() - started
    return {
        "elapsed_s": elapsed,
        "budget_overrun_s": max(0.0, elapsed - config.shared_planning_time_s),
        "saved_distance_m": max(0.0, base_distance - best_distance),
        "assignments": {
            channel: (point.x, point.y) for channel, point in best_assignments.items()
        },
    }


# =============================================================================
# 5. 模拟器运行与流程控制算法
# =============================================================================


def run_controller(controller: Problem3Controller) -> dict[str, Any]:
    """执行进入、滚动TSP规划、反馈更新和主动退出的完整主循环。"""
    controller.wall_start_s = time.monotonic()
    enter_response = controller.client.enter()
    remaining = float(enter_response["remaining_real_duration_s"])
    controller.real_deadline_s = time.monotonic() + remaining

    for belief in controller.beliefs.values():
        controller.refresh_channel(belief)

    while controller.actions_executed < controller.config.max_actions:
        controller._check_real_time()
        cleared_count = sum(belief.cleared for belief in controller.beliefs.values())
        all_certified = all(
            belief.certified_complete(len(controller.task_planner.coverage_points))
            for belief in controller.beliefs.values()
        )
        if cleared_count >= controller.config.source_count_max or all_certified:
            response = controller.client.exit()
            return controller.summary(response.get("exit_reason", "user_exit"))

        controller.task_planner.discard_stale_tasks(controller.beliefs)
        if not controller.task_planner.tasks:
            raise RuntimeError("任务池为空，但尚未满足退出条件。")
        current_key = point_key(
            controller.client.position, controller.config.point_precision
        )
        has_local_task = any(
            task.point is not None
            and task.action_kind != "exit"
            and point_key(task.point, controller.config.point_precision) == current_key
            for task in controller.task_planner.tasks.values()
        )
        if (
            not has_local_task
            and controller.config.p2_planning_mode == "bilateral_shared"
            and time.monotonic() + controller.config.shared_planning_time_s
            < controller.real_deadline_s - controller.config.real_time_reserve_s
        ):
            event = controller.optimize_shared_measurements()
            controller.shared_planning_time_s += event["elapsed_s"]
            controller.shared_planned_distance_saved_m += event["saved_distance_m"]
            if event["assignments"]:
                controller.shared_assignment_events.append(event)
                recorder = getattr(controller.client, "record_planning_event", None)
                if callable(recorder):
                    recorder(event)
            controller._check_real_time()
        task = controller.task_planner.choose_next(
            controller.client.position,
            controller.client.current_channel,
        )
        if task.action_kind == "measure":
            controller._handle_measure(task)
        elif task.action_kind == "clear":
            controller._handle_clear(task)
        else:
            response = controller.client.exit()
            return controller.summary(response.get("exit_reason", "user_exit"))
        controller.actions_executed += 1

    raise RuntimeError(
        f"达到动作上限 {controller.config.max_actions}，任务尚未完成。"
    )


# =============================================================================
# 6. 离线检查、统计与程序入口
# =============================================================================


def run_offline_check(config: Problem3Config, coverage_generator=generate_coverage_points) -> dict[str, Any]:
    points = coverage_generator(config)
    return {
        "coverage_points": [(round(point.x, 3), round(point.y, 3)) for point in points],
        "class_structure_ready": True,
        "algorithms_migrated": True,
        "route_planner": "python-tsp two_opt (open route, movement objective)",
        "p2_planning_mode": config.p2_planning_mode,
    }


def main(*, controller_type=Problem3Controller,
         coverage_generator=generate_coverage_points, problem_number=3) -> None:
    """P3/P4 共用命令行入口，覆盖点策略由调用者提供。"""
    parser = argparse.ArgumentParser(description=f"CUMCM 2026 B题问题{problem_number}控制算法")
    parser.add_argument("--run", action="store_true", help="连接模拟器并执行")
    parser.add_argument("--robot-id", help="当前登录模拟器的参赛队号")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    parser.add_argument("--log", type=Path, default=Path(f"P{problem_number}_run_log.jsonl"))
    parser.add_argument("--route-search-time", type=float, default=0.1,
                        help="每轮TSP搜索预算（秒，默认0.1）")
    parser.add_argument(
        "--p2-planning-mode",
        choices=("legacy_single", "bilateral_independent", "bilateral_shared"),
        default="bilateral_shared",
        help="P2对照模式（默认双边共享规划）",
    )
    parser.add_argument("--shared-planning-time", type=float, default=1.0,
                        help="每轮共享二测规划预算（秒，默认1.0）")
    args = parser.parse_args()

    config = Problem3Config(
        route_search_time_s=args.route_search_time,
        p2_planning_mode=args.p2_planning_mode,
        shared_planning_time_s=args.shared_planning_time,
    )
    if not args.run:
        print(json.dumps(run_offline_check(config, coverage_generator), ensure_ascii=False, indent=2))
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
    summary = controller_type(client, config).run()
    console_summary = {
        key: value for key, value in summary.items()
        if key != "shared_assignment_events"
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
