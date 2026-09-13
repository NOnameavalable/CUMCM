"""验证第四题只替换点位，继续使用 P3 完整控制循环。"""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from P3 import (
    ObservationRecord,
    Problem3Config,
    Problem3Controller,
    RouteTask,
    TaskPlanner,
    TaskType,
    refresh_channel_tasks,
)
from p4.verify_coverage import generate_stations
from utils import Point

# 根目录入口 p4.py 与几何包 p4/ 同名，以文件方式加载入口。
spec = importlib.util.spec_from_file_location("problem4_entry", Path(__file__).resolve().parents[1] / "p4.py")
p4_entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p4_entry
spec.loader.exec_module(p4_entry)


def test_stations_and_independent_p3():
    controller = p4_entry.Problem4Controller(SimpleNamespace())
    stations, outer, inner, *_ = generate_stations()
    points = controller.task_planner.coverage_points
    assert len(points) == 17
    assert [(p.x, p.y) for p in points] == [(x, y) for x, y, _ in stations]
    assert all(p.distance_to(Point(0, 0)) == pytest.approx(outer) for p in points[1:9])
    assert all(p.distance_to(Point(0, 0)) == pytest.approx(inner) for p in points[9:])
    assert len(TaskPlanner(Problem3Config()).coverage_points) == 7
    assert issubclass(controller.__class__, Problem3Controller)
    for belief in controller.beliefs.values():
        controller.refresh_channel(belief)
    assert len(controller.task_planner.tasks) == 340


def test_all_17_stations_are_required_before_no_signal_exit():
    class Client:
        position = Point(0, 0)
        current_channel = 1
        virtual_time_s = 0

        def __init__(self):
            self.visits = []

        def enter(self):
            return {"remaining_real_duration_s": 1200}

        def measure(self, point, channel):
            self.position, self.current_channel = point, channel
            self.virtual_time_s += 5
            self.visits.append((point.x, point.y, channel))
            return {"measure_result": "no_signal", "virtual_time_s": self.virtual_time_s}

        def exit(self):
            assert len(self.visits) == 340
            return {"exit_reason": "user_exit"}

    client = Client()
    controller = p4_entry.Problem4Controller(client)
    result = controller.run()
    assert result["actions_executed"] == 340
    assert len(set(client.visits)) == 340
    assert all(belief.coverage_complete(17) for belief in controller.beliefs.values())


def test_symmetric_opposite_fallback_when_second_measure_has_no_signal():
    """测试二测在示向线一侧遭遇 no_signal 时，自动翻转到对称对侧候选点重试。"""
    class MockClient:
        position = Point(0, 0)
        current_channel = 1
        virtual_time_s = 0

        def __init__(self):
            self.measured_points = []

        def enter(self):
            return {"remaining_real_duration_s": 1200}

        def measure(self, point, channel):
            self.position, self.current_channel = point, channel
            self.virtual_time_s += 5
            self.measured_points.append(point)
            # 第一次测向（在原点）返回 direction 45度
            if len(self.measured_points) == 1:
                return {"measure_result": "direction", "svd_deg": 45.0, "virtual_time_s": self.virtual_time_s}
            # 第二测返回 no_signal（落入后半球盲区）
            return {"measure_result": "no_signal", "virtual_time_s": self.virtual_time_s}

        def exit(self):
            return {"exit_reason": "user_exit"}

    client = MockClient()
    controller = p4_entry.Problem4Controller(client)
    belief = controller.beliefs[1]

    # 1. 模拟初测得到 direction 45°
    first_task = RouteTask("test-1", TaskType.COVERAGE_MEASURE, Point(0, 0), 1, coverage_index=0)
    controller._handle_measure(first_task)
    assert len(belief.direction_observations) == 1
    assert belief.second_measure_point is not None
    p1 = belief.second_measure_point

    # 2. 模拟第二测执行并返回 no_signal
    second_task = RouteTask("test-2", TaskType.SECOND_MEASURE, p1, 1)
    controller._handle_measure(second_task)

    # 应该自动切换到对称对侧候选点或深度回退点，且未标记 completed
    assert not belief.second_measure_completed
    p2 = belief.second_measure_point
    assert p2 != p1


def test_direct_clear_on_small_diameter():
    """测试候选区域直径 <= 40 米时直接派发 CENTER_CLEAR 而非 CENTER_MEASURE。"""
    controller = p4_entry.Problem4Controller(SimpleNamespace())
    belief = controller.beliefs[1]

    # 添加两次近距离方向观测，使其交会出小直径区域
    belief.observations.append(ObservationRecord(Point(0, 0), "direction", 0.0, 0.0))
    belief.observations.append(ObservationRecord(Point(100, 100), "direction", 270.0, 5.0))

    controller.refresh_channel(belief)
    tasks = list(controller.task_planner.tasks.values())
    assert len(tasks) == 1
    # 核心验证：小区域直接 CENTER_CLEAR，不进行测向
    assert tasks[0].task_type == TaskType.CENTER_CLEAR


def test_no_infinite_loop_when_center_measure_gets_no_signal():
    """测试当多边形中心点测向遭遇定向源盲区(no_signal)时，绝对不会原地死循环。"""
    controller = p4_entry.Problem4Controller(SimpleNamespace())
    belief = controller.beliefs[13]

    # 还原 session_20260912_140637 中信道 13 的两次基站探测
    belief.observations.append(ObservationRecord(Point(0.0, 0.0), "direction", 173.34, 10.0))
    belief.observations.append(ObservationRecord(Point(-850.0, 50.0), "direction", 152.0, 20.0))

    # 首次刷新：直径 > 80m，圆心未测过，派发圆心 FOLLOW_UP_MEASURE
    controller.refresh_channel(belief)
    tasks = list(controller.task_planner.tasks.values())
    assert len(tasks) == 1
    first_measure_task = tasks[0]
    assert first_measure_task.task_type == TaskType.FOLLOW_UP_MEASURE
    center_pt = first_measure_task.point

    # 模拟在圆心处测得 no_signal（落入盲区）
    belief.observations.append(ObservationRecord(center_pt, "no_signal", None, 30.0))

    # 再次刷新：圆心已测过，绝不允许再次派发相同坐标的测向任务！
    controller.refresh_channel(belief)
    new_tasks = list(controller.task_planner.tasks.values())
    assert len(new_tasks) >= 1
    # 验证新任务绝不是在该 center_pt 上的重复 measure 任务
    for t in new_tasks:
        if t.action_kind == "measure":
            assert t.point != center_pt

