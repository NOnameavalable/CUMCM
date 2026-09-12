"""验证第四题只替换点位，继续使用 P3 完整控制循环。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from P3 import Problem3Config, Problem3Controller, TaskPlanner, refresh_channel_tasks
from p4.verify_coverage import generate_stations
from utils import Point

# 根目录入口 p4.py 与几何包 p4/ 同名，以文件方式加载入口。
spec = importlib.util.spec_from_file_location("problem4_entry", Path(__file__).resolve().parents[1] / "p4.py")
p4_entry = importlib.util.module_from_spec(spec)
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
    assert controller.__class__._handle_measure is Problem3Controller._handle_measure
    for belief in controller.beliefs.values():
        refresh_channel_tasks(belief, controller.task_planner, controller.config)
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
