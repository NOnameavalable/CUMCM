"""P4混合后验、探针切换和清除闭环的行为测试。"""
import importlib.util
from pathlib import Path
import sys

import pytest

from P3 import RouteTask, TaskType
from utils import Point
from P3 import ObservationRecord
from p4.planning import plan_geometry_measurement
from utils import Region


_SPEC = importlib.util.spec_from_file_location("p4_main", Path(__file__).with_name("p4.py"))
assert _SPEC is not None and _SPEC.loader is not None
_P4 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _P4
_SPEC.loader.exec_module(_P4)


class Client:
    position = Point(0.0, 0.0)
    current_channel = 1
    virtual_time_s = 0.0

    def __init__(self):
        self.measurements = [
            {"measure_result": "direction", "svd_deg": 0.0},
            {"measure_result": "no_signal"},
            {"measure_result": "no_signal"},
        ]

    def measure(self, point, channel):
        self.position = point
        self.current_channel = channel
        self.virtual_time_s += 5.0
        return {**self.measurements.pop(0), "virtual_time_s": self.virtual_time_s}

    def clear(self, point, channel):
        self.position = point
        self.virtual_time_s += 2.0
        return {"clear_result": "no_target_in_range", "virtual_time_s": self.virtual_time_s}


@pytest.fixture()
def initialized_controller():
    controller = _P4.Problem4Controller(Client())
    controller._handle_measure(
        RouteTask("first", TaskType.COVERAGE_MEASURE, Point(0.0, 0.0), 1, coverage_index=0)
    )
    return controller


def test_first_direction_builds_joint_source_posterior(initialized_controller):
    belief = initialized_controller.beliefs[1]
    assert belief.posterior is not None
    assert 0.0 < belief.posterior.omni_probability < 1.0
    assert not hasattr(belief, "s_min")


def test_two_no_signal_results_switch_to_probe_strategy(initialized_controller):
    controller = initialized_controller
    belief = controller.beliefs[1]
    for _ in range(2):
        task = next(task for task in controller.task_planner.tasks.values() if task.channel == 1)
        controller._handle_measure(task)
    assert belief.search_mode == "probe"
    task = next(task for task in controller.task_planner.tasks.values() if task.channel == 1)
    assert "探针" in task.reason


def test_clear_failure_reduces_region_and_keeps_channel_active(initialized_controller):
    controller = initialized_controller
    belief = controller.beliefs[1]
    assert belief.posterior is not None
    point = belief.posterior.position_region.representative_point()
    old_area = belief.posterior.position_region.area
    controller._handle_clear(RouteTask("clear", TaskType.CENTER_CLEAR, point, 1))
    assert belief.posterior.position_region.area < old_area
    assert belief.failed_clear_points == [point]
    assert any(task.channel == 1 for task in controller.task_planner.tasks.values())


def test_two_directions_leave_old_p2_candidates(initialized_controller, monkeypatch):
    controller = initialized_controller
    belief = controller.beliefs[1]
    belief.observations.append(ObservationRecord(Point(100, 0), "direction", 0, 10))
    # 同轴观测留下较长区域；即使离散边缘概率为空也应继续P1测向。
    monkeypatch.setattr(belief.posterior, 'target_marginals', lambda: [])
    monkeypatch.setattr(_P4, 'plan_adaptive_measurements', lambda *a, **k: pytest.fail('returned to P2'))
    controller.refresh_channel(belief)
    task = next(t for t in controller.task_planner.tasks.values() if t.channel == 1)
    assert task.reason.startswith('P1')
    belief.observations.append(ObservationRecord(task.point, 'no_signal', None, 15))
    controller.refresh_channel(belief)
    next_task = next(t for t in controller.task_planner.tasks.values() if t.channel == 1)
    assert next_task.reason.startswith('P1')
    assert next_task.point.distance_to(task.point) > 1


def test_geometry_points_do_not_repeat_measured_center():
    region = Region.disk(Point(500, 0), 100)
    first = plan_geometry_measurement(region, [], 20)
    second = plan_geometry_measurement(region, [first], 20)
    assert first.distance_to(second) > 1
