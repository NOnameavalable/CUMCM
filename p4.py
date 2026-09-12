"""第四题：继承 P3 控制流程，使用 verify_coverage 定义的 17 个检测站。"""
from P3 import Problem3Config, Problem3Controller, main as run_cli
from p4.verify_coverage import generate_stations
from simulator import SimulatorClient
from utils import Point


def generate_coverage_points(config: Problem3Config) -> list[Point]:
    stations, *_ = generate_stations(config.target_radius, config.receive_radius_min)
    return [Point(x, y) for x, y, _ in stations]


class Problem4Controller(Problem3Controller):
    """所有测量、定位、清除、共享规划和退出逻辑均继承 P3。"""

    def __init__(self, client: SimulatorClient, config: Problem3Config | None = None):
        super().__init__(client, config)
        # 父类初始化只创建空任务池；首次任务生成发生在 run() 中。
        self.task_planner.coverage_points = generate_coverage_points(self.config)


def main() -> None:
    run_cli(controller_type=Problem4Controller,
            coverage_generator=generate_coverage_points, problem_number=4)


if __name__ == "__main__":
    main()
