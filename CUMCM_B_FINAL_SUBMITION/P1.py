"""CUMCM 2026 B 题算法主体。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from CUMCM_B_FINAL_SUBMITION.utils import DetectionSector, Point, Polygon, Region
from CUMCM_B_FINAL_SUBMITION.utils import _TOLERANCE


# ============================================================
# 作用：保存 P1 的完整连续几何结果，供 P3 取得定位区域及其中心。
# ============================================================
@dataclass(frozen=True)
class Problem1Geometry:
    polygon: Region
    diameter: float
    diameter_endpoints: tuple[Point, Point]
    diameter_circle_center: Point
    diameter_circle_covers: bool


# ============================================================
# 作用：把一次“测点 + 示向度”转换为带 ±error_deg 误差的检测夹角。
# ============================================================
def _make_detection_sector(
    x: float,
    y: float,
    bearing_deg: float,
    error_deg: float,
) -> DetectionSector:
    """由检测点、示向度和误差角构造检测夹角。"""
    if not all(isfinite(value) for value in (x, y, bearing_deg, error_deg)):
        raise ValueError("检测点坐标、示向度和误差角必须是有限数值。")
    if not 0.0 < error_deg < 90.0:
        raise ValueError("误差角必须位于 (0°, 90°) 内。")

    origin = Point(float(x), float(y))
    return DetectionSector.from_measurement(origin, float(bearing_deg), error_deg)


# ============================================================
# 作用：检验以最远点对为直径的圆是否覆盖整个定位区域。
# ============================================================
def _diameter_circle_covers(
    polygon: Polygon | Region,
    diameter: float,
    endpoints: tuple[Point, Point] | None = None,
) -> bool:
    """检验一组最远点对的直径圆；传入端点时须与给定直径对应。"""
    vertices = polygon.vertices
    if not vertices:
        raise ValueError("定位区域为空，无法检验直径圆。")
    if len(vertices) == 1:
        return True

    if endpoints is None:
        _, first, second = polygon.diameter_with_endpoints()
    else:
        first, second = endpoints
    center = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
    tolerance = max(_TOLERANCE, diameter * 1e-10)
    return all(
        center.distance_to(vertex) <= diameter / 2.0 + tolerance
        for vertex in vertices
    )


# ============================================================
# 作用：求完整定位多边形、直径端点和直径圆，供 P3 后续定位使用。
# ============================================================
def solve_problem_1_geometry(
    detection_data: list[tuple[float, float, float]],
    error_deg: float = 1.0,
    target_radius: float = 1800.0,
    receive_radius_max: float | None = None,
    near_radius: float = 0.0,
) -> Problem1Geometry:
    """返回多次示向度公共定位区域的完整连续几何结果。"""
    if len(detection_data) < 2:
        raise ValueError("第一问至少需要两个检测点。")

    sectors: list[DetectionSector] = []
    for item in detection_data:
        if len(item) != 3:
            raise ValueError("每条检测数据必须是 (x, y, 示向度) 三元组。")
        sectors.append(_make_detection_sector(*item, error_deg))

    if target_radius <= 0.0:
        raise ValueError("目标圆域半径必须为正数。")
    if receive_radius_max is not None and receive_radius_max <= 0.0:
        raise ValueError("最大接收距离必须为正数。")
    if near_radius < 0.0:
        raise ValueError("near 半径不能为负数。")

    # 连续候选区域统一由 Shapely 后端完成布尔运算。P1 的默认语义仍是
    # “目标圆域与多个示向扇区求交”；P2 如需接收距离约束会显式传入。
    polygon = Region.disk(Point(0.0, 0.0), target_radius)
    for sector in sectors:
        sector_limit = receive_radius_max or (
            sector.origin.distance_to(Point(0.0, 0.0)) + target_radius + 1.0
        )
        if near_radius >= sector_limit:
            raise ValueError("near 半径必须小于扇区截断距离。")
        polygon = polygon.intersection(
            sector.to_region(sector_limit, near_radius)
        )
        if polygon.is_empty or polygon.area <= 1e-8:
            raise ValueError("所有检测数据的公共定位区域为空或退化。")

    diameter, first, second = polygon.diameter_with_endpoints()
    center = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
    can_cover = _diameter_circle_covers(polygon, diameter, (first, second))
    return Problem1Geometry(
        polygon=polygon,
        diameter=diameter,
        diameter_endpoints=(first, second),
        diameter_circle_center=center,
        diameter_circle_covers=can_cover,
    )


# ============================================================
# 作用：保持问题 1 原接口，只返回直径及直径圆覆盖结论。
# ============================================================
def solve_problem_1(
    detection_data: list[tuple[float, float, float]],
    error_deg: float = 1.0,
) -> tuple[float, bool]:
    """求第一问的定位区域直径及直径圆覆盖结论。

    ``detection_data`` 中每项依次为检测点 x 坐标、y 坐标和示向度。
    返回 ``(定位区域直径, 直径圆能否覆盖整个定位区域)``。
    """
    result = solve_problem_1_geometry(detection_data, error_deg)
    return result.diameter, result.diameter_circle_covers


if __name__ == "__main__":
    # 示例数据只用于展示调用格式，正式使用时替换为实际检测结果。
    example_data = [
        (-1000.0, 0.0, 0.0),
        (1000.0, 0.0, 180.0),
    ]
    result_diameter, result_can_cover = solve_problem_1(example_data)
    print(f"定位区域直径：{result_diameter:.6f} 米")
    print(f"直径圆能否覆盖整个区域：{'能' if result_can_cover else '不能'}")
