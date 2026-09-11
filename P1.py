"""CUMCM 2026 B 题算法主体。"""

from __future__ import annotations

from itertools import combinations
from math import isfinite

from utils import DetectionSector, Point, Polygon, Ray, Segment, TargetArea
from utils import _TOLERANCE, _same_point, _same_segment, _unique_points


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
    return DetectionSector(
        lower_ray=Ray(origin, float(bearing_deg) - error_deg),
        upper_ray=Ray(origin, float(bearing_deg) + error_deg),
    )


def _initial_polygon(
    target_area: TargetArea,
    first: DetectionSector,
    second: DetectionSector,
) -> Polygon | None:
    """构造两个检测夹角形成的有界非退化多边形。"""
    rays = (
        first.lower_ray,
        first.upper_ray,
        second.lower_ray,
        second.upper_ray,
    )
    candidates: list[Point] = []

    # 夹角顶点位于另一个夹角内时，它也是交集多边形的候选顶点。
    if second.contains(first.origin):
        candidates.append(first.origin)
    if first.contains(second.origin):
        candidates.append(second.origin)

    for first_ray in rays[:2]:
        for second_ray in rays[2:]:
            intersection = target_area.intersect_rays(first_ray, second_ray)
            if (
                isinstance(intersection, Point)
                and first.contains(intersection)
                and second.contains(intersection)
            ):
                candidates.append(intersection)

    candidates = _unique_points(candidates)
    if len(candidates) < 3:
        return None

    edges: list[Segment] = []
    for ray in rays:
        boundary_points = [
            point
            for point in candidates
            if ray.contains(point)
        ]
        if len(boundary_points) < 2:
            continue

        start, end = max(
            combinations(boundary_points, 2),
            key=lambda pair: pair[0].distance_to(pair[1]),
        )
        edge = Segment(start, end)
        if not any(_same_segment(edge, existing) for existing in edges):
            edges.append(edge)

    if len(edges) < 3:
        return None

    # 完整闭合边界的每个顶点应恰好与两条边相连。
    vertices = _unique_points([
        point for edge in edges for point in (edge.start, edge.end)
    ])
    for vertex in vertices:
        degree = sum(
            _same_point(edge.start, vertex) or _same_point(edge.end, vertex)
            for edge in edges
        )
        if degree != 2:
            return None

    return Polygon(edges)


def _find_initial_polygon(
    target_area: TargetArea,
    sectors: list[DetectionSector],
) -> tuple[Polygon, tuple[int, int]]:
    """寻找一对能够形成有界初始定位区域的检测夹角。"""
    for first_index, second_index in combinations(range(len(sectors)), 2):
        polygon = _initial_polygon(
            target_area,
            sectors[first_index],
            sectors[second_index],
        )
        if polygon is not None:
            return polygon, (first_index, second_index)
    raise ValueError("任意两个检测夹角均无法形成有界的初始定位多边形。")


def _diameter_circle_covers(
    polygon: Polygon,
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


def solve_problem_1(
    detection_data: list[tuple[float, float, float]],
    error_deg: float = 1.0,
) -> tuple[float, bool]:
    """求第一问的定位区域直径及直径圆覆盖结论。

    ``detection_data`` 中每项依次为检测点 x 坐标、y 坐标和示向度。
    返回 ``(定位区域直径, 直径圆能否覆盖整个定位区域)``。
    """
    if len(detection_data) < 2:
        raise ValueError("第一问至少需要两个检测点。")

    sectors: list[DetectionSector] = []
    for item in detection_data:
        if len(item) != 3:
            raise ValueError("每条检测数据必须是 (x, y, 示向度) 三元组。")
        sectors.append(_make_detection_sector(*item, error_deg))

    target_area = TargetArea()
    polygon, initial_indices = _find_initial_polygon(target_area, sectors)

    for index, sector in enumerate(sectors):
        if index in initial_indices:
            continue
        polygon = target_area.intersect_detection_area(polygon, sector)
        if not polygon.edges:
            raise ValueError("所有检测数据的公共定位区域为空。")

    diameter, first, second = polygon.diameter_with_endpoints()
    can_cover = _diameter_circle_covers(polygon, diameter, (first, second))
    return diameter, can_cover


if __name__ == "__main__":
    # 示例数据只用于展示调用格式，正式使用时替换为实际检测结果。
    example_data = [
        (-1000.0, 0.0, 0.0),
        (1000.0, 0.0, 180.0),
    ]
    result_diameter, result_can_cover = solve_problem_1(example_data)
    print(f"定位区域直径：{result_diameter:.6f} 米")
    print(f"直径圆能否覆盖整个区域：{'能' if result_can_cover else '不能'}")
