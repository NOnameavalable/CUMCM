"""CUMCM 2026 B 题的平面几何对象。

所有坐标均使用题目给定的全局直角坐标系：目标圆域圆心为 (0, 0)，
x 轴正向为东，y 轴正向为北，单位为米。图形可以位于目标圆域之外。
对象负责自身的几何性质；TargetArea 负责不同图形之间的交集与裁剪。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, radians, sin
from typing import Iterable, Sequence

import numpy as np
from shapely.geometry import GeometryCollection
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


_TOLERANCE = 1e-9


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.dot(np.array([ax, ay]), np.array([by, -bx])))


def _dot(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.dot(np.array([ax, ay]), np.array([bx, by])))


def _vector(start: "Point", end: "Point") -> np.ndarray:
    return np.array([end.x - start.x, end.y - start.y], dtype=float)


def _same_point(first: "Point", second: "Point") -> bool:
    return first.distance_to(second) <= _TOLERANCE


def _unique_points(points: list[Point]) -> list[Point]:
    result: list[Point] = []
    for point in points:
        if not any(_same_point(point, existing) for existing in result):
            result.append(point)
    return result


def _same_segment(first: Segment, second: Segment) -> bool:
    return (
        _same_point(first.start, second.start) and _same_point(first.end, second.end)
    ) or (
        _same_point(first.start, second.end) and _same_point(first.end, second.start)
    )


@dataclass
class Point:
    """全局二维直角坐标系中的点。"""

    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        """返回到另一点的欧氏距离。"""
        return float(np.linalg.norm(_vector(self, other)))

    def bearing_to(self, other: "Point") -> float:
        """返回指向另一点的方位角，范围为 [0, 360)。"""
        dx, dy = other.x - self.x, other.y - self.y
        if np.linalg.norm(np.array([dx, dy])) <= _TOLERANCE:
            raise ValueError("两点重合，无法确定方位角。")
        return degrees(atan2(dy, dx)) % 360.0


@dataclass
class Segment:
    """由两个端点确定的闭线段。起点和终点可重合。"""

    start: Point
    end: Point

    def length(self) -> float:
        """返回线段长度。"""
        return self.start.distance_to(self.end)

    def point_at(self, parameter: float) -> Point:
        """返回 start + parameter * (end - start)，parameter 必须在 [0, 1]。"""
        if not -_TOLERANCE <= parameter <= 1.0 + _TOLERANCE:
            raise ValueError("线段参数必须位于 [0, 1]。")
        parameter = min(1.0, max(0.0, parameter))
        return Point(
            self.start.x + parameter * (self.end.x - self.start.x),
            self.start.y + parameter * (self.end.y - self.start.y),
        )

    def contains(self, point: Point) -> bool:
        """判断点是否位于闭线段上，包含端点。"""
        dx, dy = self.end.x - self.start.x, self.end.y - self.start.y
        length_squared = dx * dx + dy * dy
        if length_squared <= _TOLERANCE * _TOLERANCE:
            return _same_point(self.start, point)
        px, py = point.x - self.start.x, point.y - self.start.y
        if abs(_cross(dx, dy, px, py)) > _TOLERANCE * np.linalg.norm(np.array([dx, dy])):
            return False
        projection = _dot(px, py, dx, dy)
        return -_TOLERANCE <= projection <= length_squared + _TOLERANCE

    def distance_to_point(self, point: Point) -> float:
        """返回点到有限线段的最短距离。"""
        dx, dy = self.end.x - self.start.x, self.end.y - self.start.y
        length_squared = dx * dx + dy * dy
        if length_squared <= _TOLERANCE * _TOLERANCE:
            return self.start.distance_to(point)
        parameter = _dot(point.x - self.start.x, point.y - self.start.y, dx, dy)
        parameter = min(1.0, max(0.0, parameter / length_squared))
        return point.distance_to(self.point_at(parameter))


@dataclass
class Ray:
    """由起点和方位角确定的射线。"""

    origin: Point
    angle: float

    def __post_init__(self) -> None:
        self.angle %= 360.0

    def _local_coordinates(self, point: Point) -> tuple[float, float]:
        """返回点沿射线方向的投影和到支撑直线的有符号垂距。"""
        angle_rad = radians(self.angle)
        direction_x, direction_y = cos(angle_rad), sin(angle_rad)
        dx, dy = point.x - self.origin.x, point.y - self.origin.y
        return (
            _dot(dx, dy, direction_x, direction_y),
            _cross(direction_x, direction_y, dx, dy),
        )

    def contains(self, point: Point) -> bool:
        """判断点是否位于射线上，包含起点。"""
        projection, perpendicular = self._local_coordinates(point)
        return projection >= -_TOLERANCE and abs(perpendicular) <= _TOLERANCE

    def distance_to_point(self, point: Point) -> float:
        """返回点到射线的最短距离。"""
        projection, perpendicular = self._local_coordinates(point)
        return abs(perpendicular) if projection >= 0.0 else self.origin.distance_to(point)


@dataclass
class DetectionSector:
    """同一起点的两条边界射线围成的检测方向范围。

内部定义为从 lower_ray 逆时针转到 upper_ray 的区域，包含两条边界。
本题中通常由示向度减 1° 与加 1° 两条射线构成。
"""

    lower_ray: Ray
    upper_ray: Ray

    def __post_init__(self) -> None:
        if not _same_point(self.lower_ray.origin, self.upper_ray.origin):
            raise ValueError("检测范围的两条边界射线必须有相同起点。")

    @property
    def origin(self) -> Point:
        return self.lower_ray.origin

    @property
    def angle_width(self) -> float:
        """返回从下边界逆时针旋转到上边界的角宽，范围为 [0, 360)。"""
        return (self.upper_ray.angle - self.lower_ray.angle) % 360.0

    def contains(self, point: Point) -> bool:
        """判断点是否处于检测方向范围内，包含起点与两条边界。"""
        if _same_point(point, self.origin):
            return True
        bearing = self.origin.bearing_to(point)
        relative_angle = (bearing - self.lower_ray.angle) % 360.0
        return relative_angle <= self.angle_width + _TOLERANCE

    @classmethod
    def from_measurement(
        cls,
        origin: Point,
        bearing_deg: float,
        error_deg: float = 1.0,
    ) -> "DetectionSector":
        if not 0.0 < error_deg < 90.0:
            raise ValueError("示向度误差必须位于 (0°, 90°) 内。")
        return cls(Ray(origin, bearing_deg - error_deg), Ray(origin, bearing_deg + error_deg))

    def to_region(
        self,
        max_distance: float,
        min_distance: float = 0.0,
        arc_samples: int = 65,
    ) -> "Region":
        """Convert the bounded measurement sector to a continuous Region."""
        if max_distance <= 0.0 or min_distance < 0.0 or min_distance >= max_distance:
            raise ValueError("检测扇区距离范围无效。")
        if arc_samples < 3:
            raise ValueError("检测扇区圆弧采样数至少为 3。")
        angles = self.lower_ray.angle + np.linspace(0.0, self.angle_width, arc_samples)
        outer = [
            Point(
                self.origin.x + max_distance * cos(radians(float(angle))),
                self.origin.y + max_distance * sin(radians(float(angle))),
            )
            for angle in angles
        ]
        geometry = Region.from_vertices([self.origin, *outer])
        if min_distance > 0.0:
            geometry = geometry.difference(Region.disk(self.origin, min_distance))
        return geometry


@dataclass
class Polygon:
    """由无序线段集合表示的多边形边界。

    边的存放顺序和端点方向均无要求，但非退化多边形必须具有完整边界。
    空列表表示空集；一条非零线段表示退化线段；一条零长度线段表示单点。
    本题的裁剪操作以输入区域为凸集为前提。
    """

    edges: list[Segment]

    def __post_init__(self) -> None:
        self.edges = list(self.edges)

    @property
    def vertices(self) -> list[Point]:
        """收集全部边端点，并按浮点容差去重。"""
        return _unique_points([point for edge in self.edges for point in (edge.start, edge.end)])

    def contains(self, point: Point) -> bool:
        """判断点是否位于多边形内部或边界上。"""
        if not self.edges:
            return False
        if any(edge.contains(point) for edge in self.edges):
            return True

        if len(self.vertices) < 3:
            return False

        inside = False
        for edge in self.edges:
            start, end = edge.start, edge.end
            if (start.y > point.y) != (end.y > point.y):
                crossing_x = start.x + (point.y - start.y) * (end.x - start.x) / (end.y - start.y)
                if crossing_x > point.x + _TOLERANCE:
                    inside = not inside
        return inside

    def diameter(self) -> float:
        """返回区域内任意两点间的最大距离。空集时抛出 ValueError。"""
        return self.diameter_with_endpoints()[0]

    def diameter_with_endpoints(self) -> tuple[float, Point, Point]:
        """返回直径及一组最远点对；单点区域返回两个相同端点。"""
        vertices = self.vertices
        if not vertices:
            raise ValueError("空多边形不存在直径。")
        return max(
            ((first.distance_to(second), first, second) for index, first in enumerate(vertices)
             for second in vertices[index + 1:]),
            key=lambda result: result[0],
            default=(0.0, vertices[0], vertices[0]),
        )


@dataclass
class TargetArea:
    """题目的目标圆域及跨图形的几何操作。

圆心固定为全局坐标原点；管理集合只用于保存需要长期使用的对象，
交集和裁剪产生的临时图形不会自动登记。
"""

    radius: float = 1800.0
    center: Point = field(default_factory=lambda: Point(0.0, 0.0), init=False)
    points: list[Point] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    rays: list[Ray] = field(default_factory=list)
    polygons: list[Polygon] = field(default_factory=list)
    detection_sectors: list[DetectionSector] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("目标圆域半径必须为正数。")

    def add_point(self, x: float, y: float) -> Point:
        point = Point(x, y)
        self.points.append(point)
        return point

    def add_segment(self, start: Point, end: Point) -> Segment:
        segment = Segment(start, end)
        self.segments.append(segment)
        return segment

    def add_ray(self, origin: Point, angle: float) -> Ray:
        ray = Ray(origin, angle)
        self.rays.append(ray)
        return ray

    def add_polygon(self, edges: list[Segment]) -> Polygon:
        polygon = Polygon(edges)
        self.polygons.append(polygon)
        return polygon

    def add_detection_sector(self, lower_ray: Ray, upper_ray: Ray) -> DetectionSector:
        sector = DetectionSector(lower_ray, upper_ray)
        self.detection_sectors.append(sector)
        return sector

    def contains(self, point: Point) -> bool:
        """判断点是否在目标圆域内或圆周上。"""
        return self.center.distance_to(point) <= self.radius + _TOLERANCE

    def intersect_segments(self, first: Segment, second: Segment) -> Point | Segment | None:
        """返回两条闭线段的交集。重叠部分沿 first 的方向返回。"""
        px, py = first.start.x, first.start.y
        rx, ry = first.end.x - px, first.end.y - py
        qx, qy = second.start.x, second.start.y
        sx, sy = second.end.x - qx, second.end.y - qy
        first_length_sq, second_length_sq = _dot(rx, ry, rx, ry), _dot(sx, sy, sx, sy)

        if first_length_sq <= _TOLERANCE * _TOLERANCE:
            return first.start if second.contains(first.start) else None
        if second_length_sq <= _TOLERANCE * _TOLERANCE:
            return second.start if first.contains(second.start) else None

        denominator = _cross(rx, ry, sx, sy)
        q_minus_p_x, q_minus_p_y = qx - px, qy - py
        if abs(denominator) > _TOLERANCE:
            first_parameter = _cross(q_minus_p_x, q_minus_p_y, sx, sy) / denominator
            second_parameter = _cross(q_minus_p_x, q_minus_p_y, rx, ry) / denominator
            if (-_TOLERANCE <= first_parameter <= 1.0 + _TOLERANCE
                    and -_TOLERANCE <= second_parameter <= 1.0 + _TOLERANCE):
                return first.point_at(first_parameter)
            return None

        if abs(_cross(q_minus_p_x, q_minus_p_y, rx, ry)) > _TOLERANCE * np.linalg.norm(np.array([rx, ry])):
            return None
        start_parameter = _dot(q_minus_p_x, q_minus_p_y, rx, ry) / first_length_sq
        end_parameter = _dot(second.end.x - px, second.end.y - py, rx, ry) / first_length_sq
        lower = max(0.0, min(start_parameter, end_parameter))
        upper = min(1.0, max(start_parameter, end_parameter))
        if upper < lower - _TOLERANCE:
            return None
        if upper <= lower + _TOLERANCE:
            return first.point_at((lower + upper) / 2.0)
        return Segment(first.point_at(lower), first.point_at(upper))

    def intersect_ray_segment(self, ray: Ray, segment: Segment) -> Point | Segment | None:
        """返回射线和闭线段的交集。重叠部分沿 segment 的方向返回。"""
        angle_rad = radians(ray.angle)
        dx, dy = cos(angle_rad), sin(angle_rad)
        qx, qy = segment.start.x, segment.start.y
        sx, sy = segment.end.x - qx, segment.end.y - qy
        segment_length_sq = _dot(sx, sy, sx, sy)

        if segment_length_sq <= _TOLERANCE * _TOLERANCE:
            return segment.start if ray.contains(segment.start) else None

        q_minus_p_x, q_minus_p_y = qx - ray.origin.x, qy - ray.origin.y
        denominator = _cross(dx, dy, sx, sy)
        if abs(denominator) > _TOLERANCE:
            ray_parameter = _cross(q_minus_p_x, q_minus_p_y, sx, sy) / denominator
            segment_parameter = _cross(q_minus_p_x, q_minus_p_y, dx, dy) / denominator
            if ray_parameter >= -_TOLERANCE and -_TOLERANCE <= segment_parameter <= 1.0 + _TOLERANCE:
                return segment.point_at(segment_parameter)
            return None

        if abs(_cross(q_minus_p_x, q_minus_p_y, dx, dy)) > _TOLERANCE:
            return None
        start_projection = _dot(segment.start.x - ray.origin.x, segment.start.y - ray.origin.y, dx, dy)
        end_projection = _dot(segment.end.x - ray.origin.x, segment.end.y - ray.origin.y, dx, dy)
        if max(start_projection, end_projection) < -_TOLERANCE:
            return None
        if min(start_projection, end_projection) >= -_TOLERANCE:
            return segment
        origin_parameter = start_projection / (start_projection - end_projection)
        origin = segment.point_at(origin_parameter)
        if start_projection > end_projection:
            return Segment(segment.start, origin)
        return Segment(origin, segment.end)

    def intersect_rays(self, first: Ray, second: Ray) -> Point | Segment | Ray | None:
        """返回两条射线的交集。共线同向时返回交集射线。"""
        first_angle, second_angle = radians(first.angle), radians(second.angle)
        rx, ry = cos(first_angle), sin(first_angle)
        sx, sy = cos(second_angle), sin(second_angle)
        q_minus_p_x = second.origin.x - first.origin.x
        q_minus_p_y = second.origin.y - first.origin.y
        denominator = _cross(rx, ry, sx, sy)
        if abs(denominator) > _TOLERANCE:
            first_parameter = _cross(q_minus_p_x, q_minus_p_y, sx, sy) / denominator
            second_parameter = _cross(q_minus_p_x, q_minus_p_y, rx, ry) / denominator
            if first_parameter >= -_TOLERANCE and second_parameter >= -_TOLERANCE:
                return Point(first.origin.x + first_parameter * rx, first.origin.y + first_parameter * ry)
            return None

        if abs(_cross(q_minus_p_x, q_minus_p_y, rx, ry)) > _TOLERANCE:
            return None
        alignment = _dot(rx, ry, sx, sy)
        second_origin_on_first = _dot(q_minus_p_x, q_minus_p_y, rx, ry)
        if alignment > 0.0:
            return Ray(second.origin if second_origin_on_first >= 0.0 else first.origin, first.angle)
        if second_origin_on_first < -_TOLERANCE:
            return None
        if second_origin_on_first <= _TOLERANCE:
            return first.origin
        return Segment(first.origin, second.origin)

    def intersect_detection_area(
        self,
        polygon: Polygon,
        detection_area: DetectionSector,
    ) -> Polygon:
        """用检测夹角的两个支撑半平面依次裁剪凸多边形。"""
        if detection_area.angle_width > 180.0 + _TOLERANCE:
            raise ValueError("该方法只适用于角宽不超过 180° 的检测区域。")
        if detection_area.angle_width <= _TOLERANCE:
            raise ValueError("检测区域的角宽必须大于 0°。")
        if not polygon.edges:
            return Polygon([])
        intermediate = self._clip_polygon_by_half_plane(
            polygon, detection_area.lower_ray, keep_left=True
        )
        return self._clip_polygon_by_half_plane(
            intermediate, detection_area.upper_ray, keep_left=False
        )

    def _clip_polygon_by_half_plane(
        self,
        polygon: Polygon,
        boundary: Ray,
        keep_left: bool,
    ) -> Polygon:
        """用 ``boundary`` 的支撑直线裁剪凸区域，边可无序且可反向。"""
        if not polygon.edges:
            return Polygon([])

        angle_rad = radians(boundary.angle)
        direction = np.array([cos(angle_rad), sin(angle_rad)], dtype=float)
        origin = np.array([boundary.origin.x, boundary.origin.y], dtype=float)

        def signed_value(point: Point) -> float:
            offset = np.array([point.x, point.y], dtype=float) - origin
            return float(np.dot(direction, np.array([offset[1], -offset[0]])))

        def is_inside(value: float) -> bool:
            return value >= -_TOLERANCE if keep_left else value <= _TOLERANCE

        kept_edges: list[Segment] = []
        intersections: list[Point] = []
        retained_points: list[Point] = []

        for edge in polygon.edges:
            start_value = signed_value(edge.start)
            end_value = signed_value(edge.end)
            start_inside = is_inside(start_value)
            end_inside = is_inside(end_value)

            if edge.length() <= _TOLERANCE:
                if start_inside:
                    retained_points.append(edge.start)
                continue

            if start_inside and end_inside:
                kept_edges.append(edge)
                continue
            if not start_inside and not end_inside:
                continue

            denominator = start_value - end_value
            if abs(denominator) <= _TOLERANCE:
                # 容差使近乎平行的端点分居两类时，边界端点就是稳定交点。
                intersection = edge.start if abs(start_value) <= abs(end_value) else edge.end
            else:
                intersection = edge.point_at(start_value / denominator)
            intersections.append(intersection)
            inside_point = edge.start if start_inside else edge.end
            if not _same_point(inside_point, intersection):
                kept_edges.append(Segment(inside_point, intersection))
            else:
                retained_points.append(intersection)

        unique_intersections = _unique_points(intersections)
        if len(unique_intersections) >= 2:
            cut_start, cut_end = max(
                (
                    (first, second)
                    for index, first in enumerate(unique_intersections)
                    for second in unique_intersections[index + 1:]
                ),
                key=lambda pair: pair[0].distance_to(pair[1]),
            )
            if not _same_point(cut_start, cut_end):
                kept_edges.append(Segment(cut_start, cut_end))
        elif len(unique_intersections) == 1:
            retained_points.append(unique_intersections[0])

        normalized = self._normalize_edges(kept_edges)
        if normalized:
            return Polygon(normalized)

        unique_points = _unique_points(retained_points)
        if not unique_points:
            return Polygon([])
        if len(unique_points) == 1:
            return Polygon([Segment(unique_points[0], unique_points[0])])
        first, second = max(
            (
                (first, second)
                for index, first in enumerate(unique_points)
                for second in unique_points[index + 1:]
            ),
            key=lambda pair: pair[0].distance_to(pair[1]),
        )
        return Polygon([Segment(first, second)])

    @staticmethod
    def _normalize_edges(edges: list[Segment]) -> list[Segment]:
        """移除零长度边和方向无关的重复边。"""
        result: list[Segment] = []
        for edge in edges:
            if edge.length() <= _TOLERANCE:
                continue
            if not any(_same_segment(edge, existing) for existing in result):
                result.append(edge)
        return result


# Shapely is the single backend for continuous-region topology.  The legacy
# Segment/Polygon/TargetArea classes above remain temporarily for API
# compatibility; new P1/P2 code uses Region instead.


@dataclass(frozen=True)
class GeometryTolerance:
    coordinate: float = 1e-8
    distance: float = 1e-7
    angle_deg: float = 1e-10
    area: float = 1e-8


GEOMETRY_TOLERANCE = GeometryTolerance()


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float

    def __post_init__(self) -> None:
        if self.radius < 0.0:
            raise ValueError("圆半径不能为负数。")

    def contains(self, point: Point) -> bool:
        return self.center.distance_to(point) <= self.radius + GEOMETRY_TOLERANCE.distance

    def to_region(self, quad_segs: int = 64) -> "Region":
        return Region.disk(self.center, self.radius, quad_segs)


class Region:
    """Shapely geometry wrapper exposed to the problem modules."""

    __slots__ = ("_geometry",)

    def __init__(self, geometry: BaseGeometry) -> None:
        self._geometry = geometry

    @classmethod
    def empty(cls) -> "Region":
        return cls(GeometryCollection())

    @classmethod
    def disk(cls, center: Point, radius: float, quad_segs: int = 64) -> "Region":
        if radius < 0.0:
            raise ValueError("圆半径不能为负数。")
        return cls(ShapelyPoint(center.x, center.y).buffer(radius, quad_segs=quad_segs))

    @classmethod
    def from_vertices(cls, vertices: Sequence[Point]) -> "Region":
        if len(vertices) < 3:
            return cls.empty()
        return cls(ShapelyPolygon([(point.x, point.y) for point in vertices]))

    @classmethod
    def from_geometry(cls, geometry: BaseGeometry) -> "Region":
        return cls(geometry)

    @classmethod
    def union_all(cls, regions: Iterable["Region"]) -> "Region":
        return cls(unary_union([region._geometry for region in regions]))

    @property
    def is_empty(self) -> bool:
        return self._geometry.is_empty

    @property
    def area(self) -> float:
        return float(self._geometry.area)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        values = self._geometry.bounds
        return float(values[0]), float(values[1]), float(values[2]), float(values[3])

    @property
    def convex_hull(self) -> "Region":
        return Region(self._geometry.convex_hull)

    @property
    def vertices(self) -> list[Point]:
        return _unique_points([Point(x, y) for x, y in self.boundary_coordinates()])

    def intersection(self, *others: "Region") -> "Region":
        geometry = self._geometry
        for other in others:
            geometry = geometry.intersection(other._geometry)
        return Region(geometry)

    def difference(self, other: "Region") -> "Region":
        return Region(self._geometry.difference(other._geometry))

    def union(self, *others: "Region") -> "Region":
        return Region.union_all((self, *others))

    def covers(self, value: Point | "Region") -> bool:
        geometry = ShapelyPoint(value.x, value.y) if isinstance(value, Point) else value._geometry
        return bool(self._geometry.covers(geometry))

    def contains(self, point: Point) -> bool:
        return self.covers(point)

    def representative_point(self) -> Point:
        point = self._geometry.representative_point()
        return Point(float(point.x), float(point.y))

    def _polygon_parts(self) -> list[BaseGeometry]:
        if self._geometry.geom_type == "Polygon":
            return [self._geometry]
        if self._geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
            return [part for part in self._geometry.geoms if part.geom_type == "Polygon"]
        return []

    def boundary_coordinates(self) -> list[tuple[float, float]]:
        coordinates: list[tuple[float, float]] = []
        for polygon in self._polygon_parts():
            coordinates.extend((float(x), float(y)) for x, y in polygon.exterior.coords)
            for ring in polygon.interiors:
                coordinates.extend((float(x), float(y)) for x, y in ring.coords)
        if coordinates:
            return coordinates
        if hasattr(self._geometry, "coords"):
            return [(float(x), float(y)) for x, y in self._geometry.coords]
        return []

    def exterior_rings(self) -> list[list[tuple[float, float]]]:
        return [
            [(float(x), float(y)) for x, y in polygon.exterior.coords]
            for polygon in self._polygon_parts()
        ]

    def diameter_with_endpoints(self) -> tuple[float, Point, Point]:
        result = region_diameter(self)
        return result.length, result.first, result.second

    def diameter(self) -> float:
        return self.diameter_with_endpoints()[0]


@dataclass(frozen=True)
class RegionDiameter:
    length: float
    first: Point
    second: Point


def region_diameter(region: Region) -> RegionDiameter:
    vertices = region.convex_hull.vertices
    if not vertices:
        raise ValueError("空区域不存在直径。")
    length, first, second = max(
        (
            (first.distance_to(second), first, second)
            for index, first in enumerate(vertices)
            for second in vertices[index + 1:]
        ),
        key=lambda result: result[0],
        default=(0.0, vertices[0], vertices[0]),
    )
    return RegionDiameter(length, first, second)


def minimum_enclosing_circle(points: Sequence[Point], random_seed: int = 2026) -> Circle:
    """Return the minimum enclosing circle of a non-empty point sequence."""
    if not points:
        raise ValueError("空点集不存在最小覆盖圆。")

    def from_two(first: Point, second: Point) -> Circle:
        center = Point((first.x + second.x) / 2.0, (first.y + second.y) / 2.0)
        return Circle(center, center.distance_to(first))

    def from_three(first: Point, second: Point, third: Point) -> Circle | None:
        determinant = 2.0 * (
            first.x * (second.y - third.y)
            + second.x * (third.y - first.y)
            + third.x * (first.y - second.y)
        )
        if abs(determinant) <= 1e-12:
            return None
        first_sq = first.x * first.x + first.y * first.y
        second_sq = second.x * second.x + second.y * second.y
        third_sq = third.x * third.x + third.y * third.y
        center = Point(
            (
                first_sq * (second.y - third.y)
                + second_sq * (third.y - first.y)
                + third_sq * (first.y - second.y)
            ) / determinant,
            (
                first_sq * (third.x - second.x)
                + second_sq * (first.x - third.x)
                + third_sq * (second.x - first.x)
            ) / determinant,
        )
        return Circle(center, center.distance_to(first))

    def covers(circle: Circle, point: Point) -> bool:
        tolerance = max(GEOMETRY_TOLERANCE.distance, circle.radius * 1e-10)
        return circle.center.distance_to(point) <= circle.radius + tolerance

    shuffled = list(points)
    np.random.default_rng(random_seed).shuffle(shuffled)
    circle: Circle | None = None
    for index, first in enumerate(shuffled):
        if circle is not None and covers(circle, first):
            continue
        circle = Circle(first, 0.0)
        for second_index, second in enumerate(shuffled[:index]):
            if covers(circle, second):
                continue
            circle = from_two(first, second)
            for third in shuffled[:second_index]:
                if covers(circle, third):
                    continue
                circumcircle = from_three(first, second, third)
                if circumcircle is not None:
                    circle = circumcircle
                else:
                    pairs = ((first, second), (first, third), (second, third))
                    circle = max((from_two(*pair) for pair in pairs), key=lambda item: item.radius)
    assert circle is not None
    return circle


def point_key(point: Point, precision: float = 1e-6) -> tuple[int, int]:
    if precision <= 0.0:
        raise ValueError("坐标键精度必须为正数。")
    return round(point.x / precision), round(point.y / precision)
