"""完整扇形的本地模板缓存；被场地裁剪时仍调用实时 P2。

缓存仅加载本程序写入的本地 pickle，不能从不可信来源复制缓存文件。
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import warnings

import shapely
from shapely.affinity import affine_transform

from CUMCM_B_FINAL_SUBMITION.P2 import Problem2Config, Problem2Result, _validate_inputs, solve_problem_2
from CUMCM_B_FINAL_SUBMITION.utils import DetectionSector, Point, Region


CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".p2_cache"


def full_sector(position: Point, bearing: float, config: Problem2Config) -> Region:
    return DetectionSector.from_measurement(
        position, bearing, config.bearing_error_deg
    ).to_region(config.receive_radius_max, config.near_radius, config.sector_arc_samples)


def sector_is_complete(position: Point, bearing: float, config: Problem2Config) -> bool:
    """同时检查连续扇形最远点及当前 P2 多边形覆盖，保守处理边界。"""
    radial_angle = math.degrees(math.atan2(position.y, position.x))
    delta = (radial_angle - bearing + 180) % 360 - 180
    angles = [bearing - config.bearing_error_deg, bearing + config.bearing_error_deg]
    if abs(delta) <= config.bearing_error_deg:
        angles.append(radial_angle)
    for angle in angles:
        for radius in (config.near_radius, config.receive_radius_max):
            x = position.x + radius * math.cos(math.radians(angle))
            y = position.y + radius * math.sin(math.radians(angle))
            if math.hypot(x, y) >= config.target_radius - 1e-7:
                return False
    field = Region.disk(Point(0, 0), config.target_radius, config.circle_quad_segs)
    return field.covers(full_sector(position, bearing, config))


def template_cache_key(config: Problem2Config) -> str:
    # 自动随评分、序列化和几何实现变化失效，避免遗忘更新版本号。
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True, allow_nan=False).encode())
    digest.update(f"{CACHE_VERSION}:{shapely.__version__}".encode())
    for name in ("P2.py", "utils.py", "P2_cache.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def transform_template(template: Problem2Result, position: Point, bearing: float) -> Problem2Result:
    """深复制并变换所有空间量，保证多频道不会修改公共模板。"""
    angle = math.radians(bearing)
    c, s = math.cos(angle), math.sin(angle)

    def point(p: Point) -> Point:
        return Point(position.x + c * p.x - s * p.y, position.y + s * p.x + c * p.y)

    def region(r: Region) -> Region:
        return Region.from_geometry(affine_transform(
            r._geometry, [c, -s, s, c, position.x, position.y]
        ))

    candidates = []
    for candidate in template.candidates:
        cases = []
        for case in candidate.posterior_cases:
            observation = case.observation
            if observation.bearing_deg is not None:
                observation = replace(observation, bearing_deg=(observation.bearing_deg + bearing) % 360)
            cases.append(replace(case, observation=observation, clear_center=point(case.clear_center)))
        candidates.append(replace(candidate, point=point(candidate.point), posterior_cases=cases))
    mapping = {id(old): new for old, new in zip(template.candidates, candidates)}
    return replace(
        template,
        first_position=Point(position.x, position.y), first_bearing_deg=bearing % 360,
        first_feasible_region=region(template.first_feasible_region),
        safe_candidate_region=region(template.safe_candidate_region),
        recommended_region=region(template.recommended_region),
        target_cells=[replace(cell, representative=point(cell.representative), geometry=region(cell.geometry))
                      for cell in template.target_cells],
        states=[replace(state, target=point(state.target)) for state in template.states],
        candidates=candidates,
        best_candidate=mapping[id(template.best_candidate)],
        pareto_candidates=[mapping[id(candidate)] for candidate in template.pareto_candidates],
        result_source="template_reuse",
    )


def get_problem_2_result(first_position, first_bearing_deg: float,
                         config: Problem2Config | None = None, *,
                         cache_dir: Path | str | None = None) -> Problem2Result:
    config = config or Problem2Config()
    first = _validate_inputs(first_position, first_bearing_deg)
    bearing = float(first_bearing_deg) % 360
    if not sector_is_complete(first, bearing, config):
        result = solve_problem_2(first_position, bearing, config)
        result.result_source = "boundary_solve"
        return result

    key = template_cache_key(config)
    path = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_file = path / f"{key}.pickle"
    template = None
    if cache_file.exists():
        try:
            with cache_file.open("rb") as stream:
                payload = pickle.load(stream)
            if (payload["key"] == key and isinstance(payload["result"], Problem2Result)
                    and payload["result"].config == config):
                template = payload["result"]
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, KeyError, TypeError, ValueError):
            warnings.warn("P2 模板缓存无效，将重新生成。", RuntimeWarning)
    created = template is None
    if created:
        template = solve_problem_2((0, 0), 0, config,
                                  _first_region=full_sector(Point(0, 0), 0, config))
        temp_path = None
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path, suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                pickle.dump({"key": key, "result": template}, stream, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, cache_file)
        except OSError as exc:
            warnings.warn(f"P2 缓存写入失败，本次仍使用已计算结果：{exc}", RuntimeWarning)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    result = transform_template(template, first, bearing)
    if created:
        result.result_source = "template_build"
    return result
