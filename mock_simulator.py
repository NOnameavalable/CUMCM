#!/usr/bin/env python3
"""2026 年全国大学生数学建模竞赛 B 题《无线电干扰源的快速自动定位与清除》
高保真本地 Mock 模拟器服务端 + 可视化 WebUI (mock_simulator.py)

特性：
1. 100% 遵循官方协议与附件1、2物理模型（支持 P3 全向模式 与 P4 全向+定向混合模式）。
2. 纯 Python 3 标准库实现（http.server, json, math, random, argparse 等），无外部依赖。
3. 严格的虚拟时间（virtual_time_s）计算与状态机推进（移动、切频、测向、清除）。
4. 示向度均匀噪声（[-1.0°, 1.0°]）与近场（<=5m）饱和检测。
5. 朝向无关的光学精确定位与激光清除（<=20m）。
6. 支持请求幂等性（重复 request_id 返回缓存响应）。
7. 持久化日志系统：自动在 ./logs/ 目录记录每次对战数据，即时落盘，算法异常崩溃亦可完整回放。
8. 现代化内置 WebUI：打开 http://127.0.0.1:2026 即可在浏览器中享受实时对战监控与进度条历史回放，
   直观展示随机干扰源位置、定向源发射扇面、机器狗轨迹以及带有 ±1.0° 误差的测向观测锥形区域。
9. 内置 --test 自动化测试套件。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

# 确保本地回环地址请求不被系统代理拦截
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",127.0.0.1,localhost"

# 工作区日志目录与运行日志目录
WORKSPACE_DIR = Path(__file__).resolve().parent
LOGS_DIR = WORKSPACE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_LOGS_DIR = LOGS_DIR / "runtime"
RUNTIME_LOGS_DIR.mkdir(parents=True, exist_ok=True)



# =============================================================================
# 1. 物理几何与数学基础工具
# =============================================================================

def normalize_deg_360(deg: float) -> float:
    """归一化角度至 [0.0, 360.0)。"""
    deg = deg % 360.0
    if deg < 0.0:
        deg += 360.0
    return deg


def angle_difference_deg(deg1: float, deg2: float) -> float:
    """计算圆周上两角度的最短绝对角差，结果在 [0.0, 180.0] 度。"""
    diff = (deg1 - deg2 + 180.0) % 360.0 - 180.0
    return abs(diff)


def sample_area_uniform_disk(
    radius: float, rng: random.Random
) -> Tuple[float, float]:
    """在以原点为中心、半径为 radius 的圆盘内面积均匀随机采样一点 (x, y)。"""
    r = radius * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2.0 * math.pi)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y


# =============================================================================
# 2. 干扰源与会话日志引擎 (Crash-Resistant)
# =============================================================================

@dataclass
class Target:
    """单个干扰源的数据与状态。"""
    channel: int
    x: float
    y: float
    recv_radius: float
    is_directional: bool = False
    dir_angle_deg: float = 0.0  # 发射主朝向 [0, 360) 度，正东为0，正北为90
    cleared: bool = False

    def distance_to(self, px: float, py: float) -> float:
        return math.hypot(px - self.x, py - self.y)

    def is_in_coverage(self, px: float, py: float) -> bool:
        """判断位于 (px, py) 的机器狗是否落在该干扰源的有效辐射覆盖范围内。"""
        d = self.distance_to(px, py)
        if d > self.recv_radius:
            return False
        if not self.is_directional:
            return True
        # 定向源：从干扰源看机器狗的绝对角 phi
        if d < 1e-9:
            return True  # 机器狗与干扰源完全重合
        phi = normalize_deg_360(math.degrees(math.atan2(py - self.y, px - self.x)))
        diff = angle_difference_deg(phi, self.dir_angle_deg)
        return diff <= 90.0 + 1e-9  # 前向 180° 扇面（含边界）


@dataclass
class SimulationStats:
    """统计战报数据。"""
    total_actions: int = 0
    measure_count: int = 0
    measure_direction_count: int = 0
    measure_near_count: int = 0
    measure_no_signal_count: int = 0
    clear_count: int = 0
    clear_success_count: int = 0
    clear_failed_count: int = 0
    switch_channel_count: int = 0
    total_move_distance_m: float = 0.0
    total_move_time_s: float = 0.0
    total_switch_time_s: float = 0.0
    total_action_time_s: float = 0.0


class SessionLogger:
    """对战持久化日志管理器：每次动作原子落盘，保障即使异常退出亦可完整回放。"""

    def __init__(self, logs_dir: Path = LOGS_DIR, mode: str = "p4") -> None:
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.current_file: Optional[Path] = None
        self.session_data: Dict[str, Any] = {
            "session_id": "idle",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arena_id": "default",
            "robot_id": "none",
            "mode": mode.lower(),
            "seed": None,
            "arena_radius": 1800.0,
            "targets": [],
            "frames": [],
            "status": "idle",
            "summary": None,
        }

    def prepare_session(
        self,
        arena_id: str,
        robot_id: str,
        mode: str,
        seed: Optional[int],
        arena_radius: float,
        targets: Dict[int, Target],
    ) -> None:
        """待机就绪状态：记录目标点位以供实时预览，不生成具体的 session_*.json 日志文件。"""
        with self.lock:
            self.current_file = None  # 待机时不创建具体的 session_*.json
            target_records = [
                {
                    "channel": t.channel,
                    "x": round(t.x, 2),
                    "y": round(t.y, 2),
                    "recv_radius": round(t.recv_radius, 2),
                    "is_directional": t.is_directional,
                    "dir_angle_deg": round(t.dir_angle_deg, 2),
                    "cleared": False,
                }
                for ch, t in sorted(targets.items())
            ]
            self.session_data = {
                "session_id": "waiting_preview",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "arena_id": arena_id,
                "robot_id": robot_id,
                "mode": mode,
                "seed": seed,
                "arena_radius": arena_radius,
                "targets": target_records,
                "frames": [],
                "status": "waiting",
                "summary": None,
            }
            self._flush_to_disk()

    def activate_session(
        self,
        arena_id: str,
        robot_id: str,
    ) -> None:
        """开始对战：此时才正式分配 session 文件名并在 logs/ 目录下创建专属 session_*.json 日志。"""
        with self.lock:
            ts = time.strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{ts}"
            self.current_file = self.logs_dir / f"{session_id}.json"
            self.session_data["session_id"] = session_id
            self.session_data["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.session_data["arena_id"] = arena_id
            self.session_data["robot_id"] = robot_id
            self.session_data["status"] = "running"
            self.session_data["frames"] = [
                {
                    "step": 0,
                    "action": "enter",
                    "virtual_time_s": 0.0,
                    "robot_position": {"x": 0.0, "y": 0.0},
                    "channel": 1,
                    "measure_result": None,
                    "svd_deg": None,
                    "cone": None,
                    "clear_result": None,
                    "cleared_channels": [],
                    "description": f"机器狗 {robot_id} 在原点 (0.0, 0.0) 进入场地，信道默认为 1",
                }
            ]
            self._flush_to_disk()

    def start_session(
        self,
        arena_id: str,
        robot_id: str,
        mode: str,
        seed: Optional[int],
        arena_radius: float,
        targets: Dict[int, Target],
    ) -> None:
        with self.lock:
            ts = time.strftime("%Y%m%d_%H%M%S")
            session_id = f"session_{ts}"
            self.current_file = self.logs_dir / f"{session_id}.json"
            target_records = [
                {
                    "channel": t.channel,
                    "x": round(t.x, 2),
                    "y": round(t.y, 2),
                    "recv_radius": round(t.recv_radius, 2),
                    "is_directional": t.is_directional,
                    "dir_angle_deg": round(t.dir_angle_deg, 2),
                    "cleared": False,
                }
                for ch, t in sorted(targets.items())
            ]
            self.session_data = {
                "session_id": session_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "arena_id": arena_id,
                "robot_id": robot_id,
                "mode": mode,
                "seed": seed,
                "arena_radius": arena_radius,
                "targets": target_records,
                "frames": [
                    {
                        "step": 0,
                        "action": "enter",
                        "virtual_time_s": 0.0,
                        "robot_position": {"x": 0.0, "y": 0.0},
                        "channel": 1,
                        "measure_result": None,
                        "svd_deg": None,
                        "cone": None,
                        "clear_result": None,
                        "cleared_channels": [],
                        "description": "机器狗在原点 (0.0, 0.0) 进入场地，信道默认为 1",
                    }
                ],
                "status": "running",
                "summary": None,
            }
            self._flush_to_disk()

    def append_frame(
        self,
        frame: Dict[str, Any],
        targets: Dict[int, Target],
    ) -> None:
        with self.lock:
            # 记录当前已清除信道
            cleared_chs = [ch for ch, t in sorted(targets.items()) if t.cleared]
            frame["cleared_channels"] = cleared_chs
            # 更新 targets 列表中的 cleared 字段
            for t_info in self.session_data.get("targets", []):
                ch = t_info["channel"]
                if ch in targets:
                    t_info["cleared"] = targets[ch].cleared

            self.session_data["frames"].append(frame)
            self._flush_to_disk()

    def finish_session(
        self,
        exit_reason: str,
        stats: SimulationStats,
        virtual_time_s: float,
        targets: Dict[int, Target],
    ) -> None:
        with self.lock:
            cleared_chs = [ch for ch, t in sorted(targets.items()) if t.cleared]
            total_t = len(targets)
            self.session_data["status"] = "finished"
            self.session_data["summary"] = {
                "exit_reason": exit_reason,
                "total_virtual_time_s": round(virtual_time_s, 2),
                "total_targets": total_t,
                "cleared_count": len(cleared_chs),
                "clearance_rate": round(len(cleared_chs) / total_t * 100.0, 1) if total_t else 0.0,
                "total_actions": stats.total_actions,
                "measure_count": stats.measure_count,
                "clear_count": stats.clear_count,
                "clear_success_count": stats.clear_success_count,
                "total_move_distance_m": round(stats.total_move_distance_m, 1),
                "total_move_time_s": round(stats.total_move_time_s, 1),
                "total_switch_time_s": round(stats.total_switch_time_s, 1),
            }
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        try:
            # 仅在正式激活会话后才写入专属 session_*.json
            if self.current_file is not None:
                tmp_path = self.current_file.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self.session_data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.current_file)

            # 始终刷新 latest.json 供前端实时轮询
            latest_file = self.logs_dir / "latest.json"
            tmp_latest = self.logs_dir / "latest.tmp"
            with open(tmp_latest, "w", encoding="utf-8") as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_latest, latest_file)
        except Exception:
            pass

    def get_live_data(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.session_data)

    def list_sessions(self) -> List[Dict[str, Any]]:
        results = []
        for p in sorted(self.logs_dir.glob("session_*.json"), reverse=True):
            try:
                stat = p.stat()
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                summary = data.get("summary") or {}
                results.append({
                    "filename": p.name,
                    "session_id": data.get("session_id", p.stem),
                    "created_at": data.get("created_at", ""),
                    "mode": data.get("mode", "p4"),
                    "seed": data.get("seed"),
                    "robot_id": data.get("robot_id", ""),
                    "status": data.get("status", "unknown"),
                    "frames_count": len(data.get("frames", [])),
                    "cleared_count": summary.get("cleared_count"),
                    "total_targets": summary.get("total_targets", len(data.get("targets", []))),
                    "virtual_time_s": summary.get("total_virtual_time_s"),
                    "size_kb": round(stat.st_size / 1024, 1),
                })
            except Exception:
                continue
        return results

    def read_session_file(self, filename: str) -> Optional[Dict[str, Any]]:
        # 防止目录穿越
        safe_name = Path(filename).name
        target_path = self.logs_dir / safe_name
        if not target_path.is_file():
            return None
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None


# =============================================================================
# 3. 场地与仿真状态机
# =============================================================================

class Arena:
    """模拟器场地状态与仿真引擎。"""

    def __init__(
        self,
        mode: str = "p4",
        seed: Optional[int] = None,
        target_count: Optional[int] = None,
        arena_radius: float = 1800.0,
        dog_speed: float = 5.0,
        clear_radius: float = 20.0,
        near_radius: float = 5.0,
        max_virtual_duration_s: float = 360000.0,
        max_real_duration_s: float = 1200.0,
        verbose: bool = False,
        logger: Optional[SessionLogger] = None,
    ) -> None:
        self.mode = mode.lower()
        if self.mode not in ("p3", "p4"):
            raise ValueError(f"未知的模式 mode: {mode}，应为 'p3' 或 'p4'。")
        self.initial_seed = seed
        self.rng = random.Random(seed)
        self.target_count_override = target_count
        self.arena_radius = float(arena_radius)
        self.dog_speed = float(dog_speed)
        self.clear_radius = float(clear_radius)
        self.near_radius = float(near_radius)
        self.max_virtual_duration_s = float(max_virtual_duration_s)
        self.max_real_duration_s = float(max_real_duration_s)
        self.verbose = verbose
        self.logger = logger or SessionLogger(mode=self.mode)

        self.lock = threading.RLock()
        self.arena_id: str = "default"
        self.robot_id: str = "unknown"
        self.virtual_time_s: float = 0.0
        self.real_start_time_s: float = time.time()
        self.robot_x: float = 0.0
        self.robot_y: float = 0.0
        self.current_channel: int = 1
        self.targets: Dict[int, Target] = {}
        self.stats = SimulationStats()
        self.cached_responses: Dict[str, Dict[str, Any]] = {}
        self.step_counter: int = 0
        self.entered = False
        self.exited = False

        self.reset_session("default", "waiting")

    def set_mode(self, new_mode: str) -> None:
        """动态切换模拟器模式（P3 或 P4），并重新生成对应模式的干扰源。"""
        with self.lock:
            new_mode = new_mode.lower()
            if new_mode not in ("p3", "p4"):
                raise ValueError(f"无效的模式: {new_mode}")
            self.mode = new_mode
            self.reset_session(self.arena_id, "waiting")
            if self.verbose:
                print(f"[MockSimulator] 已切换至模式: {self.mode.upper()} 并刷新干扰源")

    def _init_targets(self) -> None:
        """依据规则生成干扰源。"""
        if self.target_count_override is not None:
            k = self.target_count_override
        else:
            k = self.rng.randint(10, 16)

        channels = sorted(self.rng.sample(range(1, 21), k))
        self.targets.clear()

        # P4 模式下至少有 1 个定向和 1 个全向
        if self.mode == "p4" and k >= 2:
            num_dir = self.rng.randint(max(1, k // 3), max(2, 2 * k // 3))
            dir_channels = set(self.rng.sample(channels, num_dir))
        else:
            dir_channels = set()

        for ch in channels:
            tx, ty = sample_area_uniform_disk(self.arena_radius, self.rng)
            recv_r = self.rng.uniform(1000.0, 1500.0)
            is_dir = (ch in dir_channels)
            dir_deg = self.rng.uniform(0.0, 360.0) if is_dir else 0.0
            self.targets[ch] = Target(
                channel=ch,
                x=tx,
                y=ty,
                recv_radius=recv_r,
                is_directional=is_dir,
                dir_angle_deg=dir_deg,
                cleared=False,
            )

    def reset_session(self, arena_id: str = "default", robot_id: str = "waiting") -> None:
        """初始化新会话状态，重新生成干扰源并落盘。"""
        with self.lock:
            self.arena_id = arena_id
            self.robot_id = robot_id
            self.virtual_time_s = 0.0
            self.real_start_time_s = time.time()
            self.robot_x = 0.0
            self.robot_y = 0.0
            self.current_channel = 1
            self.step_counter = 0
            self.stats = SimulationStats()
            self.cached_responses.clear()
            self.entered = (robot_id != "waiting")
            self.exited = False

            if self.initial_seed is not None:
                self.rng = random.Random(self.initial_seed)
            self._init_targets()

            # 待机时不创建具体的 session 文件，正式启动后才落盘
            if robot_id == "waiting":
                self.logger.prepare_session(
                    arena_id=arena_id,
                    robot_id=robot_id,
                    mode=self.mode,
                    seed=self.initial_seed,
                    arena_radius=self.arena_radius,
                    targets=self.targets,
                )
            else:
                self.logger.start_session(
                    arena_id=arena_id,
                    robot_id=robot_id,
                    mode=self.mode,
                    seed=self.initial_seed,
                    arena_radius=self.arena_radius,
                    targets=self.targets,
                )

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"[MockSimulator] 进入新会话: arena_id='{arena_id}', robot_id='{robot_id}'")
                print(f"  模式: {self.mode.upper()} | 种子: {self.initial_seed}")
                print(f"  目标总数: {len(self.targets)}")
                for ch, t in sorted(self.targets.items()):
                    type_str = f"定向(朝向 {t.dir_angle_deg:.1f}°)" if t.is_directional else "全向"
                    print(f"    信道 {ch:2d}: 位置=({t.x:8.2f}, {t.y:8.2f}) | 半径={t.recv_radius:6.1f}m | 类型={type_str}")
                print(f"{'='*70}\n")

    def _update_movement(self, new_x: float, new_y: float) -> float:
        """计算移动距离与耗时，并更新机器狗坐标。"""
        dx = new_x - self.robot_x
        dy = new_y - self.robot_y
        dist = math.hypot(dx, dy)
        t_move = dist / self.dog_speed
        self.robot_x = new_x
        self.robot_y = new_y
        self.stats.total_move_distance_m += dist
        self.stats.total_move_time_s += t_move
        return t_move

    def enter(self, request_id: str, arena_id: str, robot_id: str) -> Dict[str, Any]:
        """处理 /enter。"""
        with self.lock:
            if request_id in self.cached_responses:
                return self.cached_responses[request_id]

            # 如果当前处于待机状态且还未执行任何动作指令，则激活当前预览的目标点，并创建正式专属 session 日志
            if self.robot_id == "waiting" and self.step_counter == 0 and len(self.targets) > 0:
                self.arena_id = arena_id
                self.robot_id = robot_id
                self.virtual_time_s = 0.0
                self.real_start_time_s = time.time()
                self.robot_x = 0.0
                self.robot_y = 0.0
                self.current_channel = 1
                self.stats = SimulationStats()
                self.cached_responses.clear()
                self.entered = True
                self.exited = False
                self.logger.activate_session(arena_id, robot_id)
            else:
                self.reset_session(arena_id, robot_id)
                self.entered = True
                self.logger.activate_session(arena_id, robot_id)

            elapsed_real = time.time() - self.real_start_time_s
            remaining_real = max(0.0, self.max_real_duration_s - elapsed_real)

            res = {
                "accepted": True,
                "real_timestamp_ms": int(time.time() * 1000),
                "virtual_time_s": 0.0,
                "max_virtual_duration_s": int(self.max_virtual_duration_s),
                "max_real_duration_s": int(self.max_real_duration_s),
                "remaining_real_duration_s": round(remaining_real, 1),
            }
            self.cached_responses[request_id] = res
            return res

    def measure(
        self,
        request_id: str,
        pos_x: float,
        pos_y: float,
        channel: int,
    ) -> Dict[str, Any]:
        """处理 /measure。"""
        with self.lock:
            if request_id in self.cached_responses:
                return self.cached_responses[request_id]

            self.stats.total_actions += 1
            self.stats.measure_count += 1
            self.step_counter += 1

            # 1. 移动耗时
            t_move = self._update_movement(pos_x, pos_y)

            # 2. 切频耗时
            t_switch = 0.0
            if channel != self.current_channel:
                t_switch = 1.0
                self.stats.switch_channel_count += 1
                self.current_channel = channel
            self.stats.total_switch_time_s += t_switch

            # 3. 动作固定耗时 5.0s
            t_action = 5.0
            self.stats.total_action_time_s += t_action

            # 4. 推进虚拟时间
            self.virtual_time_s += t_move + t_switch + t_action
            vt = round(self.virtual_time_s, 4)

            # 5. 测向物理判断
            target = self.targets.get(channel)
            measure_result = "no_signal"
            svd_deg: Optional[float] = None
            cone_info: Optional[Dict[str, float]] = None
            gt_info = ""

            if target is not None and not target.cleared:
                d = target.distance_to(pos_x, pos_y)
                in_cov = target.is_in_coverage(pos_x, pos_y)
                gt_info = f"[GT] 目标距狗 {d:.2f}m, 覆盖={in_cov}"
                if in_cov:
                    if d <= self.near_radius:
                        measure_result = "near"
                        self.stats.measure_near_count += 1
                    else:
                        measure_result = "direction"
                        self.stats.measure_direction_count += 1
                        alpha_true = normalize_deg_360(
                            math.degrees(math.atan2(target.y - pos_y, target.x - pos_x))
                        )
                        noise = self.rng.uniform(-1.0, 1.0)
                        alpha_obs = normalize_deg_360(alpha_true + noise)
                        svd_deg = round(alpha_obs, 2)
                        # 示向度带有 ±1.0° 误差锥
                        cone_info = {
                            "center_deg": svd_deg,
                            "half_width_deg": 1.0,
                            "start_deg": normalize_deg_360(svd_deg - 1.0),
                            "end_deg": normalize_deg_360(svd_deg + 1.0),
                        }
                        gt_info += f", 真实角={alpha_true:.2f}°, 观测角={svd_deg:.2f}°"
                else:
                    self.stats.measure_no_signal_count += 1
            else:
                self.stats.measure_no_signal_count += 1
                if target is None:
                    gt_info = "[GT] 该信道无目标"
                else:
                    gt_info = "[GT] 该信道目标已被清除"

            res: Dict[str, Any] = {
                "accepted": True,
                "real_timestamp_ms": int(time.time() * 1000),
                "virtual_time_s": vt,
                "measure_result": measure_result,
            }
            if svd_deg is not None:
                res["svd_deg"] = svd_deg

            # 持久化记录当前帧
            desc = f"移动至 ({pos_x:.1f}, {pos_y:.1f})，信道 {channel} 测向 -> {measure_result}"
            if svd_deg is not None:
                desc += f" (观测角: {svd_deg:.2f}°)"
            frame_record = {
                "step": self.step_counter,
                "action": "measure",
                "virtual_time_s": vt,
                "robot_position": {"x": round(pos_x, 2), "y": round(pos_y, 2)},
                "channel": channel,
                "measure_result": measure_result,
                "svd_deg": svd_deg,
                "cone": cone_info,
                "clear_result": None,
                "description": desc,
            }
            self.logger.append_frame(frame_record, self.targets)

            if self.verbose:
                print(
                    f"[{vt:8.2f}s] MEASURE: 坐标=({pos_x:7.1f}, {pos_y:7.1f}) | 信道={channel:2d} | "
                    f"结果={measure_result:9s} | {gt_info}"
                )

            self.cached_responses[request_id] = res
            return res

    def clear(
        self,
        request_id: str,
        pos_x: float,
        pos_y: float,
        channel: int,
    ) -> Dict[str, Any]:
        """处理 /clear。"""
        with self.lock:
            if request_id in self.cached_responses:
                return self.cached_responses[request_id]

            self.stats.total_actions += 1
            self.stats.clear_count += 1
            self.step_counter += 1

            # 1. 移动耗时
            t_move = self._update_movement(pos_x, pos_y)

            # 注意：/clear 指令不改变测向机频段，不产生切频耗时
            # 2. 清除判定与动作耗时
            target = self.targets.get(channel)
            is_success = False
            gt_info = ""

            if target is not None and not target.cleared:
                d = target.distance_to(pos_x, pos_y)
                gt_info = f"[GT] 目标距狗 {d:.2f}m"
                if d <= self.clear_radius:
                    is_success = True
                    target.cleared = True
                    self.stats.clear_success_count += 1
                    t_action = 5.0  # 光学定位 3.0s + 激光清除 2.0s
                    gt_info += " -> 清除成功！"
                else:
                    self.stats.clear_failed_count += 1
                    t_action = 3.0  # 光学定位 3.0s，未发现目标
                    gt_info += f" > {self.clear_radius}m，未命中"
            else:
                self.stats.clear_failed_count += 1
                t_action = 3.0
                gt_info = "[GT] 该信道已清除或无目标"

            self.stats.total_action_time_s += t_action
            self.virtual_time_s += t_move + t_action
            vt = round(self.virtual_time_s, 4)

            clear_result = "success" if is_success else "no_target_in_range"
            res = {
                "accepted": True,
                "real_timestamp_ms": int(time.time() * 1000),
                "virtual_time_s": vt,
                "clear_result": clear_result,
            }

            # 持久化记录当前帧
            desc = f"移动至 ({pos_x:.1f}, {pos_y:.1f})，清除信道 {channel} -> {clear_result}"
            frame_record = {
                "step": self.step_counter,
                "action": "clear",
                "virtual_time_s": vt,
                "robot_position": {"x": round(pos_x, 2), "y": round(pos_y, 2)},
                "channel": channel,
                "measure_result": None,
                "svd_deg": None,
                "cone": None,
                "clear_result": clear_result,
                "description": desc,
            }
            self.logger.append_frame(frame_record, self.targets)

            if self.verbose:
                print(
                    f"[{vt:8.2f}s] CLEAR:   坐标=({pos_x:7.1f}, {pos_y:7.1f}) | 信道={channel:2d} | "
                    f"结果={clear_result:18s} | {gt_info}"
                )

            self.cached_responses[request_id] = res
            return res

    def exit(self, request_id: str) -> Dict[str, Any]:
        """处理 /exit。"""
        with self.lock:
            if request_id in self.cached_responses:
                return self.cached_responses[request_id]

            self.exited = True
            vt = round(self.virtual_time_s, 4)
            res = {
                "accepted": True,
                "real_timestamp_ms": int(time.time() * 1000),
                "virtual_time_s": vt,
                "exit_reason": "user_exit",
            }
            self.cached_responses[request_id] = res

            # 完成并总结日志
            self.logger.finish_session(
                exit_reason="user_exit",
                stats=self.stats,
                virtual_time_s=self.virtual_time_s,
                targets=self.targets,
            )

            if self.verbose:
                self.print_summary_report()

            return res

    def print_summary_report(self) -> None:
        """打印战报与结算信息。"""
        total_targets = len(self.targets)
        cleared_targets = sum(1 for t in self.targets.values() if t.cleared)
        rate = (cleared_targets / total_targets * 100.0) if total_targets else 0.0
        elapsed_real = time.time() - self.real_start_time_s

        print(f"\n{'='*30} 模拟器结算战报 {'='*30}")
        print(f"  会话信息: Arena='{self.arena_id}', Robot='{self.robot_id}'")
        print(f"  模式: {self.mode.upper()} | 种子: {self.initial_seed}")
        print(f"  总虚拟时间: {self.virtual_time_s:.2f} 秒 ({self.virtual_time_s/60:.2f} 分钟)")
        print(f"  程序运行用时: {elapsed_real:.2f} 秒")
        print(f"  干扰源清除进度: {cleared_targets} / {total_targets} ({rate:.1f}%)")
        print(f"  总移动路程: {self.stats.total_move_distance_m:.1f} 米 | 移动耗时: {self.stats.total_move_time_s:.1f} 秒")
        print(f"  频段切换次数: {self.stats.switch_channel_count} 次 | 切频耗时: {self.stats.total_switch_time_s:.1f} 秒")
        print(f"  测量次数: {self.stats.measure_count} (方向: {self.stats.measure_direction_count}, "
              f"近距: {self.stats.measure_near_count}, 无信号: {self.stats.measure_no_signal_count})")
        print(f"  清除尝试: {self.stats.clear_count} (成功: {self.stats.clear_success_count}, 失败: {self.stats.clear_failed_count})")

        cleared_list = [ch for ch, t in sorted(self.targets.items()) if t.cleared]
        remaining_list = [ch for ch, t in sorted(self.targets.items()) if not t.cleared]
        print(f"  已清除信道: {cleared_list}")
        if remaining_list:
            print(f"  未清除信道: {remaining_list}")
            for ch in remaining_list:
                t = self.targets[ch]
                d = t.distance_to(self.robot_x, self.robot_y)
                t_type = f"定向({t.dir_angle_deg:.1f}°)" if t.is_directional else "全向"
                print(f"    - 信道 {ch}: 位置=({t.x:.1f}, {t.y:.1f}) | 类型={t_type} | 距狗={d:.1f}m")
        print(f"{'='*76}\n")

    def get_status(self) -> Dict[str, Any]:
        """供调试检查当前场地信息的结构。"""
        with self.lock:
            return {
                "arena_id": self.arena_id,
                "robot_id": self.robot_id,
                "mode": self.mode,
                "virtual_time_s": round(self.virtual_time_s, 4),
                "robot_position": {"x": self.robot_x, "y": self.robot_y},
                "current_channel": self.current_channel,
                "stats": asdict(self.stats),
                "targets": {
                    ch: {
                        "x": t.x,
                        "y": t.y,
                        "recv_radius": t.recv_radius,
                        "is_directional": t.is_directional,
                        "dir_angle_deg": t.dir_angle_deg,
                        "cleared": t.cleared,
                    }
                    for ch, t in self.targets.items()
                },
            }


# =============================================================================
# 4. 内置现代化 WebUI 单页应用 (HTML5 + CSS + Canvas + Vanilla JS)
# =============================================================================

WEBUI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>2026 CUMCM B题 无线电干扰源定位与清除 · 战术态势控制台</title>
  <style>
    :root {
      --bg-dark: #090d16;
      --panel-bg: #131b2e;
      --panel-border: #1e293b;
      --accent-cyan: #06b6d4;
      --accent-blue: #3b82f6;
      --accent-orange: #f59e0b;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --accent-purple: #a855f7;
      --text-main: #f8fafc;
      --text-dim: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg-dark); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }
    
    /* 左侧地图主区域 */
    #mapContainer {
      flex: 1;
      position: relative;
      background: radial-gradient(circle at center, #111827 0%, #030712 100%);
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: hidden;
    }
    canvas { display: block; cursor: grab; }
    canvas:active { cursor: grabbing; }

    /* 地图浮动工具栏 */
    .map-overlay-top {
      position: absolute;
      top: 16px;
      left: 20px;
      background: rgba(19, 27, 46, 0.85);
      backdrop-filter: blur(12px);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 10px 18px;
      display: flex;
      align-items: center;
      gap: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    .badge-mode {
      background: #0284c7;
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      letter-spacing: 0.5px;
    }
    .map-controls-floating {
      position: absolute;
      bottom: 24px;
      left: 20px;
      display: flex;
      gap: 10px;
    }
    .btn-tool {
      background: rgba(19, 27, 46, 0.85);
      border: 1px solid var(--panel-border);
      color: var(--text-main);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 13px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-tool:hover { background: #1e293b; border-color: var(--accent-cyan); color: var(--accent-cyan); }

    /* 右侧控制看板 */
    #sidebar {
      width: 440px;
      background: var(--panel-bg);
      border-left: 1px solid var(--panel-border);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow-y: auto;
    }
    .sidebar-header {
      padding: 18px 20px;
      border-bottom: 1px solid var(--panel-border);
      background: #0f172a;
    }
    .sidebar-header h1 { font-size: 16px; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 8px; }
    .sidebar-header p { font-size: 12px; color: var(--text-dim); margin-top: 4px; }

    /* 会话选择器 */
    .session-selector-box {
      padding: 14px 20px;
      border-bottom: 1px solid var(--panel-border);
      display: flex;
      gap: 10px;
      align-items: center;
      background: rgba(15, 23, 42, 0.5);
    }
    select {
      flex: 1;
      background: #0b1120;
      color: var(--text-main);
      border: 1px solid var(--panel-border);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 13px;
      outline: none;
      cursor: pointer;
    }
    select:focus { border-color: var(--accent-cyan); }

    /* 实时模式开关 */
    .live-toggle {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--text-dim);
      cursor: pointer;
    }
    .live-dot { width: 10px; height: 10px; border-radius: 50%; background: #64748b; }
    .live-dot.active { background: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }

    /* 核心指标看板 */
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--panel-border);
    }
    .metric-card {
      background: #0b1120;
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 12px;
    }
    .metric-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 20px; font-weight: 700; margin-top: 4px; color: var(--accent-cyan); }

    /* 回放进度控制器 */
    .replay-box {
      padding: 16px 20px;
      border-bottom: 1px solid var(--panel-border);
      background: #0f172a;
    }
    .timeline-slider {
      width: 100%;
      margin: 10px 0;
      accent-color: var(--accent-cyan);
      cursor: pointer;
    }
    .replay-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 6px;
    }
    .btn-group-replay { display: flex; gap: 6px; }
    .btn-replay {
      background: #1e293b;
      border: 1px solid var(--panel-border);
      color: var(--text-main);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 13px;
      cursor: pointer;
    }
    .btn-replay:hover { background: #334155; }
    .speed-select {
      background: #1e293b;
      border: 1px solid var(--panel-border);
      color: var(--text-main);
      padding: 6px 8px;
      border-radius: 6px;
      font-size: 12px;
    }

    /* 选项卡面板 */
    .tabs-bar {
      display: flex;
      border-bottom: 1px solid var(--panel-border);
      background: #0b1120;
    }
    .tab-btn {
      flex: 1;
      padding: 10px;
      text-align: center;
      font-size: 13px;
      color: var(--text-dim);
      border: none;
      background: transparent;
      cursor: pointer;
      border-bottom: 2px solid transparent;
    }
    .tab-btn.active { color: var(--accent-cyan); border-bottom-color: var(--accent-cyan); font-weight: 600; }

    /* 目标表格与流水列表 */
    .tab-content { flex: 1; overflow-y: auto; padding: 12px 16px; font-size: 12px; }
    .target-table { width: 100%; border-collapse: collapse; }
    .target-table th, .target-table td { padding: 8px 6px; text-align: left; border-bottom: 1px solid #1e293b; }
    .target-table th { color: var(--text-dim); font-size: 11px; }
    .tag-status {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 600;
    }
    .tag-cleared { background: rgba(16, 185, 129, 0.2); color: var(--accent-green); }
    .tag-active { background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); }

    .log-item {
      padding: 8px 10px;
      border-radius: 6px;
      margin-bottom: 6px;
      background: #0b1120;
      border-left: 3px solid #334155;
      cursor: pointer;
      transition: background 0.15s;
    }
    .log-item:hover { background: #1e293b; }
    .log-item.current { border-left-color: var(--accent-cyan); background: rgba(6, 182, 212, 0.1); }
    .log-header { display: flex; justify-content: space-between; color: var(--text-dim); margin-bottom: 3px; font-size: 11px; }
    .log-body { color: var(--text-main); word-break: break-all; }
  </style>
</head>
<body>

  <!-- 左侧：战术雷达地图 -->
  <div id="mapContainer">
    <div class="map-overlay-top">
      <select id="modeSelect" onchange="onModeSelectChange(this.value)" class="badge-mode" style="border:none; cursor:pointer; outline:none;">
        <option value="p3">🟢 P3 全向模式 (7检测圆)</option>
        <option value="p4" selected>🔵 P4 混合模式 (17检测圆)</option>
      </select>
      <button class="btn-tool" onclick="resetArena()" style="background:#dc2626; color:#fff; border:none; padding:4px 12px; font-weight:600; border-radius:6px; font-size:12px; cursor:pointer; display:flex; align-items:center; gap:4px; box-shadow: 0 2px 8px rgba(220,38,38,0.4);" title="清除画面并重新随机生成干扰点">🔄 重置新局</button>
      <span style="font-size: 13px; color: var(--text-dim);" id="sessionTitle">加载中...</span>
    </div>

    <canvas id="radarCanvas"></canvas>

    <div class="map-controls-floating">
      <button class="btn-tool" onclick="resetView()">🔍 重置视角</button>
      <button class="btn-tool" id="btnToggleSectors" onclick="toggleSectors()">📡 探测扇形: 开</button>
      <button class="btn-tool" onclick="exportImage()">📸 导出论文插图</button>
    </div>
  </div>

  <!-- 右侧：控制看板 -->
  <div id="sidebar">
    <div class="sidebar-header">
      <h1>📡 2026 CUMCM B题 模拟器控制台</h1>
      <p>无线电干扰源快速自动定位与清除 · 动态态势</p>
    </div>

    <!-- 对战记录切换器 -->
    <div class="session-selector-box">
      <select id="logSelect" onchange="onSessionSelectChange()">
        <option value="__live__" selected>🔴 正在进行的实时对战</option>
      </select>
      <label class="live-toggle">
        <input type="checkbox" id="liveToggle" checked onchange="toggleLiveMode()" style="display:none;">
        <span class="live-dot active" id="liveDot"></span>
        <span>实时</span>
      </label>
    </div>

    <!-- 关键指标卡片 (6宫格) -->
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">虚拟时间 (s)</div>
        <div class="metric-value" id="valVirtTime">0.0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">干扰源清除进度</div>
        <div class="metric-value" id="valCleared">0 / 0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">平均用时 (s/目标)</div>
        <div class="metric-value" id="valAvgTime">--</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">态势状态</div>
        <div class="metric-value" id="valStatus" style="font-size: 16px;">待机就绪</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">移动路程 (m)</div>
        <div class="metric-value" id="valDistance">0.0</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">执行步数 / 总步数</div>
        <div class="metric-value" id="valStep">0 / 0</div>
      </div>
    </div>

    <!-- 实时对战启动/停止控制区域 (选实时对战时显示) -->
    <div id="startBox" class="start-box" style="display: block; padding: 14px 20px; border-bottom: 1px solid var(--panel-border); background: #0f172a;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 8px;">
        <span style="font-size: 12px; color: var(--text-dim); font-weight:600;">🤖 算法自动化控制</span>
        <span id="algoStatusBadge" style="font-size: 11px; padding: 2px 8px; border-radius: 4px; background: rgba(148, 163, 184, 0.15); color: #94a3b8;">待命中</span>
      </div>
      <button id="btnStartAlgo" onclick="toggleAlgorithm()" style="width: 100%; padding: 11px 16px; font-size: 14px; font-weight: 700; border-radius: 8px; border: none; cursor: pointer; background: #0284c7; color: #fff; display: flex; align-items: center; justify-content: center; gap: 8px; box-shadow: 0 2px 10px rgba(2, 132, 199, 0.4); transition: all 0.2s;">
        <span>🚀 开始对战</span>
      </button>
      <div style="margin-top: 8px; font-size: 11px; color: var(--text-dim); display: flex; justify-content: space-between;">
        <span>目标程序: <code id="lblScriptTarget" style="color: var(--accent-cyan); font-weight:600;">p4.py</code></span>
        <span id="algoInfoText" style="color: #64748b;">点击开始即刻计算并记录日志</span>
      </div>
      <div id="algoErrorBox" style="display: none; margin-top: 10px; padding: 8px 10px; border-radius: 6px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; font-size: 11px; font-family: monospace; white-space: pre-wrap; max-height: 120px; overflow-y: auto;"></div>
    </div>

    <!-- 历史日志回放进度条与控制按钮 (选历史日志时显示) -->
    <div id="replayBox" class="replay-box" style="display: none;">
      <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-dim);">
        <span>时间轴进度</span>
        <span id="sliderLabel">第 0 步</span>
      </div>
      <input type="range" id="frameSlider" class="timeline-slider" min="0" max="0" value="0" oninput="onSliderInput(this.value)">
      <div class="replay-actions">
        <div class="btn-group-replay">
          <button class="btn-replay" onclick="stepFirst()">|&lt;</button>
          <button class="btn-replay" onclick="stepPrev()">&lt;</button>
          <button class="btn-replay" id="btnPlay" onclick="togglePlay()">▶ 播放</button>
          <button class="btn-replay" onclick="stepNext()">&gt;</button>
          <button class="btn-replay" onclick="stepLast()">&gt;|</button>
        </div>
        <select id="playSpeed" class="speed-select" onchange="updatePlaySpeed()">
          <option value="1">1x 速度</option>
          <option value="2">2x 速度</option>
          <option value="5" selected>5x 速度</option>
          <option value="10">10x 速度</option>
        </select>
      </div>
    </div>

    <!-- 标签页栏 -->
    <div class="tabs-bar">
      <button class="tab-btn active" onclick="switchTab('tabLogs', this)">动作流水记录</button>
      <button class="tab-btn" onclick="switchTab('tabTargets', this)">干扰源真值清单</button>
    </div>

    <!-- 动作流水流 -->
    <div id="tabLogs" class="tab-content">
      <div id="logList"></div>
    </div>

    <!-- 干扰源真值列表 -->
    <div id="tabTargets" class="tab-content" style="display:none;">
      <table class="target-table">
        <thead>
          <tr>
            <th>信道</th>
            <th>类型</th>
            <th>真实坐标 (m)</th>
            <th>接收半径</th>
            <th>状态</th>
          </tr>
        </thead>
        <tbody id="targetTableBody"></tbody>
      </table>
    </div>
  </div>

  <script>
    // 全局数据状态
    let sessionData = null;
    let currentFrameIdx = 0;
    let isPlaying = false;
    let playInterval = null;
    let isLiveMode = true;
    let pollInterval = null;

    // 画布与视角状态
    const canvas = document.getElementById("radarCanvas");
    const ctx = canvas.getContext("2d");
    let viewZoom = 1.0;
    let viewPanX = 0;
    let viewPanY = 0;
    let isDragging = false;
    let dragStartX = 0;
    let dragStartY = 0;

    // 初始化页面与画布
    function resizeCanvas() {
      const container = document.getElementById("mapContainer");
      canvas.width = container.clientWidth * window.devicePixelRatio;
      canvas.height = container.clientHeight * window.devicePixelRatio;
      canvas.style.width = container.clientWidth + "px";
      canvas.style.height = container.clientHeight + "px";
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      render();
    }
    window.addEventListener("resize", resizeCanvas);

    // 视角交互：鼠标滚轮缩放与拖拽平移
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
      viewZoom = Math.min(Math.max(viewZoom * zoomFactor, 0.2), 10.0);
      render();
    });

    canvas.addEventListener("mousedown", (e) => {
      isDragging = true;
      dragStartX = e.clientX - viewPanX;
      dragStartY = e.clientY - viewPanY;
    });
    window.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      viewPanX = e.clientX - dragStartX;
      viewPanY = e.clientY - dragStartY;
      render();
    });
    window.addEventListener("mouseup", () => { isDragging = false; });

    function resetView() {
      const container = document.getElementById("mapContainer");
      const size = Math.min(container.clientWidth, container.clientHeight);
      // 默认让 1800m 圆盘占据画布 80%
      viewZoom = (size * 0.8) / 3600;
      viewPanX = container.clientWidth / 2;
      viewPanY = container.clientHeight / 2;
      render();
    }

    // 坐标映射：数学直角坐标系 (正东为+X, 正北为+Y) -> 屏幕 Canvas 像素
    function toCanvasX(x) { return viewPanX + x * viewZoom; }
    function toCanvasY(y) { return viewPanY - y * viewZoom; } // 屏幕 Y 向下，故取反

    let showSectors = true;
    function toggleSectors() {
      showSectors = !showSectors;
      const btn = document.getElementById("btnToggleSectors");
      if (btn) btn.innerText = showSectors ? "📡 探测扇形: 开" : "📡 探测扇形: 关";
      render();
    }

    // 计算 P3 (7个圆) 与 P4 (17个圆) 的检测站坐标 (半径均为 1000m)
    function getCoverageStations(mode) {
      const r_arena = 1800.0;
      const r_detect = 1000.0;
      const stations = [];

      if (mode === "p3") {
        // P3 模式：7 个检测站（原点 + 内接正六边形各边为弦的检测圆内侧圆心）
        // 半弦 R/2 = 900m
        const half_chord = r_arena / 2.0;
        const offset = Math.sqrt(r_detect * r_detect - half_chord * half_chord); // sqrt(1000^2 - 900^2) ≈ 435.89m
        const rho = r_arena * Math.cos(Math.PI / 6.0) - offset; // 1800 * cos(30°) - 435.89 ≈ 1122.96m

        stations.push({ x: 0, y: 0, label: "P0", r: r_detect });
        for (let i = 0; i < 6; i++) {
          const ang = i * Math.PI / 3.0; // 60° 步进
          stations.push({
            x: rho * Math.cos(ang),
            y: rho * Math.sin(ang),
            label: `P${i + 1}`,
            r: r_detect
          });
        }
      } else {
        // P4 模式：17 个检测站（原点 + 外切八边形8顶点 + 内接八边形8边为弦探测圆心）
        const cos_22_5 = Math.cos(22.5 * Math.PI / 180);
        const sin_22_5 = Math.sin(22.5 * Math.PI / 180);

        // 1. 中心原点
        stations.push({ x: 0, y: 0, label: "P0", r: r_detect });

        // 2. 外圈 8 点 (外切正八边形顶点)
        const r_out = r_arena / cos_22_5; // ≈ 1948.31m
        for (let i = 0; i < 8; i++) {
          const rad = (45.0 * i) * Math.PI / 180;
          stations.push({
            x: r_out * Math.cos(rad),
            y: r_out * Math.sin(rad),
            label: `P_out${i + 1}`,
            r: r_detect
          });
        }

        // 3. 内圈 8 点 (以八边形边为弦探测圆心)
        const half_chord = r_arena * sin_22_5; // ≈ 688.83m
        const h_chord = Math.sqrt(r_detect * r_detect - half_chord * half_chord); // ≈ 724.92m
        const d_edge = r_arena * cos_22_5; // ≈ 1662.98m
        const r_in = d_edge - h_chord; // ≈ 938.06m
        for (let j = 0; j < 8; j++) {
          const rad = (45.0 * j + 22.5) * Math.PI / 180;
          stations.push({
            x: r_in * Math.cos(rad),
            y: r_in * Math.sin(rad),
            label: `P_in${j + 1}`,
            r: r_detect
          });
        }
      }
      return stations;
    }

    // 主渲染流程
    function render() {
      const container = document.getElementById("mapContainer");
      const w = container.clientWidth;
      const h = container.clientHeight;
      ctx.clearRect(0, 0, w, h);

      if (!sessionData) {
        ctx.fillStyle = "#64748b";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("等待对战数据加载...", w / 2, h / 2);
        return;
      }

      const R_arena = sessionData.arena_radius || 1800.0;
      const curFrame = (sessionData.frames && sessionData.frames[currentFrameIdx]) || null;
      const clearedSet = new Set((curFrame && curFrame.cleared_channels) || []);
      const curMode = (sessionData.mode || "p4").toLowerCase();

      // 1. 绘制十字坐标轴、1800m 场地大圆边界与规划检测圆 (P3: 7圆; P4: 17圆)
      ctx.save();
      // 坐标轴线
      ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(-R_arena * 1.15), toCanvasY(0));
      ctx.lineTo(toCanvasX(R_arena * 1.15), toCanvasY(0));
      ctx.moveTo(toCanvasX(0), toCanvasY(-R_arena * 1.15));
      ctx.lineTo(toCanvasX(0), toCanvasY(R_arena * 1.15));
      ctx.stroke();

      // 1800m 场地边界圆
      ctx.beginPath();
      ctx.arc(toCanvasX(0), toCanvasY(0), R_arena * viewZoom, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.75)";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#64748b";
      ctx.font = "10px sans-serif";
      ctx.fillText("1800m 场地边界", toCanvasX(R_arena + 8), toCanvasY(0) - 4);

      // 覆盖方案检测圆 (P3: 7个圆; P4: 17个圆)
      const stations = getCoverageStations(curMode);
      stations.forEach((st) => {
        const sx = toCanvasX(st.x);
        const sy = toCanvasY(st.y);
        const rPx = st.r * viewZoom;

        // 探测圆虚线轮廓
        ctx.beginPath();
        ctx.arc(sx, sy, rPx, 0, 2 * Math.PI);
        ctx.strokeStyle = "rgba(59, 130, 246, 0.32)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // 探测圆内部淡蓝填充
        ctx.fillStyle = "rgba(59, 130, 246, 0.015)";
        ctx.fill();

        // 测站圆心标记
        ctx.beginPath();
        ctx.arc(sx, sy, 3, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(96, 165, 250, 0.8)";
        ctx.fill();

        ctx.fillStyle = "rgba(148, 163, 184, 0.7)";
        ctx.font = "9px sans-serif";
        ctx.fillText(st.label, sx + 5, sy - 5);
      });

      // 画布左下角模式提示
      ctx.fillStyle = "rgba(148, 163, 184, 0.6)";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(
        curMode === "p3" 
          ? "背景布局: P3 全覆盖规划 (7 个检测圆, 半径 1000m)" 
          : "背景布局: P4 全覆盖规划 (17 个检测圆, 半径 1000m)", 
        16, 
        h - 16
      );
      ctx.restore();

      // 2. 绘制扫描探测留下的细长扇形区域（仅在探测到干扰源时绘制并持久留在场地上，无信号不显示）
      if (showSectors && sessionData.frames && sessionData.frames.length > 0) {
        for (let i = 0; i <= currentFrameIdx && i < sessionData.frames.length; i++) {
          const f = sessionData.frames[i];
          // 仅当是 measure 动作且成功探测到干扰源 (svd_deg 有数值) 时才绘制细长扇形
          // 如果某个信道没有探测到干扰源 (measure_result === "no_signal" 或 svd_deg === null)，则不显示扇形
          if (f.action === "measure" && f.svd_deg !== null && f.svd_deg !== undefined) {
            const sx = toCanvasX(f.robot_position.x);
            const sy = toCanvasY(f.robot_position.y);
            const svd = f.svd_deg;
            const beamLen = 2400 * viewZoom;
            const halfSpanRad = 1.0 * Math.PI / 180; // ±1.0° 窄波束锥
            const centerCanvasRad = -svd * Math.PI / 180;
            const startRad = centerCanvasRad - halfSpanRad;
            const endRad = centerCanvasRad + halfSpanRad;

            const isCurrent = (i === currentFrameIdx);

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.arc(sx, sy, beamLen, startRad, endRad, false);
            ctx.closePath();

            if (isCurrent) {
              // 当前执行帧：高亮紫色发光渐变扇形
              const coneGrad = ctx.createRadialGradient(sx, sy, 10, sx, sy, beamLen);
              coneGrad.addColorStop(0, "rgba(168, 85, 247, 0.55)");
              coneGrad.addColorStop(1, "rgba(168, 85, 247, 0.02)");
              ctx.fillStyle = coneGrad;
              ctx.fill();

              ctx.strokeStyle = "rgba(232, 121, 249, 0.9)";
              ctx.lineWidth = 1.8;
              ctx.stroke();

              // 中心瞄准线
              ctx.beginPath();
              ctx.moveTo(sx, sy);
              ctx.lineTo(sx + Math.cos(centerCanvasRad) * beamLen, sy + Math.sin(centerCanvasRad) * beamLen);
              ctx.strokeStyle = "rgba(244, 114, 182, 0.9)";
              ctx.lineWidth = 1.5;
              ctx.stroke();

              // 示向度提示文字
              ctx.fillStyle = "#f3e8ff";
              ctx.font = "bold 11px sans-serif";
              ctx.fillText(`[步#${f.step}] 信道 #${f.channel} 示向度: ${svd.toFixed(2)}° (±1.0°)`, sx + 12, sy - 14);
            } else {
              // 历史扫描帧：半透明细长扇形，持久保留在场地上
              ctx.fillStyle = "rgba(168, 85, 247, 0.09)";
              ctx.fill();

              ctx.strokeStyle = "rgba(192, 132, 252, 0.28)";
              ctx.lineWidth = 0.8;
              ctx.stroke();

              // 中心虚线示向线
              ctx.beginPath();
              ctx.moveTo(sx, sy);
              ctx.lineTo(sx + Math.cos(centerCanvasRad) * beamLen, sy + Math.sin(centerCanvasRad) * beamLen);
              ctx.strokeStyle = "rgba(216, 180, 254, 0.35)";
              ctx.lineWidth = 0.8;
              ctx.setLineDash([3, 4]);
              ctx.stroke();
              ctx.setLineDash([]);

              // 信道编号微标 (沿射线方向适当距离处)
              const tagDist = 140 * viewZoom;
              ctx.fillStyle = "rgba(216, 180, 254, 0.75)";
              ctx.font = "bold 9px sans-serif";
              ctx.fillText(`#${f.channel}`, sx + Math.cos(centerCanvasRad) * tagDist, sy + Math.sin(centerCanvasRad) * tagDist);
            }
            ctx.restore();
          }
        }
      }

      // 3. 绘制干扰源点位与发射波束
      (sessionData.targets || []).forEach(t => {
        const cx = toCanvasX(t.x);
        const cy = toCanvasY(t.y);
        const isCleared = clearedSet.has(t.channel);

        // (a) 如果是定向源：绘制发射主朝向的前向 180° 扇面光晕
        if (t.is_directional) {
          ctx.save();
          const rPixel = t.recv_radius * viewZoom;
          // 数学角度 (正东0, 正北90) 转 Canvas 角度 (正东0, 正南90)
          const mainCanvasRad = -t.dir_angle_deg * Math.PI / 180;
          const startRad = mainCanvasRad - Math.PI / 2;
          const endRad = mainCanvasRad + Math.PI / 2;

          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.arc(cx, cy, rPixel, startRad, endRad, false);
          ctx.closePath();

          if (isCleared) {
            ctx.fillStyle = "rgba(16, 185, 129, 0.05)";
          } else {
            const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, rPixel);
            grad.addColorStop(0, "rgba(245, 158, 11, 0.25)");
            grad.addColorStop(1, "rgba(245, 158, 11, 0.02)");
            ctx.fillStyle = grad;
          }
          ctx.fill();
          ctx.strokeStyle = isCleared ? "rgba(16, 185, 129, 0.3)" : "rgba(245, 158, 11, 0.4)";
          ctx.lineWidth = 1;
          ctx.stroke();

          // 发射主轴箭头
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + Math.cos(mainCanvasRad) * (rPixel * 0.4), cy + Math.sin(mainCanvasRad) * (rPixel * 0.4));
          ctx.strokeStyle = isCleared ? "#10b981" : "#f59e0b";
          ctx.lineWidth = 1.5;
          ctx.stroke();
          ctx.restore();
        } else {
          // 全向源接收范围外边框
          ctx.save();
          ctx.beginPath();
          ctx.arc(cx, cy, t.recv_radius * viewZoom, 0, 2 * Math.PI);
          ctx.strokeStyle = isCleared ? "rgba(16, 185, 129, 0.15)" : "rgba(245, 158, 11, 0.15)";
          ctx.setLineDash([3, 5]);
          ctx.stroke();
          ctx.restore();
        }

        // (b) 目标中心圆标与信道
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, 9, 0, 2 * Math.PI);
        ctx.fillStyle = isCleared ? "#10b981" : "#f59e0b";
        ctx.shadowColor = isCleared ? "#10b981" : "#f59e0b";
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;

        // 信道文字
        ctx.fillStyle = "#000";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(t.channel, cx, cy);
        ctx.restore();
      });

      // 4. 绘制机器狗历史走位轨迹
      if (sessionData.frames && sessionData.frames.length > 0) {
        ctx.save();
        ctx.beginPath();
        for (let i = 0; i <= currentFrameIdx && i < sessionData.frames.length; i++) {
          const f = sessionData.frames[i];
          const px = toCanvasX(f.robot_position.x);
          const py = toCanvasY(f.robot_position.y);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = "rgba(6, 182, 212, 0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }

      // 5. 绘制激光清除特效与机器狗当前图标
      const rx = toCanvasX((curFrame && curFrame.robot_position) ? curFrame.robot_position.x : 0.0);
      const ry = toCanvasY((curFrame && curFrame.robot_position) ? curFrame.robot_position.y : 0.0);

      if (curFrame && curFrame.action === "clear") {
        ctx.save();
        const clearRadiusPx = 20.0 * viewZoom;
        ctx.beginPath();
        ctx.arc(rx, ry, Math.max(clearRadiusPx, 6), 0, 2 * Math.PI);
        const isSuccess = curFrame.clear_result === "success";
        ctx.strokeStyle = isSuccess ? "#10b981" : "#ef4444";
        ctx.lineWidth = 2.5;
        ctx.shadowColor = isSuccess ? "#10b981" : "#ef4444";
        ctx.shadowBlur = 12;
        ctx.stroke();
        ctx.restore();
      }

      // 机器狗自身图标（始终绘制在当前位置，待机时在 (0, 0)）
      ctx.save();
      ctx.beginPath();
      ctx.arc(rx, ry, 7, 0, 2 * Math.PI);
      ctx.fillStyle = "#06b6d4";
      ctx.shadowColor = "#06b6d4";
      ctx.shadowBlur = 10;
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("机器狗", rx, ry - 12);
      ctx.restore();
    }

    // 更新面板数据
    function updatePanel() {
      if (!sessionData) return;
      const frames = sessionData.frames || [];
      const totalFrames = frames.length;
      if (currentFrameIdx >= totalFrames) {
        currentFrameIdx = Math.max(0, totalFrames - 1);
      }
      const curFrame = frames[currentFrameIdx] || (frames[0] || { virtual_time_s: 0.0, cleared_channels: [], robot_position: {x: 0, y: 0} });
      
      const curMode = (sessionData.mode || "p4").toLowerCase();
      const modeSelect = document.getElementById("modeSelect");
      if (modeSelect) {
        modeSelect.value = curMode;
        if (curMode === "p3") {
          modeSelect.style.background = "#059669"; // 绿色 P3
        } else {
          modeSelect.style.background = "#0284c7"; // 蓝色 P4
        }
      }
      const lblScript = document.getElementById("lblScriptTarget");
      if (lblScript) {
        lblScript.innerText = (curMode === "p3" ? "P3.py" : "p4.py");
      }
      if (!isAlgoRunning) {
        updateAlgoUI(false, curMode);
      }
      document.getElementById("sessionTitle").innerText = `${sessionData.session_id} (${sessionData.robot_id || "待机"})`;
      
      const curTime = curFrame.virtual_time_s || 0.0;
      document.getElementById("valVirtTime").innerText = curTime.toFixed(1);
      
      const clearedList = curFrame.cleared_channels || [];
      const totalT = (sessionData.targets || []).length;
      document.getElementById("valCleared").innerText = `${clearedList.length} / ${totalT}`;

      // 平均用时 (s/目标)
      const avgEl = document.getElementById("valAvgTime");
      if (avgEl) {
        if (clearedList.length > 0) {
          avgEl.innerText = (curTime / clearedList.length).toFixed(1) + " s";
        } else {
          avgEl.innerText = "--";
        }
      }

      // 态势状态
      const statusEl = document.getElementById("valStatus");
      if (statusEl) {
        const st = sessionData.status || "idle";
        if (st === "idle" || st === "waiting") {
          statusEl.innerText = "待机就绪";
          statusEl.style.color = "#94a3b8";
        } else if (st === "running") {
          statusEl.innerText = "🚀 对战中";
          statusEl.style.color = "#38bdf8";
        } else if (st === "finished") {
          statusEl.innerText = "🏆 已完赛";
          statusEl.style.color = "#4ade80";
        } else {
          statusEl.innerText = st;
          statusEl.style.color = "#f59e0b";
        }
      }
      
      // 距离估算
      let dist = 0.0;
      for (let i = 1; i <= currentFrameIdx && i < frames.length; i++) {
        const p1 = frames[i - 1].robot_position;
        const p2 = frames[i].robot_position;
        dist += Math.hypot(p2.x - p1.x, p2.y - p1.y);
      }
      document.getElementById("valDistance").innerText = dist.toFixed(1);
      document.getElementById("valStep").innerText = `${currentFrameIdx} / ${Math.max(0, totalFrames - 1)}`;

      // 滑块同步
      const slider = document.getElementById("frameSlider");
      slider.max = Math.max(0, totalFrames - 1);
      slider.value = currentFrameIdx;
      document.getElementById("sliderLabel").innerText = `第 ${currentFrameIdx} 步 / 共 ${Math.max(0, totalFrames - 1)} 步`;

      // 渲染目标表格
      const tbody = document.getElementById("targetTableBody");
      tbody.innerHTML = (sessionData.targets || []).map(t => {
        const isCl = clearedList.includes(t.channel);
        const typeStr = t.is_directional ? `定向 (${t.dir_angle_deg.toFixed(1)}°)` : "全向";
        return `<tr>
          <td><b>#${t.channel}</b></td>
          <td>${typeStr}</td>
          <td>(${t.x.toFixed(1)}, ${t.y.toFixed(1)})</td>
          <td>${t.recv_radius.toFixed(0)}m</td>
          <td><span class="tag-status ${isCl ? 'tag-cleared' : 'tag-active'}">${isCl ? '已清除' : '未清除'}</span></td>
        </tr>`;
      }).join("");

      // 渲染动作流水日志
      const logList = document.getElementById("logList");
      logList.innerHTML = frames.map((f, idx) => {
        const isCur = idx === currentFrameIdx;
        return `<div class="log-item ${isCur ? 'current' : ''}" onclick="jumpToStep(${idx})">
          <div class="log-header">
            <span>[#${f.step}] ${f.action.toUpperCase()}</span>
            <span>${f.virtual_time_s.toFixed(1)}s</span>
          </div>
          <div class="log-body">${f.description || ''}</div>
        </div>`;
      }).join("");

      // 滚动至当前选中的日志
      const activeEl = logList.querySelector(".log-item.current");
      if (activeEl) {
        activeEl.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }

      render();
    }

    // 播放控制器交互
    function onSliderInput(val) {
      pause();
      currentFrameIdx = parseInt(val, 10);
      updatePanel();
    }
    function jumpToStep(idx) {
      pause();
      currentFrameIdx = idx;
      updatePanel();
    }
    function stepFirst() { pause(); currentFrameIdx = 0; updatePanel(); }
    function stepPrev() { pause(); if (currentFrameIdx > 0) currentFrameIdx--; updatePanel(); }
    function stepNext() {
      const maxIdx = (sessionData.frames ? sessionData.frames.length - 1 : 0);
      if (currentFrameIdx < maxIdx) { currentFrameIdx++; updatePanel(); }
    }
    function stepLast() {
      pause();
      if (sessionData && sessionData.frames) currentFrameIdx = sessionData.frames.length - 1;
      updatePanel();
    }
    function togglePlay() {
      if (isPlaying) pause();
      else play();
    }
    function play() {
      isPlaying = true;
      document.getElementById("btnPlay").innerText = "⏸ 暂停";
      const speed = parseFloat(document.getElementById("playSpeed").value) || 5;
      const intervalMs = 1000 / speed;
      clearInterval(playInterval);
      playInterval = setInterval(() => {
        const maxIdx = (sessionData.frames ? sessionData.frames.length - 1 : 0);
        if (currentFrameIdx < maxIdx) {
          currentFrameIdx++;
          updatePanel();
        } else {
          pause();
        }
      }, intervalMs);
    }
    function pause() {
      isPlaying = false;
      document.getElementById("btnPlay").innerText = "▶ 播放";
      clearInterval(playInterval);
    }
    function updatePlaySpeed() {
      if (isPlaying) play();
    }

    // 标签页切换
    function switchTab(tabId, btn) {
      document.getElementById("tabLogs").style.display = "none";
      document.getElementById("tabTargets").style.display = "none";
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.getElementById(tabId).style.display = "block";
      btn.classList.add("active");
    }

    // 切换控制面板可见性 (实时对战 vs 历史回放)
    function syncBoxVisibility(isLive) {
      const startBox = document.getElementById("startBox");
      const replayBox = document.getElementById("replayBox");
      if (startBox) startBox.style.display = isLive ? "block" : "none";
      if (replayBox) replayBox.style.display = isLive ? "none" : "block";
    }

    // 设置实时模式状态
    function setLiveMode(active) {
      isLiveMode = active;
      const dot = document.getElementById("liveDot");
      const chk = document.getElementById("liveToggle");
      if (chk) chk.checked = active;
      if (dot) {
        if (active) dot.classList.add("active");
        else dot.classList.remove("active");
      }
      if (active) {
        pause();
        fetchLiveData();
        startPolling();
      } else {
        stopPolling();
      }
    }

    // 实时模式开关切换 (点击顶部“实时”复选框)
    function toggleLiveMode() {
      const chk = document.getElementById("liveToggle");
      if (chk && chk.checked) {
        const sel = document.getElementById("logSelect");
        if (sel) sel.value = "__live__";
        onSessionSelectChange();
      } else {
        setLiveMode(false);
      }
    }
    function startPolling() {
      stopPolling();
      pollInterval = setInterval(fetchLiveData, 400);
    }
    function stopPolling() {
      clearInterval(pollInterval);
    }

    let isAlgoRunning = false;
    let lastSessionStatus = null;

    // API 请求：获取后台算法运行状态
    function checkAlgoStatus() {
      fetch("/api/algo_status")
        .then(res => res.json())
        .then(data => {
          isAlgoRunning = !!data.running;
          updateAlgoUI(data.running, data.mode, data.error);
        })
        .catch(() => {});
    }

    // 更新算法启动按钮与状态指示
    function updateAlgoUI(running, mode, error) {
      const btn = document.getElementById("btnStartAlgo");
      const badge = document.getElementById("algoStatusBadge");
      const info = document.getElementById("algoInfoText");
      const scriptLabel = document.getElementById("lblScriptTarget");
      const errorBox = document.getElementById("algoErrorBox");
      const curMode = (mode || (sessionData && sessionData.mode) || (document.getElementById("modeSelect") ? document.getElementById("modeSelect").value : "p4")).toLowerCase();

      if (scriptLabel) {
        scriptLabel.innerText = curMode === "p3" ? "P3.py" : "p4.py";
      }

      if (error && !running) {
        if (errorBox) {
          errorBox.style.display = "block";
          errorBox.innerText = "❌ " + error;
        }
      } else if (running) {
        if (errorBox) errorBox.style.display = "none";
      }

      if (running) {
        if (btn) {
          btn.innerHTML = `<span>⏹ 终止对战</span>`;
          btn.style.background = "#dc2626";
          btn.style.boxShadow = "0 2px 10px rgba(220, 38, 38, 0.4)";
        }
        if (badge) {
          badge.innerText = "运行中";
          badge.style.background = "rgba(34, 197, 94, 0.2)";
          badge.style.color = "#4ade80";
        }
        if (info) info.innerText = "正在执行算法与实时推演...";
      } else {
        if (btn) {
          btn.innerHTML = `<span>🚀 开始对战</span>`;
          btn.style.background = curMode === "p3" ? "#059669" : "#0284c7";
          btn.style.boxShadow = curMode === "p3" ? "0 2px 10px rgba(5, 150, 105, 0.4)" : "0 2px 10px rgba(2, 132, 199, 0.4)";
        }
        if (badge) {
          badge.innerText = error ? "异常退出" : "待命中";
          badge.style.background = error ? "rgba(239, 68, 68, 0.2)" : "rgba(148, 163, 184, 0.15)";
          badge.style.color = error ? "#ef4444" : "#94a3b8";
        }
        if (info) {
          if (error) {
            info.innerHTML = `<span style="color:#ef4444; font-weight:600;">⚠️ 算法执行异常，详见红框提示</span>`;
          } else {
            info.innerText = "点击开始即刻计算并记录日志";
          }
        }
      }
    }

    // 启动/停止算法自动化
    function toggleAlgorithm() {
      const btn = document.getElementById("btnStartAlgo");
      if (btn) btn.disabled = true;
      const endpoint = isAlgoRunning ? "/api/stop" : "/api/start";
      fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}"
      })
        .then(res => res.json())
        .then(data => {
          if (btn) btn.disabled = false;
          if (!data.accepted && data.error) {
            const errorBox = document.getElementById("algoErrorBox");
            if (errorBox) {
              errorBox.style.display = "block";
              errorBox.innerText = "❌ 启动失败: " + data.error;
            }
          }
          checkAlgoStatus();
          if (!isAlgoRunning) {
            setTimeout(() => {
              fetchLiveData();
              loadLogsList();
            }, 300);
          }
        })
        .catch(err => {
          console.error("Algo toggle failed:", err);
          if (btn) btn.disabled = false;
        });
    }

    // API 请求：获取实时数据
    function fetchLiveData() {
      fetch("/api/live")
        .then(res => res.json())
        .then(data => {
          if (!data) return;
          const frames = data.frames || [];
          const wasAtEnd = sessionData ? (currentFrameIdx >= (sessionData.frames ? sessionData.frames.length - 1 : 0)) : true;
          const prevStatus = lastSessionStatus;
          lastSessionStatus = data.status;
          sessionData = data;
          if (wasAtEnd) {
            currentFrameIdx = Math.max(0, frames.length - 1);
          }
          updatePanel();
          if (prevStatus && prevStatus !== data.status && (data.status === "finished" || data.status === "aborted")) {
            loadLogsList();
            checkAlgoStatus();
          }
        })
        .catch(() => {});
    }

    // 重置新局：停止正在运行的算法，清除画面并重新随机生成干扰点（不新建日志文件）
    function resetArena() {
      pause();
      fetch("/api/reset")
        .then(res => res.json())
        .then(data => {
          currentFrameIdx = 0;
          checkAlgoStatus();
          fetchLiveData();
          loadLogsList();
        })
        .catch(err => console.error("Reset failed:", err));
    }

    // API 请求：获取所有历史日志列表
    function loadLogsList() {
      return fetch("/api/logs")
        .then(res => res.json())
        .then(list => {
          const sel = document.getElementById("logSelect");
          if (!sel) return;
          const curVal = sel.value || "__live__";
          sel.innerHTML = `<option value="__live__">🔴 正在进行的实时对战</option>` + 
            list.map(item => `<option value="${item.filename}">${item.filename} [${item.status === 'finished' ? '已完成' : '未终结'}]</option>`).join("");
          if (curVal && Array.from(sel.options).some(o => o.value === curVal)) {
            sel.value = curVal;
          } else {
            sel.value = "__live__";
          }
          syncBoxVisibility(sel.value === "__live__");
        })
        .catch(err => console.error("loadLogsList error:", err));
    }

    // 切换日志回放 / 实时模式
    function onSessionSelectChange() {
      const sel = document.getElementById("logSelect");
      const val = (sel && sel.value) ? sel.value : "__live__";
      const isLive = (val === "__live__");

      syncBoxVisibility(isLive);

      if (isLive) {
        pause(); // 确保历史回放定时器停止，btnPlay 显示为 ▶ 播放
        setLiveMode(true);
        checkAlgoStatus();
      } else {
        setLiveMode(false);
        pause(); // 确保历史回放定时器停止，btnPlay 显示为 ▶ 播放
        fetch(`/api/log?file=${encodeURIComponent(val)}`)
          .then(res => res.json())
          .then(data => {
            sessionData = data;
            currentFrameIdx = 0;
            updatePanel();
          })
          .catch(err => console.error("Fetch log failed:", err));
      }
    }

    // 切换 P3 / P4 模式
    function onModeSelectChange(val) {
      pause();
      fetch('/api/set_mode?mode=' + encodeURIComponent(val))
        .then(res => res.json())
        .then(() => {
          currentFrameIdx = 0;
          checkAlgoStatus();
          fetchLiveData();
          loadLogsList();
        })
        .catch(err => console.error("Set mode failed:", err));
    }

    // 导出高质量论文插图 (PNG)
    function exportImage() {
      const link = document.createElement("a");
      link.download = `cumcm_simulation_${sessionData ? sessionData.session_id : 'figure'}_step${currentFrameIdx}.png`;
      link.href = canvas.toDataURL("image/png");
      link.click();
    }

    // 启动初始化
    window.addEventListener("DOMContentLoaded", () => {
      resetView();
      resizeCanvas();
      const sel = document.getElementById("logSelect");
      if (sel) sel.value = "__live__";
      syncBoxVisibility(true);
      setLiveMode(true);
      checkAlgoStatus();
      loadLogsList();
    });
  </script>
</body>
</html>
"""


# =============================================================================
# 5. HTTP 服务端 Handler (双重角色：机器狗 API + WebUI / REST API)
# =============================================================================

class MockSimulatorHandler(BaseHTTPRequestHandler):
    """处理机器狗算法 API 以及 Web 浏览器的 HTTP 请求。"""

    server: MockSimulatorServer

    def log_message(self, format: str, *args: Any) -> None:
        """除非开启了详细日志，否则静音标准 HTTP 访问记录。"""
        if getattr(self.server, "verbose_http", False):
            sys.stderr.write(f"[HTTP] {self.address_string()} - {format % args}\n")

    def _send_json(self, status_code: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, status_code: int, html_str: str) -> None:
        body = html_str.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        # 1. 根目录返回 WebUI 单页应用
        if path in ("", "/index.html"):
            self._send_html(200, WEBUI_HTML)
            return

        # 2. REST API: /api/live (当前对战的实时帧)
        if path == "/api/live":
            self._send_json(200, self.server.arena.logger.get_live_data())
            return

        # 3. REST API: /api/logs (列出 ./logs/ 目录下的所有历史会话)
        if path == "/api/logs":
            self._send_json(200, self.server.arena.logger.list_sessions())
            return

        # 4. REST API: /api/log?file=session_xxx.json (读取指定历史对战)
        if path == "/api/log":
            filename = query.get("file", [""])[0]
            data = self.server.arena.logger.read_session_file(filename)
            if data is not None:
                self._send_json(200, data)
            else:
                self._send_json(404, {"error": "File not found"})
            return

        # 5. 调试接口 /status
        if path == "/status":
            self._send_json(200, self.server.arena.get_status())
            return

        # 6. 切换模式接口 /api/set_mode?mode=p3 / p4
        if path == "/api/set_mode":
            mode = query.get("mode", ["p4"])[0]
            self.server.arena.set_mode(mode)
            self._send_json(200, {"accepted": True, "mode": self.server.arena.mode})
            return

        # 7. 重置新局接口 /api/reset
        if path == "/api/reset":
            self.server.stop_algorithm()
            self.server.arena.reset_session(self.server.arena.arena_id, "waiting")
            self._send_json(200, {"accepted": True, "status": "reset", "mode": self.server.arena.mode})
            return

        # 8. 算法状态接口 /api/algo_status
        if path == "/api/algo_status":
            self._send_json(200, self.server.get_algorithm_status())
            return

        # 9. 启动算法接口 /api/start
        if path == "/api/start":
            res = self.server.start_algorithm()
            self._send_json(200 if res.get("accepted") else 400, res)
            return

        # 10. 停止算法接口 /api/stop
        if path == "/api/stop":
            res = self.server.stop_algorithm()
            self._send_json(200, res)
            return

        self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        arena = self.server.arena

        # 控制与管理接口，无需请求体
        if path == "/api/start":
            res = self.server.start_algorithm()
            self._send_json(200 if res.get("accepted") else 400, res)
            return

        if path == "/api/stop":
            res = self.server.stop_algorithm()
            self._send_json(200, res)
            return

        if path == "/api/reset":
            self.server.stop_algorithm()
            arena.reset_session(arena.arena_id, "waiting")
            self._send_json(200, {"accepted": True, "status": "reset", "mode": arena.mode})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0 and path != "/exit":
            self._send_json(400, {"accepted": False, "error": "Empty request body"})
            return

        payload = {}
        if content_length > 0:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                self._send_json(400, {"accepted": False, "error": f"Invalid JSON: {exc}"})
                return

        request_id = payload.get("request_id", "")
        arena_id = payload.get("arena_id", "default")
        robot_id = payload.get("robot_id", "default_dog")

        try:
            if path == "/enter":
                response = arena.enter(request_id, arena_id, robot_id)
                self._send_json(200, response)

            elif path == "/measure":
                pos = payload.get("position", {})
                px = float(pos.get("x", 0.0))
                py = float(pos.get("y", 0.0))
                ch = int(payload.get("channel", 1))
                response = arena.measure(request_id, px, py, ch)
                self._send_json(200, response)

            elif path == "/clear":
                pos = payload.get("position", {})
                px = float(pos.get("x", 0.0))
                py = float(pos.get("y", 0.0))
                ch = int(payload.get("channel", 1))
                response = arena.clear(request_id, px, py, ch)
                self._send_json(200, response)

            elif path == "/exit":
                response = arena.exit(request_id)
                self._send_json(200, response)

            elif path == "/api/set_mode":
                mode = payload.get("mode", "p4")
                arena.set_mode(mode)
                self._send_json(200, {"accepted": True, "mode": arena.mode})

            else:
                self._send_json(404, {"accepted": False, "error": f"Unknown path: {path}"})

        except Exception as exc:
            self._send_json(500, {"accepted": False, "error": str(exc)})


class MockSimulatorServer(ThreadingHTTPServer):
    """包含 Arena 状态与后台算法进程管理的多线程 HTTP 服务端。"""

    def __init__(
        self,
        server_address: Tuple[str, int],
        arena: Arena,
        python_bin: Optional[str] = None,
        verbose_http: bool = False,
    ) -> None:
        self.arena = arena
        self.verbose_http = verbose_http
        self.algo_process: Optional[subprocess.Popen] = None
        self.algo_log_file = None
        self.algo_last_error: Optional[str] = None
        self.algo_python_bin = self._resolve_python_bin(python_bin)
        super().__init__(server_address, MockSimulatorHandler)

    @classmethod
    def _resolve_python_bin(cls, preferred_bin: Optional[str] = None) -> str:
        """寻找包含 numpy 与算法依赖的 Python 解释器。"""
        if preferred_bin:
            return preferred_bin

        candidates = []

        # 1. 检查环境变量 CONDA_PREFIX (若已激活虚拟环境)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            candidates.append(str(Path(conda_prefix) / "bin" / "python"))

        # 2. 检查 ~/.conda/environments.txt 中包含 cumcm 的环境
        env_file = Path.home() / ".conda" / "environments.txt"
        if env_file.is_file():
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and "cumcm" in line.lower():
                            candidates.append(str(Path(line) / "bin" / "python"))
            except Exception:
                pass

        # 3. 常见 miniconda / miniforge 路径下的 cumcm2026-b
        for home_sub in ["miniconda3", "miniforge3", "anaconda3", ".conda"]:
            candidates.append(str(Path.home() / home_sub / "envs" / "cumcm2026-b" / "bin" / "python"))

        # 4. PATH 中的 python 与 python3
        for py_name in ["python", "python3"]:
            which_p = shutil.which(py_name)
            if which_p:
                candidates.append(which_p)

        # 5. 当前解释器 sys.executable
        candidates.append(sys.executable)

        seen = set()
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            p_obj = Path(c)
            if not p_obj.is_file():
                continue
            # 测试能否导入 numpy
            try:
                res = subprocess.run(
                    [c, "-c", "import numpy"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                if res.returncode == 0:
                    return c
            except Exception:
                if "cumcm" in c.lower() and p_obj.is_file():
                    return c

        return sys.executable

    def get_algorithm_status(self) -> Dict[str, Any]:
        with self.arena.lock:
            running = False
            exit_code = None
            if self.algo_process is not None:
                ret = self.algo_process.poll()
                if ret is None:
                    running = True
                else:
                    exit_code = ret
                    self.algo_process = None
                    if self.algo_log_file and not self.algo_log_file.closed:
                        try:
                            self.algo_log_file.close()
                        except Exception:
                            pass
                    if ret != 0 and not self.algo_last_error:
                        algo_log_path = RUNTIME_LOGS_DIR / "algo_runner.log"
                        tail_msg = ""
                        if algo_log_path.is_file():
                            try:
                                with open(algo_log_path, "r", encoding="utf-8") as f:
                                    lines = f.readlines()
                                    tail_msg = "".join(lines[-12:]).strip()
                            except Exception:
                                pass
                        self.algo_last_error = (
                            f"算法退出 (退出码 {ret})。" + (f"\n错误日志:\n{tail_msg}" if tail_msg else "")
                        )
                        if self.arena.verbose:
                            print(f"[MockSimulator] ❌ 算法子进程异常退出 (退出码 {ret})")
                            if tail_msg:
                                print(f"[MockSimulator] 错误详情:\n{tail_msg}")

            return {
                "running": running,
                "mode": self.arena.mode,
                "python": self.algo_python_bin,
                "error": self.algo_last_error,
                "exit_code": exit_code,
            }

    def start_algorithm(self) -> Dict[str, Any]:
        with self.arena.lock:
            # 检查是否有正在运行的算法进程
            if self.algo_process is not None:
                if self.algo_process.poll() is None:
                    return {"accepted": False, "error": "算法正在运行中，请勿重复启动"}
                else:
                    self.algo_process = None

            self.algo_last_error = None

            script_name = "P3.py" if self.arena.mode == "p3" else "p4.py"
            workspace_dir = Path(__file__).resolve().parent
            script_path = workspace_dir / script_name

            if not script_path.exists():
                return {"accepted": False, "error": f"未在工作区找到算法文件: {script_name}"}

            port = self.server_address[1]
            cmd = [
                self.algo_python_bin,
                str(script_path),
                "--run",
                "--robot-id",
                f"webui-{self.arena.mode}",
                "--base-url",
                f"http://127.0.0.1:{port}",
            ]

            env = os.environ.copy()
            # 必须设置 no_proxy，避免 127.0.0.1 请求被本地代理（Clash/Surge等）拦截报错
            env["no_proxy"] = "127.0.0.1,localhost," + env.get("no_proxy", "")
            env["NO_PROXY"] = "127.0.0.1,localhost," + env.get("NO_PROXY", "")
            # 保证工作区在 PYTHONPATH 中
            env["PYTHONPATH"] = str(workspace_dir) + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")

            algo_log_path = RUNTIME_LOGS_DIR / "algo_runner.log"
            try:
                self.algo_log_file = open(algo_log_path, "w", encoding="utf-8")
                self.algo_process = subprocess.Popen(
                    cmd,
                    cwd=str(workspace_dir),
                    stdout=self.algo_log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                if self.arena.verbose:
                    print(f"[MockSimulator] 🚀 已在后台启动算法 (PID: {self.algo_process.pid})")
                    print(f"[MockSimulator] 采用解释器: {self.algo_python_bin}")
                    print(f"[MockSimulator] 命令行指令: {' '.join(cmd)}")
                    print(f"[MockSimulator] 完整运行日志实时记录至: {algo_log_path}")
                return {
                    "accepted": True,
                    "status": "started",
                    "mode": self.arena.mode,
                    "pid": self.algo_process.pid,
                    "script": script_name,
                    "python": self.algo_python_bin,
                }
            except Exception as e:
                self.algo_last_error = f"后台启动失败: {e}"
                return {"accepted": False, "error": self.algo_last_error}

    def stop_algorithm(self) -> Dict[str, Any]:
        with self.arena.lock:
            if self.algo_process is not None and self.algo_process.poll() is None:
                try:
                    self.algo_process.terminate()
                    self.algo_process.wait(timeout=1.0)
                except Exception:
                    self.algo_process.kill()
                self.algo_process = None
                if self.algo_log_file and not self.algo_log_file.closed:
                    try:
                        self.algo_log_file.close()
                    except Exception:
                        pass
                if self.arena.verbose:
                    print("[MockSimulator] ⏹ 已停止后台算法进程")
                return {"accepted": True, "status": "stopped"}
            self.algo_process = None
            return {"accepted": True, "status": "not_running"}


# =============================================================================
# 6. 自动化验证套件 (--test)
# =============================================================================

def run_self_tests() -> bool:
    """运行全套数学模型、物理规则、持久化日志与 HTTP 接口自检。"""
    print(f"\n{'='*30} 开始运行 Mock 模拟器自测套件 {'='*30}")
    all_passed = True

    def check(name: str, condition: bool, msg: str = "") -> None:
        nonlocal all_passed
        if condition:
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {msg}")
            all_passed = False

    # 1. 角度工具测试
    check("normalize_deg_360", normalize_deg_360(-30.0) == 330.0 and normalize_deg_360(370.0) == 10.0)
    check("angle_difference_deg", (
        math.isclose(angle_difference_deg(10.0, 350.0), 20.0) and
        math.isclose(angle_difference_deg(0.0, 180.0), 180.0) and
        math.isclose(angle_difference_deg(90.0, 270.0), 180.0) and
        math.isclose(angle_difference_deg(100.0, 120.0), 20.0)
    ))

    # 2. 面积均匀采样范围测试
    rng = random.Random(42)
    pts = [sample_area_uniform_disk(1800.0, rng) for _ in range(500)]
    all_in_arena = all(math.hypot(x, y) <= 1800.0 + 1e-9 for x, y in pts)
    check("sample_area_uniform_disk 范围判定", all_in_arena)

    # 3. 定向源辐射判定测试
    dir_target = Target(channel=1, x=0.0, y=0.0, recv_radius=1200.0, is_directional=True, dir_angle_deg=90.0)
    check("定向源辐射: 正前方 (0, 500)", dir_target.is_in_coverage(0.0, 500.0) is True)
    check("定向源辐射: 正东边界 (500, 0)", dir_target.is_in_coverage(500.0, 0.0) is True)
    check("定向源辐射: 正西边界 (-500, 0)", dir_target.is_in_coverage(-500.0, 0.0) is True)
    check("定向源辐射: 正后方 (0, -500)", dir_target.is_in_coverage(0.0, -500.0) is False)
    check("定向源辐射: 超出半径 (0, 1500)", dir_target.is_in_coverage(0.0, 1500.0) is False)

    # 4. 全向源与朝向无关清除测试
    omni_target = Target(channel=2, x=100.0, y=100.0, recv_radius=1200.0, is_directional=False)
    check("全向源辐射: (100, 1000)", omni_target.is_in_coverage(100.0, 1000.0) is True)
    check("全向源辐射: 超出半径", omni_target.is_in_coverage(100.0, 1500.0) is False)

    # 5. 端到端状态机与虚拟时间推进测试
    test_arena = Arena(mode="p4", seed=123, arena_radius=1800.0, verbose=False)

    # (a) 测试 /enter
    res_enter = test_arena.enter("req-enter-1", "test_arena", "test_dog")
    check("/enter 返回 accepted", res_enter.get("accepted") is True)
    check("/enter 初始 virtual_time_s == 0.0", res_enter.get("virtual_time_s") == 0.0)
    check("/enter 包含 remaining_real_duration_s", "remaining_real_duration_s" in res_enter)

    # 注入测试目标信道 5
    test_target = Target(channel=5, x=300.0, y=400.0, recv_radius=1200.0, is_directional=False)
    test_arena.targets = {5: test_target}

    # (b) 测试 /measure: 移动 (0,0) -> (300, 0), 切频 1 -> 5
    # 移动距离 300m, t_move = 300/5 = 60s; t_switch = 1s; t_action = 5s; 总增量 = 66s
    res_m1 = test_arena.measure("req-m-1", 300.0, 0.0, 5)
    check("/measure 虚拟时间推进 (66.0s)", math.isclose(res_m1.get("virtual_time_s", 0), 66.0, abs_tol=1e-3))
    check("/measure 测向有结果 (direction)", res_m1.get("measure_result") == "direction")
    svd = res_m1.get("svd_deg", 0.0)
    check("/measure 示向度准确性 (89°~91°)", 89.0 <= svd <= 91.0, f"实际 svd={svd}")

    # (c) 测试 /measure 近距 (near): 机器狗移动到 (300, 398)，距离 2m <= 5m
    res_near = test_arena.measure("req-m-2", 300.0, 398.0, 5)
    check("/measure 近距饱和 (near)", res_near.get("measure_result") == "near")
    check("/measure near 不返回 svd_deg", "svd_deg" not in res_near)
    check("/measure near 虚拟时间 (150.6s)", math.isclose(res_near.get("virtual_time_s", 0), 150.6, abs_tol=1e-3))

    # (d) 测试 /clear 失败: 在 (0, 0) 尝试清除信道 5，距离 500m > 20m
    res_c_fail = test_arena.clear("req-c-1", 0.0, 0.0, 5)
    check("/clear 范围外失败 (no_target_in_range)", res_c_fail.get("clear_result") == "no_target_in_range")

    # (e) 测试 /clear 成功: 移动到 (300, 410)，距离 10m <= 20m
    res_c_succ = test_arena.clear("req-c-2", 300.0, 410.0, 5)
    check("/clear 范围内成功 (success)", res_c_succ.get("clear_result") == "success")
    check("目标清除标记为 True", test_arena.targets[5].cleared is True)

    # (f) 测试幂等重试: 再次发送相同 request_id
    res_retry = test_arena.clear("req-c-2", 300.0, 410.0, 5)
    check("幂等重试返回一致结果", res_retry == res_c_succ)

    # (g) 测试 /exit
    res_exit = test_arena.exit("req-exit-1")
    check("/exit 正常响应", res_exit.get("accepted") is True and res_exit.get("exit_reason") == "user_exit")

    # 6. 持久化日志与异常崩溃容错测试
    live_log = test_arena.logger.get_live_data()
    check("持久化日志帧数匹配 (共 5 步)", len(live_log.get("frames", [])) == 5)
    latest_file = LOGS_DIR / "latest.json"
    check("latest.json 成功落盘并存在", latest_file.is_file())
    if latest_file.is_file():
        with open(latest_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        check("磁盘持久化日志解析成功且结构完整", disk_data.get("session_id") == live_log.get("session_id"))

    # 7. 懒加载日志与对战启动测试 (未开始对战不产生 session_*.json，仅更新 latest.json)
    existing_sessions_before = set(LOGS_DIR.glob("session_*.json"))
    lazy_arena = Arena(mode="p3", seed=999, verbose=False)
    check("待机模式下 logger.current_file 为 None", lazy_arena.logger.current_file is None)
    lazy_sessions_during = set(LOGS_DIR.glob("session_*.json"))
    check("重置待机状态不产生 session_*.json 文件", len(lazy_sessions_during - existing_sessions_before) == 0)

    # 机器狗算法正式进入 (/enter)，此时才正式落盘 session_*.json
    res_lazy_enter = lazy_arena.enter("req-lazy-enter", "lazy_arena", "robot-lazy")
    check("/enter 后 logger.current_file 成功激活", lazy_arena.logger.current_file is not None)
    check("激活后 session_*.json 成功落盘", lazy_arena.logger.current_file.is_file() if lazy_arena.logger.current_file else False)
    if lazy_arena.logger.current_file and lazy_arena.logger.current_file.is_file():
        lazy_arena.logger.current_file.unlink(missing_ok=True)

    print(f"\n自测总结: {'全部通过!' if all_passed else '存在失败项，请检查！'}")
    print(f"{'='*70}\n")
    return all_passed


# =============================================================================
# 7. CLI 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2026 CUMCM B题 本地高保真 Mock 模拟器服务端 + 可视化 WebUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听的主机地址")
    parser.add_argument("--port", type=int, default=2026, help="监听的端口号")
    parser.add_argument(
        "--mode",
        choices=["p3", "p4"],
        default="p4",
        help="模拟器模式：'p3' 为全向干扰源；'p4' 为全向与定向混合干扰源",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机数种子（用于复现固定算例）")
    parser.add_argument("--targets", type=int, default=None, help="干扰源数量（默认在 [10, 16] 随机）")
    parser.add_argument("--arena-radius", type=float, default=1800.0, help="场地半径 (m)")
    parser.add_argument(
        "--dog-speed",
        "--robot-speed",
        dest="dog_speed",
        type=float,
        default=5.0,
        help="机器狗移动速度 (m/s)",
    )
    parser.add_argument("--clear-radius", type=float, default=20.0, help="激光清除判定半径 (m)")
    parser.add_argument("--near-radius", type=float, default=5.0, help="近距饱和判定半径 (m)")
    parser.add_argument(
        "--python-bin",
        default=None,
        help="执行算法程序的 Python 解释器路径 (默认自动探测 cumcm2026-b 或当前具备 numpy 的环境)",
    )
    parser.add_argument("--headless", action="store_true", help="无头运行模式（不弹出提示或专注于后台批处理）")
    parser.add_argument("--verbose", action="store_true", help="打印详细执行日志与战报")
    parser.add_argument("--verbose-http", action="store_true", help="打印 HTTP 访问底层日志")
    parser.add_argument("--test", action="store_true", help="执行内置单元与协议自测并退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.test:
        success = run_self_tests()
        sys.exit(0 if success else 1)

    arena = Arena(
        mode=args.mode,
        seed=args.seed,
        target_count=args.targets,
        arena_radius=args.arena_radius,
        dog_speed=args.dog_speed,
        clear_radius=args.clear_radius,
        near_radius=args.near_radius,
        verbose=args.verbose,
    )

    server_address = (args.host, args.port)
    try:
        server = MockSimulatorServer(
            server_address,
            arena,
            python_bin=args.python_bin,
            verbose_http=args.verbose_http,
        )
    except OSError as exc:
        sys.stderr.write(f"启动失败: 无法绑定到 {args.host}:{args.port} ({exc})\n")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  2026 CUMCM B题 Mock 模拟器服务端 + 可视化 WebUI 已启动")
    print(f"  可视化控制台: http://{args.host}:{args.port}/ (浏览器直接打开查看)")
    print(f"  运行模式: {args.mode.upper()} ({'全向干扰源' if args.mode == 'p3' else '全向 + 定向混合干扰源'})")
    print(f"  算法执行 Python: {server.algo_python_bin}")
    if args.seed is not None:
        print(f"  随机种子: {args.seed}")
    print(f"  持久化日志: 已启用 (保存在 {LOGS_DIR})")
    print(f"  详细输出: {'开启' if args.verbose else '关闭 (使用 --verbose 可查看每步真值对比)'}")
    print(f"  支持接口: POST /enter, /measure, /clear, /exit  |  GET /api/logs, /api/live")
    print(f"  按 Ctrl+C 可终止服务")
    print(f"{'='*70}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MockSimulator] 接收到终止信号，正在关闭服务...")
    finally:
        server.shutdown()
        server.server_close()
        print("[MockSimulator] 服务已安全关闭。")


if __name__ == "__main__":
    main()
