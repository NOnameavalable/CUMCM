"""验证“外切正八边形顶点 + 内接正八边形边为弦 + 中心原点”17检测站全覆盖脚本。

精确几何定义：
1. 场地为半径 R_arena = 1800 米的大圆；
2. 外圈 8 个检测圆心：位于场地外切正八边形的 8 个顶点上
   R_out = 1800 / cos(22.5°) ≈ 1948.31 米；
3. 内圈 8 个检测圆心：内接正八边形的边是每个半径 1000 米探测圆的“弦”
   - 内接八边形边长 (弦长): L = 2 * 1800 * sin(22.5°) ≈ 1377.66 米
   - 半弦长: L / 2 ≈ 688.83 米
   - 探测圆心到弦的距离 (弦心距): h = sqrt(1000^2 - (L/2)^2) ≈ 724.92 米
   - 八边形边心距: d_edge = 1800 * cos(22.5°) ≈ 1662.98 米
   - 内圈探测圆心轨道半径: R_in = d_edge - h ≈ 938.06 米
   - 特征：探测圆圆周恰好精确穿过内接八边形该边的两个端点！
4. 中心检测圆心：位于原点 (0, 0)，半径 1000 米；
5. 在 1800 米圆域内高密度随机枚举 25,000 个干扰源，进行 100% 覆盖验证，并输出高清图像。
"""

from __future__ import annotations

import math
import random
import struct
import time
import zlib
from pathlib import Path


def generate_stations(
    r_arena: float = 1800.0,
    r_detect: float = 1000.0,
) -> tuple[list[tuple[float, float, str]], float, float, float, float]:
    """根据严格“边为弦”的几何定义计算17个检测站点的精确坐标。"""
    cos_22_5 = math.cos(math.radians(22.5))
    sin_22_5 = math.sin(math.radians(22.5))

    # 外圈：外切正八边形顶点
    r_out = r_arena / cos_22_5  # ≈ 1948.31m

    # 内圈：内接正八边形的边为1000m圆的弦
    half_chord = r_arena * sin_22_5  # ≈ 688.83m
    chord_len = 2.0 * half_chord  # ≈ 1377.66m
    h_chord = math.sqrt(r_detect**2 - half_chord**2)  # ≈ 724.92m
    d_edge = r_arena * cos_22_5  # ≈ 1662.98m
    r_in = d_edge - h_chord  # ≈ 938.06m

    stations: list[tuple[float, float, str]] = [(0.0, 0.0, "中心原点 P0")]

    # 外圈 8 点 (外切顶点)
    for i in range(8):
        deg = 45.0 * i
        rad = math.radians(deg)
        stations.append((r_out * math.cos(rad), r_out * math.sin(rad), f"外切八边形顶点 P_out_{i+1}"))

    # 内圈 8 点 (以八边形边为弦的圆心)
    for j in range(8):
        deg = 45.0 * j + 22.5
        rad = math.radians(deg)
        stations.append((r_in * math.cos(rad), r_in * math.sin(rad), f"边为弦探测圆心 P_in_{j+1}"))

    return stations, r_out, r_in, chord_len, h_chord


def sample_targets(sample_count: int = 25000, r_arena: float = 1800.0, seed: int = 2026):
    """面积均匀随机生成干扰源。"""
    random.seed(seed)
    targets = []
    for _ in range(sample_count):
        r = r_arena * math.sqrt(random.random())
        theta = random.random() * 2.0 * math.pi
        targets.append((r * math.cos(theta), r * math.sin(theta)))
    return targets


def encode_png(width: int, height: int, rgb_buffer: bytearray) -> bytes:
    """零依赖标准 PNG 编码。"""
    line_bytes = width * 3
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)
        start = y * line_bytes
        raw_data.extend(rgb_buffer[start : start + line_bytes])

    compressed = zlib.compress(bytes(raw_data), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def render_verification():
    width, height = 1800, 1800
    r_arena = 1800.0
    r_detect = 1000.0
    sample_count = 25000

    stations, r_out, r_in, chord_len, h_chord = generate_stations(r_arena, r_detect)
    targets = sample_targets(sample_count, r_arena)

    # 1. 检验覆盖
    covered_targets = []
    uncovered_targets = []
    max_min_dist = 0.0

    for x, y in targets:
        min_dist = min(math.hypot(x - sx, y - sy) for sx, sy, _ in stations)
        if min_dist > max_min_dist:
            max_min_dist = min_dist
        if min_dist <= r_detect:
            covered_targets.append((x, y))
        else:
            uncovered_targets.append((x, y))

    cov_rate = len(covered_targets) / len(targets) * 100.0
    print(f"=== 【内接八边形边为弦】严格几何验证结果 ===")
    print(f"外切正八边形顶点半径 R_out: {r_out:.2f} 米")
    print(f"内接正八边形边长 (弦长 L): {chord_len:.2f} 米 (半弦: {chord_len/2:.2f} 米)")
    print(f"探测圆心至弦的距离 (弦心距 h): {h_chord:.2f} 米")
    print(f"内圈圆心轨道半径 R_in (d_edge - h): {r_in:.2f} 米")
    print(f"随机干扰源样本总数: {len(targets)}")
    print(f"覆盖点数 (绿色): {len(covered_targets)} ({cov_rate:.2f}%)")
    print(f"未覆盖点数 (红色): {len(uncovered_targets)} ({len(uncovered_targets)/len(targets)*100:.2f}%)")
    print(f"全场点距最近检测站最大距离: {max_min_dist:.2f} 米 (安全门限: {r_detect} 米)")

    # 2. 画布初始化 (深空黑蓝背景: #090d16 -> 9, 13, 22)
    buf = bytearray([9, 13, 22] * (width * height))

    view_range = 2300.0
    scale = width / (2.0 * view_range)
    ox = width / 2.0
    oy = height / 2.0

    def to_screen(wx: float, wy: float) -> tuple[int, int]:
        return int(ox + wx * scale), int(oy - wy * scale)

    def set_pixel(u: int, v: int, r: int, g: int, b: int) -> None:
        if 0 <= u < width and 0 <= v < height:
            idx = (v * width + u) * 3
            buf[idx] = r
            buf[idx + 1] = g
            buf[idx + 2] = b

    def draw_line(x1: float, y1: float, x2: float, y2: float, r: int, g: int, b: int, thickness: int = 1):
        u1, v1 = to_screen(x1, y1)
        u2, v2 = to_screen(x2, y2)
        dist = int(math.hypot(u2 - u1, v2 - v1))
        steps = max(1, dist * 2)
        for s in range(steps + 1):
            t = s / steps
            u = int(u1 + t * (u2 - u1))
            v = int(v1 + t * (v2 - v1))
            for dx in range(-thickness // 2, thickness // 2 + 1):
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    set_pixel(u + dx, v + dy, r, g, b)

    def draw_circle(cx: float, cy: float, radius: float, r: int, g: int, b: int, thickness: int = 2):
        u_c, v_c = to_screen(cx, cy)
        r_px = radius * scale
        steps = max(100, int(2 * math.pi * r_px * 2))
        for step in range(steps):
            ang = 2 * math.pi * step / steps
            px = int(u_c + r_px * math.cos(ang))
            py = int(v_c - r_px * math.sin(ang))
            for dx in range(-thickness // 2, thickness // 2 + 1):
                for dy in range(-thickness // 2, thickness // 2 + 1):
                    set_pixel(px + dx, py + dy, r, g, b)

    # 3. 绘制内接正八边形（弦线集合，高亮珊瑚橙: #f97316 -> 249, 115, 22）
    inscribed_pts = []
    for k in range(8):
        rad = math.radians(45.0 * k)
        inscribed_pts.append((r_arena * math.cos(rad), r_arena * math.sin(rad)))
    for k in range(8):
        pA = inscribed_pts[k]
        pB = inscribed_pts[(k + 1) % 8]
        draw_line(pA[0], pA[1], pB[0], pB[1], 249, 115, 22, thickness=3)

    # 4. 绘制 17 个检测圆的覆盖边界 (高科技冷蓝: #38bdf8 -> 56, 189, 248)
    for sx, sy, _ in stations:
        draw_circle(sx, sy, r_detect, 56, 189, 248, thickness=2)

    # 5. 绘制所有随机干扰源点
    # 覆盖到的点：翡翠绿 (#22c55e -> 34, 197, 94)
    for tx, ty in covered_targets:
        u, v = to_screen(tx, ty)
        set_pixel(u, v, 34, 197, 94)
        set_pixel(u + 1, v, 34, 197, 94)
        set_pixel(u, v + 1, 34, 197, 94)
        set_pixel(u + 1, v + 1, 34, 197, 94)

    # 未覆盖到的点：警示红 (#ef4444 -> 239, 68, 68)
    for tx, ty in uncovered_targets:
        u, v = to_screen(tx, ty)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                set_pixel(u + dx, v + dy, 239, 68, 68)

    # 6. 绘制 1800m 场地大圆边界 (高亮琥珀金: #fbbf24 -> 251, 191, 36)
    draw_circle(0.0, 0.0, r_arena, 251, 191, 36, thickness=4)

    # 7. 绘制 17 个检测站点中心标点 (内亮白外水蓝)
    for sx, sy, name in stations:
        u, v = to_screen(sx, sy)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                d2 = dx * dx + dy * dy
                if d2 <= 9:
                    set_pixel(u + dx, v + dy, 255, 255, 255)
                elif d2 <= 20:
                    set_pixel(u + dx, v + dy, 14, 165, 233)

    png_bytes = encode_png(width, height, buf)
    target_path = Path("coverage_verification.png")
    target_path.write_bytes(png_bytes)
    print(f"高清图像已成功更新至: {target_path.resolve()} (文件大小: {len(png_bytes)/1024:.1f} KB)")


if __name__ == "__main__":
    t0 = time.time()
    render_verification()
    print(f"全部执行完毕，耗时: {time.time() - t0:.2f} 秒。")
