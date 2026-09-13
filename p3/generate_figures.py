"""生成第7章（问题三）论文高清图表与数据。"""
import sys
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon as PolygonPatch, Circle as MplCircle, FancyArrowPatch, Rectangle
import shapely.geometry as sg

# 确保中文字体
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for preferred_font in ('PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'STHeiti', 'Microsoft YaHei', 'Noto Sans SC', 'SimHei'):
    if preferred_font in available_fonts:
        plt.rcParams['font.sans-serif'] = [preferred_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

out_dirs = [
    Path('/Users/ray_zhang/Developer/CUMCM/p3/figures'),
    Path('/Users/ray_zhang/Developer/CUMCM/paper/problem3/figures'),
]
for d in out_dirs:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 图 7-1：自动搜索与清除的闭环反馈控制流程图
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, xy, w, h, text, facecolor='#e3f2fd', edgecolor='#1976d2', fontsize=10, bold=False):
    box = Rectangle((xy[0] - w/2, xy[1] - h/2), w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=3)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fontsize, fontweight=weight, color='#212121', zorder=4)

def draw_arrow(ax, p1, p2, label=None, label_offset=(0, 0), color='#37474f', style='->', lw=1.5):
    arrow = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, color=color, linewidth=lw, zorder=2)
    ax.add_patch(arrow)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=9, color=color, ha='center', va='center', backgroundcolor='white', zorder=5)

# 绘制流程框
draw_box(ax, (6, 9.3), 6.5, 0.7, "系统初始化：原点 (0,0)、初始信道 1\n初始化 20 信道先验信度 & 140 初始搜索任务池", facecolor='#e8f5e9', edgecolor='#2e7d32', bold=True)
draw_box(ax, (6, 8.0), 5.5, 0.6, "更新有效任务池与信道版本", facecolor='#e3f2fd', edgecolor='#1565c0')
draw_box(ax, (6, 6.8), 6.0, 0.7, "任务完成条件判定\n(已清除16个 或 20信道均满足已清除/全七点未发现)", facecolor='#fff3e0', edgecolor='#ef6c00', bold=True)
draw_box(ax, (10.5, 6.8), 2.2, 0.6, "触发 /exit 正常退出\n输出评测统计", facecolor='#c8e6c9', edgecolor='#2e7d32', bold=True)

draw_box(ax, (6, 5.4), 6.8, 0.8, "原地任务优先调度\n(同点先清除；再优先测当前信道，避免频段切换开销)\n离开当前点前：共享二测优化 + 开放路径 TSP (2-opt)", facecolor='#e1f5fe', edgecolor='#0288d1')
draw_box(ax, (6, 4.0), 4.2, 0.6, "串行执行动作：/measure 或 /clear", facecolor='#f3e5f5', edgecolor='#7b1fa2', bold=True)

# 测量分支
draw_box(ax, (2.6, 2.7), 3.8, 1.1, "【测向反馈 /measure】\n· near：目标≤5m，原地直清\n· 首次方向：P2 双侧搜索二测点\n· 多次方向：P1 连续凸多边形交会\n· 无信号：登记覆盖 / 二测回退", facecolor='#ede7f6', edgecolor='#512da8')

# 清除分支
draw_box(ax, (9.4, 2.7), 3.8, 1.1, "【清除反馈 /clear】\n· success：成功清除，注销该信道\n· failure (未命中)：\n  精确扣除 20m 失败圆盘\n  在残余区域内部代表点补清", facecolor='#fce4ec', edgecolor='#c2185b')

# 状态闭环汇合
draw_box(ax, (6, 1.2), 6.0, 0.7, "更新信道状态机、版本号与多边形可行域\n清理失效任务，重建后续任务", facecolor='#e0f2f1', edgecolor='#00796b', bold=True)

# 侧边运行保护注释
draw_box(ax, (10.5, 9.3), 2.4, 0.7, "【运行安全保护】\n截止时刻前 10s 截断\n最大执行 2000 动作", facecolor='#ffebee', edgecolor='#d32f2f', fontsize=8)

# 连线
draw_arrow(ax, (6, 8.95), (6, 8.3))
draw_arrow(ax, (6, 7.7), (6, 7.15))
draw_arrow(ax, (9.0, 6.8), (9.4, 6.8), label="满足", label_offset=(0, 0.2))
draw_arrow(ax, (6, 6.45), (6, 5.8), label="未完成", label_offset=(0.35, 0))
draw_arrow(ax, (6, 5.0), (6, 4.3))

draw_arrow(ax, (4.5, 3.7), (2.6, 3.25))
draw_arrow(ax, (7.5, 3.7), (9.4, 3.25))

draw_arrow(ax, (2.6, 2.15), (4.5, 1.55))
draw_arrow(ax, (9.4, 2.15), (7.5, 1.55))

# 闭环向上箭头
draw_arrow(ax, (3.0, 1.2), (0.7, 1.2), style='-')
draw_arrow(ax, (0.7, 1.2), (0.7, 8.0), style='-')
draw_arrow(ax, (0.7, 8.0), (3.25, 8.0), label="闭环状态反馈迭代", label_offset=(0, 0.25))

plt.title("图 7-1  全向干扰源自主搜索、定位与清除的闭环控制流程图", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
for d in out_dirs:
    fig.savefig(d / 'fig7-1_control_flowchart.png', dpi=300)
    fig.savefig(d / 'fig7-1_control_flowchart.pdf')
plt.close(fig)
print('Fig 7-1 generated successfully.')

# =============================================================================
# 图 7-2：(a) 七点覆盖几何布局  (b) 共享二测前后开放路线对比
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)

# (a) 七点覆盖几何布局
R = 1800.0
r0 = 1000.0
offset = math.sqrt(r0**2 - (R/2.0)**2)
rho = R * math.cos(math.pi / 6.0) - offset  # ~1122.956 m

circle_R = MplCircle((0, 0), R, fill=False, edgecolor='#212121', linewidth=2.0, linestyle='-', label=r'目标圆域边界 $B_0 (R=1800\,\mathrm{m})$')
ax1.add_patch(circle_R)

# 7个覆盖圆
centers = [(0.0, 0.0)] + [
    (rho * math.cos(k * math.pi / 3.0), rho * math.sin(k * math.pi / 3.0))
    for k in range(6)
]

for idx, (cx, cy) in enumerate(centers):
    c_patch = MplCircle((cx, cy), r0, facecolor='#90caf9', alpha=0.22, edgecolor='#1976d2', linewidth=1.2, linestyle='-')
    ax1.add_patch(c_patch)
    ax1.plot(cx, cy, 'o', color='#0d47a1', markersize=6)
    name = f'$S_{idx}$' if idx > 0 else '$S_0(0,0)$'
    ax1.text(cx + 45, cy + 45, name, fontsize=10, fontweight='bold', color='#0d47a1')

# 辅助内接六边形
hex_pts = [
    (R * math.cos(k * math.pi / 3.0), R * math.sin(k * math.pi / 3.0))
    for k in range(7)
]
hex_xs, hex_ys = zip(*hex_pts)
ax1.plot(hex_xs, hex_ys, color='#757575', linestyle='--', linewidth=1.2, label=r'内接正六边形 (边长 $R$)')

# 标注几何关系线段
ax1.plot([0, centers[1][0]], [0, centers[1][1]], color='#d32f2f', linestyle='-', linewidth=1.5, label=r'外点偏置半径 $\rho \approx 1122.96\,\mathrm{m}$')
chord_p1 = (R * math.cos(-math.pi/6), R * math.sin(-math.pi/6))
chord_p2 = (R * math.cos(math.pi/6), R * math.sin(math.pi/6))
ax1.plot([centers[1][0], chord_p1[0]], [centers[1][1], chord_p1[1]], color='#2e7d32', linestyle=':', linewidth=1.5, label=r'覆盖接收半径 $r_0=1000\,\mathrm{m}$')
ax1.plot([centers[1][0], chord_p2[0]], [centers[1][1], chord_p2[1]], color='#2e7d32', linestyle=':', linewidth=1.5)

ax1.set_xlim(-2200, 2200)
ax1.set_ylim(-2200, 2200)
ax1.set_aspect('equal')
ax1.grid(True, linestyle=':', alpha=0.5)
ax1.set_xlabel('$x$ 坐标 / m', fontsize=11)
ax1.set_ylabel('$y$ 坐标 / m', fontsize=11)
ax1.set_title('(a) 七点覆盖几何布局与最小接收半径保证', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

# (b) 共享二测前后开放路线对比
p_now = np.array([0.0, 0.0])
p_ch1_ind = np.array([320.0, 1150.0])
p_ch17_ind = np.array([210.0, 780.0])
p_shared = np.array([270.45, 947.27])
other_tasks = [
    np.array([rho, 0.0]),
    np.array([rho * math.cos(math.pi/3), rho * math.sin(math.pi/3)]),
    np.array([-rho, 0.0]),
]

route_ind = [p_now, p_ch17_ind, p_ch1_ind, other_tasks[1], other_tasks[0], other_tasks[2]]
dist_ind = sum(np.linalg.norm(route_ind[i+1] - route_ind[i]) for i in range(len(route_ind)-1))

route_shared = [p_now, p_shared, other_tasks[1], other_tasks[0], other_tasks[2]]
dist_shared = sum(np.linalg.norm(route_shared[i+1] - route_shared[i]) for i in range(len(route_shared)-1))
saved = dist_ind - dist_shared

r_ind_x = [p[0] for p in route_ind]
r_ind_y = [p[1] for p in route_ind]
ax2.plot(r_ind_x, r_ind_y, color='#78909c', linestyle='--', linewidth=1.6, alpha=0.85, label=f'独立方案开放路线 ($L={dist_ind:.1f}\,\mathrm{{m}}$)')

r_sh_x = [p[0] for p in route_shared]
r_sh_y = [p[1] for p in route_shared]
ax2.plot(r_sh_x, r_sh_y, color='#d84315', linestyle='-', linewidth=2.2, label=f'共享方案开放路线 ($L={dist_shared:.1f}\,\mathrm{{m}}$, 节约 ${saved:.1f}\,\mathrm{{m}}$)')

ax2.scatter([p_now[0]], [p_now[1]], marker='o', s=100, color='black', zorder=5, label='机器狗当前位置 $x_{\\mathrm{now}}(0,0)$')
ax2.scatter([p_ch1_ind[0], p_ch17_ind[0]], [p_ch1_ind[1], p_ch17_ind[1]], marker='s', s=80, color='#1e88e5', zorder=5, label='独立二测点 (信道 1 与信道 17)')
ax2.scatter([p_shared[0]], [p_shared[1]], marker='*', s=220, color='#e53935', edgecolor='black', linewidth=0.8, zorder=6, label='经复评通过的共享二测点')

for ot in other_tasks:
    ax2.scatter([ot[0]], [ot[1]], marker='^', s=70, color='#546e7a', zorder=4)
ax2.scatter([], [], marker='^', s=70, color='#546e7a', label='其余待执行搜索任务点')

for i in range(len(route_shared)-1):
    mid = (route_shared[i] + route_shared[i+1]) / 2.0
    dx = (route_shared[i+1][0] - route_shared[i][0]) * 0.05
    dy = (route_shared[i+1][1] - route_shared[i][1]) * 0.05
    ax2.annotate('', xy=(mid[0] + dx, mid[1] + dy), xytext=(mid[0] - dx, mid[1] - dy),
                 arrowprops=dict(arrowstyle="->", color='#d84315', lw=1.5))

ax2.set_xlim(-1500, 1600)
ax2.set_ylim(-500, 1600)
ax2.set_aspect('equal')
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.set_xlabel('$x$ 坐标 / m', fontsize=11)
ax2.set_ylabel('$y$ 坐标 / m', fontsize=11)
ax2.set_title('(b) 同一任务快照下共享二测与开放 TSP 路线对比', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=8.5, framealpha=0.9)

plt.suptitle("图 7-2  七点几何覆盖布局与共享二测路径规划效果对比", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
for d in out_dirs:
    fig.savefig(d / 'fig7-2_coverage_and_routing.png', dpi=300)
    fig.savefig(d / 'fig7-2_coverage_and_routing.pdf')
plt.close(fig)
print('Fig 7-2 generated successfully.')

# =============================================================================
# 图 7-3：单频道交会定位与清除分支示例 (三面板 a, b, c)
# =============================================================================
fig, (ax3a, ax3b, ax3c) = plt.subplots(1, 3, figsize=(18, 5.8), dpi=300)

# (a) 两次方向交会区域
s1 = np.array([0.0, 0.0])
th1 = 186.06
s2 = np.array([-845.15139195, -403.18365019])
th2 = 95.97

def sector_poly(s, theta, length=1200.0, err=1.0):
    t_rad = math.radians(theta)
    t_min = math.radians(theta - err)
    t_max = math.radians(theta + err)
    pts = [s]
    for ang in np.linspace(t_min, t_max, 30):
        pts.append(s + length * np.array([math.cos(ang), math.sin(ang)]))
    return sg.Polygon(pts)

poly1 = sector_poly(s1, th1, 1200)
poly2 = sector_poly(s2, th2, 1200)
inter_a = poly1.intersection(poly2)

ax3a.plot(s1[0], s1[1], 'ks', markersize=7, label=r'测点 $S_1(0,0)$')
ax3a.plot(s2[0], s2[1], 'bs', markersize=7, label=r'二测点 $S_2(-845.2, -403.2)$')

ax3a.plot([s1[0], s1[0] + 1100*math.cos(math.radians(th1))], [s1[1], s1[1] + 1100*math.sin(math.radians(th1))], 'k--', linewidth=1.2, label=r'示向中心线 $\theta_1=186.1^\circ$')
ax3a.plot([s2[0], s2[0] + 800*math.cos(math.radians(th2))], [s2[1], s2[1] + 800*math.sin(math.radians(th2))], 'b--', linewidth=1.2, label=r'示向中心线 $\theta_2=96.0^\circ$')

patch1 = PolygonPatch(list(poly1.exterior.coords), facecolor='#b0bec5', alpha=0.3, edgecolor='none')
patch2 = PolygonPatch(list(poly2.exterior.coords), facecolor='#90caf9', alpha=0.3, edgecolor='none')
ax3a.add_patch(patch1)
ax3a.add_patch(patch2)

if not inter_a.is_empty:
    patch_int = PolygonPatch(list(inter_a.exterior.coords), facecolor='#e53935', alpha=0.6, edgecolor='#b71c1c', linewidth=1.5, label=r'二次交会区域 $\Omega_2$')
    ax3a.add_patch(patch_int)

ax3a.set_xlim(-1100, 100)
ax3a.set_ylim(-600, 100)
ax3a.set_aspect('equal')
ax3a.grid(True, linestyle=':', alpha=0.5)
ax3a.set_xlabel('$x$ 坐标 / m', fontsize=11)
ax3a.set_ylabel('$y$ 坐标 / m', fontsize=11)
ax3a.set_title('(a) 双测向角带交会几何约束', fontsize=12, fontweight='bold')
ax3a.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

# (b) 局部放大后验区域及直径圆全覆盖判据
pA = np.array([-877.29, -93.23])
pB = np.array([-884.45, -81.75])
pC = (pA + pB) / 2.0
p3_poly = np.array([pA, pB, [-881.5, -92.0]])

patch_b = PolygonPatch(p3_poly, facecolor='#ffcdd2', edgecolor='#d32f2f', linewidth=1.5, label=r'定位区域 $\Omega_3$ ($D=13.53\,\mathrm{m}$)')
ax3b.add_patch(patch_b)

circle_diam = MplCircle((pC[0], pC[1]), 6.76, fill=False, edgecolor='#2e7d32', linestyle='--', linewidth=1.8, label=r'直径圆 ($r=6.76\,\mathrm{m}\leq20\,\mathrm{m}$)')
circle_clear = MplCircle((pC[0], pC[1]), 20.0, fill=True, facecolor='#a5d6a7', alpha=0.25, edgecolor='#1b5e20', linewidth=1.5, label=r'光学清除圆 ($r_{\mathrm{clear}}=20\,\mathrm{m}$)')
ax3b.add_patch(circle_clear)
ax3b.add_patch(circle_diam)

ax3b.plot(pC[0], pC[1], 'ro', markersize=6, label=r'直径圆心 $c(-880.87, -87.49)$')
ax3b.plot([pA[0], pB[0]], [pA[1], pB[1]], 'k-', linewidth=1.5, label='最远点对直径线段')

ax3b.set_xlim(-910, -850)
ax3b.set_ylim(-115, -60)
ax3b.set_aspect('equal')
ax3b.grid(True, linestyle=':', alpha=0.5)
ax3b.set_xlabel('$x$ 坐标 / m', fontsize=11)
ax3b.set_ylabel('$y$ 坐标 / m', fontsize=11)
ax3b.set_title('(b) 直径圆覆盖判据满足，直接触发清除', fontsize=12, fontweight='bold')
ax3b.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

# (c) 几何示意：小直径但未覆盖分支与失败残余区扣除
pt1 = np.array([-18.0, 0.0])
pt2 = np.array([18.0, 0.0])
pt3 = np.array([0.0, 24.0])
tri = sg.Polygon([pt1, pt2, pt3])
disk_clear = sg.Point(0, 0).buffer(20.0)
rem_poly = tri.difference(disk_clear)
rep_pt = np.array([rem_poly.representative_point().x, rem_poly.representative_point().y])

patch_c_orig = PolygonPatch(list(tri.exterior.coords), facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=1.2, linestyle=':', label=r'原候选区 ($D=36\,\mathrm{m}\leq40\,\mathrm{m}$)')
ax3c.add_patch(patch_c_orig)

c_disk_fail = MplCircle((0, 0), 20.0, facecolor='#ffcdd2', alpha=0.35, edgecolor='#d32f2f', linewidth=1.5, linestyle='--', label=r'失败清除圆盘 $B(c, 20\,\mathrm{m})$')
ax3c.add_patch(c_disk_fail)

if not rem_poly.is_empty:
    patch_rem = PolygonPatch(list(rem_poly.exterior.coords), facecolor='#ba68c8', alpha=0.55, edgecolor='#4a148c', linewidth=1.8, label=r'残余清除区 $\Omega_{\mathrm{rem}} = \Omega \setminus B(c,20)$')
    ax3c.add_patch(patch_rem)

ax3c.plot(0, 0, 'rx', markersize=8, markeredgewidth=2, label=r'原圆心 $c(0,0)$ (初测/清除失败)')
ax3c.plot(rep_pt[0], rep_pt[1], 'm*', markersize=12, label=r'残余区内部代表点 $p_{\mathrm{rem}}$')

ax3c.set_xlim(-25, 25)
ax3c.set_ylim(-22, 28)
ax3c.set_aspect('equal')
ax3c.grid(True, linestyle=':', alpha=0.5)
ax3c.set_xlabel('$x$ 坐标 / m', fontsize=11)
ax3c.set_ylabel('$y$ 坐标 / m', fontsize=11)
ax3c.set_title('(c) 几何示意：未覆盖原点补测与残余区扣除', fontsize=12, fontweight='bold')
ax3c.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

plt.suptitle("图 7-3  多点示向交会收敛、直径圆覆盖判定与残余区清除机制", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
for d in out_dirs:
    fig.savefig(d / 'fig7-3_localization_and_clearing.png', dpi=300)
    fig.savefig(d / 'fig7-3_localization_and_clearing.pdf')
plt.close(fig)
print('Fig 7-3 generated successfully.')
