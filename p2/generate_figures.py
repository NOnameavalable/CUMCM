"""生成第6章（问题二）论文高清图表与数据。"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon as PolygonPatch, Circle as MplCircle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from P2 import (
    solve_problem_2,
    Problem2Config,
    evaluate_problem_2_candidate,
    adjusted_worst_radius,
    _plot_geometry,
    _simulate_observation,
    _update_posterior_states,
    _project_target_points,
    _minimum_enclosing_circle,
    _state_arrays,
    Observation,
)
from utils import Point, DetectionSector, Region

# 确保中文字体
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for preferred_font in ('PingFang SC', 'Arial Unicode MS', 'Heiti SC', 'STHeiti', 'Microsoft YaHei', 'Noto Sans SC', 'SimHei'):
    if preferred_font in available_fonts:
        plt.rcParams['font.sans-serif'] = [preferred_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

print('Solving Problem 2 baseline scenario (S1=(0,0), theta1=0 deg)...')
res = solve_problem_2((0.0, 0.0), 0.0)
best = res.best_candidate
print(f'Best candidate: a={best.a}, b={best.b}, worst_radius={best.worst_radius:.3f}')

# ==================== 图 6-1 绘制 ====================
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

_plot_geometry(ax, res.first_feasible_region, facecolor='#b0bec5', alpha=0.35, edgecolor='#78909c', linewidth=1.2, label=r'首次可行域 $\Omega_1$')
_plot_geometry(ax, res.safe_candidate_region, facecolor='#a5d6a7', alpha=0.25, edgecolor='#388e3c', linewidth=1.2, linestyle='--', label=r'保证接收区域 $\mathcal{S}$')
_plot_geometry(ax, res.recommended_region, facecolor='#ce93d8', alpha=0.35, edgecolor='#7b1fa2', linewidth=1.5, label=r'推荐近优区域 $\mathcal{C}$')

pts_in = [c for c in res.candidates if c.guaranteed_signal]
pts_out = [c for c in res.candidates if not c.guaranteed_signal]

sc_out = ax.scatter([c.a for c in pts_out], [c.b for c in pts_out], c=[c.worst_radius for c in pts_out], cmap='viridis_r', s=16, alpha=0.22, vmin=40, vmax=350)
sc_in = ax.scatter([c.a for c in pts_in], [c.b for c in pts_in], c=[c.worst_radius for c in pts_in], cmap='viridis_r', s=24, alpha=0.85, vmin=40, vmax=350, edgecolors='none')

cbar = fig.colorbar(sc_in, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label(r'离散情景最坏覆盖半径 $\widehat{R}_{\max}(S_2)$ / m', fontsize=11)

ax.scatter([0], [0], marker='s', s=80, color='black', zorder=5, label=r'首次检测点 $S_1(0,0)$')
ax.scatter([best.a], [best.b], marker='*', s=220, color='#d32f2f', edgecolor='black', linewidth=0.8, zorder=6, label=f'算法最优点 $S_2^*(800, 600)$\n' + r'$R_{\max}=52.11$ m')

# 局部放大近优点簇
axins = inset_axes(ax, width='32%', height='32%', loc='lower right', borderpad=2)
_plot_geometry(axins, res.safe_candidate_region, facecolor='#a5d6a7', alpha=0.2, edgecolor='#388e3c', linewidth=1.0, linestyle='--')
_plot_geometry(axins, res.recommended_region, facecolor='#ce93d8', alpha=0.4, edgecolor='#7b1fa2', linewidth=1.5)

fine_pts = [c for c in res.candidates if abs(c.a - best.a) <= 180 and abs(c.b - best.b) <= 180]
axins.scatter([c.a for c in fine_pts], [c.b for c in fine_pts], c=[c.worst_radius for c in fine_pts], cmap='viridis_r', s=45, alpha=0.9, vmin=40, vmax=350, edgecolors='k', linewidths=0.5)
axins.scatter([best.a], [best.b], marker='*', s=200, color='#d32f2f', edgecolor='black', zorder=6)
axins.set_xlim(best.a - 120, best.a + 120)
axins.set_ylim(best.b - 120, best.b + 120)
axins.set_title('近优点簇与 50 m 细网格', fontsize=9)
axins.grid(True, linestyle=':', alpha=0.6)
mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.5', lw=0.8)

ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('沿示向线局部纵向坐标 a / m', fontsize=11)
ax.set_ylabel('沿示向线局部横向坐标 b / m', fontsize=11)
ax.set_title(r'图 6-1：第二检测点评价分布与推荐区域（$S_1=(0,0), \theta_1=0^\circ$）', fontsize=13, fontweight='bold', pad=12)
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(loc='upper left', framealpha=0.9, fontsize=9.5)

fig.savefig('/Users/ray_zhang/Developer/CUMCM/paper/problem2/figures/fig6-1_candidate_evaluation.png', dpi=300, bbox_inches='tight')
fig.savefig('/Users/ray_zhang/Developer/CUMCM/p2/figures/fig6-1_candidate_evaluation.png', dpi=300, bbox_inches='tight')
fig.savefig('/Users/ray_zhang/Developer/CUMCM/paper/problem2/figures/fig6-1_candidate_evaluation.pdf', bbox_inches='tight')
fig.savefig('/Users/ray_zhang/Developer/CUMCM/p2/figures/fig6-1_candidate_evaluation.pdf', bbox_inches='tight')
plt.close(fig)
print('Fig 6-1 saved successfully.')

# ==================== 图 6-2 绘制 ====================
z_sample = Point(1000.0, 10.0)
err_sample = 0.0

pt_base = Point(800.0, 0.0)
pt_opt = Point(800.0, 600.0)

arrays = _state_arrays(res.states)
cfg = res.config

def get_demo_posterior(cand_pt):
    bearing = cand_pt.bearing_to(z_sample)
    obs = Observation('direction', bearing + err_sample)
    post_states, _ = _update_posterior_states(res.states, arrays, cand_pt, obs, cfg)
    pts, cell_radii = _project_target_points(post_states)
    circ = _minimum_enclosing_circle(pts, cfg.random_seed)
    max_delta = max(cell_radii) if cell_radii else 0.0
    R_hat = circ.radius + max_delta
    sec = DetectionSector.from_measurement(cand_pt, obs.bearing_deg, cfg.bearing_error_deg).to_region(
        cfg.receive_radius_max, cfg.near_radius, cfg.sector_arc_samples
    )
    return obs, post_states, pts, circ, max_delta, R_hat, sec

obs_b, states_b, pts_b, circ_b, d_b, R_hat_b, sec_b = get_demo_posterior(pt_base)
obs_o, states_o, pts_o, circ_o, d_o, R_hat_o, sec_o = get_demo_posterior(pt_opt)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)

for ax, pt, name, obs, pts, circ, R_hat, sec, is_opt in [
    (ax1, pt_base, r'（a）沿示向线基准点 $S_2(800, 0)$', obs_b, pts_b, circ_b, R_hat_b, sec_b, False),
    (ax2, pt_opt, r'（b）优化第二检测点 $S_2^*(800, 600)$', obs_o, pts_o, circ_o, R_hat_o, sec_o, True)
]:
    _plot_geometry(ax, res.first_feasible_region, facecolor='#b0bec5', alpha=0.3, edgecolor='#78909c', linewidth=1.0, label=r'首次可行域 $\Omega_1$')
    _plot_geometry(ax, sec, facecolor='#ffe0b2', alpha=0.35, edgecolor='#ff9800', linewidth=1.2, label=r'第二次测向误差扇区 $W_2$')
    
    ax.scatter([p.x for p in pts], [p.y for p in pts], c='#1565c0', s=20, alpha=0.85, zorder=4, label=f'相容目标代表点 (N={len(pts)})')
    
    circle_samp = MplCircle((circ.center.x, circ.center.y), circ.radius, fill=False, edgecolor='#1e88e5', linestyle='--', linewidth=1.5, zorder=5, label=f'最小覆盖圆 ($\\rho_o={circ.radius:.1f}$ m)')
    circle_cons = MplCircle((circ.center.x, circ.center.y), R_hat, fill=False, edgecolor='#e53935', linestyle='-', linewidth=1.8, zorder=5, label=f'误差修正圆 ($\\widehat{{R}}_o={R_hat:.1f}$ m)')
    ax.add_patch(circle_samp)
    ax.add_patch(circle_cons)
    
    ax.scatter([0], [0], marker='s', s=70, color='black', zorder=6, label=r'首次检测点 $S_1$')
    ax.scatter([pt.x], [pt.y], marker='^' if not is_opt else '*', s=120 if not is_opt else 160, color='#ff6f00' if not is_opt else '#d32f2f', edgecolor='black', zorder=6, label=r'第二检测点 $S_2$')
    ax.scatter([circ.center.x], [circ.center.y], marker='+', s=100, color='#b71c1c', linewidth=2, zorder=6, label=r'后验覆盖圆心 $c_o$')
    ax.scatter([z_sample.x], [z_sample.y], marker='x', s=80, color='green', linewidth=2, zorder=6, label='示意真实目标')
    
    scale_x = circ.center.x - 70
    scale_y = circ.center.y - 70
    ax.plot([scale_x, scale_x + 20], [scale_y, scale_y], color='black', linewidth=2.5, zorder=7)
    ax.text(scale_x + 10, scale_y - 8, '20 m', ha='center', va='top', fontsize=9, fontweight='bold')
    
    span = 140 if not is_opt else 90
    ax.set_xlim(circ.center.x - span, circ.center.x + span)
    ax.set_ylim(circ.center.y - span, circ.center.y + span)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('全局坐标 x / m', fontsize=11)
    ax.set_ylabel('全局坐标 y / m', fontsize=11)
    status_str = "未达标" if R_hat > 20 else "达标"
    ax.set_title(f'{name}\n' + r'后验修正半径 $\widehat{R}_o=' + f'{R_hat:.1f}$ m' + f'（清除判据 $\\leq 20$ m: {status_str}）', fontsize=10.5, fontweight='bold', pad=8)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)

fig.suptitle(r'图 6-2：第二次观测后相容区域收缩与后验覆盖圆对比（示意情景：$z=(1000, 10), \varepsilon_2=0^\circ$）', fontsize=13, fontweight='bold', y=0.98)
fig.savefig('/Users/ray_zhang/Developer/CUMCM/paper/problem2/figures/fig6-2_posterior_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig('/Users/ray_zhang/Developer/CUMCM/p2/figures/fig6-2_posterior_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig('/Users/ray_zhang/Developer/CUMCM/paper/problem2/figures/fig6-2_posterior_comparison.pdf', bbox_inches='tight')
fig.savefig('/Users/ray_zhang/Developer/CUMCM/p2/figures/fig6-2_posterior_comparison.pdf', bbox_inches='tight')
plt.close(fig)
print('Fig 6-2 saved successfully.')
