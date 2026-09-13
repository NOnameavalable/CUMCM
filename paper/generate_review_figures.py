"""Rebuild corrected paper figures from algorithm outputs and explicit examples.

Run with the project's Python environment. No simulator requests or model changes.
Outputs are limited to paper/problem*/figures and figure_review_values.json.
"""
from pathlib import Path
from types import SimpleNamespace
import sys
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cumcm-paper-matplotlib')
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Wedge, Polygon as Patch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from P1 import solve_problem_1_geometry
from P2 import (Problem2Config, Observation, _build_first_feasible_region,
                _sample_target_cells, _expand_receive_radius_states, _state_arrays,
                _update_posterior_states, _project_target_points)
from P3 import Problem3Config, generate_coverage_points, ObservationRecord
from p4.belief import MixedSourceBelief
from p4.planning import plan_probe
from p4.verify_coverage import generate_stations
from utils import Point, Region, DetectionSector, minimum_enclosing_circle

for font in [Path('/mnt/c/Windows/Fonts/msyh.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]:
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
        break
plt.rcParams.update({'font.size':10, 'axes.unicode_minus':False, 'pdf.fonttype':42})
VALUES = {}

def xy(p): return (p.x, p.y)
def polygon(ax, region, color, label=None, alpha=.3):
    geom = region._geometry if isinstance(region, Region) else region
    parts = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)
    for i, part in enumerate(parts):
        if part.geom_type == 'Polygon':
            ax.add_patch(Patch(np.asarray(part.exterior.coords), facecolor=color, edgecolor=color,
                               alpha=alpha, linewidth=1.2, label=label if i == 0 else None))
def circle(ax, center, radius, color, label=None, style='--', fill=False):
    ax.add_patch(Circle(xy(center), radius, edgecolor=color, facecolor=color if fill else 'none',
                        alpha=.15 if fill else 1, linestyle=style, linewidth=1.4, label=label))
def axes(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x / m'); ax.set_ylabel('y / m'); ax.grid(alpha=.25, linestyle=':')
def save(fig, problem, name):
    directory = ROOT / 'paper' / f'problem{problem}' / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf'):
        fig.savefig(directory/f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

cfg = Problem2Config()
first = Point(0,0)
region = _build_first_feasible_region(first, 0, cfg)
cells = _sample_target_cells(region, cfg)
states = _expand_receive_radius_states(cells, first, cfg)
result = SimpleNamespace(config=cfg, target_cells=cells, states=states, first_feasible_region=region)
truth = Point(1000,10)

# Figure 6-2: full geometry on the same scale, then distinct posterior close-ups.
fig, axs = plt.subplots(2,2,figsize=(13,10),layout='constrained')
for col, sensor in enumerate((Point(800,0),Point(800,600))):
    obs = Observation('direction', sensor.bearing_to(truth))
    post,_ = _update_posterior_states(states,_state_arrays(states),sensor,obs,cfg)
    points, offsets = _project_target_points(post)
    mec = minimum_enclosing_circle(points)
    radius = mec.radius + max(offsets)
    VALUES[f'p2_{col}'] = {'sample_radius':mec.radius,'corrected_radius':radius,'targets':len(points)}
    sector = DetectionSector.from_measurement(sensor,obs.bearing_deg,1).to_region(1500,5)
    for row in range(2):
        ax=axs[row,col]
        polygon(ax,region,'#78909c','首次可行域',.15)
        polygon(ax,sector,'#ef9b25','第二测向扇区',.18)
        ax.scatter([p.x for p in points],[p.y for p in points],s=18,c='#1565c0',label='相容代表点',zorder=4)
        circle(ax,mec.center,mec.radius,'#1565c0',f'代表点最小圆：{mec.radius:.2f} m')
        circle(ax,mec.center,radius,'#c62828',f'修正圆：{radius:.2f} m','-')
        ax.scatter([truth.x],[truth.y],marker='x',c='green',s=80,zorder=7,label='示例真值')
        ax.scatter([mec.center.x],[mec.center.y],marker='+',c='#c62828',s=65,zorder=6)
        if row==0:
            ax.scatter([0,sensor.x],[0,sensor.y],c=['black','#e46d00'],marker='s',s=35,zorder=6)
            ax.annotate('S1',(0,0),xytext=(0,12),textcoords='offset points')
            ax.annotate('S2',xy(sensor),xytext=(8,8),textcoords='offset points')
            ax.plot([0,truth.x,sensor.x],[0,truth.y,sensor.y],color='#555',ls=':',lw=1)
            ax.set_xlim(-100,1600); ax.set_ylim(-400,800)
            axes(ax,f'({chr(97+col)}) '+('沿线基准 (800,0)' if col==0 else '侧移方案 (800,600)')+' · 全景同尺度')
        else:
            span=radius*1.2
            ax.set_xlim(mec.center.x-span,mec.center.x+span)
            ax.set_ylim(mec.center.y-span,mec.center.y+span)
            axes(ax,f'({chr(99+col)}) 后验放大 · 修正半径 {radius:.2f} m > 20 m')
    axs[0,col].legend(fontsize=8,loc='upper left')
fig.suptitle('图 6-2  固定观测示例：上排同尺度全景，下排分别放大（尺度不同）',fontsize=13)
save(fig,2,'fig6-2_posterior_comparison')

# Figure 7-2: exact coverage layout plus explicitly illustrative fixed routes.
stations=generate_coverage_points(Problem3Config())
rho=stations[1].distance_to(first)
fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
ax=axs[0]
for p in stations: circle(ax,p,1000,'#559ac4',fill=True)
circle(ax,first,1800,'black',style='-')
verts=[(1800*math.cos(k*math.pi/3+math.pi/6),1800*math.sin(k*math.pi/3+math.pi/6)) for k in range(7)]
ax.plot(*np.array(verts).T,color='gray',ls='--',label='扇区交界构成的内接六边形')
ax.scatter([p.x for p in stations],[p.y for p in stations],s=24,c='#1565c0',label='七个测站')
ax.plot([0,rho],[0,0],color='#d32f2f',label=f'外点半径 {rho:.2f} m')
ax.set_xlim(-2200,2200);ax.set_ylim(-2200,2200);axes(ax,'(a) 七点覆盖：接收半径 1000 m');ax.legend(fontsize=8,loc='lower left')
ind=[(0,0),(210,780),(320,1150),xy(stations[2]),xy(stations[1]),xy(stations[4])]
shared=[(0,0),(270.45,947.27),xy(stations[2]),xy(stations[1]),xy(stations[4])]
length=lambda route:sum(math.dist(a,b) for a,b in zip(route,route[1:]))
li,ls=length(ind),length(shared)
VALUES['route_example']={'independent':li,'shared':ls,'saved':li-ls,'source':'constructed illustration, not log snapshot'}
ax=axs[1]
ax.plot(*np.array(ind).T,'--s',c='#5585aa',label=f'独立路线 {li:.1f} m')
ax.plot(*np.array(shared).T,'-o',c='#d84315',label=f'共享路线 {ls:.1f} m')
ax.scatter([270.45],[947.27],marker='*',c='#d84315',edgecolor='black',s=150,zorder=5,label='示意共享点')
ax.scatter([0],[0],c='black',s=60,zorder=5,label='起点')
axes(ax,f'(b) 构造路线示意 · 节约 {li-ls:.1f} m')
ax.set_xlim(-1400,1400);ax.set_ylim(-500,1700);ax.legend(fontsize=9,loc='upper left')
fig.suptitle('图 7-2  七点覆盖与共享测点的路线示意（不代表日志快照或最优路线）',fontsize=13)
save(fig,3,'fig7-2_coverage_and_routing')

# Figure 7-3: compute both regions from the same physical settings as P3/P4.
data=[(0,0,186.06),(-845.15139195,-403.18365019,95.97),(-877.29411572,-93.23217920,122.92)]
geometries=[solve_problem_1_geometry(data[:n],receive_radius_max=1500,near_radius=5) for n in (2,3)]
g2,g3=geometries
VALUES['physical_geometry']=[{'area':g.polygon.area,'diameter':g.diameter,'center':xy(g.diameter_circle_center),'mec_radius':minimum_enclosing_circle(g.polygon.convex_hull.vertices).radius} for g in geometries]
fig,axs=plt.subplots(1,3,figsize=(16,6),layout='constrained')
ax=axs[0]
for i,(x,y,t) in enumerate(data[:2]):
    sec=DetectionSector.from_measurement(Point(x,y),t,1).to_region(1500,5)
    polygon(ax,sec,['#78909c','#2196f3'][i],alpha=.2)
    ax.scatter([x],[y],marker='s',s=35,c='black');ax.annotate(f'S{i+1}',(x,y),xytext=(8,8),textcoords='offset points')
polygon(ax,g2.polygon,'#d32f2f')
ax.set_xlim(-1100,100);ax.set_ylim(-650,150);axes(ax,f'(a) 两次测向 · 面积 {g2.polygon.area:.2f} m²')
inset=ax.inset_axes([.56,.05,.40,.38]);polygon(inset,g2.polygon,'#d32f2f',alpha=.45)
b=g2.polygon.bounds;inset.set_xlim(b[0]-5,b[2]+5);inset.set_ylim(b[1]-5,b[3]+5);inset.set_aspect('equal');inset.tick_params(labelsize=7);inset.set_title('交集放大',fontsize=8)
ax=axs[1];polygon(ax,g3.polygon,'#d32f2f','第三测后物理可行域',.5)
circle(ax,g3.diameter_circle_center,g3.diameter/2,'#23823b',f'新直径圆 r={g3.diameter/2:.2f} m')
circle(ax,g2.diameter_circle_center,20,'#1565c0','原圆心的 20 m 清除圆','-')
ax.scatter(*xy(g2.diameter_circle_center),marker='x',s=80,c='#1565c0',label='实际待清除的原圆心')
ax.scatter(*xy(g3.diameter_circle_center),marker='+',s=65,c='#23823b',label='新直径圆心')
a,b=g3.diameter_endpoints;ax.plot([a.x,b.x],[a.y,b.y],c='black')
ax.set_xlim(-904,-850);ax.set_ylim(-118,-64);axes(ax,f'(b) 第三测：D={g3.diameter:.2f} m，面积={g3.polygon.area:.2f} m²');ax.legend(fontsize=8,loc='lower left')
ax=axs[2]
tri=Region.from_geometry(__import__('shapely').geometry.Polygon([(-18,0),(18,0),(0,24)]))
rem=tri.difference(Region.disk(first,20))
polygon(ax,tri,'#edaf37','锐角三角形 D=36 m',.3);polygon(ax,rem,'#9c27b0','残余区',.6)
circle(ax,first,18,'#23823b','直径圆 r=18 m');circle(ax,first,20,'#c62828','失败清除圆 r=20 m','-')
p=rem.representative_point();ax.scatter(*xy(p),marker='*',c='#9c27b0',s=100)
ax.set_xlim(-26,26);ax.set_ylim(-26,30);axes(ax,'(c) 构造示例：锐角三角形与失败扣除');ax.legend(fontsize=8,loc='lower left')
fig.suptitle('图 7-3  物理约束下的交会结果与独立残余区示例',fontsize=13)
save(fig,3,'fig7-3_localization_and_clearing')

# Mixed state example: all updates go through the current implementation.
def mixed():return MixedSourceBelief(result,bearing_error_deg=1,near_radius=5,clear_radius=20)
def update(b,p,kind,bearing=None):
    assert b.update_measurement(ObservationRecord(p,kind,bearing,0)), 'example posterior inconsistent'
def snapshot(b):return (b.target_marginals(),b.omni_probability)
b=mixed();update(b,first,'direction',0);snaps=[snapshot(b)]
update(b,Point(1200,200),'no_signal');snaps.append(snapshot(b))
update(b,Point(600,0),'direction',Point(600,0).bearing_to(truth));snaps.append(snapshot(b))
VALUES['mixed_example_omni_weights']=[s[1] for s in snaps]
fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
ax=axs[0];ax.add_patch(Wedge((0,0),1000,90,270,facecolor='#a5d6a7',alpha=.5,label='可见半圆（朝向 180°）'))
circle(ax,first,1000,'gray','距离上限')
ax.annotate('',(-650,0),(0,0),arrowprops={'arrowstyle':'->','color':'green'})
for label,p,col in [('A：有信号',(-600,250),'green'),('B：背向无信号',(500,250),'#c62828'),('C：超距无信号',(-1200,-450),'#c62828')]:
    ax.scatter(*p,c=col);ax.annotate(label,p,xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
ax.scatter(0,0,marker='*',s=100,c='black');axes(ax,'(a) 距离与朝向的两种无信号原因');ax.set_xlim(-1550,1400);ax.set_ylim(-1250,1250)
ax=axs[1]
for i,(marg,_) in enumerate(snaps):
    ax.scatter([p.x for p,w in marg],[i for _ in marg],s=[5+600*w for p,w in marg],alpha=.65,label=['首测方向','二测无信号','再次方向'][i])
ax.set_yticks(range(3),['首测方向','二测无信号','再次方向']);ax.set_xlabel('候选位置的 x 坐标 / m');ax.set_title('(b) 位置边际权重：点面积表示权重');ax.grid(alpha=.2)
ax=axs[2];omni=np.array([s[1] for s in snaps]);ax.bar(range(3),omni,label='全向假设',color='#559ac4');ax.bar(range(3),1-omni,bottom=omni,label='定向假设',color='#b39ddb')
ax.set_xticks(range(3),['首测方向','二测无信号','再次方向']);ax.set_ylim(0,1);ax.set_ylabel('归一化权重');ax.set_title('(c) 源类型假设权重');ax.legend(fontsize=9)
fig.suptitle('图 8-1  定向盲区与混合后验更新（固定观测示例，非实验统计）',fontsize=13)
save(fig,4,'p4_belief_update')

# Station coordinates and a reproducible probe example.
b=mixed();update(b,first,'direction',0)
for p in (Point(1200,200),Point(1250,-200)):update(b,p,'no_signal')
marg=b.target_marginals();depths=sorted((p.x,w) for p,w in marg);cum=np.cumsum([w for _,w in depths]);quantiles=[depths[min(np.searchsorted(cum,q*cum[-1]),len(depths)-1)][0] for q in (.25,.5,.75)]
probe=plan_probe(b,first,0,Point(1250,-200),[first,Point(1200,200),Point(1250,-200)])
assert probe is not None
VALUES['probe_example']={'quantiles':quantiles,'selected':xy(probe.point)}
station_data,rout,rin,*_=generate_stations()
fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
ax=axs[0];circle(ax,first,1800,'black','目标边界 1800 m','-')
for radius in (rout,rin):circle(ax,first,radius,'gray')
for subset,marker,color,label in [(station_data[1:9],'^','#e67700',f'外圈 {rout:.2f} m'),(station_data[9:],'o','#1565c0',f'内圈 {rin:.2f} m')]:
    ax.scatter([s[0] for s in subset],[s[1] for s in subset],marker=marker,c=color,s=45,label=label)
ax.scatter(0,0,c='black',marker='s',label='中心站');ax.set_xlim(-2250,2250);ax.set_ylim(-2250,2250);axes(ax,'(a) 当前代码生成的 17 站布局');ax.legend(fontsize=9,loc='lower left')
ax=axs[1];ax.axhline(0,color='gray',ls='--',label='首测示向轴')
route=[(0,0),(1200,200),(1250,-200),xy(probe.point)]
ax.plot(*np.array(route).T,ls=':',c='gray',label='示意机动路线')
ax.scatter([1200,1250],[200,-200],marker='x',s=65,c='#c62828',label='连续无信号测点')
for value in sorted(set(quantiles)):
    qs='/'.join(f'{q}%' for q,d in zip((25,50,75),quantiles) if d==value)
    ax.scatter(value,0,s=45,c='#1565c0');ax.annotate(qs,(value,0),xytext=(0,18),textcoords='offset points',ha='center')
ax.scatter(*xy(probe.point),marker='*',s=160,c='#c62828',zorder=5,label='规划器选择的探针')
ax.scatter(0,0,c='black',marker='s');ax.set_xlim(-100,1500);ax.set_ylim(-500,500);axes(ax,'(b) 后验加权分位点与选中探针');ax.legend(fontsize=8,loc='lower left')
fig.suptitle('图 8-2  17 站巡检布局与固定后验示例的探针选点',fontsize=13)
save(fig,4,'p4_inspection_probe')

fig,axs=plt.subplots(1,3,figsize=(15,5),layout='constrained')
for i,g in enumerate(geometries):
    ax=axs[i];mec=minimum_enclosing_circle(g.polygon.convex_hull.vertices)
    polygon(ax,g.polygon,'#1565c0','物理可行域',.4);circle(ax,mec.center,mec.radius,'#23823b',f'最小外接圆 {mec.radius:.2f} m');circle(ax,mec.center,20,'#c62828','20 m 清除圆','-')
    ax.scatter(*xy(mec.center),marker='+',c='black');ax.set_xlim(mec.center.x-26,mec.center.x+26);ax.set_ylim(mec.center.y-26,mec.center.y+26)
    axes(ax,f'({chr(97+i)}) {i+2} 次测向 · 已满足几何覆盖');ax.legend(fontsize=8,loc='lower left')
ax=axs[2];polygon(ax,tri,'#edaf37','独立构造的试射前区域',.3);polygon(ax,rem,'#9c27b0','失败后残余区',.6);circle(ax,first,20,'#c62828','被扣除的 20 m 圆盘','-');ax.set_xlim(-26,26);ax.set_ylim(-26,30);axes(ax,'(c) 独立失败试射示意');ax.legend(fontsize=8,loc='lower left')
fig.suptitle('图 8-3  连续区域最小外接圆与失败扣除（面板 c 为独立示例）',fontsize=13)
save(fig,4,'p4_geometry_clear')
(ROOT/'paper/figure_review_values.json').write_text(json.dumps(VALUES,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(VALUES,ensure_ascii=False,indent=2))
