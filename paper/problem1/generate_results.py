"""从仓库运行日志复现问题一算例，并生成论文表格、结果图。"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cumcm-mpl')
import sys, json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as Patch
from P1 import solve_problem_1_geometry
from utils import Region, Point, DetectionSector
OUT = Path(__file__).resolve().parent
rows=[]
for number,line in enumerate((ROOT/'P3_run_log.jsonl').read_text().splitlines(),1):
    obj=json.loads(line); req=obj.get('request',{}); res=obj.get('response',{})
    if obj.get('path')=='/measure' and req.get('channel')==4 and res.get('accepted') and 'svd_deg' in res:
        p=req['position']; row=(p['x'],p['y'],res['svd_deg'])
        if row[:2] not in [r['measurement'][:2] for r in rows]:
            rows.append({'line':number,'measurement':row})
        if len(rows)==3: break
assert len(rows)==3
data=[r['measurement'] for r in rows]
r=solve_problem_1_geometry(data)
xy=lambda p:[p.x,p.y]
areas=[]
region=Region.disk(Point(0,0),1800)
for x,y,t in data:
    region=region.intersection(DetectionSector.from_measurement(Point(x,y),t).to_region(math.hypot(x,y)+1801))
    areas.append(region.area)
result={'source':'P3_run_log.jsonl','channel':4,'input':rows,'area':r.polygon.area,'vertices':len(r.polygon.vertices),'diameter':r.diameter,'A':xy(r.diameter_endpoints[0]),'B':xy(r.diameter_endpoints[1]),'C':xy(r.diameter_circle_center),'covers':r.diameter_circle_covers,'incremental_areas':areas}
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
with (OUT/'tables.tex').open('w') as f:
    f.write(r'\begin{table}[htbp]\centering\caption{主算例测向输入数据}\label{tab:input}\begin{tabular}{crrrc}\toprule 检测点 & $x_i/\mathrm m$ & $y_i/\mathrm m$ & $\theta_i/(^\circ)$ & 误差范围\\\midrule'+'\n')
    for i,(x,y,t) in enumerate(data,1): f.write(f'$S_{i}$ & {x:.8f} & {y:.8f} & {t:.2f} & $\\pm1^\\circ$ \\\\\n')
    f.write(r'\bottomrule\end{tabular}\par\smallskip{\small 数据来源：仓库 \texttt{P3\_run\_log.jsonl}，频道 4，按顺序选取前三个不同检测位置；计算使用日志原始精度。}\end{table}'+'\n')
    f.write(r'\begin{table}[htbp]\centering\caption{主算例计算结果}\label{tab:result}\begin{tabular}{lr}\toprule 指标 & 数值\\\midrule'+'\n')
    vals=[('定位区域面积 / $\\mathrm{m}^2$',f'{r.polygon.area:.2f}'),('边界顶点数',str(len(r.polygon.vertices))),('直径 $D$ / $\\mathrm m$',f'{r.diameter:.2f}')]
    vals += [(f'{name} / $\\mathrm m$',f'$({p.x:.2f}, {p.y:.2f})$') for name,p in [('端点 $A$',r.diameter_endpoints[0]),('端点 $B$',r.diameter_endpoints[1]),('圆心 $C$',r.diameter_circle_center)]]
    vals += [('半径 $D/2$ / $\\mathrm m$',f'{r.diameter/2:.2f}'),('直径圆是否覆盖','是' if r.diameter_circle_covers else '否')]
    for k,v in vals: f.write(f'{k} & {v} \\\\\n')
    f.write(r'\bottomrule\end{tabular}\end{table}'+'\n')
(OUT/'numbers.tex').write_text('\\newcommand{\\CaseDiameter}{%.2f}\n\\newcommand{\\CaseRadius}{%.2f}\n\\newcommand{\\FirstArea}{%.2f}\n\\newcommand{\\SecondArea}{%.2f}\n\\newcommand{\\FinalArea}{%.2f}\n'%(r.diameter,r.diameter/2,*areas))
fig,axs=plt.subplots(1,2,figsize=(10.5,4.7),gridspec_kw={'width_ratios':[1,1]})
verts=[xy(p) for p in r.polygon.vertices]
for ax in axs:
    ax.set_aspect('equal'); ax.set_xlabel('x / m (East)'); ax.set_ylabel('y / m (North)'); ax.grid(alpha=.2)
    ax.add_patch(Patch(verts,facecolor='#c0392b',alpha=.3,edgecolor='#c0392b'))
ax=axs[0]; ax.add_patch(Circle((0,0),1800,fill=False,color='#3670b2'))
for i,(x,y,t) in enumerate(data,1):
    ax.plot(x,y,'o',color='#27825c'); ax.annotate(f'$S_{i}$',(x,y),xytext=(5,5),textcoords='offset points')
    L=math.hypot(x,y)+1801
    for dt in [-1,0,1]:
        ang=math.radians(t+dt); ax.plot([x,x+L*math.cos(ang)],[y,y+L*math.sin(ang)],color='#e67e22',ls='--' if dt else '-',lw=.65)
ax.set_xlim(-1900,1900);ax.set_ylim(-1900,1900);ax.set_title('(a) Measurement geometry')
ax=axs[1]; a,b=r.diameter_endpoints;c=r.diameter_circle_center
ax.add_patch(Circle(xy(c),r.diameter/2,fill=False,color='#3670b2',ls='--'))
ax.plot([a.x,b.x],[a.y,b.y],color='#c0392b',lw=2)
for label,p in [('A',a),('B',b),('C',c)]:
    ax.plot(p.x,p.y,'ko',ms=3);ax.annotate(label,xy(p),xytext=(5,5),textcoords='offset points')
for p in r.polygon.vertices:
    if c.distance_to(p)>r.diameter/2+max(1e-9,r.diameter*1e-10): ax.plot(p.x,p.y,'o',color='#8e44ad')
d=r.diameter*.7; ax.set_xlim(c.x-d,c.x+d);ax.set_ylim(c.y-d,c.y+d);ax.set_title(f'(b) D = {r.diameter:.2f} m; covers: {r.diameter_circle_covers}')
fig.tight_layout();fig.savefig(OUT/'figures/case.pdf');fig.savefig(OUT/'figures/case.png',dpi=180)
print(json.dumps(result,ensure_ascii=False,indent=2))
